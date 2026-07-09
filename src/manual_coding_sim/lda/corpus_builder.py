"""Построение LDA-корпуса по априорным признакам главы 3.

Построитель корпуса читает ``prior_features.csv``, проверяет отсутствие
утечки фактических данных, преобразует признаки в токены и сохраняет набор
артефактов, необходимых для последующего обучения ``LDA_prior``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from manual_coding_sim.lda.config import LdaTokenizationConfig
from manual_coding_sim.lda.leakage_guard import LeakageGuard
from manual_coding_sim.lda.tokenization import FeatureTokenizer, TokenizedFeature


@dataclass(frozen=True)
class LdaCorpusBuilderConfig:
    """Параметры построения корпуса для основной LDA-модели."""

    tokenization: LdaTokenizationConfig = LdaTokenizationConfig()
    overwrite: bool = True
    encoding: str = "utf-8"


@dataclass(frozen=True)
class LdaCorpusBuildResult:
    """Результат построения корпуса и пути к сохраненным артефактам."""

    document_count: int
    token_count_before_filter: int
    token_count_after_filter: int
    dictionary_path: Path
    token_map_path: Path
    corpus_path: Path
    metadata_path: Path
    leakage_report_path: Path
    corpus_hash: str


class LdaCorpusBuilder:
    """Строит токенизированный корпус ``LDA_prior`` из ``prior_features.csv``."""

    def __init__(
        self,
        config: LdaCorpusBuilderConfig | None = None,
        leakage_guard: LeakageGuard | None = None,
    ) -> None:
        """Создать построитель корпуса главы 4."""

        self.config = config or LdaCorpusBuilderConfig()
        self.leakage_guard = leakage_guard or LeakageGuard()

    def build_from_csv(
        self,
        prior_features_path: str | Path,
        output_dir: str | Path,
    ) -> LdaCorpusBuildResult:
        """Построить и сохранить корпус по CSV-файлу априорных признаков."""

        prior_path = Path(prior_features_path)
        output_path = Path(output_dir)
        rows = self._read_csv(prior_path)
        if not rows:
            msg = f"Файл {prior_path} не содержит строк априорных признаков."
            raise ValueError(msg)

        self.leakage_guard.validate_prior_input(
            source_paths=[prior_path],
            columns=rows[0].keys(),
        )
        output_path.mkdir(parents=True, exist_ok=True)

        tokenizer = FeatureTokenizer(config=self.config.tokenization).fit(rows)
        tokenized_documents = tokenizer.transform(rows)
        document_token_counts = self._count_tokens_by_document(tokenized_documents)
        document_frequency = self._calculate_document_frequency(document_token_counts)
        dictionary = self._build_dictionary(document_frequency, len(rows))
        filtered_counts = self._filter_document_counts(document_token_counts, dictionary)
        corpus_rows = self._build_corpus_rows(rows, filtered_counts, dictionary)
        corpus_hash = self._calculate_corpus_hash(corpus_rows)

        token_map_path = output_path / "token_map.json"
        dictionary_path = output_path / "dictionary.json"
        corpus_path = output_path / "corpus_prior.csv"
        metadata_path = output_path / "corpus_metadata.json"
        leakage_report_path = output_path / "leakage_report.json"

        self._ensure_can_write(
            paths=[
                token_map_path,
                dictionary_path,
                corpus_path,
                metadata_path,
                leakage_report_path,
            ]
        )
        self._write_json(token_map_path, tokenizer.to_token_map())
        self._write_json(dictionary_path, self._dictionary_to_json(dictionary))
        self._write_corpus_csv(corpus_path, corpus_rows)

        metadata = self._build_metadata(
            prior_path=prior_path,
            document_count=len(rows),
            token_count_before_filter=sum(sum(counts.values()) for counts in document_token_counts),
            token_count_after_filter=sum(row["count"] for row in corpus_rows),
            dictionary=dictionary,
            corpus_hash=corpus_hash,
        )
        self._write_json(metadata_path, metadata)
        leakage_result = self.leakage_guard.check_prior_input(
            source_paths=[prior_path],
            columns=rows[0].keys(),
        )
        self._write_json(leakage_report_path, leakage_result.to_dict())

        return LdaCorpusBuildResult(
            document_count=len(rows),
            token_count_before_filter=metadata["token_count_before_filter"],
            token_count_after_filter=metadata["token_count_after_filter"],
            dictionary_path=dictionary_path,
            token_map_path=token_map_path,
            corpus_path=corpus_path,
            metadata_path=metadata_path,
            leakage_report_path=leakage_report_path,
            corpus_hash=corpus_hash,
        )

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        """Прочитать CSV-файл в список словарей."""

        if not path.exists():
            msg = f"Файл априорных признаков не найден: {path}"
            raise FileNotFoundError(msg)
        with path.open("r", encoding=self.config.encoding, newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            if reader.fieldnames is None:
                msg = f"Файл {path} не содержит заголовок CSV."
                raise ValueError(msg)
            return [dict(row) for row in reader]

    def _count_tokens_by_document(
        self,
        tokenized_documents: Sequence[Sequence[TokenizedFeature]],
    ) -> list[dict[str, int]]:
        """Посчитать частоты токенов внутри каждого документа."""

        result: list[dict[str, int]] = []
        for document_tokens in tokenized_documents:
            counts: dict[str, int] = {}
            for tokenized_feature in document_tokens:
                counts[tokenized_feature.token] = counts.get(tokenized_feature.token, 0) + 1
            result.append(counts)
        return result

    def _calculate_document_frequency(
        self,
        document_token_counts: Sequence[Mapping[str, int]],
    ) -> dict[str, int]:
        """Рассчитать документную частоту каждого токена."""

        document_frequency: dict[str, int] = {}
        for token_counts in document_token_counts:
            for token in token_counts:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        return document_frequency

    def _build_dictionary(
        self,
        document_frequency: Mapping[str, int],
        document_count: int,
    ) -> dict[str, dict[str, int]]:
        """Построить словарь токенов после фильтрации по document frequency."""

        max_document_frequency = max(
            1,
            int(document_count * self.config.tokenization.df_max_ratio),
        )
        allowed_tokens = [
            token
            for token, frequency in document_frequency.items()
            if self.config.tokenization.df_min <= frequency <= max_document_frequency
        ]
        dictionary: dict[str, dict[str, int]] = {}
        for token_id, token in enumerate(sorted(allowed_tokens)):
            dictionary[token] = {
                "token_id": token_id,
                "document_frequency": int(document_frequency[token]),
            }
        return dictionary

    def _filter_document_counts(
        self,
        document_token_counts: Sequence[Mapping[str, int]],
        dictionary: Mapping[str, Mapping[str, int]],
    ) -> list[dict[str, int]]:
        """Удалить из документов токены, не вошедшие в итоговый словарь."""

        allowed_tokens = set(dictionary)
        result: list[dict[str, int]] = []
        for token_counts in document_token_counts:
            filtered = {
                token: count
                for token, count in token_counts.items()
                if token in allowed_tokens
            }
            result.append(filtered)
        return result

    def _build_corpus_rows(
        self,
        source_rows: Sequence[Mapping[str, str]],
        filtered_counts: Sequence[Mapping[str, int]],
        dictionary: Mapping[str, Mapping[str, int]],
    ) -> list[dict[str, object]]:
        """Сформировать строки длинного CSV-корпуса."""

        corpus_rows: list[dict[str, object]] = []
        for document_index, token_counts in enumerate(filtered_counts):
            source_row = source_rows[document_index]
            for token in sorted(token_counts):
                dictionary_row = dictionary[token]
                corpus_rows.append(
                    {
                        "document_index": document_index,
                        "run_id": source_row.get("run_id", ""),
                        "protocol_id": source_row.get("protocol_id", f"doc_{document_index:06d}"),
                        "scenario_id": source_row.get("scenario_id", ""),
                        "token_id": dictionary_row["token_id"],
                        "token": token,
                        "count": int(token_counts[token]),
                    }
                )
        return corpus_rows

    def _dictionary_to_json(
        self,
        dictionary: Mapping[str, Mapping[str, int]],
    ) -> dict[str, object]:
        """Преобразовать словарь токенов в JSON-структуру."""

        tokens = [
            {
                "token_id": values["token_id"],
                "token": token,
                "document_frequency": values["document_frequency"],
            }
            for token, values in sorted(
                dictionary.items(),
                key=lambda item: item[1]["token_id"],
            )
        ]
        return {"token_count": len(tokens), "tokens": tokens}

    def _build_metadata(
        self,
        prior_path: Path,
        document_count: int,
        token_count_before_filter: int,
        token_count_after_filter: int,
        dictionary: Mapping[str, Mapping[str, int]],
        corpus_hash: str,
    ) -> dict[str, object]:
        """Сформировать метаданные построенного корпуса."""

        return {
            "source_path": str(prior_path),
            "document_count": document_count,
            "token_count_before_filter": token_count_before_filter,
            "token_count_after_filter": token_count_after_filter,
            "dictionary_token_count": len(dictionary),
            "df_min": self.config.tokenization.df_min,
            "df_max_ratio": self.config.tokenization.df_max_ratio,
            "numeric_strategy": self.config.tokenization.numeric_strategy,
            "numeric_bins": self.config.tokenization.numeric_bins,
            "corpus_hash": corpus_hash,
        }

    def _calculate_corpus_hash(self, corpus_rows: Sequence[Mapping[str, object]]) -> str:
        """Рассчитать стабильный SHA-256 хеш строк корпуса."""

        payload = json.dumps(list(corpus_rows), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _write_corpus_csv(
        self,
        path: Path,
        corpus_rows: Sequence[Mapping[str, object]],
    ) -> None:
        """Сохранить корпус в длинном CSV-формате."""

        fieldnames = [
            "document_index",
            "run_id",
            "protocol_id",
            "scenario_id",
            "token_id",
            "token",
            "count",
        ]
        with path.open("w", encoding=self.config.encoding, newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(corpus_rows)

    def _write_json(self, path: Path, payload: Mapping[str, object]) -> None:
        """Сохранить JSON-файл с человекочитаемым форматированием."""

        with path.open("w", encoding=self.config.encoding) as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
            file_obj.write("\n")

    def _ensure_can_write(self, paths: Sequence[Path]) -> None:
        """Проверить, что существующие файлы можно перезаписать."""

        if self.config.overwrite:
            return
        existing_paths = [path for path in paths if path.exists()]
        if existing_paths:
            msg = "Запрещена перезапись существующих файлов: "
            msg += ", ".join(str(path) for path in existing_paths)
            raise FileExistsError(msg)
