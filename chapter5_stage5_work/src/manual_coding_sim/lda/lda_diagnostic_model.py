"""Диагностические LDA-модели главы 4.

Модуль реализует вспомогательные модели ``LDA_diag`` и ``LDA_full``.
Обе модели предназначены только для анализа устойчивости и постфактум
интерпретации структуры признаков. Они не являются входом для априорного
прогноза качества и не должны перезаписывать артефакты ``LDA_prior``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

from manual_coding_sim.lda.config import LdaTokenizationConfig
from manual_coding_sim.lda.leakage_guard import LeakageGuard
from manual_coding_sim.lda.matrix_builder import (
    LdaDocumentMetadata,
    LdaMatrixBuilder,
    LdaMatrixBuildResult,
    LdaVocabularyItem,
)
from manual_coding_sim.lda.tokenization import (
    DEFAULT_IDENTIFIER_COLUMNS,
    FeatureTokenizer,
    TokenizedFeature,
)

_DIAGNOSTIC_KINDS = frozenset({"diag", "full"})


@dataclass(frozen=True)
class LdaDiagnosticModelConfig:
    """Параметры обучения одной диагностической LDA-модели."""

    diagnostic_kind: str
    n_components: int
    tokenization: LdaTokenizationConfig = LdaTokenizationConfig()
    doc_topic_prior: float | None = None
    topic_word_prior: float | None = None
    learning_method: str = "batch"
    max_iter: int = 100
    random_state: int = 42
    evaluate_every: int = -1
    n_jobs: int | None = None
    encoding: str = "utf-8"
    overwrite: bool = True

    def validate(self) -> None:
        """Проверить корректность параметров диагностической модели."""

        if self.diagnostic_kind not in _DIAGNOSTIC_KINDS:
            msg = "diagnostic_kind должен иметь значение 'diag' или 'full'."
            raise ValueError(msg)
        if self.n_components < 2:
            msg = "n_components должен быть не меньше 2."
            raise ValueError(msg)
        allowed_methods = {"batch", "online"}
        if self.learning_method not in allowed_methods:
            msg = (
                "learning_method должен иметь значение "
                f"из множества {sorted(allowed_methods)}."
            )
            raise ValueError(msg)
        if self.max_iter < 1:
            msg = "max_iter должен быть положительным целым числом."
            raise ValueError(msg)
        if self.doc_topic_prior is not None and self.doc_topic_prior <= 0:
            msg = "doc_topic_prior должен быть положительным числом или None."
            raise ValueError(msg)
        if self.topic_word_prior is not None and self.topic_word_prior <= 0:
            msg = "topic_word_prior должен быть положительным числом или None."
            raise ValueError(msg)
        self.tokenization.validate()


@dataclass(frozen=True)
class LdaDiagnosticTrainingResult:
    """Пути к артефактам, созданным диагностической LDA-моделью."""

    diagnostic_kind: str
    model_name: str
    model_path: Path
    theta_path: Path
    topic_word_path: Path
    metadata_path: Path
    corpus_path: Path
    dictionary_path: Path
    token_map_path: Path
    corpus_metadata_path: Path
    document_count: int
    token_count: int
    n_components: int
    corpus_hash: str
    model_hash: str
    allowed_for_apriori_forecast: bool


class LdaDiagnosticModel:
    """Обучает ``LDA_diag`` или ``LDA_full`` в изолированном режиме."""

    def __init__(self, config: LdaDiagnosticModelConfig) -> None:
        """Создать объект обучения диагностической LDA-модели."""

        config.validate()
        self.config = config
        self.matrix_builder = LdaMatrixBuilder()
        self.leakage_guard = LeakageGuard()

    def fit_from_csv(
        self,
        prior_features_path: str | Path,
        extension_features_path: str | Path,
        data_dir: str | Path,
        models_dir: str | Path,
        reports_dir: str | Path,
    ) -> LdaDiagnosticTrainingResult:
        """Построить диагностический корпус и обучить LDA-модель.

        ``prior_features_path`` всегда проверяется как априорный источник.
        ``extension_features_path`` используется только в диагностическом
        режиме и не может породить ``theta_prior.csv`` или ``lda_prior.joblib``.
        """

        prior_path = Path(prior_features_path)
        extension_path = Path(extension_features_path)
        data_path = Path(data_dir)
        models_path = Path(models_dir)
        reports_path = Path(reports_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        models_path.mkdir(parents=True, exist_ok=True)
        reports_path.mkdir(parents=True, exist_ok=True)

        prior_rows = self._read_csv(prior_path)
        extension_rows = self._read_csv(extension_path)
        self._validate_prior_rows(prior_path, prior_rows)
        merged_rows = self._merge_rows(prior_rows, extension_rows)

        corpus_artifacts = self._build_and_save_corpus(
            rows=merged_rows,
            data_dir=data_path,
            prior_path=prior_path,
            extension_path=extension_path,
        )
        matrix_result = self.matrix_builder.build_from_files(
            corpus_path=corpus_artifacts["corpus_path"],
            dictionary_path=corpus_artifacts["dictionary_path"],
            metadata_path=corpus_artifacts["metadata_path"],
        )
        self._validate_matrix(matrix_result)

        model = self._create_model()
        theta = model.fit_transform(matrix_result.matrix)
        theta = self._normalize_rows(theta)
        topic_word = self._normalize_rows(model.components_)

        model_path = models_path / f"lda_{self.config.diagnostic_kind}.joblib"
        theta_path = reports_path / f"theta_{self.config.diagnostic_kind}.csv"
        topic_word_path = reports_path / f"topic_word_{self.config.diagnostic_kind}.csv"
        metadata_path = reports_path / "lda_diagnostic_metadata.json"
        self._ensure_can_write(
            paths=[
                model_path,
                theta_path,
                topic_word_path,
                metadata_path,
            ]
        )
        self._ensure_not_prior_artifacts(
            paths=[model_path, theta_path, topic_word_path, metadata_path]
        )

        joblib.dump(model, model_path)
        self._write_theta_csv(theta_path, matrix_result.documents, theta)
        self._write_topic_word_csv(
            topic_word_path,
            matrix_result.vocabulary,
            topic_word,
        )
        model_hash = self._calculate_file_hash(model_path)
        metadata = self._build_metadata(
            matrix_result=matrix_result,
            prior_path=prior_path,
            extension_path=extension_path,
            corpus_hash=str(corpus_artifacts["corpus_hash"]),
            model_hash=model_hash,
        )
        self._write_diagnostic_metadata(metadata_path, metadata)

        return LdaDiagnosticTrainingResult(
            diagnostic_kind=self.config.diagnostic_kind,
            model_name=self._model_name,
            model_path=model_path,
            theta_path=theta_path,
            topic_word_path=topic_word_path,
            metadata_path=metadata_path,
            corpus_path=Path(corpus_artifacts["corpus_path"]),
            dictionary_path=Path(corpus_artifacts["dictionary_path"]),
            token_map_path=Path(corpus_artifacts["token_map_path"]),
            corpus_metadata_path=Path(corpus_artifacts["metadata_path"]),
            document_count=matrix_result.matrix.shape[0],
            token_count=matrix_result.matrix.shape[1],
            n_components=self.config.n_components,
            corpus_hash=str(corpus_artifacts["corpus_hash"]),
            model_hash=model_hash,
            allowed_for_apriori_forecast=False,
        )

    @property
    def _model_name(self) -> str:
        """Вернуть научное имя диагностической модели."""

        if self.config.diagnostic_kind == "diag":
            return "LDA_diag"
        return "LDA_full"

    @property
    def _extension_prefix(self) -> str:
        """Вернуть префикс для признаков дополнительного источника."""

        if self.config.diagnostic_kind == "diag":
            return "diag"
        return "fact"

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        """Прочитать CSV-файл в список словарей."""

        if not path.exists():
            msg = f"Файл диагностических входных данных не найден: {path}"
            raise FileNotFoundError(msg)
        with path.open("r", encoding=self.config.encoding, newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            if reader.fieldnames is None:
                msg = f"Файл {path} не содержит заголовок CSV."
                raise ValueError(msg)
            return [dict(row) for row in reader]

    def _validate_prior_rows(
        self,
        prior_path: Path,
        prior_rows: Sequence[Mapping[str, str]],
    ) -> None:
        """Проверить априорную часть диагностического корпуса."""

        if not prior_rows:
            msg = "Априорная часть диагностического корпуса не должна быть пустой."
            raise ValueError(msg)
        self.leakage_guard.validate_prior_input(
            source_paths=[prior_path],
            columns=prior_rows[0].keys(),
        )

    def _merge_rows(
        self,
        prior_rows: Sequence[Mapping[str, str]],
        extension_rows: Sequence[Mapping[str, str]],
    ) -> list[dict[str, object]]:
        """Объединить априорные и диагностические признаки построчно."""

        if not extension_rows:
            msg = "Дополнительная часть диагностического корпуса не должна быть пустой."
            raise ValueError(msg)
        if len(prior_rows) != len(extension_rows):
            msg = "Число строк prior и diagnostic/full признаков должно совпадать."
            raise ValueError(msg)

        merged_rows: list[dict[str, object]] = []
        for row_index, (prior_row, extension_row) in enumerate(
            zip(prior_rows, extension_rows, strict=True)
        ):
            self._validate_identifier_compatibility(
                prior_row=prior_row,
                extension_row=extension_row,
                row_index=row_index,
            )
            merged_row = dict(prior_row)
            for column, value in extension_row.items():
                if column in DEFAULT_IDENTIFIER_COLUMNS:
                    continue
                merged_row[self._extension_column_name(column)] = value
            merged_rows.append(merged_row)
        return merged_rows

    def _validate_identifier_compatibility(
        self,
        prior_row: Mapping[str, object],
        extension_row: Mapping[str, object],
        row_index: int,
    ) -> None:
        """Проверить совпадение общих идентификаторов двух строк."""

        checked_columns = ("run_id", "protocol_id", "scenario_id")
        for column in checked_columns:
            prior_value = str(prior_row.get(column, ""))
            extension_value = str(extension_row.get(column, ""))
            if prior_value and extension_value and prior_value != extension_value:
                msg = (
                    f"Несовпадение идентификатора {column} в строке {row_index}: "
                    f"{prior_value!r} != {extension_value!r}."
                )
                raise ValueError(msg)

    def _extension_column_name(self, column: str) -> str:
        """Сформировать безопасное имя диагностического признака."""

        prefix = self._extension_prefix
        if column.startswith(f"{prefix}_"):
            return column
        return f"{prefix}_{column}"

    def _build_and_save_corpus(
        self,
        rows: Sequence[Mapping[str, object]],
        data_dir: Path,
        prior_path: Path,
        extension_path: Path,
    ) -> dict[str, object]:
        """Построить и сохранить корпус диагностической модели."""

        tokenizer = FeatureTokenizer(config=self.config.tokenization)
        tokenizer.fit(rows)
        tokenized_documents = tokenizer.transform(rows)
        document_token_counts = self._count_tokens_by_document(tokenized_documents)
        document_frequency = self._calculate_document_frequency(document_token_counts)
        dictionary = self._build_dictionary(document_frequency, len(rows))
        if not dictionary:
            msg = "После фильтрации диагностический словарь оказался пустым."
            raise ValueError(msg)

        filtered_counts = self._filter_document_counts(document_token_counts, dictionary)
        corpus_rows = self._build_corpus_rows(rows, filtered_counts, dictionary)
        if not corpus_rows:
            msg = "После фильтрации диагностический корпус оказался пустым."
            raise ValueError(msg)

        suffix = self.config.diagnostic_kind
        corpus_path = data_dir / f"corpus_{suffix}.csv"
        dictionary_path = data_dir / f"dictionary_{suffix}.json"
        token_map_path = data_dir / f"token_map_{suffix}.json"
        metadata_path = data_dir / f"corpus_metadata_{suffix}.json"
        corpus_hash = self._calculate_corpus_hash(corpus_rows)

        dictionary_payload = self._build_dictionary_payload(dictionary)
        token_map_payload = tokenizer.to_token_map()
        token_map_payload["model_name"] = self._model_name
        token_map_payload["diagnostic_kind"] = self.config.diagnostic_kind
        metadata_payload = self._build_corpus_metadata(
            rows=rows,
            document_frequency=document_frequency,
            dictionary=dictionary,
            prior_path=prior_path,
            extension_path=extension_path,
            corpus_hash=corpus_hash,
        )
        self._ensure_can_write(
            paths=[corpus_path, dictionary_path, token_map_path, metadata_path]
        )
        self._write_corpus_csv(corpus_path, corpus_rows)
        self._write_json(dictionary_path, dictionary_payload)
        self._write_json(token_map_path, token_map_payload)
        self._write_json(metadata_path, metadata_payload)

        return {
            "corpus_path": corpus_path,
            "dictionary_path": dictionary_path,
            "token_map_path": token_map_path,
            "metadata_path": metadata_path,
            "corpus_hash": corpus_hash,
        }

    def _count_tokens_by_document(
        self,
        tokenized_documents: Sequence[Sequence[TokenizedFeature]],
    ) -> list[dict[str, int]]:
        """Посчитать частоты токенов внутри каждого документа."""

        result: list[dict[str, int]] = []
        for document_tokens in tokenized_documents:
            counts: dict[str, int] = {}
            for tokenized_feature in document_tokens:
                token = tokenized_feature.token
                counts[token] = counts.get(token, 0) + 1
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
        """Построить диагностический словарь после фильтрации токенов."""

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
        """Удалить из документов токены, не вошедшие в словарь."""

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
        source_rows: Sequence[Mapping[str, object]],
        filtered_counts: Sequence[Mapping[str, int]],
        dictionary: Mapping[str, Mapping[str, int]],
    ) -> list[dict[str, object]]:
        """Сформировать строки длинного диагностического CSV-корпуса."""

        corpus_rows: list[dict[str, object]] = []
        for document_index, token_counts in enumerate(filtered_counts):
            source_row = source_rows[document_index]
            for token in sorted(token_counts):
                dictionary_row = dictionary[token]
                corpus_rows.append(
                    {
                        "document_index": document_index,
                        "run_id": source_row.get("run_id", ""),
                        "protocol_id": source_row.get(
                            "protocol_id", f"doc_{document_index:06d}"
                        ),
                        "scenario_id": source_row.get("scenario_id", ""),
                        "token_id": dictionary_row["token_id"],
                        "token": token,
                        "count": int(token_counts[token]),
                    }
                )
        return corpus_rows

    def _build_dictionary_payload(
        self,
        dictionary: Mapping[str, Mapping[str, int]],
    ) -> dict[str, object]:
        """Сформировать JSON-представление диагностического словаря."""

        tokens = [
            {
                "token": token,
                "token_id": int(payload["token_id"]),
                "document_frequency": int(payload["document_frequency"]),
            }
            for token, payload in sorted(
                dictionary.items(), key=lambda item: item[1]["token_id"]
            )
        ]
        return {
            "model_name": self._model_name,
            "diagnostic_kind": self.config.diagnostic_kind,
            "tokens": tokens,
        }

    def _build_corpus_metadata(
        self,
        rows: Sequence[Mapping[str, object]],
        document_frequency: Mapping[str, int],
        dictionary: Mapping[str, Mapping[str, int]],
        prior_path: Path,
        extension_path: Path,
        corpus_hash: str,
    ) -> dict[str, object]:
        """Сформировать метаданные диагностического корпуса."""

        return {
            "model_name": self._model_name,
            "diagnostic_kind": self.config.diagnostic_kind,
            "diagnostic_only": True,
            "allowed_for_apriori_forecast": False,
            "document_count": len(rows),
            "token_count_before_filter": len(document_frequency),
            "token_count_after_filter": len(dictionary),
            "df_min": self.config.tokenization.df_min,
            "df_max_ratio": self.config.tokenization.df_max_ratio,
            "numeric_strategy": self.config.tokenization.numeric_strategy,
            "numeric_bins": self.config.tokenization.numeric_bins,
            "prior_features_path": str(prior_path),
            "extension_features_path": str(extension_path),
            "extension_prefix": self._extension_prefix,
            "corpus_hash": corpus_hash,
        }

    def _create_model(self) -> LatentDirichletAllocation:
        """Создать объект scikit-learn LDA для диагностического режима."""

        return LatentDirichletAllocation(
            n_components=self.config.n_components,
            doc_topic_prior=self.config.doc_topic_prior,
            topic_word_prior=self.config.topic_word_prior,
            learning_method=self.config.learning_method,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
            evaluate_every=self.config.evaluate_every,
            n_jobs=self.config.n_jobs,
        )

    def _validate_matrix(self, matrix_result: LdaMatrixBuildResult) -> None:
        """Проверить пригодность диагностической матрицы для обучения."""

        matrix = matrix_result.matrix
        if matrix.ndim != 2:
            msg = "Матрица диагностической LDA должна быть двумерной."
            raise ValueError(msg)
        if matrix.shape[0] < 2:
            msg = "Для диагностической LDA требуется не менее двух документов."
            raise ValueError(msg)
        if matrix.shape[1] < 2:
            msg = "Для диагностической LDA требуется не менее двух токенов."
            raise ValueError(msg)
        if self.config.n_components > matrix.shape[1]:
            msg = "n_components не должен превышать число токенов словаря."
            raise ValueError(msg)
        if np.any(matrix < 0):
            msg = "Матрица диагностической LDA не должна содержать отрицательных значений."
            raise ValueError(msg)
        empty_documents = np.where(matrix.sum(axis=1) <= 0)[0]
        if empty_documents.size:
            msg = "В диагностическом LDA-корпусе есть документы без токенов: "
            msg += ", ".join(str(int(index)) for index in empty_documents)
            raise ValueError(msg)

    def _normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        """Нормировать строки матрицы до суммы 1."""

        row_sums = matrix.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            msg = "Невозможно нормировать строку с нулевой суммой."
            raise ValueError(msg)
        return matrix / row_sums

    def _write_theta_csv(
        self,
        path: Path,
        documents: Sequence[LdaDocumentMetadata],
        theta: np.ndarray,
    ) -> None:
        """Сохранить диагностические распределения ``θ`` по документам."""

        theta_columns = [f"theta_{index}" for index in range(theta.shape[1])]
        fieldnames = [
            "model_name",
            "diagnostic_kind",
            "document_index",
            "run_id",
            "protocol_id",
            "scenario_id",
            *theta_columns,
            "selected_k",
            "doc_topic_prior",
            "topic_word_prior",
            "random_state",
            "diagnostic_only",
            "allowed_for_apriori_forecast",
        ]
        with path.open("w", encoding=self.config.encoding, newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for document, theta_row in zip(documents, theta, strict=True):
                row: dict[str, object] = {
                    "model_name": self._model_name,
                    "diagnostic_kind": self.config.diagnostic_kind,
                    "document_index": document.document_index,
                    "run_id": document.run_id,
                    "protocol_id": document.protocol_id,
                    "scenario_id": document.scenario_id,
                    "selected_k": self.config.n_components,
                    "doc_topic_prior": self._format_optional_float(
                        self.config.doc_topic_prior
                    ),
                    "topic_word_prior": self._format_optional_float(
                        self.config.topic_word_prior
                    ),
                    "random_state": self.config.random_state,
                    "diagnostic_only": "true",
                    "allowed_for_apriori_forecast": "false",
                }
                for column_name, value in zip(theta_columns, theta_row, strict=True):
                    row[column_name] = f"{float(value):.12g}"
                writer.writerow(row)

    def _write_topic_word_csv(
        self,
        path: Path,
        vocabulary: Sequence[LdaVocabularyItem],
        topic_word: np.ndarray,
    ) -> None:
        """Сохранить диагностические распределения ``φ_k`` по токенам."""

        fieldnames = [
            "model_name",
            "diagnostic_kind",
            "topic_id",
            "token_id",
            "token",
            "document_frequency",
            "weight",
        ]
        with path.open("w", encoding=self.config.encoding, newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for topic_id in range(topic_word.shape[0]):
                for vocabulary_item in vocabulary:
                    writer.writerow(
                        {
                            "model_name": self._model_name,
                            "diagnostic_kind": self.config.diagnostic_kind,
                            "topic_id": topic_id,
                            "token_id": vocabulary_item.token_id,
                            "token": vocabulary_item.token,
                            "document_frequency": vocabulary_item.document_frequency,
                            "weight": f"{float(topic_word[topic_id, vocabulary_item.token_id]):.12g}",
                        }
                    )

    def _build_metadata(
        self,
        matrix_result: LdaMatrixBuildResult,
        prior_path: Path,
        extension_path: Path,
        corpus_hash: str,
        model_hash: str,
    ) -> dict[str, object]:
        """Сформировать метаданные обучения диагностической модели."""

        return {
            "model_name": self._model_name,
            "diagnostic_kind": self.config.diagnostic_kind,
            "diagnostic_only": True,
            "allowed_for_apriori_forecast": False,
            "n_components": self.config.n_components,
            "doc_topic_prior": self.config.doc_topic_prior,
            "topic_word_prior": self.config.topic_word_prior,
            "learning_method": self.config.learning_method,
            "max_iter": self.config.max_iter,
            "random_state": self.config.random_state,
            "evaluate_every": self.config.evaluate_every,
            "document_count": int(matrix_result.matrix.shape[0]),
            "token_count": int(matrix_result.matrix.shape[1]),
            "corpus_hash": corpus_hash,
            "matrix_corpus_hash": matrix_result.corpus_hash,
            "model_hash": model_hash,
            "prior_features_path": str(prior_path),
            "extension_features_path": str(extension_path),
            "theta_path": f"theta_{self.config.diagnostic_kind}.csv",
            "topic_word_path": f"topic_word_{self.config.diagnostic_kind}.csv",
            "model_path": f"lda_{self.config.diagnostic_kind}.joblib",
            "theta_columns": [
                f"theta_{index}" for index in range(self.config.n_components)
            ],
        }

    def _write_diagnostic_metadata(
        self,
        path: Path,
        metadata: Mapping[str, object],
    ) -> None:
        """Сохранить агрегированные метаданные диагностических моделей."""

        payload: dict[str, object]
        if path.exists() and self.config.overwrite:
            with path.open("r", encoding=self.config.encoding) as file_obj:
                loaded = json.load(file_obj)
            payload = loaded if isinstance(loaded, dict) else {}
        else:
            payload = {}
        payload.setdefault("diagnostic_only", True)
        payload.setdefault("allowed_for_apriori_forecast", False)
        models = payload.setdefault("models", {})
        if not isinstance(models, dict):
            models = {}
            payload["models"] = models
        models[self.config.diagnostic_kind] = dict(metadata)
        self._write_json(path, payload)

    def _calculate_corpus_hash(self, corpus_rows: Sequence[Mapping[str, object]]) -> str:
        """Рассчитать стабильный SHA-256 хеш строк диагностического корпуса."""

        payload = json.dumps(list(corpus_rows), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _calculate_file_hash(self, path: Path) -> str:
        """Рассчитать SHA-256 хеш сохраненной модели."""

        digest = hashlib.sha256()
        with path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _write_corpus_csv(
        self,
        path: Path,
        corpus_rows: Sequence[Mapping[str, object]],
    ) -> None:
        """Сохранить диагностический корпус в длинном CSV-формате."""

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

    def _format_optional_float(self, value: float | None) -> str:
        """Преобразовать необязательный числовой параметр в CSV-строку."""

        if value is None:
            return ""
        return f"{float(value):.12g}"

    def _ensure_can_write(self, paths: Sequence[Path]) -> None:
        """Проверить, что существующие диагностические файлы можно перезаписать."""

        if self.config.overwrite:
            return
        existing_paths = [path for path in paths if path.exists()]
        if existing_paths:
            msg = "Запрещена перезапись существующих диагностических файлов: "
            msg += ", ".join(str(path) for path in existing_paths)
            raise FileExistsError(msg)

    def _ensure_not_prior_artifacts(self, paths: Sequence[Path]) -> None:
        """Защитить основные артефакты ``LDA_prior`` от случайной записи."""

        forbidden_names = {"lda_prior.joblib", "theta_prior.csv", "topic_word.csv"}
        conflicting = [path for path in paths if path.name in forbidden_names]
        if conflicting:
            msg = "Диагностическая модель не должна записывать артефакты LDA_prior: "
            msg += ", ".join(str(path) for path in conflicting)
            raise RuntimeError(msg)
