"""Преобразование длинного LDA-корпуса в матрицу документ--токен.

Модуль не выполняет обучение LDA. Его задача — восстановить из
``corpus_prior.csv`` числовую матрицу, словарь токенов и метаданные
документов, необходимые для обучения основной модели ``LDA_prior``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class LdaDocumentMetadata:
    """Служебные идентификаторы одного документа LDA-корпуса."""

    document_index: int
    run_id: str
    protocol_id: str
    scenario_id: str


@dataclass(frozen=True)
class LdaVocabularyItem:
    """Одна запись словаря токенов LDA-корпуса."""

    token_id: int
    token: str
    document_frequency: int


@dataclass(frozen=True)
class LdaMatrixBuildResult:
    """Результат восстановления матрицы документ--токен."""

    matrix: np.ndarray
    documents: tuple[LdaDocumentMetadata, ...]
    vocabulary: tuple[LdaVocabularyItem, ...]
    corpus_hash: str | None


class LdaMatrixBuilder:
    """Восстанавливает плотную матрицу признаков из длинного CSV-корпуса."""

    required_corpus_columns = {
        "document_index",
        "run_id",
        "protocol_id",
        "scenario_id",
        "token_id",
        "token",
        "count",
    }

    def build_from_files(
        self,
        corpus_path: str | Path,
        dictionary_path: str | Path,
        metadata_path: str | Path | None = None,
    ) -> LdaMatrixBuildResult:
        """Построить матрицу по файлам корпуса, словаря и метаданных."""

        corpus_file = Path(corpus_path)
        dictionary_file = Path(dictionary_path)
        metadata_file = Path(metadata_path) if metadata_path is not None else None

        vocabulary = self._read_dictionary(dictionary_file)
        corpus_rows = self._read_corpus(corpus_file)
        metadata = self._read_metadata(metadata_file)
        document_count = self._resolve_document_count(corpus_rows, metadata)
        matrix = self._build_matrix(corpus_rows, document_count, len(vocabulary))
        documents = self._build_documents(corpus_rows, document_count)

        return LdaMatrixBuildResult(
            matrix=matrix,
            documents=tuple(documents),
            vocabulary=tuple(vocabulary),
            corpus_hash=self._resolve_corpus_hash(metadata),
        )

    def _read_dictionary(self, path: Path) -> list[LdaVocabularyItem]:
        """Прочитать JSON-словарь токенов."""

        if not path.exists():
            msg = f"Файл словаря LDA не найден: {path}"
            raise FileNotFoundError(msg)
        with path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)

        tokens = payload.get("tokens")
        if not isinstance(tokens, list):
            msg = f"Файл {path} не содержит список tokens."
            raise ValueError(msg)

        vocabulary: list[LdaVocabularyItem] = []
        for item in tokens:
            vocabulary.append(
                LdaVocabularyItem(
                    token_id=int(item["token_id"]),
                    token=str(item["token"]),
                    document_frequency=int(item.get("document_frequency", 0)),
                )
            )
        vocabulary.sort(key=lambda item: item.token_id)
        self._validate_vocabulary(vocabulary)
        return vocabulary

    def _validate_vocabulary(self, vocabulary: Sequence[LdaVocabularyItem]) -> None:
        """Проверить непрерывность идентификаторов токенов."""

        expected_ids = list(range(len(vocabulary)))
        actual_ids = [item.token_id for item in vocabulary]
        if actual_ids != expected_ids:
            msg = "Идентификаторы токенов должны идти подряд от 0."
            raise ValueError(msg)

    def _read_corpus(self, path: Path) -> list[dict[str, str]]:
        """Прочитать длинный CSV-корпус."""

        if not path.exists():
            msg = f"Файл корпуса LDA не найден: {path}"
            raise FileNotFoundError(msg)
        with path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            if reader.fieldnames is None:
                msg = f"Файл {path} не содержит заголовок CSV."
                raise ValueError(msg)
            missing_columns = self.required_corpus_columns.difference(reader.fieldnames)
            if missing_columns:
                msg = "В корпусе отсутствуют обязательные колонки: "
                msg += ", ".join(sorted(missing_columns))
                raise ValueError(msg)
            return [dict(row) for row in reader]

    def _read_metadata(self, path: Path | None) -> dict[str, object]:
        """Прочитать метаданные корпуса, если они доступны."""

        if path is None:
            return {}
        if not path.exists():
            msg = f"Файл метаданных LDA-корпуса не найден: {path}"
            raise FileNotFoundError(msg)
        with path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        if not isinstance(payload, dict):
            msg = f"Файл {path} должен содержать JSON-объект."
            raise ValueError(msg)
        return payload

    def _resolve_document_count(
        self,
        corpus_rows: Sequence[Mapping[str, str]],
        metadata: Mapping[str, object],
    ) -> int:
        """Определить число документов с учетом возможных пустых документов."""

        metadata_count = metadata.get("document_count")
        if metadata_count is not None:
            return int(metadata_count)
        if not corpus_rows:
            msg = "Невозможно определить число документов по пустому корпусу без metadata."
            raise ValueError(msg)
        max_index = max(int(row["document_index"]) for row in corpus_rows)
        return max_index + 1

    def _build_matrix(
        self,
        corpus_rows: Sequence[Mapping[str, str]],
        document_count: int,
        token_count: int,
    ) -> np.ndarray:
        """Построить плотную матрицу частот токенов."""

        if document_count < 1:
            msg = "Число документов должно быть положительным."
            raise ValueError(msg)
        if token_count < 1:
            msg = "Словарь LDA не должен быть пустым."
            raise ValueError(msg)

        matrix = np.zeros((document_count, token_count), dtype=np.float64)
        for row in corpus_rows:
            document_index = int(row["document_index"])
            token_id = int(row["token_id"])
            count = float(row["count"])
            if not 0 <= document_index < document_count:
                msg = f"document_index выходит за границы матрицы: {document_index}"
                raise ValueError(msg)
            if not 0 <= token_id < token_count:
                msg = f"token_id выходит за границы словаря: {token_id}"
                raise ValueError(msg)
            if count < 0:
                msg = "Частота токена не может быть отрицательной."
                raise ValueError(msg)
            matrix[document_index, token_id] += count
        return matrix

    def _build_documents(
        self,
        corpus_rows: Sequence[Mapping[str, str]],
        document_count: int,
    ) -> list[LdaDocumentMetadata]:
        """Восстановить метаданные документов по строкам корпуса."""

        by_index: dict[int, LdaDocumentMetadata] = {}
        for row in corpus_rows:
            document_index = int(row["document_index"])
            candidate = LdaDocumentMetadata(
                document_index=document_index,
                run_id=row.get("run_id", ""),
                protocol_id=row.get("protocol_id", f"doc_{document_index:06d}"),
                scenario_id=row.get("scenario_id", ""),
            )
            existing = by_index.get(document_index)
            if existing is not None and existing != candidate:
                msg = f"Для документа {document_index} обнаружены несовместимые метаданные."
                raise ValueError(msg)
            by_index[document_index] = candidate

        documents: list[LdaDocumentMetadata] = []
        for document_index in range(document_count):
            documents.append(
                by_index.get(
                    document_index,
                    LdaDocumentMetadata(
                        document_index=document_index,
                        run_id="",
                        protocol_id=f"doc_{document_index:06d}",
                        scenario_id="",
                    ),
                )
            )
        return documents

    def _resolve_corpus_hash(self, metadata: Mapping[str, object]) -> str | None:
        """Вернуть хеш корпуса из метаданных, если он был сохранен."""

        corpus_hash = metadata.get("corpus_hash")
        if corpus_hash is None:
            return None
        return str(corpus_hash)
