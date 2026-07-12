"""Обучение основной LDA-модели по априорному корпусу.

Модель ``LDA_prior`` использует только корпус, построенный из
``prior_features.csv``. Фактические признаки и целевые показатели качества
не передаются в этот модуль и не могут участвовать в обучении основной
априорной модели главы 4.
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

from manual_coding_sim.lda.matrix_builder import (
    LdaDocumentMetadata,
    LdaMatrixBuilder,
    LdaMatrixBuildResult,
    LdaVocabularyItem,
)


@dataclass(frozen=True)
class LdaPriorModelConfig:
    """Параметры обучения основной модели ``LDA_prior``."""

    n_components: int
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
        """Проверить корректность параметров обучения."""

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


@dataclass(frozen=True)
class LdaPriorTrainingResult:
    """Пути к артефактам, созданным после обучения ``LDA_prior``."""

    model_path: Path
    theta_prior_path: Path
    topic_word_path: Path
    metadata_path: Path
    document_count: int
    token_count: int
    n_components: int
    corpus_hash: str | None
    model_hash: str


class LdaPriorModel:
    """Обучает ``LDA_prior`` и сохраняет латентные профили документов."""

    def __init__(self, config: LdaPriorModelConfig) -> None:
        """Создать объект обучения основной LDA-модели."""

        config.validate()
        self.config = config
        self.matrix_builder = LdaMatrixBuilder()

    def fit_from_artifacts(
        self,
        corpus_path: str | Path,
        dictionary_path: str | Path,
        metadata_path: str | Path,
        models_dir: str | Path,
        reports_dir: str | Path,
    ) -> LdaPriorTrainingResult:
        """Обучить модель по сохраненным артефактам LDA-корпуса."""

        matrix_result = self.matrix_builder.build_from_files(
            corpus_path=corpus_path,
            dictionary_path=dictionary_path,
            metadata_path=metadata_path,
        )
        self._validate_matrix(matrix_result)

        model = self._create_model()
        theta = model.fit_transform(matrix_result.matrix)
        theta = self._normalize_rows(theta)
        topic_word = self._normalize_rows(model.components_)

        models_path = Path(models_dir)
        reports_path = Path(reports_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        reports_path.mkdir(parents=True, exist_ok=True)

        model_path = models_path / "lda_prior.joblib"
        theta_prior_path = reports_path / "theta_prior.csv"
        topic_word_path = reports_path / "topic_word.csv"
        metadata_output_path = reports_path / "lda_prior_metadata.json"

        self._ensure_can_write(
            paths=[
                model_path,
                theta_prior_path,
                topic_word_path,
                metadata_output_path,
            ]
        )
        joblib.dump(model, model_path)
        self._write_theta_csv(theta_prior_path, matrix_result.documents, theta)
        self._write_topic_word_csv(
            topic_word_path,
            matrix_result.vocabulary,
            topic_word,
        )
        model_hash = self._calculate_file_hash(model_path)
        metadata = self._build_metadata(
            matrix_result=matrix_result,
            model_hash=model_hash,
        )
        self._write_json(metadata_output_path, metadata)

        return LdaPriorTrainingResult(
            model_path=model_path,
            theta_prior_path=theta_prior_path,
            topic_word_path=topic_word_path,
            metadata_path=metadata_output_path,
            document_count=matrix_result.matrix.shape[0],
            token_count=matrix_result.matrix.shape[1],
            n_components=self.config.n_components,
            corpus_hash=matrix_result.corpus_hash,
            model_hash=model_hash,
        )

    def _create_model(self) -> LatentDirichletAllocation:
        """Создать объект scikit-learn LDA с зафиксированными параметрами."""

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
        """Проверить пригодность матрицы для обучения LDA."""

        matrix = matrix_result.matrix
        if matrix.ndim != 2:
            msg = "Матрица LDA должна быть двумерной."
            raise ValueError(msg)
        if matrix.shape[0] < 2:
            msg = "Для обучения LDA требуется не менее двух документов."
            raise ValueError(msg)
        if matrix.shape[1] < 2:
            msg = "Для обучения LDA требуется не менее двух токенов."
            raise ValueError(msg)
        if self.config.n_components > matrix.shape[1]:
            msg = "n_components не должен превышать число токенов словаря."
            raise ValueError(msg)
        if np.any(matrix < 0):
            msg = "Матрица LDA не должна содержать отрицательных значений."
            raise ValueError(msg)
        empty_documents = np.where(matrix.sum(axis=1) <= 0)[0]
        if empty_documents.size:
            msg = "В LDA-корпусе есть документы без токенов: "
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
        """Сохранить распределения ``θ_prior(A_i)`` по документам."""

        theta_columns = [f"theta_{index}" for index in range(theta.shape[1])]
        fieldnames = [
            "document_index",
            "run_id",
            "protocol_id",
            "scenario_id",
            *theta_columns,
            "selected_k",
            "doc_topic_prior",
            "topic_word_prior",
            "random_state",
        ]
        with path.open("w", encoding=self.config.encoding, newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for document, theta_row in zip(documents, theta, strict=True):
                row: dict[str, object] = {
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
        """Сохранить распределения ``φ_k`` по токенам словаря."""

        fieldnames = [
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
        model_hash: str,
    ) -> dict[str, object]:
        """Сформировать JSON-метаданные обучения ``LDA_prior``."""

        return {
            "model_name": "LDA_prior",
            "n_components": self.config.n_components,
            "doc_topic_prior": self.config.doc_topic_prior,
            "topic_word_prior": self.config.topic_word_prior,
            "learning_method": self.config.learning_method,
            "max_iter": self.config.max_iter,
            "random_state": self.config.random_state,
            "evaluate_every": self.config.evaluate_every,
            "document_count": int(matrix_result.matrix.shape[0]),
            "token_count": int(matrix_result.matrix.shape[1]),
            "corpus_hash": matrix_result.corpus_hash,
            "model_hash": model_hash,
            "theta_columns": [
                f"theta_{index}" for index in range(self.config.n_components)
            ],
        }

    def _write_json(self, path: Path, payload: Mapping[str, object]) -> None:
        """Сохранить JSON-файл с параметрами обучения."""

        with path.open("w", encoding=self.config.encoding) as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
            file_obj.write("\n")

    def _calculate_file_hash(self, path: Path) -> str:
        """Рассчитать SHA-256 хеш сохраненной модели."""

        digest = hashlib.sha256()
        with path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _format_optional_float(self, value: float | None) -> str:
        """Преобразовать необязательный числовой параметр в CSV-строку."""

        if value is None:
            return ""
        return f"{float(value):.12g}"

    def _ensure_can_write(self, paths: Sequence[Path]) -> None:
        """Проверить, что существующие артефакты можно перезаписать."""

        if self.config.overwrite:
            return
        existing_paths = [path for path in paths if path.exists()]
        if existing_paths:
            msg = "Запрещена перезапись существующих файлов: "
            msg += ", ".join(str(path) for path in existing_paths)
            raise FileExistsError(msg)
