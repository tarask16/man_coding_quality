"""Метрики качества основной LDA-модели главы 4.

Модуль рассчитывает первичные машинные показатели для уже обученной
``LDA_prior``: perplexity, простую topic coherence по совместной
встречаемости top-токенов и topic diversity. Подбор числа факторов ``K`` и
анализ устойчивости по нескольким seed-ам намеренно вынесены в следующие
этапы, чтобы не смешивать разные задачи главы 4.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np

from manual_coding_sim.lda.matrix_builder import (
    LdaMatrixBuilder,
    LdaMatrixBuildResult,
    LdaVocabularyItem,
)


@dataclass(frozen=True)
class LdaTopicMetricsConfig:
    """Параметры расчета метрик для ``LDA_prior``."""

    top_n: int = 10
    encoding: str = "utf-8"
    overwrite: bool = True

    def validate(self) -> None:
        """Проверить корректность параметров расчета."""

        if self.top_n < 2:
            msg = "top_n должен быть не меньше 2."
            raise ValueError(msg)


@dataclass(frozen=True)
class LdaTopicMetricsResult:
    """Пути и числовые значения метрик LDA-модели."""

    metrics_csv_path: Path
    metrics_json_path: Path
    perplexity: float
    mean_coherence: float
    topic_diversity: float
    document_count: int
    token_count: int
    n_components: int
    top_n: int
    corpus_hash: str | None
    model_hash: str


class LdaTopicMetricsEvaluator:
    """Рассчитывает метрики для обученной модели ``LDA_prior``."""

    def __init__(self, config: LdaTopicMetricsConfig | None = None) -> None:
        """Создать объект расчета метрик LDA."""

        self.config = config or LdaTopicMetricsConfig()
        self.config.validate()
        self.matrix_builder = LdaMatrixBuilder()

    def evaluate_from_artifacts(
        self,
        model_path: str | Path,
        corpus_path: str | Path,
        dictionary_path: str | Path,
        metadata_path: str | Path,
        reports_dir: str | Path,
    ) -> LdaTopicMetricsResult:
        """Рассчитать метрики по сохраненным артефактам ``LDA_prior``."""

        model_file = Path(model_path)
        matrix_result = self.matrix_builder.build_from_files(
            corpus_path=corpus_path,
            dictionary_path=dictionary_path,
            metadata_path=metadata_path,
        )
        model = self._load_model(model_file)
        self._validate_inputs(model=model, matrix_result=matrix_result)

        topic_word = self._normalize_rows(np.asarray(model.components_, dtype=float))
        top_token_ids = self._top_token_ids(topic_word)
        topic_coherence = self._calculate_topic_coherence(
            matrix=matrix_result.matrix,
            top_token_ids=top_token_ids,
        )
        mean_coherence = float(np.mean(topic_coherence))
        topic_diversity = self._calculate_topic_diversity(top_token_ids)
        perplexity = float(model.perplexity(matrix_result.matrix))

        reports_path = Path(reports_dir)
        reports_path.mkdir(parents=True, exist_ok=True)
        metrics_csv_path = reports_path / "topic_metrics.csv"
        metrics_json_path = reports_path / "topic_metrics.json"
        self._ensure_can_write([metrics_csv_path, metrics_json_path])

        model_hash = self._calculate_file_hash(model_file)
        payload = self._build_json_payload(
            model=model,
            matrix_result=matrix_result,
            model_hash=model_hash,
            perplexity=perplexity,
            mean_coherence=mean_coherence,
            topic_diversity=topic_diversity,
            topic_coherence=topic_coherence,
            top_token_ids=top_token_ids,
            topic_word=topic_word,
        )
        self._write_json(metrics_json_path, payload)
        self._write_csv(metrics_csv_path, payload)

        return LdaTopicMetricsResult(
            metrics_csv_path=metrics_csv_path,
            metrics_json_path=metrics_json_path,
            perplexity=perplexity,
            mean_coherence=mean_coherence,
            topic_diversity=topic_diversity,
            document_count=matrix_result.matrix.shape[0],
            token_count=matrix_result.matrix.shape[1],
            n_components=int(model.n_components),
            top_n=len(top_token_ids[0]),
            corpus_hash=matrix_result.corpus_hash,
            model_hash=model_hash,
        )

    def _load_model(self, model_path: Path):
        """Загрузить обученную LDA-модель из файла joblib."""

        if not model_path.exists():
            msg = f"Файл LDA-модели не найден: {model_path}"
            raise FileNotFoundError(msg)
        return joblib.load(model_path)

    def _validate_inputs(self, model, matrix_result: LdaMatrixBuildResult) -> None:
        """Проверить согласованность модели, корпуса и словаря."""

        matrix = matrix_result.matrix
        if matrix.ndim != 2:
            msg = "Матрица корпуса для метрик должна быть двумерной."
            raise ValueError(msg)
        if matrix.shape[0] < 1 or matrix.shape[1] < 2:
            msg = "Для расчета метрик нужен непустой корпус и минимум два токена."
            raise ValueError(msg)
        if not hasattr(model, "components_"):
            msg = "Модель не содержит components_; вероятно, она не обучена."
            raise ValueError(msg)
        components = np.asarray(model.components_, dtype=float)
        if components.ndim != 2:
            msg = "components_ обученной LDA-модели должны быть двумерными."
            raise ValueError(msg)
        if components.shape[1] != matrix.shape[1]:
            msg = "Размер словаря модели не совпадает с размером корпуса."
            raise ValueError(msg)
        if np.any(matrix < 0):
            msg = "Матрица корпуса не должна содержать отрицательные частоты."
            raise ValueError(msg)

    def _normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        """Нормировать строки матрицы до суммы 1."""

        row_sums = matrix.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            msg = "Невозможно нормировать распределение темы с нулевой суммой."
            raise ValueError(msg)
        return matrix / row_sums

    def _top_token_ids(self, topic_word: np.ndarray) -> list[list[int]]:
        """Получить идентификаторы top-N токенов для каждой темы."""

        actual_top_n = min(self.config.top_n, topic_word.shape[1])
        top_token_ids: list[list[int]] = []
        for topic_weights in topic_word:
            ordered_ids = np.argsort(topic_weights)[::-1]
            topic_token_ids = [
                int(token_id) for token_id in ordered_ids[:actual_top_n]
            ]
            top_token_ids.append(topic_token_ids)
        return top_token_ids

    def _calculate_topic_coherence(
        self,
        matrix: np.ndarray,
        top_token_ids: Sequence[Sequence[int]],
    ) -> list[float]:
        """Рассчитать простую coherence по совместной встречаемости токенов.

        Используется сглаженная мера, близкая к ``u_mass``: для каждой темы
        суммируются логарифмы отношения совместной документной частоты пары
        top-токенов к документной частоте более раннего токена.
        """

        binary_matrix = matrix > 0
        document_frequency = binary_matrix.sum(axis=0)
        topic_scores: list[float] = []
        for token_ids in top_token_ids:
            pair_scores: list[float] = []
            for later_position in range(1, len(token_ids)):
                later_token_id = token_ids[later_position]
                for earlier_token_id in token_ids[:later_position]:
                    joint_frequency = np.logical_and(
                        binary_matrix[:, later_token_id],
                        binary_matrix[:, earlier_token_id],
                    ).sum()
                    denominator = max(float(document_frequency[earlier_token_id]), 1.0)
                    score = math.log(
                        (float(joint_frequency) + 1.0) / denominator
                    )
                    pair_scores.append(score)
            topic_scores.append(float(np.mean(pair_scores)) if pair_scores else 0.0)
        return topic_scores

    def _calculate_topic_diversity(
        self,
        top_token_ids: Sequence[Sequence[int]],
    ) -> float:
        """Рассчитать долю уникальных top-токенов среди всех тем."""

        flattened = [token_id for topic_ids in top_token_ids for token_id in topic_ids]
        if not flattened:
            msg = "Невозможно рассчитать topic diversity без top-токенов."
            raise ValueError(msg)
        return float(len(set(flattened)) / len(flattened))

    def _build_json_payload(
        self,
        model,
        matrix_result: LdaMatrixBuildResult,
        model_hash: str,
        perplexity: float,
        mean_coherence: float,
        topic_diversity: float,
        topic_coherence: Sequence[float],
        top_token_ids: Sequence[Sequence[int]],
        topic_word: np.ndarray,
    ) -> dict[str, object]:
        """Сформировать JSON-отчет с метриками и top-токенами."""

        vocabulary_by_id = {item.token_id: item for item in matrix_result.vocabulary}
        topics = []
        for topic_id, token_ids in enumerate(top_token_ids):
            topics.append(
                {
                    "topic_id": topic_id,
                    "coherence": float(topic_coherence[topic_id]),
                    "top_tokens": self._build_top_token_payload(
                        token_ids=token_ids,
                        topic_weights=topic_word[topic_id],
                        vocabulary_by_id=vocabulary_by_id,
                    ),
                }
            )

        return {
            "model_name": "LDA_prior",
            "n_components": int(model.n_components),
            "top_n": len(top_token_ids[0]),
            "document_count": int(matrix_result.matrix.shape[0]),
            "token_count": int(matrix_result.matrix.shape[1]),
            "corpus_hash": matrix_result.corpus_hash,
            "model_hash": model_hash,
            "perplexity": perplexity,
            "mean_coherence": mean_coherence,
            "topic_diversity": topic_diversity,
            "topics": topics,
        }

    def _build_top_token_payload(
        self,
        token_ids: Sequence[int],
        topic_weights: np.ndarray,
        vocabulary_by_id: Mapping[int, LdaVocabularyItem],
    ) -> list[dict[str, object]]:
        """Сформировать список top-токенов одной темы для JSON-отчета."""

        top_tokens = []
        for rank, token_id in enumerate(token_ids, start=1):
            item = vocabulary_by_id[int(token_id)]
            top_tokens.append(
                {
                    "rank": rank,
                    "token_id": item.token_id,
                    "token": item.token,
                    "document_frequency": item.document_frequency,
                    "weight": float(topic_weights[item.token_id]),
                }
            )
        return top_tokens

    def _write_json(self, path: Path, payload: Mapping[str, object]) -> None:
        """Сохранить JSON-отчет метрик."""

        with path.open("w", encoding=self.config.encoding) as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
            file_obj.write("\n")

    def _write_csv(self, path: Path, payload: Mapping[str, object]) -> None:
        """Сохранить компактный CSV-отчет метрик."""

        fieldnames = ["metric", "value"]
        rows = [
            ("model_name", payload["model_name"]),
            ("n_components", payload["n_components"]),
            ("top_n", payload["top_n"]),
            ("document_count", payload["document_count"]),
            ("token_count", payload["token_count"]),
            ("perplexity", payload["perplexity"]),
            ("mean_coherence", payload["mean_coherence"]),
            ("topic_diversity", payload["topic_diversity"]),
            ("corpus_hash", payload["corpus_hash"] or ""),
            ("model_hash", payload["model_hash"]),
        ]
        with path.open("w", encoding=self.config.encoding, newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for metric, value in rows:
                writer.writerow({"metric": metric, "value": value})

    def _calculate_file_hash(self, path: Path) -> str:
        """Рассчитать SHA-256 хеш файла модели."""

        digest = hashlib.sha256()
        with path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _ensure_can_write(self, paths: Sequence[Path]) -> None:
        """Проверить возможность записи итоговых файлов."""

        if self.config.overwrite:
            return
        existing_paths = [path for path in paths if path.exists()]
        if existing_paths:
            msg = "Запрещена перезапись существующих файлов: "
            msg += ", ".join(str(path) for path in existing_paths)
            raise FileExistsError(msg)
