"""Анализ устойчивости латентных факторов модели ``LDA_prior``.

Модуль обучает несколько экземпляров основной априорной LDA-модели с одним
и тем же числом факторов ``K``, но с разными значениями ``random_state``.
После обучения темы сравниваются по распределениям ``φ_k``. Итоговые отчеты
показывают, насколько воспроизводима структура латентных факторов качества.

Модуль не строит диагностические модели ``LDA_diag`` и ``LDA_full``. Они
должны реализовываться отдельным этапом, чтобы не смешивать анализ
устойчивости априорной модели с последующим анализом фактических признаков.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np

from manual_coding_sim.lda.lda_prior_model import (
    LdaPriorModel,
    LdaPriorModelConfig,
)


@dataclass(frozen=True)
class LdaTopicStabilityConfig:
    """Параметры анализа устойчивости тем ``LDA_prior``."""

    n_components: int
    random_states: tuple[int, ...]
    doc_topic_prior: float | None = None
    topic_word_prior: float | None = None
    learning_method: str = "batch"
    max_iter: int = 100
    evaluate_every: int = -1
    n_jobs: int | None = None
    encoding: str = "utf-8"
    overwrite: bool = True

    def validate(self) -> None:
        """Проверить корректность параметров анализа устойчивости."""

        if self.n_components < 2:
            msg = "n_components должен быть не меньше 2."
            raise ValueError(msg)
        if len(self.random_states) < 2:
            msg = "Для анализа устойчивости нужно не менее двух random_state."
            raise ValueError(msg)
        if len(set(self.random_states)) != len(self.random_states):
            msg = "random_states не должен содержать повторяющиеся значения."
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
        allowed_methods = {"batch", "online"}
        if self.learning_method not in allowed_methods:
            msg = (
                "learning_method должен иметь значение "
                f"из множества {sorted(allowed_methods)}."
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class LdaTopicStabilityRun:
    """Артефакты одного запуска LDA с заданным ``random_state``."""

    random_state: int
    model_path: Path
    theta_prior_path: Path
    topic_word_path: Path
    metadata_path: Path
    model_hash: str


@dataclass(frozen=True)
class LdaTopicStabilityResult:
    """Итоговые артефакты и значения устойчивости тем."""

    report_csv_path: Path
    report_json_path: Path
    report_md_path: Path
    n_components: int
    random_states: tuple[int, ...]
    reference_random_state: int
    mean_stability: float
    min_stability: float
    runs: tuple[LdaTopicStabilityRun, ...]


class LdaTopicStabilityAnalyzer:
    """Оценивает устойчивость тем между несколькими запусками ``LDA_prior``."""

    def __init__(self, config: LdaTopicStabilityConfig) -> None:
        """Создать анализатор устойчивости латентных факторов."""

        config.validate()
        self.config = config

    def analyze_from_artifacts(
        self,
        corpus_path: str | Path,
        dictionary_path: str | Path,
        metadata_path: str | Path,
        models_dir: str | Path,
        reports_dir: str | Path,
    ) -> LdaTopicStabilityResult:
        """Выполнить анализ устойчивости по сохраненным артефактам корпуса."""

        models_path = Path(models_dir)
        reports_path = Path(reports_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        reports_path.mkdir(parents=True, exist_ok=True)

        report_csv_path = reports_path / "topic_stability_report.csv"
        report_json_path = reports_path / "topic_stability_report.json"
        report_md_path = reports_path / "topic_stability_report.md"
        self._ensure_can_write(
            [report_csv_path, report_json_path, report_md_path]
        )

        runs: list[LdaTopicStabilityRun] = []
        topic_matrices: list[np.ndarray] = []
        for random_state in self.config.random_states:
            run = self._fit_seed_model(
                random_state=random_state,
                corpus_path=corpus_path,
                dictionary_path=dictionary_path,
                metadata_path=metadata_path,
                models_dir=models_path,
                reports_dir=reports_path,
            )
            runs.append(run)
            topic_matrices.append(self._load_topic_word_matrix(run.model_path))

        reference_index = 0
        reference_random_state = self.config.random_states[reference_index]
        per_topic_payload = self._calculate_reference_topic_stability(
            reference_matrix=topic_matrices[reference_index],
            compared_matrices=topic_matrices[1:],
            compared_states=self.config.random_states[1:],
        )
        pairwise_payload = self._calculate_pairwise_run_stability(
            topic_matrices=topic_matrices,
            random_states=self.config.random_states,
        )
        per_topic_stability = [
            float(item["mean_similarity"]) for item in per_topic_payload
        ]
        mean_stability = float(np.mean(per_topic_stability))
        min_stability = float(np.min(per_topic_stability))

        payload = self._build_json_payload(
            runs=runs,
            reference_random_state=reference_random_state,
            mean_stability=mean_stability,
            min_stability=min_stability,
            per_topic_payload=per_topic_payload,
            pairwise_payload=pairwise_payload,
        )
        self._write_json(report_json_path, payload)
        self._write_csv(report_csv_path, payload)
        self._write_markdown(report_md_path, payload)

        return LdaTopicStabilityResult(
            report_csv_path=report_csv_path,
            report_json_path=report_json_path,
            report_md_path=report_md_path,
            n_components=self.config.n_components,
            random_states=self.config.random_states,
            reference_random_state=reference_random_state,
            mean_stability=mean_stability,
            min_stability=min_stability,
            runs=tuple(runs),
        )

    def _fit_seed_model(
        self,
        random_state: int,
        corpus_path: str | Path,
        dictionary_path: str | Path,
        metadata_path: str | Path,
        models_dir: Path,
        reports_dir: Path,
    ) -> LdaTopicStabilityRun:
        """Обучить отдельную ``LDA_prior`` для одного ``random_state``."""

        run_name = f"seed_{random_state:05d}"
        run_models_dir = models_dir / "topic_stability" / run_name
        run_reports_dir = reports_dir / "topic_stability" / run_name
        training_result = LdaPriorModel(
            LdaPriorModelConfig(
                n_components=self.config.n_components,
                doc_topic_prior=self.config.doc_topic_prior,
                topic_word_prior=self.config.topic_word_prior,
                learning_method=self.config.learning_method,
                max_iter=self.config.max_iter,
                random_state=random_state,
                evaluate_every=self.config.evaluate_every,
                n_jobs=self.config.n_jobs,
                encoding=self.config.encoding,
                overwrite=self.config.overwrite,
            )
        ).fit_from_artifacts(
            corpus_path=corpus_path,
            dictionary_path=dictionary_path,
            metadata_path=metadata_path,
            models_dir=run_models_dir,
            reports_dir=run_reports_dir,
        )
        return LdaTopicStabilityRun(
            random_state=random_state,
            model_path=training_result.model_path,
            theta_prior_path=training_result.theta_prior_path,
            topic_word_path=training_result.topic_word_path,
            metadata_path=training_result.metadata_path,
            model_hash=training_result.model_hash,
        )

    def _load_topic_word_matrix(self, model_path: Path) -> np.ndarray:
        """Загрузить нормированную матрицу ``φ_k`` из обученной LDA-модели."""

        model = joblib.load(model_path)
        if not hasattr(model, "components_"):
            msg = "Модель не содержит components_; вероятно, она не обучена."
            raise ValueError(msg)
        matrix = np.asarray(model.components_, dtype=float)
        if matrix.ndim != 2:
            msg = "Матрица components_ должна быть двумерной."
            raise ValueError(msg)
        if matrix.shape[0] != self.config.n_components:
            msg = "Число тем в модели не совпадает с n_components анализа."
            raise ValueError(msg)
        row_sums = matrix.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            msg = "В модели обнаружена тема с нулевой суммой весов."
            raise ValueError(msg)
        return matrix / row_sums

    def _calculate_reference_topic_stability(
        self,
        reference_matrix: np.ndarray,
        compared_matrices: Sequence[np.ndarray],
        compared_states: Sequence[int],
    ) -> list[dict[str, object]]:
        """Рассчитать устойчивость каждой темы относительно опорного запуска."""

        per_topic_payload: list[dict[str, object]] = []
        for topic_id, topic_vector in enumerate(reference_matrix):
            matches = []
            for compared_matrix, random_state in zip(
                compared_matrices,
                compared_states,
                strict=True,
            ):
                similarities = self._cosine_similarity_to_all(
                    vector=topic_vector,
                    matrix=compared_matrix,
                )
                best_topic_id = int(np.argmax(similarities))
                best_similarity = float(similarities[best_topic_id])
                matches.append(
                    {
                        "random_state": int(random_state),
                        "matched_topic_id": best_topic_id,
                        "similarity": best_similarity,
                    }
                )
            topic_scores = [float(item["similarity"]) for item in matches]
            per_topic_payload.append(
                {
                    "topic_id": topic_id,
                    "mean_similarity": float(np.mean(topic_scores)),
                    "min_similarity": float(np.min(topic_scores)),
                    "matches": matches,
                }
            )
        return per_topic_payload

    def _calculate_pairwise_run_stability(
        self,
        topic_matrices: Sequence[np.ndarray],
        random_states: Sequence[int],
    ) -> list[dict[str, object]]:
        """Рассчитать среднюю устойчивость для всех пар запусков."""

        pairwise_payload: list[dict[str, object]] = []
        for left_index, left_matrix in enumerate(topic_matrices):
            for right_index in range(left_index + 1, len(topic_matrices)):
                right_matrix = topic_matrices[right_index]
                directional_left = self._directional_topic_similarity(
                    source_matrix=left_matrix,
                    target_matrix=right_matrix,
                )
                directional_right = self._directional_topic_similarity(
                    source_matrix=right_matrix,
                    target_matrix=left_matrix,
                )
                mean_similarity = float(
                    (np.mean(directional_left) + np.mean(directional_right)) / 2.0
                )
                pairwise_payload.append(
                    {
                        "left_random_state": int(random_states[left_index]),
                        "right_random_state": int(random_states[right_index]),
                        "mean_similarity": mean_similarity,
                        "min_similarity": float(
                            min(np.min(directional_left), np.min(directional_right))
                        ),
                    }
                )
        return pairwise_payload

    def _directional_topic_similarity(
        self,
        source_matrix: np.ndarray,
        target_matrix: np.ndarray,
    ) -> list[float]:
        """Сопоставить каждую тему источника с ближайшей темой целевого запуска."""

        scores: list[float] = []
        for topic_vector in source_matrix:
            similarities = self._cosine_similarity_to_all(
                vector=topic_vector,
                matrix=target_matrix,
            )
            scores.append(float(np.max(similarities)))
        return scores

    def _cosine_similarity_to_all(
        self,
        vector: np.ndarray,
        matrix: np.ndarray,
    ) -> np.ndarray:
        """Рассчитать cosine similarity вектора со всеми строками матрицы."""

        vector_norm = float(np.linalg.norm(vector))
        matrix_norms = np.linalg.norm(matrix, axis=1)
        denominator = matrix_norms * vector_norm
        if vector_norm <= 0 or np.any(matrix_norms <= 0):
            msg = "Невозможно рассчитать cosine similarity для нулевого вектора."
            raise ValueError(msg)
        similarities = matrix @ vector / denominator
        return np.clip(similarities, 0.0, 1.0)

    def _build_json_payload(
        self,
        runs: Sequence[LdaTopicStabilityRun],
        reference_random_state: int,
        mean_stability: float,
        min_stability: float,
        per_topic_payload: Sequence[Mapping[str, object]],
        pairwise_payload: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        """Сформировать JSON-отчет анализа устойчивости."""

        return {
            "model_name": "LDA_prior",
            "analysis_method": "topic_word_cosine_similarity",
            "n_components": self.config.n_components,
            "random_states": list(self.config.random_states),
            "reference_random_state": reference_random_state,
            "mean_stability": mean_stability,
            "min_stability": min_stability,
            "doc_topic_prior": self.config.doc_topic_prior,
            "topic_word_prior": self.config.topic_word_prior,
            "learning_method": self.config.learning_method,
            "max_iter": self.config.max_iter,
            "runs": [self._run_to_payload(run) for run in runs],
            "per_topic_stability": list(per_topic_payload),
            "pairwise_run_stability": list(pairwise_payload),
        }

    def _run_to_payload(self, run: LdaTopicStabilityRun) -> dict[str, object]:
        """Преобразовать сведения о запуске в JSON-совместимый словарь."""

        return {
            "random_state": run.random_state,
            "model_path": str(run.model_path),
            "theta_prior_path": str(run.theta_prior_path),
            "topic_word_path": str(run.topic_word_path),
            "metadata_path": str(run.metadata_path),
            "model_hash": run.model_hash,
        }

    def _write_json(self, path: Path, payload: Mapping[str, object]) -> None:
        """Сохранить JSON-отчет устойчивости тем."""

        with path.open("w", encoding=self.config.encoding) as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
            file_obj.write("\n")

    def _write_csv(self, path: Path, payload: Mapping[str, object]) -> None:
        """Сохранить CSV-отчет устойчивости тем."""

        fieldnames = [
            "row_type",
            "topic_id",
            "left_random_state",
            "right_random_state",
            "mean_similarity",
            "min_similarity",
            "matched_runs_count",
        ]
        rows = []
        rows.append(
            {
                "row_type": "summary",
                "topic_id": "",
                "left_random_state": "",
                "right_random_state": "",
                "mean_similarity": payload["mean_stability"],
                "min_similarity": payload["min_stability"],
                "matched_runs_count": len(payload["random_states"]),
            }
        )
        for topic_item in payload["per_topic_stability"]:
            rows.append(
                {
                    "row_type": "topic",
                    "topic_id": topic_item["topic_id"],
                    "left_random_state": payload["reference_random_state"],
                    "right_random_state": "",
                    "mean_similarity": topic_item["mean_similarity"],
                    "min_similarity": topic_item["min_similarity"],
                    "matched_runs_count": len(topic_item["matches"]),
                }
            )
        for pair_item in payload["pairwise_run_stability"]:
            rows.append(
                {
                    "row_type": "pairwise_run",
                    "topic_id": "",
                    "left_random_state": pair_item["left_random_state"],
                    "right_random_state": pair_item["right_random_state"],
                    "mean_similarity": pair_item["mean_similarity"],
                    "min_similarity": pair_item["min_similarity"],
                    "matched_runs_count": "",
                }
            )
        with path.open("w", encoding=self.config.encoding, newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_markdown(self, path: Path, payload: Mapping[str, object]) -> None:
        """Сохранить человекочитаемый Markdown-отчет устойчивости."""

        lines = [
            "# Отчет устойчивости латентных факторов LDA_prior",
            "",
            f"- Число факторов K: {payload['n_components']}",
            "- Проверенные random_state: "
            + ", ".join(str(value) for value in payload["random_states"]),
            f"- Опорный random_state: {payload['reference_random_state']}",
            f"- Средняя устойчивость: {payload['mean_stability']:.6f}",
            f"- Минимальная устойчивость: {payload['min_stability']:.6f}",
            "",
            "## Устойчивость тем относительно опорного запуска",
            "",
            "| Тема | Среднее сходство | Минимальное сходство |",
            "|---:|---:|---:|",
        ]
        for topic_item in payload["per_topic_stability"]:
            lines.append(
                f"| {topic_item['topic_id']} | "
                f"{float(topic_item['mean_similarity']):.6f} | "
                f"{float(topic_item['min_similarity']):.6f} |"
            )
        lines.extend(
            [
                "",
                "## Попарная устойчивость запусков",
                "",
                "| random_state 1 | random_state 2 | "
                "Среднее сходство | Минимальное сходство |",
                "|---:|---:|---:|---:|",
            ]
        )
        for pair_item in payload["pairwise_run_stability"]:
            lines.append(
                f"| {pair_item['left_random_state']} | "
                f"{pair_item['right_random_state']} | "
                f"{float(pair_item['mean_similarity']):.6f} | "
                f"{float(pair_item['min_similarity']):.6f} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding=self.config.encoding)

    def _ensure_can_write(self, paths: Sequence[Path]) -> None:
        """Проверить возможность записи итоговых отчетов."""

        if self.config.overwrite:
            return
        existing_paths = [path for path in paths if path.exists()]
        if existing_paths:
            msg = "Запрещена перезапись существующих файлов: "
            msg += ", ".join(str(path) for path in existing_paths)
            raise FileExistsError(msg)
