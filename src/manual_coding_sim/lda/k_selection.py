"""Подбор числа латентных факторов для основной модели ``LDA_prior``.

Модуль перебирает несколько значений ``K``, для каждого значения обучает
отдельную модель ``LDA_prior`` на одном и том же априорном корпусе и
рассчитывает метрики качества тем. Итогом является воспроизводимый отчет,
который позволяет обосновать выбор числа латентных факторов в главе 4.

Модуль намеренно не выполняет анализ устойчивости по нескольким
``random_state``: эта задача выделяется в следующий программный этап.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from manual_coding_sim.lda.lda_prior_model import (
    LdaPriorModel,
    LdaPriorModelConfig,
)
from manual_coding_sim.lda.topic_metrics import (
    LdaTopicMetricsConfig,
    LdaTopicMetricsEvaluator,
)


@dataclass(frozen=True)
class LdaKSelectionConfig:
    """Параметры перебора числа латентных факторов ``K``."""

    k_values: tuple[int, ...]
    doc_topic_prior: float | None = None
    topic_word_prior: float | None = None
    learning_method: str = "batch"
    max_iter: int = 100
    random_state: int = 42
    evaluate_every: int = -1
    n_jobs: int | None = None
    top_n: int = 10
    coherence_weight: float = 1.0
    inverse_perplexity_weight: float = 1.0
    diversity_weight: float = 1.0
    encoding: str = "utf-8"
    overwrite: bool = True

    def validate(self) -> None:
        """Проверить корректность параметров подбора ``K``."""

        if not self.k_values:
            msg = "k_values не должен быть пустым."
            raise ValueError(msg)
        if len(set(self.k_values)) != len(self.k_values):
            msg = "k_values не должен содержать повторяющиеся значения."
            raise ValueError(msg)
        if any(k_value < 2 for k_value in self.k_values):
            msg = "Каждое значение K должно быть не меньше 2."
            raise ValueError(msg)
        if self.max_iter < 1:
            msg = "max_iter должен быть положительным целым числом."
            raise ValueError(msg)
        if self.top_n < 2:
            msg = "top_n должен быть не меньше 2."
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
        weights = (
            self.coherence_weight,
            self.inverse_perplexity_weight,
            self.diversity_weight,
        )
        if any(weight < 0 for weight in weights):
            msg = "Веса интегральной оценки не должны быть отрицательными."
            raise ValueError(msg)
        if sum(weights) <= 0:
            msg = "Хотя бы один вес интегральной оценки должен быть положительным."
            raise ValueError(msg)


@dataclass(frozen=True)
class LdaKSelectionCandidate:
    """Результаты обучения и оценки одного значения ``K``."""

    k: int
    perplexity: float
    mean_coherence: float
    topic_diversity: float
    normalized_coherence: float
    normalized_inverse_perplexity: float
    normalized_topic_diversity: float
    selection_score: float
    model_path: Path
    metrics_json_path: Path
    model_hash: str


@dataclass(frozen=True)
class LdaKSelectionResult:
    """Итоговые артефакты и рекомендованное значение ``K``."""

    report_csv_path: Path
    report_json_path: Path
    report_md_path: Path
    recommended_k: int
    candidates: tuple[LdaKSelectionCandidate, ...]


class LdaKSelector:
    """Подбирает число латентных факторов для модели ``LDA_prior``."""

    def __init__(self, config: LdaKSelectionConfig) -> None:
        """Создать объект подбора числа латентных факторов."""

        config.validate()
        self.config = config

    def select_from_artifacts(
        self,
        corpus_path: str | Path,
        dictionary_path: str | Path,
        metadata_path: str | Path,
        models_dir: str | Path,
        reports_dir: str | Path,
    ) -> LdaKSelectionResult:
        """Выполнить перебор ``K`` по сохраненным артефактам корпуса."""

        models_path = Path(models_dir)
        reports_path = Path(reports_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        reports_path.mkdir(parents=True, exist_ok=True)

        report_csv_path = reports_path / "k_selection_report.csv"
        report_json_path = reports_path / "k_selection_report.json"
        report_md_path = reports_path / "k_selection_report.md"
        self._ensure_can_write(
            [report_csv_path, report_json_path, report_md_path]
        )

        raw_candidates = []
        for k_value in self.config.k_values:
            candidate = self._fit_and_evaluate_k(
                k_value=k_value,
                corpus_path=corpus_path,
                dictionary_path=dictionary_path,
                metadata_path=metadata_path,
                models_dir=models_path,
                reports_dir=reports_path,
            )
            raw_candidates.append(candidate)

        candidates = self._score_candidates(raw_candidates)
        recommended = self._select_recommended_candidate(candidates)
        payload = self._build_json_payload(recommended, candidates)
        self._write_json(report_json_path, payload)
        self._write_csv(report_csv_path, candidates, recommended.k)
        self._write_markdown(report_md_path, candidates, recommended.k)

        return LdaKSelectionResult(
            report_csv_path=report_csv_path,
            report_json_path=report_json_path,
            report_md_path=report_md_path,
            recommended_k=recommended.k,
            candidates=tuple(candidates),
        )

    def _fit_and_evaluate_k(
        self,
        k_value: int,
        corpus_path: str | Path,
        dictionary_path: str | Path,
        metadata_path: str | Path,
        models_dir: Path,
        reports_dir: Path,
    ) -> dict[str, object]:
        """Обучить и оценить отдельную модель для одного значения ``K``."""

        run_name = f"k_{k_value:03d}"
        candidate_models_dir = models_dir / "k_selection" / run_name
        candidate_reports_dir = reports_dir / "k_selection" / run_name

        training_result = LdaPriorModel(
            LdaPriorModelConfig(
                n_components=k_value,
                doc_topic_prior=self.config.doc_topic_prior,
                topic_word_prior=self.config.topic_word_prior,
                learning_method=self.config.learning_method,
                max_iter=self.config.max_iter,
                random_state=self.config.random_state,
                evaluate_every=self.config.evaluate_every,
                n_jobs=self.config.n_jobs,
                encoding=self.config.encoding,
                overwrite=self.config.overwrite,
            )
        ).fit_from_artifacts(
            corpus_path=corpus_path,
            dictionary_path=dictionary_path,
            metadata_path=metadata_path,
            models_dir=candidate_models_dir,
            reports_dir=candidate_reports_dir,
        )

        metrics_result = LdaTopicMetricsEvaluator(
            LdaTopicMetricsConfig(
                top_n=self.config.top_n,
                encoding=self.config.encoding,
                overwrite=self.config.overwrite,
            )
        ).evaluate_from_artifacts(
            model_path=training_result.model_path,
            corpus_path=corpus_path,
            dictionary_path=dictionary_path,
            metadata_path=metadata_path,
            reports_dir=candidate_reports_dir,
        )

        return {
            "k": k_value,
            "perplexity": metrics_result.perplexity,
            "mean_coherence": metrics_result.mean_coherence,
            "topic_diversity": metrics_result.topic_diversity,
            "model_path": training_result.model_path,
            "metrics_json_path": metrics_result.metrics_json_path,
            "model_hash": training_result.model_hash,
        }

    def _score_candidates(
        self,
        raw_candidates: Sequence[Mapping[str, object]],
    ) -> list[LdaKSelectionCandidate]:
        """Рассчитать нормированные метрики и интегральную оценку."""

        coherence_values = [float(item["mean_coherence"]) for item in raw_candidates]
        perplexity_values = [float(item["perplexity"]) for item in raw_candidates]
        diversity_values = [float(item["topic_diversity"]) for item in raw_candidates]
        inverse_perplexity_values = [1.0 / value for value in perplexity_values]

        normalized_coherence = self._min_max_normalize(coherence_values)
        normalized_inverse_perplexity = self._min_max_normalize(
            inverse_perplexity_values
        )
        normalized_diversity = self._min_max_normalize(diversity_values)
        weight_sum = (
            self.config.coherence_weight
            + self.config.inverse_perplexity_weight
            + self.config.diversity_weight
        )

        candidates: list[LdaKSelectionCandidate] = []
        for index, raw_candidate in enumerate(raw_candidates):
            score = (
                self.config.coherence_weight * normalized_coherence[index]
                + self.config.inverse_perplexity_weight
                * normalized_inverse_perplexity[index]
                + self.config.diversity_weight * normalized_diversity[index]
            ) / weight_sum
            candidates.append(
                LdaKSelectionCandidate(
                    k=int(raw_candidate["k"]),
                    perplexity=float(raw_candidate["perplexity"]),
                    mean_coherence=float(raw_candidate["mean_coherence"]),
                    topic_diversity=float(raw_candidate["topic_diversity"]),
                    normalized_coherence=float(normalized_coherence[index]),
                    normalized_inverse_perplexity=float(
                        normalized_inverse_perplexity[index]
                    ),
                    normalized_topic_diversity=float(normalized_diversity[index]),
                    selection_score=float(score),
                    model_path=Path(str(raw_candidate["model_path"])),
                    metrics_json_path=Path(str(raw_candidate["metrics_json_path"])),
                    model_hash=str(raw_candidate["model_hash"]),
                )
            )
        return candidates

    def _min_max_normalize(self, values: Sequence[float]) -> list[float]:
        """Нормировать набор значений в диапазон от 0 до 1."""

        if not values:
            msg = "Невозможно нормировать пустой список значений."
            raise ValueError(msg)
        if any(not math.isfinite(value) for value in values):
            msg = "Все значения для нормировки должны быть конечными числами."
            raise ValueError(msg)
        min_value = min(values)
        max_value = max(values)
        if math.isclose(min_value, max_value, rel_tol=0.0, abs_tol=1e-15):
            return [1.0 for _ in values]
        return [(value - min_value) / (max_value - min_value) for value in values]

    def _select_recommended_candidate(
        self,
        candidates: Sequence[LdaKSelectionCandidate],
    ) -> LdaKSelectionCandidate:
        """Выбрать рекомендуемое ``K`` по максимальной интегральной оценке."""

        if not candidates:
            msg = "Невозможно выбрать K без кандидатов."
            raise ValueError(msg)
        return max(candidates, key=lambda item: (item.selection_score, -item.k))

    def _build_json_payload(
        self,
        recommended: LdaKSelectionCandidate,
        candidates: Sequence[LdaKSelectionCandidate],
    ) -> dict[str, object]:
        """Сформировать JSON-отчет подбора ``K``."""

        return {
            "model_name": "LDA_prior",
            "selection_method": "weighted_min_max_score",
            "recommended_k": recommended.k,
            "random_state": self.config.random_state,
            "max_iter": self.config.max_iter,
            "top_n": self.config.top_n,
            "weights": {
                "coherence": self.config.coherence_weight,
                "inverse_perplexity": self.config.inverse_perplexity_weight,
                "topic_diversity": self.config.diversity_weight,
            },
            "candidates": [self._candidate_to_payload(item) for item in candidates],
        }

    def _candidate_to_payload(
        self,
        candidate: LdaKSelectionCandidate,
    ) -> dict[str, object]:
        """Преобразовать результат одного ``K`` в JSON-совместимый объект."""

        return {
            "k": candidate.k,
            "perplexity": candidate.perplexity,
            "mean_coherence": candidate.mean_coherence,
            "topic_diversity": candidate.topic_diversity,
            "normalized_coherence": candidate.normalized_coherence,
            "normalized_inverse_perplexity": candidate.normalized_inverse_perplexity,
            "normalized_topic_diversity": candidate.normalized_topic_diversity,
            "selection_score": candidate.selection_score,
            "model_path": str(candidate.model_path),
            "metrics_json_path": str(candidate.metrics_json_path),
            "model_hash": candidate.model_hash,
        }

    def _write_json(self, path: Path, payload: Mapping[str, object]) -> None:
        """Сохранить JSON-отчет выбора числа факторов."""

        with path.open("w", encoding=self.config.encoding) as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
            file_obj.write("\n")

    def _write_csv(
        self,
        path: Path,
        candidates: Sequence[LdaKSelectionCandidate],
        recommended_k: int,
    ) -> None:
        """Сохранить табличный CSV-отчет подбора ``K``."""

        fieldnames = [
            "k",
            "perplexity",
            "mean_coherence",
            "topic_diversity",
            "normalized_coherence",
            "normalized_inverse_perplexity",
            "normalized_topic_diversity",
            "selection_score",
            "is_recommended",
            "model_path",
            "metrics_json_path",
            "model_hash",
        ]
        with path.open("w", encoding=self.config.encoding, newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for candidate in candidates:
                writer.writerow(
                    {
                        "k": candidate.k,
                        "perplexity": self._format_float(candidate.perplexity),
                        "mean_coherence": self._format_float(
                            candidate.mean_coherence
                        ),
                        "topic_diversity": self._format_float(
                            candidate.topic_diversity
                        ),
                        "normalized_coherence": self._format_float(
                            candidate.normalized_coherence
                        ),
                        "normalized_inverse_perplexity": self._format_float(
                            candidate.normalized_inverse_perplexity
                        ),
                        "normalized_topic_diversity": self._format_float(
                            candidate.normalized_topic_diversity
                        ),
                        "selection_score": self._format_float(
                            candidate.selection_score
                        ),
                        "is_recommended": str(candidate.k == recommended_k).lower(),
                        "model_path": str(candidate.model_path),
                        "metrics_json_path": str(candidate.metrics_json_path),
                        "model_hash": candidate.model_hash,
                    }
                )

    def _write_markdown(
        self,
        path: Path,
        candidates: Sequence[LdaKSelectionCandidate],
        recommended_k: int,
    ) -> None:
        """Сохранить краткий Markdown-отчет для главы 4."""

        lines = [
            "# Отчет подбора числа латентных факторов K",
            "",
            "Модель: `LDA_prior`.",
            "",
            f"Рекомендуемое значение: **K = {recommended_k}**.",
            "",
            "| K | Perplexity | Coherence | Topic diversity | Score | Рекомендовано |",
            "|---:|---:|---:|---:|---:|:---:|",
        ]
        for candidate in candidates:
            lines.append(
                "| "
                f"{candidate.k} | "
                f"{self._format_float(candidate.perplexity)} | "
                f"{self._format_float(candidate.mean_coherence)} | "
                f"{self._format_float(candidate.topic_diversity)} | "
                f"{self._format_float(candidate.selection_score)} | "
                f"{'да' if candidate.k == recommended_k else 'нет'} |"
            )
        lines.extend(
            [
                "",
                "Интегральная оценка построена по нормированным значениям "
                "coherence, inverse perplexity и topic diversity.",
                "Анализ устойчивости по нескольким random_state выполняется "
                "отдельным этапом.",
                "",
            ]
        )
        path.write_text("\n".join(lines), encoding=self.config.encoding)

    def _format_float(self, value: float) -> str:
        """Стабильно отформатировать вещественное значение для CSV/Markdown."""

        return f"{value:.12g}"

    def _ensure_can_write(self, paths: Sequence[Path]) -> None:
        """Проверить возможность записи итоговых файлов подбора ``K``."""

        if self.config.overwrite:
            return
        existing_paths = [path for path in paths if path.exists()]
        if existing_paths:
            msg = "Запрещена перезапись существующих файлов: "
            msg += ", ".join(str(path) for path in existing_paths)
            raise FileExistsError(msg)
