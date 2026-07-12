"""Bootstrap-анализ статистической устойчивости результатов главы 6.

Модуль реализует этап 10 программного контура главы 6. Для неизмененных
прогнозов модели главы 5 и baseline-моделей выполняется парный bootstrap
по сценариям. Формируются доверительные интервалы основных метрик и
доверительные интервалы разностей между моделью главы 5 и baseline.

Фактические показатели используются исключительно для внешней проверки.
На этом этапе модели не переобучаются, прогнозы не перекалибровываются,
пороговые значения классов не изменяются.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from manual_coding_sim.validation.chapter6_config import Chapter6ValidationConfig


MODEL_ORDER: tuple[str, ...] = (
    "mean_baseline",
    "prior_only_baseline",
    "theta_only_baseline",
    "chapter5_model",
)
BASELINE_MODELS: tuple[str, ...] = MODEL_ORDER[:-1]
METRIC_ORDER: tuple[str, ...] = (
    "mae",
    "rmse",
    "spearman",
    "kendall",
    "accuracy",
    "macro_f1",
)
METRIC_DIRECTIONS: dict[str, str] = {
    "mae": "min",
    "rmse": "min",
    "spearman": "max",
    "kendall": "max",
    "accuracy": "max",
    "macro_f1": "max",
}
CLASS_LABELS: tuple[str, ...] = ("low", "medium", "high")
UNIT_INTERVAL_TOLERANCE = 1e-12


class BootstrapAnalysisError(ValueError):
    """Ошибка bootstrap-анализа статистической устойчивости."""


@dataclass(frozen=True)
class BootstrapAnalysisResult:
    """Результат bootstrap-анализа этапа 10."""

    confidence_intervals: pd.DataFrame
    model_differences: pd.DataFrame
    report: dict[str, Any]
    confidence_intervals_path: Path | None
    model_differences_path: Path | None
    json_path: Path | None
    markdown_path: Path | None

    @property
    def passed(self) -> bool:
        """Вернуть итоговый статус этапа 10."""

        return bool(self.report["passed"])


class BootstrapAnalysisValidator:
    """Выполнить парный bootstrap по единицам ``scenario_id``."""

    def __init__(
        self,
        config: Chapter6ValidationConfig,
        project_root: str | Path = ".",
    ) -> None:
        """Сохранить конфигурацию и корень проекта."""

        self.config = config
        self.project_root = Path(project_root)
        self.config.validate()

    def validate(
        self,
        predictions: pd.DataFrame | None = None,
    ) -> BootstrapAnalysisResult:
        """Рассчитать доверительные интервалы метрик и разностей моделей."""

        source = predictions.copy() if predictions is not None else self._load_predictions()
        source = source.reset_index(drop=True)
        self._validate_source(source)

        sampling_unit = self.config.bootstrap.sampling_unit
        grouped_indices = self._group_indices(source, sampling_unit)
        point_metrics = self._calculate_all_model_metrics(source)
        bootstrap_values = self._run_bootstrap(source, grouped_indices)

        confidence_intervals = self._build_confidence_interval_table(
            point_metrics=point_metrics,
            bootstrap_values=bootstrap_values,
        )
        model_differences = self._build_difference_table(
            point_metrics=point_metrics,
            bootstrap_values=bootstrap_values,
        )

        finite_intervals = self._numeric_frame_is_finite(confidence_intervals)
        finite_differences = self._numeric_frame_is_finite(model_differences)
        all_resamples_valid = bool(
            (confidence_intervals["valid_resamples"] == self.config.bootstrap.resamples).all()
            and (
                model_differences["valid_resamples"]
                == self.config.bootstrap.resamples
            ).all()
        )
        passed = bool(
            len(source) == self._expected_row_count()
            and finite_intervals
            and finite_differences
            and all_resamples_valid
        )

        chapter5_rows = confidence_intervals[
            confidence_intervals["model"] == "chapter5_model"
        ]
        stable_chapter5_wins = int(
            (model_differences["conclusion"] == "chapter5_model_favored").sum()
        )
        stable_baseline_wins = int(
            (model_differences["conclusion"] == "baseline_favored").sum()
        )
        inconclusive = int(
            (model_differences["conclusion"] == "no_stable_difference").sum()
        )

        report = {
            "stage": 10,
            "report_type": "bootstrap_statistical_stability",
            "passed": passed,
            "row_count": int(len(source)),
            "expected_row_count": self._expected_row_count(),
            "sampling": {
                "method": "paired_cluster_percentile_bootstrap",
                "sampling_unit": sampling_unit,
                "sampling_unit_count": int(len(grouped_indices)),
                "resamples": int(self.config.bootstrap.resamples),
                "confidence_level": float(self.config.bootstrap.confidence_level),
                "random_seed": int(self.config.bootstrap.random_seed),
                "sampled_units_per_resample": int(len(grouped_indices)),
                "replacement": True,
            },
            "metrics": list(METRIC_ORDER),
            "metric_directions": dict(METRIC_DIRECTIONS),
            "model_order": list(MODEL_ORDER),
            "delta_definition": "metric_chapter5_model - metric_baseline",
            "chapter5_confidence_intervals": chapter5_rows.to_dict(orient="records"),
            "confidence_intervals": confidence_intervals.to_dict(orient="records"),
            "model_differences": model_differences.to_dict(orient="records"),
            "summary": {
                "stable_chapter5_wins": stable_chapter5_wins,
                "stable_baseline_wins": stable_baseline_wins,
                "no_stable_difference": inconclusive,
                "comparison_count": int(len(model_differences)),
            },
            "methodological_checks": {
                "paired_resamples_used_for_all_models": True,
                "fixed_stage9_predictions_used": True,
                "models_refitted_inside_bootstrap": False,
                "chapter5_prediction_modified": False,
                "quality_thresholds_modified": False,
                "factual_values_used_only_for_external_validation": True,
            },
            "interpretation": {
                "confidence_intervals": (
                    "Использованы двусторонние percentile-доверительные интервалы."
                ),
                "difference_rule": (
                    "Для MAE и RMSE отрицательная разность в пользу модели главы 5; "
                    "для Spearman, Kendall, Accuracy и Macro F1 положительная "
                    "разность в пользу модели главы 5."
                ),
                "stable_difference": (
                    "Различие считается статистически устойчивым на заданном уровне "
                    "доверия, если доверительный интервал разности не включает ноль."
                ),
                "mean_baseline": (
                    "В bootstrap используются зафиксированные out-of-fold прогнозы "
                    "mean baseline этапа 9; повторное обучение внутри resample не "
                    "выполняется."
                ),
            },
        }

        return BootstrapAnalysisResult(
            confidence_intervals=confidence_intervals,
            model_differences=model_differences,
            report=report,
            confidence_intervals_path=None,
            model_differences_path=None,
            json_path=None,
            markdown_path=None,
        )

    def validate_and_save(
        self,
        predictions: pd.DataFrame | None = None,
    ) -> BootstrapAnalysisResult:
        """Выполнить анализ и сохранить CSV-, JSON- и Markdown-артефакты."""

        result = self.validate(predictions=predictions)
        confidence_intervals_path = self._resolve_output_path(
            (
                "bootstrap_confidence_intervals_path",
                "bootstrap_confidence_intervals",
            ),
            "reports/chapter6/bootstrap_confidence_intervals.csv",
        )
        model_differences_path = self._resolve_output_path(
            (
                "bootstrap_model_differences_path",
                "bootstrap_model_differences",
            ),
            "reports/chapter6/bootstrap_model_differences.csv",
        )
        json_path = self._resolve_output_path(
            (
                "bootstrap_report_json_path",
                "bootstrap_report_json",
            ),
            "reports/chapter6/bootstrap_report.json",
        )
        markdown_path = self._resolve_output_path(
            (
                "bootstrap_report_md_path",
                "bootstrap_report_md",
            ),
            "reports/chapter6/bootstrap_report.md",
        )

        for path in (
            confidence_intervals_path,
            model_differences_path,
            json_path,
            markdown_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)

        result.confidence_intervals.to_csv(
            confidence_intervals_path,
            index=False,
            encoding="utf-8",
        )
        result.model_differences.to_csv(
            model_differences_path,
            index=False,
            encoding="utf-8",
        )
        json_path.write_text(
            json.dumps(result.report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        markdown_path.write_text(
            self._build_markdown_report(result.report),
            encoding="utf-8",
        )

        return BootstrapAnalysisResult(
            confidence_intervals=result.confidence_intervals,
            model_differences=result.model_differences,
            report=result.report,
            confidence_intervals_path=confidence_intervals_path,
            model_differences_path=model_differences_path,
            json_path=json_path,
            markdown_path=markdown_path,
        )

    def _load_predictions(self) -> pd.DataFrame:
        """Загрузить зафиксированные прогнозы этапа 9."""

        path = self._resolve_output_path(
            ("baseline_predictions_path", "baseline_predictions"),
            "reports/chapter6/baseline_predictions.csv",
        )
        if not path.exists():
            raise FileNotFoundError(
                "Baseline-прогнозы этапа 9 не найдены: "
                f"{path}. Сначала выполните --compare-baselines."
            )
        try:
            return pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as error:
            raise BootstrapAnalysisError(
                f"Не удалось прочитать baseline-прогнозы: {path}"
            ) from error

    def _validate_source(self, source: pd.DataFrame) -> None:
        """Проверить структуру и числовую корректность входных прогнозов."""

        required = [
            *self._join_keys(),
            self.config.bootstrap.sampling_unit,
            "q_fact",
            *MODEL_ORDER,
        ]
        missing = [column for column in required if column not in source.columns]
        if missing:
            raise BootstrapAnalysisError(
                "В baseline_predictions.csv отсутствуют обязательные колонки: "
                + ", ".join(dict.fromkeys(missing))
            )

        if len(source) != self._expected_row_count():
            raise BootstrapAnalysisError(
                "Некорректное число строк baseline_predictions.csv: "
                f"ожидалось {self._expected_row_count()}, получено {len(source)}."
            )

        keys = list(self._join_keys())
        if source[keys].isna().any().any():
            raise BootstrapAnalysisError(
                "Составной ключ содержит пропущенные значения."
            )
        if source.duplicated(subset=keys).any():
            raise BootstrapAnalysisError(
                "Составной ключ scenario_id, protocol_id не является уникальным."
            )
        if source[self.config.bootstrap.sampling_unit].isna().any():
            raise BootstrapAnalysisError(
                "Единица bootstrap-выборки содержит пропущенные значения."
            )
        if source[self.config.bootstrap.sampling_unit].nunique(dropna=False) < 2:
            raise BootstrapAnalysisError(
                "Для bootstrap требуется не менее двух независимых единиц выборки."
            )

        numeric_columns = ["q_fact", *MODEL_ORDER]
        try:
            numeric = source[numeric_columns].apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as error:
            raise BootstrapAnalysisError(
                "Фактическое качество и прогнозы моделей должны быть числовыми."
            ) from error

        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise BootstrapAnalysisError(
                "Входные прогнозы содержат NaN, inf или -inf."
            )
        if (
            (values < -UNIT_INTERVAL_TOLERANCE)
            | (values > 1.0 + UNIT_INTERVAL_TOLERANCE)
        ).any():
            raise BootstrapAnalysisError(
                "Фактическое качество и прогнозы должны находиться в диапазоне [0; 1]."
            )

    def _group_indices(
        self,
        source: pd.DataFrame,
        sampling_unit: str,
    ) -> list[np.ndarray]:
        """Сгруппировать индексы строк по единице bootstrap-выборки."""

        grouped: list[np.ndarray] = []
        for _, group in source.groupby(sampling_unit, sort=False, dropna=False):
            grouped.append(group.index.to_numpy(dtype=int))
        return grouped

    def _run_bootstrap(
        self,
        source: pd.DataFrame,
        grouped_indices: Sequence[np.ndarray],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Сформировать распределения метрик на парных bootstrap-выборках."""

        rng = np.random.default_rng(self.config.bootstrap.random_seed)
        unit_count = len(grouped_indices)
        values: dict[str, dict[str, np.ndarray]] = {
            model: {
                metric: np.empty(self.config.bootstrap.resamples, dtype=float)
                for metric in METRIC_ORDER
            }
            for model in MODEL_ORDER
        }

        factual_full = source["q_fact"].to_numpy(dtype=float)
        predicted_full = {
            model: source[model].to_numpy(dtype=float) for model in MODEL_ORDER
        }

        for resample_index in range(self.config.bootstrap.resamples):
            selected_units = rng.integers(0, unit_count, size=unit_count)
            row_indices = np.concatenate(
                [grouped_indices[int(index)] for index in selected_units]
            )
            factual = factual_full[row_indices]
            for model in MODEL_ORDER:
                metrics = self._calculate_metrics(
                    predicted=predicted_full[model][row_indices],
                    factual=factual,
                )
                for metric in METRIC_ORDER:
                    values[model][metric][resample_index] = metrics[metric]

        return values

    def _calculate_all_model_metrics(
        self,
        source: pd.DataFrame,
    ) -> dict[str, dict[str, float]]:
        """Рассчитать точечные оценки метрик по полному корпусу."""

        factual = source["q_fact"].to_numpy(dtype=float)
        return {
            model: self._calculate_metrics(
                predicted=source[model].to_numpy(dtype=float),
                factual=factual,
            )
            for model in MODEL_ORDER
        }

    def _calculate_metrics(
        self,
        predicted: np.ndarray,
        factual: np.ndarray,
    ) -> dict[str, float]:
        """Рассчитать шесть метрик, включенных в bootstrap-анализ."""

        errors = predicted - factual
        factual_classes = np.asarray(
            [self._quality_class(value) for value in factual],
            dtype=object,
        )
        predicted_classes = np.asarray(
            [self._quality_class(value) for value in predicted],
            dtype=object,
        )
        classification = self._classification_metrics(
            factual_classes=factual_classes,
            predicted_classes=predicted_classes,
        )
        return {
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(math.sqrt(float(np.mean(np.square(errors))))),
            "spearman": self._spearman(predicted, factual),
            "kendall": self._kendall_tau_b(predicted, factual),
            "accuracy": classification["accuracy"],
            "macro_f1": classification["macro_f1"],
        }

    def _build_confidence_interval_table(
        self,
        point_metrics: Mapping[str, Mapping[str, float]],
        bootstrap_values: Mapping[str, Mapping[str, np.ndarray]],
    ) -> pd.DataFrame:
        """Сформировать таблицу доверительных интервалов всех моделей."""

        rows: list[dict[str, Any]] = []
        for model in MODEL_ORDER:
            for metric in METRIC_ORDER:
                distribution = bootstrap_values[model][metric]
                lower, upper = self._percentile_interval(distribution)
                rows.append(
                    {
                        "model": model,
                        "metric": metric,
                        "direction": METRIC_DIRECTIONS[metric],
                        "point_estimate": float(point_metrics[model][metric]),
                        "bootstrap_mean": float(np.mean(distribution)),
                        "bootstrap_std": float(np.std(distribution, ddof=1)),
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "confidence_level": float(
                            self.config.bootstrap.confidence_level
                        ),
                        "resamples": int(self.config.bootstrap.resamples),
                        "valid_resamples": int(np.isfinite(distribution).sum()),
                    }
                )
        return pd.DataFrame(rows)

    def _build_difference_table(
        self,
        point_metrics: Mapping[str, Mapping[str, float]],
        bootstrap_values: Mapping[str, Mapping[str, np.ndarray]],
    ) -> pd.DataFrame:
        """Сформировать парные bootstrap-разности с каждым baseline."""

        rows: list[dict[str, Any]] = []
        chapter5 = "chapter5_model"
        for baseline in BASELINE_MODELS:
            for metric in METRIC_ORDER:
                distribution = (
                    bootstrap_values[chapter5][metric]
                    - bootstrap_values[baseline][metric]
                )
                lower, upper = self._percentile_interval(distribution)
                point_delta = float(
                    point_metrics[chapter5][metric]
                    - point_metrics[baseline][metric]
                )
                includes_zero = bool(lower <= 0.0 <= upper)
                conclusion = self._difference_conclusion(
                    direction=METRIC_DIRECTIONS[metric],
                    lower=lower,
                    upper=upper,
                )
                rows.append(
                    {
                        "baseline": baseline,
                        "metric": metric,
                        "direction": METRIC_DIRECTIONS[metric],
                        "delta_definition": (
                            "metric_chapter5_model - metric_baseline"
                        ),
                        "point_delta": point_delta,
                        "bootstrap_mean_delta": float(np.mean(distribution)),
                        "bootstrap_std_delta": float(np.std(distribution, ddof=1)),
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "confidence_level": float(
                            self.config.bootstrap.confidence_level
                        ),
                        "resamples": int(self.config.bootstrap.resamples),
                        "valid_resamples": int(np.isfinite(distribution).sum()),
                        "ci_includes_zero": includes_zero,
                        "conclusion": conclusion,
                    }
                )
        return pd.DataFrame(rows)

    def _percentile_interval(self, values: np.ndarray) -> tuple[float, float]:
        """Рассчитать двусторонний percentile-интервал."""

        alpha = (1.0 - self.config.bootstrap.confidence_level) / 2.0
        lower = float(np.quantile(values, alpha))
        upper = float(np.quantile(values, 1.0 - alpha))
        return lower, upper

    def _difference_conclusion(
        self,
        direction: str,
        lower: float,
        upper: float,
    ) -> str:
        """Интерпретировать доверительный интервал парной разности."""

        if lower <= 0.0 <= upper:
            return "no_stable_difference"
        if direction == "min":
            return "chapter5_model_favored" if upper < 0.0 else "baseline_favored"
        return "chapter5_model_favored" if lower > 0.0 else "baseline_favored"

    def _classification_metrics(
        self,
        factual_classes: np.ndarray,
        predicted_classes: np.ndarray,
    ) -> dict[str, float]:
        """Рассчитать Accuracy и Macro F1 по трем фиксированным классам."""

        accuracy = float(np.mean(factual_classes == predicted_classes))
        f1_values: list[float] = []
        for label in CLASS_LABELS:
            factual_mask = factual_classes == label
            predicted_mask = predicted_classes == label
            true_positive = float(np.sum(factual_mask & predicted_mask))
            support = float(np.sum(factual_mask))
            predicted_count = float(np.sum(predicted_mask))
            precision = self._safe_divide(true_positive, predicted_count)
            recall = self._safe_divide(true_positive, support)
            f1_values.append(
                self._safe_divide(2.0 * precision * recall, precision + recall)
            )
        return {
            "accuracy": accuracy,
            "macro_f1": float(np.mean(f1_values)),
        }

    def _quality_class(self, value: float) -> str:
        """Преобразовать непрерывную оценку в low, medium или high."""

        numeric = float(value)
        if numeric < self.config.decision_thresholds.low_max:
            return "low"
        if numeric < self.config.decision_thresholds.high_min:
            return "medium"
        return "high"

    def _pearson(self, left: np.ndarray, right: np.ndarray) -> float:
        """Рассчитать корреляцию Пирсона с защитой от нулевой дисперсии."""

        left_centered = left - float(np.mean(left))
        right_centered = right - float(np.mean(right))
        denominator = math.sqrt(
            float(np.sum(np.square(left_centered)))
            * float(np.sum(np.square(right_centered)))
        )
        if denominator == 0.0:
            return 0.0
        return float(np.sum(left_centered * right_centered) / denominator)

    def _spearman(self, left: np.ndarray, right: np.ndarray) -> float:
        """Рассчитать корреляцию Спирмена по средним рангам."""

        left_ranks = pd.Series(left).rank(method="average").to_numpy(dtype=float)
        right_ranks = pd.Series(right).rank(method="average").to_numpy(dtype=float)
        return self._pearson(left_ranks, right_ranks)

    def _kendall_tau_b(self, left: np.ndarray, right: np.ndarray) -> float:
        """Рассчитать Kendall tau-b с учетом совпадающих рангов."""

        concordant = 0
        discordant = 0
        ties_left = 0
        ties_right = 0
        for first in range(len(left) - 1):
            left_diff = left[first + 1 :] - left[first]
            right_diff = right[first + 1 :] - right[first]
            left_sign = np.sign(left_diff)
            right_sign = np.sign(right_diff)
            products = left_sign * right_sign
            concordant += int(np.sum(products > 0))
            discordant += int(np.sum(products < 0))
            ties_left += int(np.sum((left_sign == 0) & (right_sign != 0)))
            ties_right += int(np.sum((right_sign == 0) & (left_sign != 0)))

        denominator = math.sqrt(
            float(concordant + discordant + ties_left)
            * float(concordant + discordant + ties_right)
        )
        if denominator == 0.0:
            return 0.0
        return float((concordant - discordant) / denominator)

    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """Выполнить деление, возвращая ноль при нулевом знаменателе."""

        if denominator == 0.0:
            return 0.0
        return float(numerator / denominator)

    def _numeric_frame_is_finite(self, frame: pd.DataFrame) -> bool:
        """Проверить конечность всех числовых полей таблицы."""

        numeric = frame.select_dtypes(include=[np.number])
        return bool(np.isfinite(numeric.to_numpy(dtype=float)).all())

    def _join_keys(self) -> tuple[str, ...]:
        """Вернуть составной ключ главы 6."""

        return tuple(self.config.merge.key_columns)

    def _expected_row_count(self) -> int:
        """Вернуть ожидаемое число сценариев."""

        return int(self.config.merge.expected_row_count)

    def _resolve_output_path(
        self,
        names: Sequence[str],
        fallback: str,
    ) -> Path:
        """Разрешить выходной путь без изменения API конфигурации этапа 1."""

        outputs = self.config.outputs
        for name in names:
            if hasattr(outputs, name):
                value = getattr(outputs, name)
                if value is not None:
                    path = Path(value)
                    return path if path.is_absolute() else self.project_root / path
        fallback_path = Path(fallback)
        return (
            fallback_path
            if fallback_path.is_absolute()
            else self.project_root / fallback_path
        )

    def _build_markdown_report(self, report: Mapping[str, Any]) -> str:
        """Сформировать человекочитаемый Markdown-отчет этапа 10."""

        sampling = report["sampling"]
        summary = report["summary"]
        lines = [
            "# Bootstrap-анализ статистической устойчивости",
            "",
            f"- Этап: **{report['stage']}**",
            f"- Статус: **{'пройден' if report['passed'] else 'не пройден'}**",
            f"- Сценариев: **{report['row_count']}**",
            f"- Bootstrap-повторов: **{sampling['resamples']}**",
            f"- Уровень доверия: **{sampling['confidence_level']:.2%}**",
            f"- Единица выборки: **{sampling['sampling_unit']}**",
            f"- Random seed: **{sampling['random_seed']}**",
            "",
            "## Доверительные интервалы модели главы 5",
            "",
            "| Метрика | Значение | Bootstrap mean | CI lower | CI upper |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in report["chapter5_confidence_intervals"]:
            lines.append(
                "| {metric} | {point_estimate:.10f} | {bootstrap_mean:.10f} | "
                "{ci_lower:.10f} | {ci_upper:.10f} |".format(**row)
            )

        lines.extend(
            [
                "",
                "## Парные разности модели главы 5 и baseline",
                "",
                "Разность определяется как `metric_chapter5_model - metric_baseline`.",
                "",
                "| Baseline | Метрика | Δ | CI lower | CI upper | Вывод |",
                "|---|---|---:|---:|---:|---|",
            ]
        )
        for row in report["model_differences"]:
            lines.append(
                "| {baseline} | {metric} | {point_delta:.10f} | "
                "{ci_lower:.10f} | {ci_upper:.10f} | {conclusion} |".format(
                    **row
                )
            )

        lines.extend(
            [
                "",
                "## Сводка статистически устойчивых различий",
                "",
                f"- Преимуществ модели главы 5: **{summary['stable_chapter5_wins']}**",
                f"- Преимуществ baseline: **{summary['stable_baseline_wins']}**",
                f"- Неустойчивых различий: **{summary['no_stable_difference']}**",
                "",
                "## Методическое ограничение",
                "",
                "Использован парный percentile-bootstrap по сценариям. "
                "Внутри повторных выборок применялись уже зафиксированные "
                "out-of-fold прогнозы этапа 9. Модели не переобучались, "
                "прогноз главы 5 не изменялся.",
                "",
            ]
        )
        return "\n".join(lines)
