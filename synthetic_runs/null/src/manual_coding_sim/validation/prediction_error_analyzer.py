"""Диагностический анализ ошибок априорного прогноза в главе 6.

Модуль реализует этап 11 программного контура главы 6. Он формирует
ранжированный список сценариев с наибольшей абсолютной ошибкой, оценивает
связь ошибок с неопределенностью, доминирующим LDA-фактором, классами
качества, условиями выполнения и фактическими диагностическими признаками.
Параметры модели главы 5 при этом не изменяются.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from manual_coding_sim.validation.chapter6_config import Chapter6ValidationConfig


CLASS_LABELS: tuple[str, ...] = ("low", "medium", "high")
THETA_COLUMNS: tuple[str, ...] = ("theta_0", "theta_1", "theta_2")
TOP_ERROR_COUNT = 10
UNIT_INTERVAL_TOLERANCE = 1e-12

REQUIRED_NUMERIC_COLUMNS: tuple[str, ...] = (
    "q_pred",
    "q_fact",
    "uncertainty_score",
    *THETA_COLUMNS,
    "prior_condition_noise_level_norm",
    "prior_condition_time_pressure_norm",
    "fact_duration_sec",
    "fact_error_count",
    "fact_recheck_count",
    "fact_reject_count",
    "fact_success",
)

UNIT_INTERVAL_COLUMNS: tuple[str, ...] = (
    "q_pred",
    "q_fact",
    "uncertainty_score",
    *THETA_COLUMNS,
    "prior_condition_noise_level_norm",
    "prior_condition_time_pressure_norm",
    "fact_success",
)

DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "uncertainty_score",
    "fact_duration_sec",
    "fact_error_count",
    "fact_recheck_count",
    "fact_reject_count",
    "fact_success",
    "prior_condition_noise_level_norm",
    "prior_condition_time_pressure_norm",
)

GROUP_DIMENSION_ORDER: tuple[str, ...] = (
    "factual_class",
    "predicted_class",
    "dominant_factor",
    "uncertainty_quantile",
    "noise_level",
    "time_pressure",
    "fact_success",
    "fact_error_count",
    "fact_recheck_count",
    "fact_reject_count",
    "fact_duration_quantile",
    "error_direction",
)


class PredictionErrorAnalysisError(ValueError):
    """Ошибка диагностического анализа ошибок прогноза."""


@dataclass(frozen=True)
class PredictionErrorAnalysisResult:
    """Результат анализа ошибок этапа 11."""

    top_errors: pd.DataFrame
    group_analysis: pd.DataFrame
    report: dict[str, Any]
    top_errors_path: Path | None
    group_analysis_path: Path | None
    json_path: Path | None
    markdown_path: Path | None

    @property
    def passed(self) -> bool:
        """Вернуть итоговый статус этапа."""

        return bool(self.report["passed"])


class PredictionErrorAnalyzer:
    """Анализатор сценариев и факторов, связанных с ошибкой прогноза."""

    def __init__(
        self,
        config: Chapter6ValidationConfig,
        project_root: str | Path = ".",
    ) -> None:
        """Сохранить конфигурацию и корень проекта."""

        self.config = config
        self.project_root = Path(project_root)
        self.config.validate()

    def analyze(
        self,
        dataset: pd.DataFrame | None = None,
    ) -> PredictionErrorAnalysisResult:
        """Выполнить полный диагностический анализ ошибок прогноза."""

        source = dataset.copy() if dataset is not None else self._load_dataset()
        prepared = self._validate_and_prepare(source)
        details = self._build_error_details(prepared)
        top_errors = self._build_top_errors(details)
        group_analysis = self._build_group_analysis(details)
        correlations = self._calculate_diagnostic_correlations(details)
        summary = self._build_summary(
            details=details,
            top_errors=top_errors,
            group_analysis=group_analysis,
            correlations=correlations,
        )

        all_group_counts_valid = all(
            int(
                group_analysis.loc[
                    group_analysis["analysis_dimension"] == dimension,
                    "count",
                ].sum()
            )
            == len(details)
            for dimension in GROUP_DIMENSION_ORDER
        )
        finite_group_metrics = bool(
            np.isfinite(
                group_analysis[
                    [
                        "mae",
                        "rmse",
                        "bias",
                        "median_absolute_error",
                        "max_absolute_error",
                        "q_pred_mean",
                        "q_fact_mean",
                        "uncertainty_mean",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        )
        finite_correlations = all(
            math.isfinite(float(value))
            for row in correlations
            for key, value in row.items()
            if key not in {"variable", "description"}
        )
        passed = (
            len(details) == self._expected_row_count()
            and len(top_errors) == min(TOP_ERROR_COUNT, len(details))
            and top_errors["absolute_error"].is_monotonic_decreasing
            and all_group_counts_valid
            and finite_group_metrics
            and finite_correlations
        )

        report = {
            "stage": 11,
            "report_type": "prediction_error_analysis",
            "passed": passed,
            "row_count": int(len(details)),
            "expected_row_count": self._expected_row_count(),
            "top_error_count": int(len(top_errors)),
            "error_definition": "prediction_error = q_pred - q_fact",
            "absolute_error_definition": "absolute_error = abs(prediction_error)",
            "summary": summary,
            "uncertainty_relation": self._relation_for(
                correlations,
                "uncertainty_score",
            ),
            "diagnostic_correlations": correlations,
            "group_dimensions": list(GROUP_DIMENSION_ORDER),
            "group_highlights": self._build_group_highlights(group_analysis),
            "top_errors": self._records(top_errors),
            "methodological_checks": {
                "chapter5_prediction_modified": False,
                "quality_thresholds_modified": False,
                "factual_values_used_only_for_external_validation": True,
                "dominant_factor_recalculated_from_theta": True,
                "stored_classes_revalidated": True,
                "analysis_is_diagnostic_not_calibrating": True,
            },
            "methodological_note": (
                "Анализ описывает структуру ошибок зафиксированного прогноза "
                "главы 5. Фактические показатели применяются только для "
                "внешней проверки и не используются для изменения модели, "
                "порогов или весов."
            ),
        }

        return PredictionErrorAnalysisResult(
            top_errors=top_errors,
            group_analysis=group_analysis,
            report=report,
            top_errors_path=None,
            group_analysis_path=None,
            json_path=None,
            markdown_path=None,
        )

    def analyze_and_save(
        self,
        dataset: pd.DataFrame | None = None,
    ) -> PredictionErrorAnalysisResult:
        """Выполнить анализ и сохранить CSV-, JSON- и Markdown-артефакты."""

        result = self.analyze(dataset=dataset)
        top_errors_path = self._resolve_output_path(
            ("top_prediction_errors_path", "top_prediction_errors"),
            "reports/chapter6/top_prediction_errors.csv",
        )
        group_analysis_path = self._resolve_output_path(
            ("error_group_analysis_path", "error_group_analysis"),
            "reports/chapter6/error_group_analysis.csv",
        )
        json_path = self._resolve_output_path(
            ("prediction_error_analysis_json_path", "prediction_error_analysis_json"),
            "reports/chapter6/prediction_error_analysis.json",
        )
        markdown_path = self._resolve_output_path(
            ("prediction_error_analysis_md_path", "prediction_error_analysis_md"),
            "reports/chapter6/prediction_error_analysis.md",
        )

        for path in (
            top_errors_path,
            group_analysis_path,
            json_path,
            markdown_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)

        result.top_errors.to_csv(top_errors_path, index=False, encoding="utf-8")
        result.group_analysis.to_csv(
            group_analysis_path,
            index=False,
            encoding="utf-8",
        )
        json_path.write_text(
            json.dumps(result.report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        markdown_path.write_text(
            self._build_markdown_report(result.report, result.group_analysis),
            encoding="utf-8",
        )

        return PredictionErrorAnalysisResult(
            top_errors=result.top_errors,
            group_analysis=result.group_analysis,
            report=result.report,
            top_errors_path=top_errors_path,
            group_analysis_path=group_analysis_path,
            json_path=json_path,
            markdown_path=markdown_path,
        )

    def _load_dataset(self) -> pd.DataFrame:
        """Загрузить проверочный датасет этапа 3."""

        path = self._resolve_output_path(
            ("validation_dataset_path", "validation_dataset"),
            "reports/chapter6/validation_dataset.csv",
        )
        if not path.exists():
            raise FileNotFoundError(
                "Проверочный датасет этапа 3 не найден: "
                f"{path}. Сначала выполните --build-validation-dataset."
            )
        try:
            return pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as error:
            raise PredictionErrorAnalysisError(
                f"Не удалось прочитать проверочный датасет: {path}"
            ) from error

    def _validate_and_prepare(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Проверить структуру, диапазоны и сохраненные аннотации."""

        required = [
            *self._join_keys(),
            *REQUIRED_NUMERIC_COLUMNS,
            "q_pred_class",
            "q_fact_class",
            "theta_dominant_topic",
        ]
        missing = [column for column in required if column not in dataset.columns]
        if missing:
            raise PredictionErrorAnalysisError(
                "В проверочном датасете отсутствуют обязательные колонки: "
                + ", ".join(missing)
            )
        if len(dataset) != self._expected_row_count():
            raise PredictionErrorAnalysisError(
                "Число строк проверочного датасета не соответствует "
                f"ожидаемому: {len(dataset)} вместо "
                f"{self._expected_row_count()}."
            )

        keys = list(self._join_keys())
        if dataset[keys].isna().any().any():
            raise PredictionErrorAnalysisError(
                "Составной ключ содержит пропущенные значения."
            )
        if dataset.duplicated(keys).any():
            raise PredictionErrorAnalysisError(
                "Составной ключ scenario_id + protocol_id не является уникальным."
            )

        prepared = dataset.copy()
        for column in REQUIRED_NUMERIC_COLUMNS:
            try:
                prepared[column] = pd.to_numeric(prepared[column], errors="raise")
            except (TypeError, ValueError) as error:
                raise PredictionErrorAnalysisError(
                    f"Колонка {column} содержит нечисловые значения."
                ) from error

        numeric_values = prepared[list(REQUIRED_NUMERIC_COLUMNS)].to_numpy(
            dtype=float
        )
        if np.isnan(numeric_values).any():
            raise PredictionErrorAnalysisError(
                "Числовые колонки содержат NaN."
            )
        if not np.isfinite(numeric_values).all():
            raise PredictionErrorAnalysisError(
                "Числовые колонки содержат inf или -inf."
            )

        for column in UNIT_INTERVAL_COLUMNS:
            values = prepared[column].to_numpy(dtype=float)
            if (
                values.min() < -UNIT_INTERVAL_TOLERANCE
                or values.max() > 1.0 + UNIT_INTERVAL_TOLERANCE
            ):
                raise PredictionErrorAnalysisError(
                    f"Колонка {column} содержит значения вне диапазона [0; 1]."
                )

        count_columns = (
            "fact_error_count",
            "fact_recheck_count",
            "fact_reject_count",
        )
        for column in count_columns:
            values = prepared[column].to_numpy(dtype=float)
            if (values < 0.0).any():
                raise PredictionErrorAnalysisError(
                    f"Колонка {column} содержит отрицательные значения."
                )
            if not np.allclose(values, np.round(values), atol=1e-12, rtol=0.0):
                raise PredictionErrorAnalysisError(
                    f"Колонка {column} должна содержать целые счетчики."
                )

        if (prepared["fact_duration_sec"] < 0.0).any():
            raise PredictionErrorAnalysisError(
                "Колонка fact_duration_sec содержит отрицательные значения."
            )
        if not prepared["fact_success"].isin((0, 1, 0.0, 1.0)).all():
            raise PredictionErrorAnalysisError(
                "Колонка fact_success должна содержать только 0 или 1."
            )

        q_pred_classes = prepared["q_pred"].map(self._quality_class)
        q_fact_classes = prepared["q_fact"].map(self._quality_class)
        if not prepared["q_pred_class"].astype(str).equals(q_pred_classes):
            raise PredictionErrorAnalysisError(
                "Сохраненная колонка q_pred_class не согласована с порогами."
            )
        if not prepared["q_fact_class"].astype(str).equals(q_fact_classes):
            raise PredictionErrorAnalysisError(
                "Сохраненная колонка q_fact_class не согласована с порогами."
            )

        dominant_factor = prepared[list(THETA_COLUMNS)].idxmax(axis=1)
        if not prepared["theta_dominant_topic"].astype(str).equals(
            dominant_factor
        ):
            raise PredictionErrorAnalysisError(
                "Сохраненная доминирующая тема не согласована с theta_0..theta_2."
            )
        return prepared

    def _build_error_details(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Добавить ошибки, аналитические группы и диагностические признаки."""

        details = dataset.copy()
        details["prediction_error"] = details["q_pred"] - details["q_fact"]
        details["absolute_error"] = details["prediction_error"].abs()
        details["squared_error"] = details["prediction_error"] ** 2
        details["error_direction"] = np.select(
            [
                details["prediction_error"] < -1e-15,
                details["prediction_error"] > 1e-15,
            ],
            ["underestimation", "overestimation"],
            default="exact",
        )
        details["dominant_factor"] = details[list(THETA_COLUMNS)].idxmax(axis=1)
        details["uncertainty_quantile"] = self._rank_quantiles(
            details["uncertainty_score"]
        )
        details["fact_duration_quantile"] = self._rank_quantiles(
            details["fact_duration_sec"]
        )
        details["noise_level"] = details[
            "prior_condition_noise_level_norm"
        ].map(self._normalized_level)
        details["time_pressure"] = details[
            "prior_condition_time_pressure_norm"
        ].map(self._normalized_level)
        details["fact_success_group"] = details["fact_success"].map(
            {0: "unsuccessful", 1: "successful", 0.0: "unsuccessful", 1.0: "successful"}
        )
        details["fact_error_count_group"] = details["fact_error_count"].map(
            self._error_count_group
        )
        details["fact_recheck_count_group"] = details[
            "fact_recheck_count"
        ].map(self._recheck_count_group)
        details["fact_reject_count_group"] = details[
            "fact_reject_count"
        ].map(self._reject_count_group)
        return details

    def _build_top_errors(self, details: pd.DataFrame) -> pd.DataFrame:
        """Сформировать top-10 сценариев по абсолютной ошибке."""

        columns = [
            *self._join_keys(),
            "q_pred",
            "q_fact",
            "prediction_error",
            "absolute_error",
            "squared_error",
            "error_direction",
            "q_pred_class",
            "q_fact_class",
            "uncertainty_score",
            "uncertainty_quantile",
            "dominant_factor",
            *THETA_COLUMNS,
            "prior_condition_noise_level_norm",
            "noise_level",
            "prior_condition_time_pressure_norm",
            "time_pressure",
            "fact_duration_sec",
            "fact_error_count",
            "fact_recheck_count",
            "fact_reject_count",
            "fact_success",
        ]
        top = (
            details.sort_values(
                ["absolute_error", *self._join_keys()],
                ascending=[False, True, True],
                kind="mergesort",
            )
            .head(TOP_ERROR_COUNT)
            .loc[:, columns]
            .reset_index(drop=True)
        )
        top.insert(0, "error_rank", np.arange(1, len(top) + 1, dtype=int))
        return top

    def _build_group_analysis(self, details: pd.DataFrame) -> pd.DataFrame:
        """Рассчитать метрики ошибок для всех диагностических срезов."""

        group_specs: tuple[tuple[str, str, Sequence[str]], ...] = (
            ("factual_class", "q_fact_class", CLASS_LABELS),
            ("predicted_class", "q_pred_class", CLASS_LABELS),
            ("dominant_factor", "dominant_factor", THETA_COLUMNS),
            (
                "uncertainty_quantile",
                "uncertainty_quantile",
                ("Q1", "Q2", "Q3", "Q4"),
            ),
            ("noise_level", "noise_level", ("low", "medium", "high")),
            (
                "time_pressure",
                "time_pressure",
                ("low", "medium", "high"),
            ),
            (
                "fact_success",
                "fact_success_group",
                ("unsuccessful", "successful"),
            ),
            (
                "fact_error_count",
                "fact_error_count_group",
                ("0", "1-2", "3-5", "6+"),
            ),
            (
                "fact_recheck_count",
                "fact_recheck_count_group",
                ("0", "1", "2+"),
            ),
            (
                "fact_reject_count",
                "fact_reject_count_group",
                ("0", "1+"),
            ),
            (
                "fact_duration_quantile",
                "fact_duration_quantile",
                ("Q1", "Q2", "Q3", "Q4"),
            ),
            (
                "error_direction",
                "error_direction",
                ("underestimation", "exact", "overestimation"),
            ),
        )

        rows: list[dict[str, Any]] = []
        for dimension, column, labels in group_specs:
            for label in labels:
                group = details.loc[details[column].astype(str) == str(label)]
                if group.empty:
                    continue
                rows.append(
                    self._group_metrics(
                        analysis_dimension=dimension,
                        group_label=str(label),
                        group=group,
                    )
                )
        frame = pd.DataFrame(rows)
        frame["share"] = frame["count"] / float(len(details))
        return frame

    @staticmethod
    def _group_metrics(
        analysis_dimension: str,
        group_label: str,
        group: pd.DataFrame,
    ) -> dict[str, Any]:
        """Рассчитать метрики одного диагностического среза."""

        errors = group["prediction_error"].to_numpy(dtype=float)
        absolute = np.abs(errors)
        return {
            "analysis_dimension": analysis_dimension,
            "group": group_label,
            "count": int(len(group)),
            "share": 0.0,
            "mae": float(np.mean(absolute)),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "bias": float(np.mean(errors)),
            "median_absolute_error": float(np.median(absolute)),
            "max_absolute_error": float(np.max(absolute)),
            "q_pred_mean": float(group["q_pred"].mean()),
            "q_fact_mean": float(group["q_fact"].mean()),
            "uncertainty_mean": float(group["uncertainty_score"].mean()),
            "underestimation_count": int((errors < -1e-15).sum()),
            "overestimation_count": int((errors > 1e-15).sum()),
            "exact_count": int((np.abs(errors) <= 1e-15).sum()),
        }

    def _calculate_diagnostic_correlations(
        self,
        details: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Оценить связь диагностических признаков с ошибками."""

        descriptions = {
            "uncertainty_score": "Априорная оценка неопределенности",
            "fact_duration_sec": "Фактическая длительность выполнения",
            "fact_error_count": "Фактическое число ошибок",
            "fact_recheck_count": "Фактическое число перепроверок",
            "fact_reject_count": "Фактическое число отказов",
            "fact_success": "Фактический признак успешности",
            "prior_condition_noise_level_norm": "Априорный уровень шума",
            "prior_condition_time_pressure_norm": "Априорное давление времени",
        }
        rows: list[dict[str, Any]] = []
        for variable in DIAGNOSTIC_COLUMNS:
            predictor = details[variable].to_numpy(dtype=float)
            signed_error = details["prediction_error"].to_numpy(dtype=float)
            absolute_error = details["absolute_error"].to_numpy(dtype=float)
            rows.append(
                {
                    "variable": variable,
                    "description": descriptions[variable],
                    "pearson_signed_error": self._safe_correlation(
                        predictor,
                        signed_error,
                        "pearson",
                    ),
                    "spearman_signed_error": self._safe_correlation(
                        predictor,
                        signed_error,
                        "spearman",
                    ),
                    "kendall_signed_error": self._safe_correlation(
                        predictor,
                        signed_error,
                        "kendall",
                    ),
                    "pearson_absolute_error": self._safe_correlation(
                        predictor,
                        absolute_error,
                        "pearson",
                    ),
                    "spearman_absolute_error": self._safe_correlation(
                        predictor,
                        absolute_error,
                        "spearman",
                    ),
                    "kendall_absolute_error": self._safe_correlation(
                        predictor,
                        absolute_error,
                        "kendall",
                    ),
                }
            )
        return rows

    @staticmethod
    def _safe_correlation(
        left: np.ndarray,
        right: np.ndarray,
        method: str,
    ) -> float:
        """Рассчитать корреляцию и вернуть ноль для константного признака."""

        if np.unique(left).size < 2 or np.unique(right).size < 2:
            return 0.0
        if method == "pearson":
            value = pearsonr(left, right).statistic
        elif method == "spearman":
            value = spearmanr(left, right).statistic
        elif method == "kendall":
            value = kendalltau(left, right, variant="b").statistic
        else:
            raise ValueError(f"Неизвестный метод корреляции: {method}")
        return 0.0 if value is None or not math.isfinite(float(value)) else float(value)

    def _build_summary(
        self,
        details: pd.DataFrame,
        top_errors: pd.DataFrame,
        group_analysis: pd.DataFrame,
        correlations: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Сформировать сводные показатели этапа 11."""

        errors = details["prediction_error"].to_numpy(dtype=float)
        absolute = np.abs(errors)
        total_absolute = float(np.sum(absolute))
        top_absolute = float(top_errors["absolute_error"].sum())

        diagnostic_candidates = [
            row for row in correlations if row["variable"] != "uncertainty_score"
        ]
        strongest_diagnostic = max(
            diagnostic_candidates,
            key=lambda row: abs(float(row["spearman_absolute_error"])),
        )
        topic_rows = group_analysis.loc[
            group_analysis["analysis_dimension"] == "dominant_factor"
        ]
        worst_topic = topic_rows.loc[topic_rows["mae"].idxmax()]
        uncertainty_rows = group_analysis.loc[
            group_analysis["analysis_dimension"] == "uncertainty_quantile"
        ]
        worst_uncertainty = uncertainty_rows.loc[
            uncertainty_rows["mae"].idxmax()
        ]
        top_topic = str(top_errors["dominant_factor"].mode().iloc[0])

        return {
            "mae": float(np.mean(absolute)),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "bias": float(np.mean(errors)),
            "median_absolute_error": float(np.median(absolute)),
            "max_absolute_error": float(np.max(absolute)),
            "underestimation_count": int((errors < -1e-15).sum()),
            "overestimation_count": int((errors > 1e-15).sum()),
            "exact_count": int((np.abs(errors) <= 1e-15).sum()),
            "top10_absolute_error_sum": top_absolute,
            "top10_share_of_total_absolute_error": (
                top_absolute / total_absolute if total_absolute > 0.0 else 0.0
            ),
            "top10_most_frequent_dominant_factor": top_topic,
            "worst_dominant_factor_by_mae": str(worst_topic["group"]),
            "worst_dominant_factor_mae": float(worst_topic["mae"]),
            "worst_uncertainty_quantile_by_mae": str(worst_uncertainty["group"]),
            "worst_uncertainty_quantile_mae": float(worst_uncertainty["mae"]),
            "strongest_diagnostic_absolute_error_relation": {
                "variable": strongest_diagnostic["variable"],
                "description": strongest_diagnostic["description"],
                "spearman": float(
                    strongest_diagnostic["spearman_absolute_error"]
                ),
            },
        }

    @staticmethod
    def _relation_for(
        correlations: Sequence[Mapping[str, Any]],
        variable: str,
    ) -> dict[str, Any]:
        """Вернуть корреляции заданного диагностического признака."""

        for row in correlations:
            if row["variable"] == variable:
                return dict(row)
        raise PredictionErrorAnalysisError(
            f"Не найдены корреляции для признака {variable}."
        )

    @staticmethod
    def _build_group_highlights(
        group_analysis: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Определить группу с максимальной MAE в каждом срезе."""

        highlights: list[dict[str, Any]] = []
        for dimension in GROUP_DIMENSION_ORDER:
            rows = group_analysis.loc[
                group_analysis["analysis_dimension"] == dimension
            ]
            worst = rows.loc[rows["mae"].idxmax()]
            highlights.append(
                {
                    "analysis_dimension": dimension,
                    "worst_group_by_mae": str(worst["group"]),
                    "count": int(worst["count"]),
                    "mae": float(worst["mae"]),
                    "bias": float(worst["bias"]),
                }
            )
        return highlights

    @staticmethod
    def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        """Преобразовать DataFrame в JSON-совместимые записи."""

        return json.loads(frame.to_json(orient="records", force_ascii=False))

    def _build_markdown_report(
        self,
        report: Mapping[str, Any],
        group_analysis: pd.DataFrame,
    ) -> str:
        """Сформировать человекочитаемый отчет этапа 11."""

        summary = report["summary"]
        uncertainty = report["uncertainty_relation"]
        status = "выполнен" if report["passed"] else "не выполнен"
        lines = [
            "# Анализ ошибок априорного прогноза",
            "",
            "## Итоговый статус",
            "",
            f"Расчетный этап **{status}**.",
            "",
            f"- этап: {report['stage']};",
            f"- сценариев: {report['row_count']};",
            f"- сценариев в top-10: {report['top_error_count']};",
            f"- определение ошибки: `{report['error_definition']}`.",
            "",
            "## Сводные показатели",
            "",
            "| Показатель | Значение |",
            "|---|---:|",
            f"| MAE | {summary['mae']:.10f} |",
            f"| RMSE | {summary['rmse']:.10f} |",
            f"| Bias | {summary['bias']:.10f} |",
            (
                "| Максимальная абсолютная ошибка | "
                f"{summary['max_absolute_error']:.10f} |"
            ),
            f"| Занижений | {summary['underestimation_count']} |",
            f"| Завышений | {summary['overestimation_count']} |",
            (
                "| Доля суммарной абсолютной ошибки в top-10 | "
                f"{summary['top10_share_of_total_absolute_error']:.10f} |"
            ),
            "",
            "## Связь ошибки с неопределенностью",
            "",
            "| Корреляция | Signed error | Absolute error |",
            "|---|---:|---:|",
            (
                "| Pearson | "
                f"{uncertainty['pearson_signed_error']:.10f} | "
                f"{uncertainty['pearson_absolute_error']:.10f} |"
            ),
            (
                "| Spearman | "
                f"{uncertainty['spearman_signed_error']:.10f} | "
                f"{uncertainty['spearman_absolute_error']:.10f} |"
            ),
            (
                "| Kendall tau-b | "
                f"{uncertainty['kendall_signed_error']:.10f} | "
                f"{uncertainty['kendall_absolute_error']:.10f} |"
            ),
            "",
            "## Top-10 сценариев по абсолютной ошибке",
            "",
            (
                "| Ранг | Сценарий | Протокол | Q_pred | Q_fact | "
                "Ошибка | Абс. ошибка | Фактор | Неопределенность |"
            ),
            "|---:|---|---|---:|---:|---:|---:|---|---:|",
        ]
        for row in report["top_errors"]:
            lines.append(
                "| {error_rank} | {scenario_id} | {protocol_id} | "
                "{q_pred:.10f} | {q_fact:.10f} | {prediction_error:.10f} | "
                "{absolute_error:.10f} | {dominant_factor} | "
                "{uncertainty_score:.10f} |".format(**row)
            )

        lines.extend(
            [
                "",
                "## Диагностические корреляции с абсолютной ошибкой",
                "",
                "| Признак | Pearson | Spearman | Kendall |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in report["diagnostic_correlations"]:
            lines.append(
                "| {description} | {pearson_absolute_error:.10f} | "
                "{spearman_absolute_error:.10f} | "
                "{kendall_absolute_error:.10f} |".format(**row)
            )

        lines.extend(
            [
                "",
                "## Группы с максимальной MAE",
                "",
                "| Срез | Группа | N | MAE | Bias |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in report["group_highlights"]:
            lines.append(
                "| {analysis_dimension} | {worst_group_by_mae} | {count} | "
                "{mae:.10f} | {bias:.10f} |".format(**row)
            )

        lines.extend(
            [
                "",
                "## Полная таблица групповых метрик",
                "",
                "| Срез | Группа | N | MAE | RMSE | Bias | Max AE |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in group_analysis.to_dict(orient="records"):
            lines.append(
                "| {analysis_dimension} | {group} | {count} | {mae:.10f} | "
                "{rmse:.10f} | {bias:.10f} | "
                "{max_absolute_error:.10f} |".format(**row)
            )

        lines.extend(
            [
                "",
                "## Методическое ограничение",
                "",
                str(report["methodological_note"]),
                "",
            ]
        )
        return "\n".join(lines)

    def _quality_class(self, value: float) -> str:
        """Определить класс качества по зафиксированным порогам."""

        if value < self.config.decision_thresholds.low_max:
            return "low"
        if value < self.config.decision_thresholds.high_min:
            return "medium"
        return "high"

    @staticmethod
    def _rank_quantiles(values: pd.Series) -> pd.Series:
        """Разделить значения на четыре равные по числу наблюдений группы."""

        ranks = values.rank(method="first")
        return pd.qcut(ranks, q=4, labels=("Q1", "Q2", "Q3", "Q4")).astype(str)

    @staticmethod
    def _normalized_level(value: float) -> str:
        """Преобразовать нормированный показатель в low, medium или high."""

        if value <= 1.0 / 3.0 + UNIT_INTERVAL_TOLERANCE:
            return "low"
        if value <= 2.0 / 3.0 + UNIT_INTERVAL_TOLERANCE:
            return "medium"
        return "high"

    @staticmethod
    def _error_count_group(value: float) -> str:
        """Сгруппировать фактическое число ошибок."""

        count = int(round(value))
        if count == 0:
            return "0"
        if count <= 2:
            return "1-2"
        if count <= 5:
            return "3-5"
        return "6+"

    @staticmethod
    def _recheck_count_group(value: float) -> str:
        """Сгруппировать число перепроверок."""

        count = int(round(value))
        if count == 0:
            return "0"
        if count == 1:
            return "1"
        return "2+"

    @staticmethod
    def _reject_count_group(value: float) -> str:
        """Сгруппировать число отказов."""

        return "0" if int(round(value)) == 0 else "1+"

    def _resolve_output_path(
        self,
        candidates: Sequence[str],
        default: str,
    ) -> Path:
        """Разрешить путь артефакта с поддержкой нескольких версий API."""

        configured: Any | None = None
        outputs = self.config.outputs
        for name in candidates:
            if hasattr(outputs, name):
                configured = getattr(outputs, name)
                break
        path = Path(configured) if configured is not None else Path(default)
        return path if path.is_absolute() else self.project_root / path

    def _join_keys(self) -> tuple[str, ...]:
        """Вернуть ключи объединения из конфигурации."""

        keys = getattr(self.config.merge, "key_columns", None)
        if keys is None:
            keys = getattr(self.config.merge, "keys", None)
        if keys is None:
            raise PredictionErrorAnalysisError(
                "В конфигурации отсутствуют ключи объединения."
            )
        return tuple(keys)

    def _expected_row_count(self) -> int:
        """Вернуть ожидаемое число сценариев."""

        return int(self.config.merge.expected_row_count)


__all__ = [
    "CLASS_LABELS",
    "DIAGNOSTIC_COLUMNS",
    "GROUP_DIMENSION_ORDER",
    "PredictionErrorAnalysisError",
    "PredictionErrorAnalysisResult",
    "PredictionErrorAnalyzer",
    "THETA_COLUMNS",
    "TOP_ERROR_COUNT",
]
