"""Проверка интервального прогноза качества в главе 6.

Модуль реализует этап 8 программного контура главы 6. Для каждого сценария
проверяется попадание фактического качества ``q_fact`` в априорный интервал
``[q_pred_lower; q_pred_upper]``. Дополнительно рассчитываются ширина
интервала, направление промаха, расстояние до ближайшей границы и срезы
по классам качества, доминирующему LDA-фактору и квартилям неопределенности.
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


CLASS_LABELS: tuple[str, ...] = ("low", "medium", "high")
THETA_COLUMNS: tuple[str, ...] = ("theta_0", "theta_1", "theta_2")
UNCERTAINTY_QUANTILES: tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4")
REQUIRED_VALUE_COLUMNS: tuple[str, ...] = (
    "q_pred",
    "q_fact",
    "q_pred_lower",
    "q_pred_upper",
    "uncertainty_score",
    *THETA_COLUMNS,
)
UNIT_INTERVAL_TOLERANCE = 1e-12


class IntervalPredictionValidationError(ValueError):
    """Ошибка проверки интервального прогноза."""


@dataclass(frozen=True)
class IntervalPredictionValidationResult:
    """Результат проверки интервального прогноза этапа 8."""

    details: pd.DataFrame
    report: dict[str, Any]
    details_path: Path | None
    json_path: Path | None
    markdown_path: Path | None

    @property
    def passed(self) -> bool:
        """Вернуть итоговый статус этапа."""

        return bool(self.report["passed"])


class IntervalPredictionValidator:
    """Валидатор покрытия и информативности прогнозных интервалов."""

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
        dataset: pd.DataFrame | None = None,
    ) -> IntervalPredictionValidationResult:
        """Рассчитать покрытие, ширину интервалов и аналитические срезы."""

        source = dataset.copy() if dataset is not None else self._load_dataset()
        numeric = self._validate_and_convert_source(source)

        keys = list(self._join_keys())
        details = numeric[keys + list(REQUIRED_VALUE_COLUMNS)].copy()
        details["interval_width"] = (
            details["q_pred_upper"] - details["q_pred_lower"]
        )
        details["interval_center"] = (
            details["q_pred_lower"] + details["q_pred_upper"]
        ) / 2.0
        details["is_covered"] = (
            (details["q_fact"] >= details["q_pred_lower"])
            & (details["q_fact"] <= details["q_pred_upper"])
        )
        details["miss_direction"] = np.select(
            [
                details["q_fact"] < details["q_pred_lower"],
                details["q_fact"] > details["q_pred_upper"],
            ],
            ["below_lower", "above_upper"],
            default="covered",
        )
        details["distance_to_interval"] = np.select(
            [
                details["q_fact"] < details["q_pred_lower"],
                details["q_fact"] > details["q_pred_upper"],
            ],
            [
                details["q_pred_lower"] - details["q_fact"],
                details["q_fact"] - details["q_pred_upper"],
            ],
            default=0.0,
        ).astype(float)
        details["q_pred_class"] = details["q_pred"].map(self._quality_class)
        details["q_fact_class"] = details["q_fact"].map(self._quality_class)
        details["dominant_factor"] = details[list(THETA_COLUMNS)].idxmax(axis=1)
        details["uncertainty_quantile"] = self._assign_uncertainty_quantiles(
            details["uncertainty_score"]
        )

        self._validate_existing_annotations(source, details)

        metrics = self._calculate_interval_metrics(details)
        slices = {
            "by_factual_class": self._calculate_slices(
                details,
                "q_fact_class",
                CLASS_LABELS,
            ),
            "by_predicted_class": self._calculate_slices(
                details,
                "q_pred_class",
                CLASS_LABELS,
            ),
            "by_dominant_factor": self._calculate_slices(
                details,
                "dominant_factor",
                THETA_COLUMNS,
            ),
            "by_uncertainty_quantile": self._calculate_slices(
                details,
                "uncertainty_quantile",
                UNCERTAINTY_QUANTILES,
                include_uncertainty_bounds=True,
            ),
        }

        finite_metrics = all(
            math.isfinite(float(value)) for value in metrics.values()
        )
        slice_counts_are_valid = all(
            sum(int(row["count"]) for row in rows) == len(details)
            for rows in slices.values()
        )
        passed = (
            len(details) == self._expected_row_count()
            and metrics["covered_count"] + metrics["miss_count"] == len(details)
            and metrics["miss_lower_count"] + metrics["miss_upper_count"]
            == metrics["miss_count"]
            and finite_metrics
            and slice_counts_are_valid
        )

        report = {
            "stage": 8,
            "report_type": "interval_prediction_validation",
            "passed": passed,
            "row_count": int(len(details)),
            "expected_row_count": self._expected_row_count(),
            "coverage_condition": (
                "q_pred_lower <= q_fact <= q_pred_upper"
            ),
            "distance_definition": (
                "0 внутри интервала; иначе расстояние до ближайшей границы"
            ),
            "metrics": metrics,
            "slices": slices,
            "uncertainty_quantile_method": (
                "Четыре равные по числу сценариев группы по рангу "
                "uncertainty_score: Q1 — наименьшая неопределенность, "
                "Q4 — наибольшая."
            ),
            "interpretation": {
                "coverage_rate": (
                    "Доля сценариев, для которых фактическое качество попало "
                    "в априорный прогнозный интервал."
                ),
                "mean_interval_width": (
                    "Средняя ширина интервала характеризует его "
                    "информативность: чрезмерно широкий интервал может давать "
                    "высокое покрытие без достаточной точности."
                ),
                "miss_directions": (
                    "below_lower означает, что фактическое качество ниже "
                    "интервала; above_upper — что фактическое качество выше "
                    "верхней границы интервала."
                ),
            },
            "methodological_note": (
                "Интервалы главы 5 проверяются без изменения их границ и без "
                "подгонки параметров по фактическим данным. Низкое покрытие "
                "является экспериментальным результатом, а не программной "
                "ошибкой этапа."
            ),
        }

        return IntervalPredictionValidationResult(
            details=details,
            report=report,
            details_path=None,
            json_path=None,
            markdown_path=None,
        )

    def validate_and_save(
        self,
        dataset: pd.DataFrame | None = None,
    ) -> IntervalPredictionValidationResult:
        """Выполнить проверку и сохранить CSV-, JSON- и Markdown-артефакты."""

        result = self.validate(dataset=dataset)
        details_path = self._resolve_output_path(
            ("interval_coverage_details_path", "interval_coverage_details"),
            "reports/chapter6/interval_coverage_details.csv",
        )
        json_path = self._resolve_output_path(
            ("interval_coverage_report_json_path", "interval_coverage_report_json"),
            "reports/chapter6/interval_coverage_report.json",
        )
        markdown_path = self._resolve_output_path(
            ("interval_coverage_report_md_path", "interval_coverage_report_md"),
            "reports/chapter6/interval_coverage_report.md",
        )

        for path in (details_path, json_path, markdown_path):
            path.parent.mkdir(parents=True, exist_ok=True)

        result.details.to_csv(details_path, index=False, encoding="utf-8")
        json_path.write_text(
            json.dumps(result.report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        markdown_path.write_text(
            self._build_markdown_report(result.report),
            encoding="utf-8",
        )

        return IntervalPredictionValidationResult(
            details=result.details,
            report=result.report,
            details_path=details_path,
            json_path=json_path,
            markdown_path=markdown_path,
        )

    def _load_dataset(self) -> pd.DataFrame:
        """Загрузить проверочный датасет, сформированный на этапе 3."""

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
            raise IntervalPredictionValidationError(
                f"Не удалось прочитать проверочный датасет: {path}"
            ) from error

    def _validate_and_convert_source(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Проверить структуру датасета и преобразовать числовые колонки."""

        required = [*self._join_keys(), *REQUIRED_VALUE_COLUMNS]
        missing = [column for column in required if column not in dataset.columns]
        if missing:
            raise IntervalPredictionValidationError(
                "В validation_dataset.csv отсутствуют обязательные колонки: "
                + ", ".join(missing)
            )

        if len(dataset) != self._expected_row_count():
            raise IntervalPredictionValidationError(
                "Некорректное число строк проверочного датасета: "
                f"ожидалось {self._expected_row_count()}, получено {len(dataset)}."
            )

        keys = list(self._join_keys())
        if dataset[keys].isna().any().any():
            raise IntervalPredictionValidationError(
                "Составной ключ содержит пропущенные значения."
            )
        if dataset.duplicated(subset=keys).any():
            raise IntervalPredictionValidationError(
                "Составной ключ проверочного датасета не является уникальным."
            )

        converted = dataset.copy()
        try:
            for column in REQUIRED_VALUE_COLUMNS:
                converted[column] = pd.to_numeric(
                    converted[column],
                    errors="raise",
                )
        except (TypeError, ValueError) as error:
            raise IntervalPredictionValidationError(
                "Колонки интервального прогноза должны содержать числовые значения."
            ) from error

        values = converted[list(REQUIRED_VALUE_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise IntervalPredictionValidationError(
                "Интервальный прогноз содержит NaN, inf или -inf."
            )
        if (
            (values < -UNIT_INTERVAL_TOLERANCE)
            | (values > 1.0 + UNIT_INTERVAL_TOLERANCE)
        ).any():
            raise IntervalPredictionValidationError(
                "Показатели качества, неопределенности и theta должны "
                "находиться в диапазоне [0; 1]."
            )

        invalid_order = converted["q_pred_lower"] > converted["q_pred_upper"]
        if invalid_order.any():
            raise IntervalPredictionValidationError(
                "Нижняя граница прогнозного интервала превышает верхнюю."
            )

        q_pred_outside = (
            (converted["q_pred"] < converted["q_pred_lower"])
            | (converted["q_pred"] > converted["q_pred_upper"])
        )
        if q_pred_outside.any():
            raise IntervalPredictionValidationError(
                "Точечный прогноз q_pred должен находиться внутри собственного "
                "прогнозного интервала."
            )

        return converted

    def _validate_existing_annotations(
        self,
        source: pd.DataFrame,
        calculated: pd.DataFrame,
    ) -> None:
        """Проверить сохраненные классы и доминирующий латентный фактор."""

        comparisons = (
            ("q_pred_class", calculated["q_pred_class"], set(CLASS_LABELS)),
            ("q_fact_class", calculated["q_fact_class"], set(CLASS_LABELS)),
            (
                "theta_dominant_topic",
                calculated["dominant_factor"],
                set(THETA_COLUMNS),
            ),
        )
        for column, expected, allowed in comparisons:
            if column not in source.columns:
                continue
            existing = source[column].astype(str)
            invalid = sorted(set(existing) - allowed)
            if invalid:
                raise IntervalPredictionValidationError(
                    f"Колонка {column} содержит неизвестные значения: "
                    + ", ".join(invalid)
                )
            mismatch_count = int(
                (existing.to_numpy() != expected.astype(str).to_numpy()).sum()
            )
            if mismatch_count:
                raise IntervalPredictionValidationError(
                    f"Колонка {column} не соответствует повторному расчету: "
                    f"расхождений {mismatch_count}."
                )

    def _assign_uncertainty_quantiles(self, values: pd.Series) -> pd.Series:
        """Разделить сценарии на четыре равные группы по рангу неопределенности."""

        if len(values) < len(UNCERTAINTY_QUANTILES):
            raise IntervalPredictionValidationError(
                "Для квартильного анализа требуется не менее четырех сценариев."
            )
        ranks = values.rank(method="first")
        try:
            quantiles = pd.qcut(
                ranks,
                q=len(UNCERTAINTY_QUANTILES),
                labels=UNCERTAINTY_QUANTILES,
            )
        except ValueError as error:
            raise IntervalPredictionValidationError(
                "Не удалось сформировать квартили uncertainty_score."
            ) from error
        return quantiles.astype(str)

    def _calculate_interval_metrics(
        self,
        details: pd.DataFrame,
    ) -> dict[str, float | int]:
        """Рассчитать основные метрики покрытия прогнозного интервала."""

        covered = details["is_covered"].astype(bool)
        missed = ~covered
        distances = details["distance_to_interval"].to_numpy(dtype=float)
        miss_distances = distances[missed.to_numpy()]
        return {
            "coverage_rate": float(covered.mean()),
            "covered_count": int(covered.sum()),
            "miss_count": int(missed.sum()),
            "mean_interval_width": float(details["interval_width"].mean()),
            "median_interval_width": float(details["interval_width"].median()),
            "min_interval_width": float(details["interval_width"].min()),
            "max_interval_width": float(details["interval_width"].max()),
            "miss_lower_count": int(
                (details["miss_direction"] == "below_lower").sum()
            ),
            "miss_upper_count": int(
                (details["miss_direction"] == "above_upper").sum()
            ),
            "mean_distance_to_interval": float(distances.mean()),
            "mean_miss_distance": (
                float(miss_distances.mean()) if len(miss_distances) else 0.0
            ),
            "max_distance_to_interval": float(distances.max()),
        }

    def _calculate_slices(
        self,
        details: pd.DataFrame,
        group_column: str,
        order: Sequence[str],
        *,
        include_uncertainty_bounds: bool = False,
    ) -> list[dict[str, Any]]:
        """Рассчитать метрики покрытия внутри заданных аналитических групп."""

        rows: list[dict[str, Any]] = []
        for label in order:
            group = details.loc[details[group_column].astype(str) == str(label)]
            if group.empty:
                continue
            missed = group.loc[~group["is_covered"]]
            row: dict[str, Any] = {
                "group": str(label),
                "count": int(len(group)),
                "covered_count": int(group["is_covered"].sum()),
                "coverage_rate": float(group["is_covered"].mean()),
                "mean_interval_width": float(group["interval_width"].mean()),
                "median_interval_width": float(group["interval_width"].median()),
                "miss_lower_count": int(
                    (group["miss_direction"] == "below_lower").sum()
                ),
                "miss_upper_count": int(
                    (group["miss_direction"] == "above_upper").sum()
                ),
                "mean_distance_to_interval": float(
                    group["distance_to_interval"].mean()
                ),
                "mean_miss_distance": (
                    float(missed["distance_to_interval"].mean())
                    if len(missed)
                    else 0.0
                ),
                "mean_uncertainty_score": float(
                    group["uncertainty_score"].mean()
                ),
                "mean_q_pred": float(group["q_pred"].mean()),
                "mean_q_fact": float(group["q_fact"].mean()),
            }
            if include_uncertainty_bounds:
                row["uncertainty_score_min"] = float(
                    group["uncertainty_score"].min()
                )
                row["uncertainty_score_max"] = float(
                    group["uncertainty_score"].max()
                )
            rows.append(row)
        return rows

    def _quality_class(self, value: float) -> str:
        """Преобразовать непрерывное качество в один из трех классов."""

        numeric = float(value)
        if numeric < self.config.decision_thresholds.low_max:
            return "low"
        if numeric < self.config.decision_thresholds.high_min:
            return "medium"
        return "high"

    def _build_markdown_report(self, report: Mapping[str, Any]) -> str:
        """Сформировать человекочитаемый отчет этапа 8."""

        status = "выполнен" if report["passed"] else "не выполнен"
        metrics = report["metrics"]
        lines = [
            "# Проверка интервального прогноза качества",
            "",
            "## Итоговый статус",
            "",
            f"Расчетный этап **{status}**.",
            "",
            f"- этап: {report['stage']};",
            f"- сценариев: {report['row_count']};",
            f"- условие покрытия: `{report['coverage_condition']}`.",
            "",
            "## Основные метрики",
            "",
            "| Метрика | Значение |",
            "|---|---:|",
            f"| Coverage rate | {metrics['coverage_rate']:.10f} |",
            f"| Покрыто сценариев | {metrics['covered_count']} |",
            f"| Всего промахов | {metrics['miss_count']} |",
            (
                "| Средняя ширина интервала | "
                f"{metrics['mean_interval_width']:.10f} |"
            ),
            (
                "| Медианная ширина интервала | "
                f"{metrics['median_interval_width']:.10f} |"
            ),
            f"| Факт ниже нижней границы | {metrics['miss_lower_count']} |",
            f"| Факт выше верхней границы | {metrics['miss_upper_count']} |",
            (
                "| Среднее расстояние до интервала | "
                f"{metrics['mean_distance_to_interval']:.10f} |"
            ),
            (
                "| Среднее расстояние среди промахов | "
                f"{metrics['mean_miss_distance']:.10f} |"
            ),
            "",
        ]

        slice_titles = (
            ("by_factual_class", "По фактическому классу"),
            ("by_predicted_class", "По прогнозному классу"),
            ("by_dominant_factor", "По доминирующему LDA-фактору"),
            ("by_uncertainty_quantile", "По квартилю неопределенности"),
        )
        for key, title in slice_titles:
            lines.extend(
                [
                    f"## {title}",
                    "",
                    (
                        "| Группа | N | Coverage | Средняя ширина | "
                        "Ниже | Выше | Среднее расстояние |"
                    ),
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in report["slices"][key]:
                lines.append(
                    "| {group} | {count} | {coverage_rate:.10f} | "
                    "{mean_interval_width:.10f} | {miss_lower_count} | "
                    "{miss_upper_count} | {mean_distance_to_interval:.10f} |".format(
                        **row
                    )
                )
            lines.append("")

        lines.extend(
            [
                "## Методическое ограничение",
                "",
                report["methodological_note"],
                "",
                "## Квартили неопределенности",
                "",
                report["uncertainty_quantile_method"],
                "",
            ]
        )
        return "\n".join(lines)

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
            raise IntervalPredictionValidationError(
                "В конфигурации отсутствуют ключи объединения."
            )
        return tuple(keys)

    def _expected_row_count(self) -> int:
        """Вернуть ожидаемое число сценариев."""

        return int(self.config.merge.expected_row_count)


__all__ = [
    "CLASS_LABELS",
    "THETA_COLUMNS",
    "UNCERTAINTY_QUANTILES",
    "IntervalPredictionValidationError",
    "IntervalPredictionValidationResult",
    "IntervalPredictionValidator",
]
