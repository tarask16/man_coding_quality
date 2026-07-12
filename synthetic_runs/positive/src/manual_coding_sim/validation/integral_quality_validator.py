"""Проверка согласованности фактического интегрального качества главы 6.

Модуль реализует этап 4 программного контура главы 6. Готовый показатель
``integral_quality`` сохраняется как основной фактический показатель
``Q_fact``. Независимая контрольная агрегация шести частных критериев
используется только для проверки внутренней согласованности и не заменяет
целевую переменную главы 3.
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


QUALITY_WEIGHTS: dict[str, float] = {
    "q_acc": 0.1666666667,
    "q_time": 0.1666666667,
    "q_effort": 0.1666666667,
    "q_res": 0.1666666667,
    "q_rep": 0.1666666666,
    "q_fit": 0.1666666666,
}
DEFAULT_CONSISTENCY_TOLERANCE = 0.05
DEFAULT_Q_FACT_ALIAS_TOLERANCE = 1e-12


class IntegralQualityValidationError(ValueError):
    """Ошибка проверки фактического интегрального качества."""


@dataclass(frozen=True)
class IntegralQualityValidationResult:
    """Результат контрольного расчета фактического качества."""

    details: pd.DataFrame
    report: dict[str, Any]
    csv_path: Path | None
    json_path: Path | None
    markdown_path: Path | None

    @property
    def passed(self) -> bool:
        """Вернуть итоговый статус проверки."""

        return bool(self.report["passed"])


class IntegralQualityValidator:
    """Валидатор основного и контрольного показателей ``Q_fact``."""

    def __init__(
        self,
        config: Chapter6ValidationConfig,
        project_root: str | Path = ".",
        *,
        tolerance: float | None = None,
    ) -> None:
        """Сохранить конфигурацию, корень проекта и числовой допуск."""

        self.config = config
        self.project_root = Path(project_root)
        self.config.validate()
        self.tolerance = self._resolve_tolerance(tolerance)

    def validate(
        self,
        dataset: pd.DataFrame | None = None,
    ) -> IntegralQualityValidationResult:
        """Проверить согласованность ``integral_quality`` и контрольной суммы."""

        source = dataset.copy() if dataset is not None else self._load_dataset()
        self._validate_source(source)

        keys = list(self._join_keys())
        quality_columns = list(QUALITY_WEIGHTS)
        selected_columns = keys + quality_columns + ["integral_quality", "q_fact"]
        details = source[selected_columns].copy()

        for column in quality_columns + ["integral_quality", "q_fact"]:
            details[column] = pd.to_numeric(details[column], errors="raise")

        control = pd.Series(0.0, index=details.index, dtype=float)
        for column, weight in QUALITY_WEIGHTS.items():
            control = control + details[column] * weight

        details["q_fact_control"] = control
        details["consistency_difference"] = (
            details["q_fact_control"] - details["integral_quality"]
        )
        details["absolute_difference"] = details["consistency_difference"].abs()
        details["within_tolerance"] = (
            details["absolute_difference"] <= self.tolerance
        )

        alias_difference = (details["q_fact"] - details["integral_quality"]).abs()
        max_alias_difference = float(alias_difference.max())
        max_absolute_difference = float(details["absolute_difference"].max())
        mean_absolute_difference = float(details["absolute_difference"].mean())
        mean_signed_difference = float(details["consistency_difference"].mean())
        rmse_difference = float(
            math.sqrt(float(np.mean(np.square(details["consistency_difference"]))))
        )
        within_count = int(details["within_tolerance"].sum())
        outside_count = int(len(details) - within_count)
        passed = (
            len(details) == self._expected_row_count()
            and max_alias_difference <= DEFAULT_Q_FACT_ALIAS_TOLERANCE
            and outside_count == 0
        )

        report = {
            "stage": 4,
            "report_type": "integral_quality_consistency_report",
            "passed": passed,
            "row_count": int(len(details)),
            "expected_row_count": self._expected_row_count(),
            "q_fact_source": "integral_quality",
            "control_aggregation": "weighted_sum_of_partial_quality_criteria",
            "criterion_columns": quality_columns,
            "weights": QUALITY_WEIGHTS.copy(),
            "weight_sum": float(sum(QUALITY_WEIGHTS.values())),
            "consistency_tolerance": self.tolerance,
            "q_fact_alias_tolerance": DEFAULT_Q_FACT_ALIAS_TOLERANCE,
            "metrics": {
                "mean_signed_difference": mean_signed_difference,
                "mean_absolute_difference": mean_absolute_difference,
                "max_absolute_difference": max_absolute_difference,
                "rmse_difference": rmse_difference,
                "max_q_fact_alias_difference": max_alias_difference,
                "within_tolerance_count": within_count,
                "outside_tolerance_count": outside_count,
                "consistency_rate": float(within_count / len(details)),
            },
            "methodological_note": (
                "integral_quality остается основной фактической целевой "
                "переменной. Контрольная агрегация используется только для "
                "диагностики согласованности и не подменяет Q_fact."
            ),
        }

        return IntegralQualityValidationResult(
            details=details,
            report=report,
            csv_path=None,
            json_path=None,
            markdown_path=None,
        )

    def validate_and_save(
        self,
        dataset: pd.DataFrame | None = None,
    ) -> IntegralQualityValidationResult:
        """Выполнить проверку и сохранить CSV-, JSON- и Markdown-артефакты."""

        result = self.validate(dataset=dataset)
        csv_path = self._resolve_output_path(
            (
                "integral_quality_consistency_path",
                "integral_quality_consistency",
            ),
            "reports/chapter6/integral_quality_consistency.csv",
        )
        json_path = self._resolve_output_path(
            (
                "integral_quality_report_json_path",
                "integral_quality_consistency_report_json_path",
                "integral_quality_report_json",
            ),
            "reports/chapter6/integral_quality_consistency_report.json",
        )
        markdown_path = self._resolve_output_path(
            (
                "integral_quality_report_md_path",
                "integral_quality_consistency_report_md_path",
                "integral_quality_report_md",
            ),
            "reports/chapter6/integral_quality_consistency_report.md",
        )

        for path in (csv_path, json_path, markdown_path):
            path.parent.mkdir(parents=True, exist_ok=True)

        result.details.to_csv(csv_path, index=False, encoding="utf-8")
        json_path.write_text(
            json.dumps(result.report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        markdown_path.write_text(
            self._build_markdown_report(result.report),
            encoding="utf-8",
        )

        return IntegralQualityValidationResult(
            details=result.details,
            report=result.report,
            csv_path=csv_path,
            json_path=json_path,
            markdown_path=markdown_path,
        )

    def _load_dataset(self) -> pd.DataFrame:
        """Загрузить сформированный на этапе 3 проверочный датасет."""

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
            raise IntegralQualityValidationError(
                f"Не удалось прочитать проверочный датасет: {path}"
            ) from error

    def _validate_source(self, dataset: pd.DataFrame) -> None:
        """Проверить структуру и числовую корректность входного датасета."""

        required = [
            *self._join_keys(),
            *QUALITY_WEIGHTS,
            "integral_quality",
            "q_fact",
        ]
        missing = [column for column in required if column not in dataset.columns]
        if missing:
            raise IntegralQualityValidationError(
                "В validation_dataset.csv отсутствуют обязательные колонки: "
                + ", ".join(missing)
            )

        if len(dataset) != self._expected_row_count():
            raise IntegralQualityValidationError(
                "Некорректное число строк проверочного датасета: "
                f"ожидалось {self._expected_row_count()}, получено {len(dataset)}."
            )

        keys = list(self._join_keys())
        if dataset[keys].isna().any().any():
            raise IntegralQualityValidationError(
                "Составной ключ содержит пропущенные значения."
            )
        if dataset.duplicated(subset=keys).any():
            raise IntegralQualityValidationError(
                "Составной ключ scenario_id, protocol_id не является уникальным."
            )

        numeric_columns = list(QUALITY_WEIGHTS) + ["integral_quality", "q_fact"]
        try:
            numeric = dataset[numeric_columns].apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as error:
            raise IntegralQualityValidationError(
                "Показатели качества должны содержать только числовые значения."
            ) from error
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise IntegralQualityValidationError(
                "Показатели качества содержат NaN, inf или -inf."
            )
        if ((numeric < 0.0) | (numeric > 1.0)).any().any():
            raise IntegralQualityValidationError(
                "Показатели качества должны находиться в диапазоне [0; 1]."
            )

        alias_difference = (numeric["q_fact"] - numeric["integral_quality"]).abs()
        if float(alias_difference.max()) > DEFAULT_Q_FACT_ALIAS_TOLERANCE:
            raise IntegralQualityValidationError(
                "Колонка q_fact не является точной копией integral_quality."
            )

    def _build_markdown_report(self, report: Mapping[str, Any]) -> str:
        """Сформировать человекочитаемый отчет этапа 4."""

        metrics = report["metrics"]
        status = "пройдена" if report["passed"] else "не пройдена"
        weight_rows = "\n".join(
            f"| `{name}` | {weight:.10f} |"
            for name, weight in report["weights"].items()
        )
        return f"""# Проверка фактического интегрального качества

## Итоговый статус

Проверка **{status}**.

- этап: {report['stage']};
- сценариев: {report['row_count']};
- основной фактический показатель: `integral_quality`;
- числовой допуск контрольной агрегации: {report['consistency_tolerance']:.6f};
- сценариев вне допуска: {metrics['outside_tolerance_count']}.

## Весовая схема контрольной агрегации

| Критерий | Вес |
|---|---:|
{weight_rows}

Сумма весов: **{report['weight_sum']:.10f}**.

## Метрики расхождения

| Метрика | Значение |
|---|---:|
| Среднее signed-расхождение | {metrics['mean_signed_difference']:.10f} |
| Среднее абсолютное расхождение | {metrics['mean_absolute_difference']:.10f} |
| Максимальное абсолютное расхождение | {metrics['max_absolute_difference']:.10f} |
| RMSE расхождения | {metrics['rmse_difference']:.10f} |
| Расхождение `q_fact` и `integral_quality` | {metrics['max_q_fact_alias_difference']:.12g} |
| Доля сценариев в пределах допуска | {metrics['consistency_rate']:.6f} |

## Методическое ограничение

{report['methodological_note']}
"""

    def _resolve_tolerance(self, explicit: float | None) -> float:
        """Разрешить допуск из аргумента, конфигурации или значения по умолчанию."""

        if explicit is not None:
            tolerance = float(explicit)
        else:
            configured = getattr(
                getattr(self.config, "validation", object()),
                "integral_quality_tolerance",
                DEFAULT_CONSISTENCY_TOLERANCE,
            )
            tolerance = float(configured)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise IntegralQualityValidationError(
                "Допуск согласованности integral_quality должен быть "
                "неотрицательным конечным числом."
            )
        return tolerance

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
            raise IntegralQualityValidationError(
                "В конфигурации отсутствуют ключи объединения."
            )
        return tuple(keys)

    def _expected_row_count(self) -> int:
        """Вернуть ожидаемое число сценариев."""

        return int(self.config.merge.expected_row_count)


__all__ = [
    "DEFAULT_CONSISTENCY_TOLERANCE",
    "IntegralQualityValidationError",
    "IntegralQualityValidationResult",
    "IntegralQualityValidator",
    "QUALITY_WEIGHTS",
]
