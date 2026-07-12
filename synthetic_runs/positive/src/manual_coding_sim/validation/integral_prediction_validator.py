"""Расчет метрик интегрального априорного прогноза главы 6.

Модуль реализует этап 5 программного контура главы 6. Он сопоставляет
зафиксированный прогноз ``q_pred`` с фактическим показателем ``q_fact`` и
формирует построчные ошибки, основные метрики точности и ранговой
согласованности. Значения прогноза и фактического качества не изменяются.
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


REQUIRED_VALUE_COLUMNS: tuple[str, ...] = (
    "q_pred",
    "q_fact",
)
UNIT_INTERVAL_TOLERANCE = 1e-12


class IntegralPredictionValidationError(ValueError):
    """Ошибка расчета метрик интегрального прогноза."""


@dataclass(frozen=True)
class IntegralPredictionValidationResult:
    """Результат проверки интегрального априорного прогноза."""

    errors: pd.DataFrame
    report: dict[str, Any]
    csv_path: Path | None
    json_path: Path | None
    markdown_path: Path | None

    @property
    def passed(self) -> bool:
        """Вернуть итоговый статус расчетного этапа."""

        return bool(self.report["passed"])


class IntegralPredictionValidator:
    """Валидатор точности и согласованности ``q_pred`` и ``q_fact``."""

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
    ) -> IntegralPredictionValidationResult:
        """Рассчитать построчные ошибки и метрики интегрального прогноза."""

        source = dataset.copy() if dataset is not None else self._load_dataset()
        self._validate_source(source)

        keys = list(self._join_keys())
        errors = source[keys + list(REQUIRED_VALUE_COLUMNS)].copy()
        errors["q_pred"] = pd.to_numeric(errors["q_pred"], errors="raise")
        errors["q_fact"] = pd.to_numeric(errors["q_fact"], errors="raise")
        errors["prediction_error"] = errors["q_pred"] - errors["q_fact"]
        errors["absolute_error"] = errors["prediction_error"].abs()
        errors["squared_error"] = np.square(errors["prediction_error"])

        predicted = errors["q_pred"].to_numpy(dtype=float)
        factual = errors["q_fact"].to_numpy(dtype=float)
        signed_errors = errors["prediction_error"].to_numpy(dtype=float)
        absolute_errors = errors["absolute_error"].to_numpy(dtype=float)

        metrics = {
            "mae": float(np.mean(absolute_errors)),
            "rmse": float(math.sqrt(float(np.mean(np.square(signed_errors))))),
            "bias": float(np.mean(signed_errors)),
            "median_absolute_error": float(np.median(absolute_errors)),
            "max_absolute_error": float(np.max(absolute_errors)),
            "pearson": self._pearson(predicted, factual),
            "spearman": self._spearman(predicted, factual),
            "kendall": self._kendall_tau_b(predicted, factual),
            "r2": self._r2_score(predicted, factual),
            "q_pred_mean": float(np.mean(predicted)),
            "q_fact_mean": float(np.mean(factual)),
            "q_pred_std": float(np.std(predicted, ddof=1)),
            "q_fact_std": float(np.std(factual, ddof=1)),
        }

        finite_metrics = all(math.isfinite(value) for value in metrics.values())
        passed = len(errors) == self._expected_row_count() and finite_metrics
        report = {
            "stage": 5,
            "report_type": "integral_prediction_metrics",
            "passed": passed,
            "row_count": int(len(errors)),
            "expected_row_count": self._expected_row_count(),
            "prediction_column": "q_pred",
            "factual_column": "q_fact",
            "error_definition": "prediction_error = q_pred - q_fact",
            "metrics": metrics,
            "interpretation": {
                "bias": (
                    "Отрицательное значение означает систематическое "
                    "занижение качества; положительное — завышение."
                ),
                "r2": (
                    "Отрицательное значение допустимо и означает, что "
                    "точечный прогноз хуже постоянного прогноза средним "
                    "фактическим качеством по сумме квадратов ошибок."
                ),
                "correlations": (
                    "Pearson оценивает линейную связь, Spearman и Kendall — "
                    "ранговую согласованность сценариев."
                ),
            },
            "methodological_note": (
                "Метрики рассчитываются для неизмененного прогноза главы 5. "
                "Фактические данные используются только для внешней проверки "
                "и не применяются для корректировки весов или формулы Q_pred."
            ),
        }

        return IntegralPredictionValidationResult(
            errors=errors,
            report=report,
            csv_path=None,
            json_path=None,
            markdown_path=None,
        )

    def validate_and_save(
        self,
        dataset: pd.DataFrame | None = None,
    ) -> IntegralPredictionValidationResult:
        """Рассчитать метрики и сохранить CSV-, JSON- и Markdown-артефакты."""

        result = self.validate(dataset=dataset)
        csv_path = self._resolve_output_path(
            (
                "integral_prediction_errors_path",
                "integral_prediction_errors",
            ),
            "reports/chapter6/integral_prediction_errors.csv",
        )
        json_path = self._resolve_output_path(
            (
                "validation_metrics_json_path",
                "validation_metrics_json",
            ),
            "reports/chapter6/validation_metrics.json",
        )
        markdown_path = self._resolve_output_path(
            (
                "validation_metrics_md_path",
                "validation_metrics_md",
            ),
            "reports/chapter6/validation_metrics.md",
        )

        for path in (csv_path, json_path, markdown_path):
            path.parent.mkdir(parents=True, exist_ok=True)

        result.errors.to_csv(csv_path, index=False, encoding="utf-8")
        json_path.write_text(
            json.dumps(result.report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        markdown_path.write_text(
            self._build_markdown_report(result.report),
            encoding="utf-8",
        )

        return IntegralPredictionValidationResult(
            errors=result.errors,
            report=result.report,
            csv_path=csv_path,
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
            raise IntegralPredictionValidationError(
                f"Не удалось прочитать проверочный датасет: {path}"
            ) from error

    def _validate_source(self, dataset: pd.DataFrame) -> None:
        """Проверить структуру, ключи и числовые значения датасета."""

        required = [*self._join_keys(), *REQUIRED_VALUE_COLUMNS]
        missing = [column for column in required if column not in dataset.columns]
        if missing:
            raise IntegralPredictionValidationError(
                "В validation_dataset.csv отсутствуют обязательные колонки: "
                + ", ".join(missing)
            )

        if len(dataset) != self._expected_row_count():
            raise IntegralPredictionValidationError(
                "Некорректное число строк проверочного датасета: "
                f"ожидалось {self._expected_row_count()}, получено {len(dataset)}."
            )

        keys = list(self._join_keys())
        if dataset[keys].isna().any().any():
            raise IntegralPredictionValidationError(
                "Составной ключ содержит пропущенные значения."
            )
        if dataset.duplicated(subset=keys).any():
            raise IntegralPredictionValidationError(
                "Составной ключ scenario_id, protocol_id не является уникальным."
            )

        try:
            numeric = dataset[list(REQUIRED_VALUE_COLUMNS)].apply(
                pd.to_numeric,
                errors="raise",
            )
        except (TypeError, ValueError) as error:
            raise IntegralPredictionValidationError(
                "q_pred и q_fact должны содержать только числовые значения."
            ) from error

        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise IntegralPredictionValidationError(
                "q_pred или q_fact содержат NaN, inf или -inf."
            )
        if (
            (values < -UNIT_INTERVAL_TOLERANCE)
            | (values > 1.0 + UNIT_INTERVAL_TOLERANCE)
        ).any():
            raise IntegralPredictionValidationError(
                "q_pred и q_fact должны находиться в диапазоне [0; 1]."
            )

        if numeric["q_pred"].nunique(dropna=False) < 2:
            raise IntegralPredictionValidationError(
                "Для расчета корреляций q_pred должен содержать не менее "
                "двух различных значений."
            )
        if numeric["q_fact"].nunique(dropna=False) < 2:
            raise IntegralPredictionValidationError(
                "Для расчета корреляций q_fact должен содержать не менее "
                "двух различных значений."
            )

    def _pearson(self, left: np.ndarray, right: np.ndarray) -> float:
        """Рассчитать коэффициент корреляции Pearson."""

        centered_left = left - float(np.mean(left))
        centered_right = right - float(np.mean(right))
        denominator = math.sqrt(
            float(np.sum(np.square(centered_left)))
            * float(np.sum(np.square(centered_right)))
        )
        if denominator == 0.0:
            raise IntegralPredictionValidationError(
                "Корреляция Pearson не определена для постоянной выборки."
            )
        return float(np.sum(centered_left * centered_right) / denominator)

    def _spearman(self, left: np.ndarray, right: np.ndarray) -> float:
        """Рассчитать Spearman через средние ранги для совпадающих значений."""

        left_ranks = pd.Series(left).rank(method="average").to_numpy(dtype=float)
        right_ranks = pd.Series(right).rank(method="average").to_numpy(dtype=float)
        return self._pearson(left_ranks, right_ranks)

    def _kendall_tau_b(self, left: np.ndarray, right: np.ndarray) -> float:
        """Рассчитать коэффициент Kendall tau-b с учетом связанных рангов."""

        concordant = 0
        discordant = 0
        tied_left_only = 0
        tied_right_only = 0

        for index in range(len(left) - 1):
            left_delta = left[index + 1 :] - left[index]
            right_delta = right[index + 1 :] - right[index]
            left_sign = np.sign(left_delta)
            right_sign = np.sign(right_delta)
            products = left_sign * right_sign

            concordant += int(np.sum(products > 0))
            discordant += int(np.sum(products < 0))
            tied_left_only += int(
                np.sum((left_sign == 0) & (right_sign != 0))
            )
            tied_right_only += int(
                np.sum((right_sign == 0) & (left_sign != 0))
            )

        denominator = math.sqrt(
            (concordant + discordant + tied_left_only)
            * (concordant + discordant + tied_right_only)
        )
        if denominator == 0.0:
            raise IntegralPredictionValidationError(
                "Корреляция Kendall не определена для постоянной выборки."
            )
        return float((concordant - discordant) / denominator)

    def _r2_score(self, predicted: np.ndarray, factual: np.ndarray) -> float:
        """Рассчитать коэффициент детерминации относительно среднего факта."""

        residual_sum = float(np.sum(np.square(predicted - factual)))
        total_sum = float(np.sum(np.square(factual - float(np.mean(factual)))))
        if total_sum == 0.0:
            raise IntegralPredictionValidationError(
                "R² не определен при нулевой дисперсии q_fact."
            )
        return float(1.0 - residual_sum / total_sum)

    def _build_markdown_report(self, report: Mapping[str, Any]) -> str:
        """Сформировать человекочитаемый отчет этапа 5."""

        metrics = report["metrics"]
        status = "выполнен" if report["passed"] else "не выполнен"
        bias_direction = (
            "систематическое занижение"
            if metrics["bias"] < 0.0
            else "систематическое завышение"
            if metrics["bias"] > 0.0
            else "систематическое смещение отсутствует"
        )
        return f"""# Метрики интегрального априорного прогноза

## Итоговый статус

Расчетный этап **{status}**.

- этап: {report['stage']};
- сценариев: {report['row_count']};
- прогнозный показатель: `q_pred`;
- фактический показатель: `q_fact`;
- определение ошибки: `q_pred - q_fact`.

## Метрики точности и согласованности

| Метрика | Значение |
|---|---:|
| MAE | {metrics['mae']:.10f} |
| RMSE | {metrics['rmse']:.10f} |
| Bias | {metrics['bias']:.10f} |
| Median Absolute Error | {metrics['median_absolute_error']:.10f} |
| Max Absolute Error | {metrics['max_absolute_error']:.10f} |
| Pearson | {metrics['pearson']:.10f} |
| Spearman | {metrics['spearman']:.10f} |
| Kendall tau-b | {metrics['kendall']:.10f} |
| R² | {metrics['r2']:.10f} |

## Описательная статистика

| Показатель | Среднее | Стандартное отклонение |
|---|---:|---:|
| `q_pred` | {metrics['q_pred_mean']:.10f} | {metrics['q_pred_std']:.10f} |
| `q_fact` | {metrics['q_fact_mean']:.10f} | {metrics['q_fact_std']:.10f} |

## Интерпретация

Знак Bias указывает на **{bias_direction}** прогноза относительно
фактического качества. Отрицательное значение R² является допустимым
результатом внешней проверки и означает, что по сумме квадратов ошибок
точечный прогноз уступает постоянному прогнозу средним фактическим
качеством. Ранговые коэффициенты следует рассматривать отдельно от
калибровки шкалы: высокая ранговая согласованность может сочетаться с
заметным систематическим смещением.

## Методическое ограничение

{report['methodological_note']}
"""

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
            raise IntegralPredictionValidationError(
                "В конфигурации отсутствуют ключи объединения."
            )
        return tuple(keys)

    def _expected_row_count(self) -> int:
        """Вернуть ожидаемое число сценариев."""

        return int(self.config.merge.expected_row_count)


__all__ = [
    "IntegralPredictionValidationError",
    "IntegralPredictionValidationResult",
    "IntegralPredictionValidator",
    "REQUIRED_VALUE_COLUMNS",
]
