"""Проверка частных прогнозных критериев качества главы 6.

Модуль реализует этап 6 программного контура главы 6. Для каждого из шести
частных критериев сопоставляется зафиксированный априорный прогноз главы 5
с соответствующим фактическим показателем. Фактические данные используются
только для внешней проверки и не применяются для изменения формул прогноза.
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


CRITERION_MAPPINGS: tuple[tuple[str, str, str, str], ...] = (
    ("q_acc", "q_acc_pred", "q_acc", "Точность восстановления сообщения"),
    ("q_time", "q_time_pred", "q_time", "Временная эффективность"),
    ("q_effort", "q_effort_pred", "q_effort", "Трудоемкость ручной процедуры"),
    ("q_res", "q_res_pred", "q_res", "Результативность и устойчивость"),
    ("q_rep", "q_rep_pred", "q_rep", "Повторяемость результата"),
    ("q_fit", "q_fit_pred", "q_fit", "Соответствие условиям применения"),
)
UNIT_INTERVAL_TOLERANCE = 1e-12


class PartialCriteriaValidationError(ValueError):
    """Ошибка проверки частных прогнозных критериев."""


@dataclass(frozen=True)
class PartialCriteriaValidationResult:
    """Результат проверки шести частных критериев качества."""

    metrics_table: pd.DataFrame
    report: dict[str, Any]
    csv_path: Path | None
    json_path: Path | None
    markdown_path: Path | None

    @property
    def passed(self) -> bool:
        """Вернуть итоговый статус этапа 6."""

        return bool(self.report["passed"])


class PartialCriteriaValidator:
    """Валидатор частных прогнозных и фактических критериев качества."""

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
    ) -> PartialCriteriaValidationResult:
        """Рассчитать метрики для всех шести пар частных критериев."""

        source = dataset.copy() if dataset is not None else self._load_dataset()
        self._validate_source(source)

        records: list[dict[str, Any]] = []
        for criterion, predicted_column, factual_column, description in (
            CRITERION_MAPPINGS
        ):
            predicted = pd.to_numeric(
                source[predicted_column], errors="raise"
            ).to_numpy(dtype=float)
            factual = pd.to_numeric(
                source[factual_column], errors="raise"
            ).to_numpy(dtype=float)
            error = predicted - factual
            absolute_error = np.abs(error)

            records.append(
                {
                    "criterion": criterion,
                    "predicted_column": predicted_column,
                    "factual_column": factual_column,
                    "description": description,
                    "mae": float(np.mean(absolute_error)),
                    "rmse": float(math.sqrt(float(np.mean(np.square(error))))),
                    "bias": float(np.mean(error)),
                    "pearson": self._pearson(predicted, factual),
                    "spearman": self._spearman(predicted, factual),
                    "kendall": self._kendall_tau_b(predicted, factual),
                    "r2": self._r2_score(predicted, factual),
                    "predicted_mean": float(np.mean(predicted)),
                    "factual_mean": float(np.mean(factual)),
                    "predicted_std": float(np.std(predicted, ddof=1)),
                    "factual_std": float(np.std(factual, ddof=1)),
                    "max_absolute_error": float(np.max(absolute_error)),
                }
            )

        metrics_table = pd.DataFrame.from_records(records)
        finite_columns = (
            "mae",
            "rmse",
            "bias",
            "pearson",
            "spearman",
            "kendall",
            "r2",
            "predicted_mean",
            "factual_mean",
            "predicted_std",
            "factual_std",
            "max_absolute_error",
        )
        finite_metrics = np.isfinite(
            metrics_table[list(finite_columns)].to_numpy(dtype=float)
        ).all()
        passed = (
            len(source) == self._expected_row_count()
            and len(metrics_table) == len(CRITERION_MAPPINGS)
            and bool(finite_metrics)
        )

        best_mae_row = metrics_table.loc[metrics_table["mae"].idxmin()]
        worst_mae_row = metrics_table.loc[metrics_table["mae"].idxmax()]
        best_spearman_row = metrics_table.loc[
            metrics_table["spearman"].idxmax()
        ]
        worst_spearman_row = metrics_table.loc[
            metrics_table["spearman"].idxmin()
        ]

        report = {
            "stage": 6,
            "report_type": "partial_criteria_validation",
            "passed": passed,
            "row_count": int(len(source)),
            "expected_row_count": self._expected_row_count(),
            "criterion_count": int(len(metrics_table)),
            "expected_criterion_count": len(CRITERION_MAPPINGS),
            "error_definition": "criterion_error = q_j_pred - q_j_fact",
            "criterion_mappings": [
                {
                    "criterion": criterion,
                    "predicted_column": predicted_column,
                    "factual_column": factual_column,
                    "description": description,
                }
                for criterion, predicted_column, factual_column, description in (
                    CRITERION_MAPPINGS
                )
            ],
            "metrics": metrics_table.to_dict(orient="records"),
            "summary": {
                "mean_mae": float(metrics_table["mae"].mean()),
                "mean_rmse": float(metrics_table["rmse"].mean()),
                "mean_absolute_bias": float(metrics_table["bias"].abs().mean()),
                "mean_spearman": float(metrics_table["spearman"].mean()),
                "mean_kendall": float(metrics_table["kendall"].mean()),
                "best_mae_criterion": str(best_mae_row["criterion"]),
                "best_mae": float(best_mae_row["mae"]),
                "worst_mae_criterion": str(worst_mae_row["criterion"]),
                "worst_mae": float(worst_mae_row["mae"]),
                "best_spearman_criterion": str(
                    best_spearman_row["criterion"]
                ),
                "best_spearman": float(best_spearman_row["spearman"]),
                "worst_spearman_criterion": str(
                    worst_spearman_row["criterion"]
                ),
                "worst_spearman": float(worst_spearman_row["spearman"]),
            },
            "interpretation": {
                "bias": (
                    "Отрицательное значение означает систематическое "
                    "занижение частного критерия, положительное — завышение."
                ),
                "ranking": (
                    "MAE характеризует точность шкалы, а Spearman и Kendall "
                    "показывают сохранение порядка сценариев."
                ),
                "r2": (
                    "Отрицательное значение R² допустимо при внешней проверке "
                    "и указывает на неудовлетворительную калибровку шкалы."
                ),
            },
            "methodological_note": (
                "Все метрики рассчитываются для неизмененных частных прогнозов "
                "главы 5. Фактические критерии используются только для внешней "
                "проверки и не участвуют в корректировке весов или формул."
            ),
        }

        return PartialCriteriaValidationResult(
            metrics_table=metrics_table,
            report=report,
            csv_path=None,
            json_path=None,
            markdown_path=None,
        )

    def validate_and_save(
        self,
        dataset: pd.DataFrame | None = None,
    ) -> PartialCriteriaValidationResult:
        """Рассчитать метрики и сохранить CSV-, JSON- и Markdown-отчеты."""

        result = self.validate(dataset=dataset)
        csv_path = self._resolve_output_path(
            (
                "partial_criteria_validation_path",
                "partial_criteria_validation",
            ),
            "reports/chapter6/partial_criteria_validation.csv",
        )
        json_path = self._resolve_output_path(
            (
                "partial_criteria_validation_report_json_path",
                "partial_criteria_validation_report_json",
            ),
            "reports/chapter6/partial_criteria_validation_report.json",
        )
        markdown_path = self._resolve_output_path(
            (
                "partial_criteria_validation_report_md_path",
                "partial_criteria_validation_report_md",
            ),
            "reports/chapter6/partial_criteria_validation_report.md",
        )

        for path in (csv_path, json_path, markdown_path):
            path.parent.mkdir(parents=True, exist_ok=True)

        result.metrics_table.to_csv(csv_path, index=False, encoding="utf-8")
        json_path.write_text(
            json.dumps(result.report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        markdown_path.write_text(
            self._build_markdown_report(result.report),
            encoding="utf-8",
        )

        return PartialCriteriaValidationResult(
            metrics_table=result.metrics_table,
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
            raise PartialCriteriaValidationError(
                f"Не удалось прочитать проверочный датасет: {path}"
            ) from error

    def _validate_source(self, dataset: pd.DataFrame) -> None:
        """Проверить структуру, ключи и значения проверочного датасета."""

        required_columns = list(self._join_keys())
        for _, predicted_column, factual_column, _ in CRITERION_MAPPINGS:
            required_columns.extend((predicted_column, factual_column))

        missing = [
            column for column in required_columns if column not in dataset.columns
        ]
        if missing:
            raise PartialCriteriaValidationError(
                "В validation_dataset.csv отсутствуют обязательные колонки: "
                + ", ".join(missing)
            )

        if len(dataset) != self._expected_row_count():
            raise PartialCriteriaValidationError(
                "Некорректное число строк проверочного датасета: "
                f"ожидалось {self._expected_row_count()}, получено {len(dataset)}."
            )

        keys = list(self._join_keys())
        if dataset[keys].isna().any().any():
            raise PartialCriteriaValidationError(
                "Составной ключ содержит пропущенные значения."
            )
        if dataset.duplicated(subset=keys).any():
            raise PartialCriteriaValidationError(
                "Составной ключ scenario_id, protocol_id не является уникальным."
            )

        value_columns = [
            column
            for _, predicted_column, factual_column, _ in CRITERION_MAPPINGS
            for column in (predicted_column, factual_column)
        ]
        try:
            numeric = dataset[value_columns].apply(
                pd.to_numeric,
                errors="raise",
            )
        except (TypeError, ValueError) as error:
            raise PartialCriteriaValidationError(
                "Частные прогнозные и фактические критерии должны быть числовыми."
            ) from error

        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise PartialCriteriaValidationError(
                "Частные критерии содержат NaN, inf или -inf."
            )
        if (
            (values < -UNIT_INTERVAL_TOLERANCE)
            | (values > 1.0 + UNIT_INTERVAL_TOLERANCE)
        ).any():
            raise PartialCriteriaValidationError(
                "Частные критерии должны находиться в диапазоне [0; 1]."
            )

        for _, predicted_column, factual_column, _ in CRITERION_MAPPINGS:
            if numeric[predicted_column].nunique(dropna=False) < 2:
                raise PartialCriteriaValidationError(
                    f"Для расчета корреляций {predicted_column} должен содержать "
                    "не менее двух различных значений."
                )
            if numeric[factual_column].nunique(dropna=False) < 2:
                raise PartialCriteriaValidationError(
                    f"Для расчета корреляций {factual_column} должен содержать "
                    "не менее двух различных значений."
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
            raise PartialCriteriaValidationError(
                "Корреляция Pearson не определена для постоянной выборки."
            )
        return float(np.sum(centered_left * centered_right) / denominator)

    def _spearman(self, left: np.ndarray, right: np.ndarray) -> float:
        """Рассчитать Spearman через средние ранги связанных значений."""

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
            raise PartialCriteriaValidationError(
                "Корреляция Kendall не определена для постоянной выборки."
            )
        return float((concordant - discordant) / denominator)

    def _r2_score(self, predicted: np.ndarray, factual: np.ndarray) -> float:
        """Рассчитать R² относительно среднего фактического критерия."""

        residual_sum = float(np.sum(np.square(predicted - factual)))
        total_sum = float(np.sum(np.square(factual - float(np.mean(factual)))))
        if total_sum == 0.0:
            raise PartialCriteriaValidationError(
                "R² не определен при нулевой дисперсии фактического критерия."
            )
        return float(1.0 - residual_sum / total_sum)

    def _build_markdown_report(self, report: Mapping[str, Any]) -> str:
        """Сформировать человекочитаемый отчет этапа 6."""

        status = "выполнен" if report["passed"] else "не выполнен"
        lines = [
            "# Проверка частных прогнозных критериев",
            "",
            "## Итоговый статус",
            "",
            f"Расчетный этап **{status}**.",
            "",
            f"- этап: {report['stage']};",
            f"- сценариев: {report['row_count']};",
            f"- проверено критериев: {report['criterion_count']};",
            "- определение ошибки: `q_j_pred - q_j_fact`.",
            "",
            "## Метрики частных критериев",
            "",
            (
                "| Критерий | MAE | RMSE | Bias | Pearson | Spearman | "
                "Kendall | R² |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in report["metrics"]:
            lines.append(
                "| {criterion} | {mae:.10f} | {rmse:.10f} | {bias:.10f} | "
                "{pearson:.10f} | {spearman:.10f} | {kendall:.10f} | "
                "{r2:.10f} |".format(**row)
            )

        summary = report["summary"]
        lines.extend(
            [
                "",
                "## Сводная интерпретация",
                "",
                f"- среднее MAE по критериям: {summary['mean_mae']:.10f};",
                f"- среднее RMSE: {summary['mean_rmse']:.10f};",
                (
                    "- средний модуль Bias: "
                    f"{summary['mean_absolute_bias']:.10f};"
                ),
                f"- средний Spearman: {summary['mean_spearman']:.10f};",
                f"- средний Kendall: {summary['mean_kendall']:.10f};",
                (
                    "- наименьший MAE: "
                    f"`{summary['best_mae_criterion']}` "
                    f"({summary['best_mae']:.10f});"
                ),
                (
                    "- наибольший MAE: "
                    f"`{summary['worst_mae_criterion']}` "
                    f"({summary['worst_mae']:.10f});"
                ),
                (
                    "- наибольший Spearman: "
                    f"`{summary['best_spearman_criterion']}` "
                    f"({summary['best_spearman']:.10f});"
                ),
                (
                    "- наименьший Spearman: "
                    f"`{summary['worst_spearman_criterion']}` "
                    f"({summary['worst_spearman']:.10f})."
                ),
                "",
                "## Методическое ограничение",
                "",
                report["methodological_note"],
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
            raise PartialCriteriaValidationError(
                "В конфигурации отсутствуют ключи объединения."
            )
        return tuple(keys)

    def _expected_row_count(self) -> int:
        """Вернуть ожидаемое число сценариев."""

        return int(self.config.merge.expected_row_count)


__all__ = [
    "CRITERION_MAPPINGS",
    "PartialCriteriaValidationError",
    "PartialCriteriaValidationResult",
    "PartialCriteriaValidator",
]
