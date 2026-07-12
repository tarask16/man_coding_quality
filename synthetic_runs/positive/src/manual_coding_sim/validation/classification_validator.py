"""Проверка классификации уровней качества в главе 6.

Модуль реализует этап 7 программного контура главы 6. Непрерывные значения
``q_pred`` и ``q_fact`` переводятся в классы ``low``, ``medium`` и ``high``
по порогам, зафиксированным до внешней проверки. Затем рассчитываются
матрица ошибок, общие классификационные метрики и показатели по классам.
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
REQUIRED_VALUE_COLUMNS: tuple[str, ...] = ("q_pred", "q_fact")
UNIT_INTERVAL_TOLERANCE = 1e-12


class ClassificationValidationError(ValueError):
    """Ошибка проверки классификации уровней качества."""


@dataclass(frozen=True)
class ClassificationValidationResult:
    """Результат классификационной проверки этапа 7."""

    predictions: pd.DataFrame
    confusion_matrix: pd.DataFrame
    report: dict[str, Any]
    predictions_path: Path | None
    confusion_matrix_path: Path | None
    json_path: Path | None
    markdown_path: Path | None

    @property
    def passed(self) -> bool:
        """Вернуть итоговый статус этапа."""

        return bool(self.report["passed"])


class ClassificationValidator:
    """Валидатор классов качества ``low``, ``medium`` и ``high``."""

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
    ) -> ClassificationValidationResult:
        """Рассчитать классы, матрицу ошибок и классификационные метрики."""

        source = dataset.copy() if dataset is not None else self._load_dataset()
        self._validate_source(source)

        keys = list(self._join_keys())
        predictions = source[keys + list(REQUIRED_VALUE_COLUMNS)].copy()
        predictions["q_pred"] = pd.to_numeric(
            predictions["q_pred"], errors="raise"
        )
        predictions["q_fact"] = pd.to_numeric(
            predictions["q_fact"], errors="raise"
        )
        predictions["q_pred_class"] = predictions["q_pred"].map(
            self._quality_class
        )
        predictions["q_fact_class"] = predictions["q_fact"].map(
            self._quality_class
        )

        self._validate_existing_classes(source, predictions)

        predictions["is_correct"] = (
            predictions["q_pred_class"] == predictions["q_fact_class"]
        )
        predictions["error_direction"] = np.where(
            predictions["is_correct"],
            "correct",
            predictions["q_fact_class"] + "->" + predictions["q_pred_class"],
        )
        predictions["class_distance"] = [
            abs(CLASS_LABELS.index(factual) - CLASS_LABELS.index(predicted))
            for factual, predicted in zip(
                predictions["q_fact_class"],
                predictions["q_pred_class"],
                strict=True,
            )
        ]
        predictions["is_critical_error"] = predictions["error_direction"].isin(
            ("low->high", "high->low")
        )

        confusion = self._build_confusion_matrix(predictions)
        per_class = self._calculate_per_class_metrics(confusion)
        general_metrics = self._calculate_general_metrics(confusion, per_class)

        predicted_distribution = {
            label: int((predictions["q_pred_class"] == label).sum())
            for label in CLASS_LABELS
        }
        factual_distribution = {
            label: int((predictions["q_fact_class"] == label).sum())
            for label in CLASS_LABELS
        }
        critical_errors = {
            "low_to_high": int(confusion.loc["low", "high"]),
            "high_to_low": int(confusion.loc["high", "low"]),
            "total": int(predictions["is_critical_error"].sum()),
        }
        error_directions = {
            direction: int(count)
            for direction, count in predictions["error_direction"]
            .value_counts()
            .sort_index()
            .items()
        }

        finite_metrics = all(
            math.isfinite(float(value)) for value in general_metrics.values()
        ) and all(
            math.isfinite(float(value))
            for row in per_class
            for key, value in row.items()
            if key not in {"class_label"}
        )
        passed = (
            len(predictions) == self._expected_row_count()
            and int(confusion.to_numpy().sum()) == len(predictions)
            and finite_metrics
        )

        report = {
            "stage": 7,
            "report_type": "classification_validation",
            "passed": passed,
            "row_count": int(len(predictions)),
            "expected_row_count": self._expected_row_count(),
            "class_labels": list(CLASS_LABELS),
            "thresholds": {
                "low": f"Q < {self.config.decision_thresholds.low_max}",
                "medium": (
                    f"{self.config.decision_thresholds.low_max} <= Q < "
                    f"{self.config.decision_thresholds.high_min}"
                ),
                "high": f"Q >= {self.config.decision_thresholds.high_min}",
                "low_max": float(self.config.decision_thresholds.low_max),
                "high_min": float(self.config.decision_thresholds.high_min),
            },
            "metrics": general_metrics,
            "per_class_metrics": per_class,
            "predicted_class_distribution": predicted_distribution,
            "factual_class_distribution": factual_distribution,
            "critical_errors": critical_errors,
            "error_directions": error_directions,
            "confusion_matrix": {
                factual: {
                    predicted: int(confusion.loc[factual, predicted])
                    for predicted in CLASS_LABELS
                }
                for factual in CLASS_LABELS
            },
            "interpretation": {
                "balanced_accuracy": (
                    "Средняя полнота по трем фактическим классам; применяется "
                    "вместе с Accuracy из-за выраженного дисбаланса классов."
                ),
                "macro_f1": (
                    "Среднее F1 без учета размера классов; каждый класс имеет "
                    "одинаковый вклад."
                ),
                "critical_errors": (
                    "low->high означает опасное завышение качества, high->low — "
                    "сильное занижение качества."
                ),
            },
            "methodological_note": (
                "Пороговые значения полностью совпадают с главой 5 и не "
                "корректируются по фактическим классам. Фактические данные "
                "используются только для внешней классификационной проверки."
            ),
        }

        return ClassificationValidationResult(
            predictions=predictions,
            confusion_matrix=confusion,
            report=report,
            predictions_path=None,
            confusion_matrix_path=None,
            json_path=None,
            markdown_path=None,
        )

    def validate_and_save(
        self,
        dataset: pd.DataFrame | None = None,
    ) -> ClassificationValidationResult:
        """Выполнить проверку и сохранить CSV-, JSON- и Markdown-артефакты."""

        result = self.validate(dataset=dataset)
        predictions_path = self._resolve_output_path(
            ("classification_predictions_path", "classification_predictions"),
            "reports/chapter6/classification_predictions.csv",
        )
        confusion_path = self._resolve_output_path(
            ("confusion_matrix_path", "confusion_matrix"),
            "reports/chapter6/confusion_matrix.csv",
        )
        json_path = self._resolve_output_path(
            ("classification_report_json_path", "classification_report_json"),
            "reports/chapter6/classification_report.json",
        )
        markdown_path = self._resolve_output_path(
            ("classification_report_md_path", "classification_report_md"),
            "reports/chapter6/classification_report.md",
        )

        for path in (
            predictions_path,
            confusion_path,
            json_path,
            markdown_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)

        result.predictions.to_csv(
            predictions_path,
            index=False,
            encoding="utf-8",
        )
        confusion_for_csv = result.confusion_matrix.copy()
        confusion_for_csv.insert(
            0,
            "factual_class",
            confusion_for_csv.index,
        )
        confusion_for_csv.to_csv(
            confusion_path,
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

        return ClassificationValidationResult(
            predictions=result.predictions,
            confusion_matrix=result.confusion_matrix,
            report=result.report,
            predictions_path=predictions_path,
            confusion_matrix_path=confusion_path,
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
            raise ClassificationValidationError(
                f"Не удалось прочитать проверочный датасет: {path}"
            ) from error

    def _validate_source(self, dataset: pd.DataFrame) -> None:
        """Проверить структуру, ключи и непрерывные показатели качества."""

        required = [*self._join_keys(), *REQUIRED_VALUE_COLUMNS]
        missing = [column for column in required if column not in dataset.columns]
        if missing:
            raise ClassificationValidationError(
                "В validation_dataset.csv отсутствуют обязательные колонки: "
                + ", ".join(missing)
            )

        if len(dataset) != self._expected_row_count():
            raise ClassificationValidationError(
                "Некорректное число строк проверочного датасета: "
                f"ожидалось {self._expected_row_count()}, получено {len(dataset)}."
            )

        keys = list(self._join_keys())
        if dataset[keys].isna().any().any():
            raise ClassificationValidationError(
                "Составной ключ содержит пропущенные значения."
            )
        if dataset.duplicated(subset=keys).any():
            raise ClassificationValidationError(
                "Составной ключ scenario_id, protocol_id не является уникальным."
            )

        try:
            numeric = dataset[list(REQUIRED_VALUE_COLUMNS)].apply(
                pd.to_numeric,
                errors="raise",
            )
        except (TypeError, ValueError) as error:
            raise ClassificationValidationError(
                "q_pred и q_fact должны содержать только числовые значения."
            ) from error

        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ClassificationValidationError(
                "q_pred или q_fact содержат NaN, inf или -inf."
            )
        if (
            (values < -UNIT_INTERVAL_TOLERANCE)
            | (values > 1.0 + UNIT_INTERVAL_TOLERANCE)
        ).any():
            raise ClassificationValidationError(
                "q_pred и q_fact должны находиться в диапазоне [0; 1]."
            )

        factual_classes = numeric["q_fact"].map(self._quality_class)
        missing_classes = [
            label for label in CLASS_LABELS if label not in set(factual_classes)
        ]
        if missing_classes:
            raise ClassificationValidationError(
                "Фактическая выборка не содержит обязательные классы: "
                + ", ".join(missing_classes)
            )

    def _validate_existing_classes(
        self,
        source: pd.DataFrame,
        calculated: pd.DataFrame,
    ) -> None:
        """Проверить согласованность классов, сохраненных на этапе 3."""

        comparisons = (
            ("q_pred_class", calculated["q_pred_class"]),
            ("q_fact_class", calculated["q_fact_class"]),
        )
        for column, expected in comparisons:
            if column not in source.columns:
                continue
            existing = source[column].astype(str)
            invalid_labels = sorted(set(existing) - set(CLASS_LABELS))
            if invalid_labels:
                raise ClassificationValidationError(
                    f"Колонка {column} содержит неизвестные классы: "
                    + ", ".join(invalid_labels)
                )
            mismatch_count = int((existing.to_numpy() != expected.to_numpy()).sum())
            if mismatch_count:
                raise ClassificationValidationError(
                    f"Колонка {column} не соответствует зафиксированным порогам: "
                    f"расхождений {mismatch_count}."
                )

    def _quality_class(self, value: float) -> str:
        """Преобразовать непрерывное качество в один из трех классов."""

        numeric = float(value)
        if numeric < self.config.decision_thresholds.low_max:
            return "low"
        if numeric < self.config.decision_thresholds.high_min:
            return "medium"
        return "high"

    def _build_confusion_matrix(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Построить матрицу: строки — факт, столбцы — прогноз."""

        confusion = pd.crosstab(
            predictions["q_fact_class"],
            predictions["q_pred_class"],
            dropna=False,
        )
        confusion = confusion.reindex(
            index=CLASS_LABELS,
            columns=CLASS_LABELS,
            fill_value=0,
        ).astype(int)
        confusion.index.name = "factual_class"
        confusion.columns.name = "predicted_class"
        return confusion

    def _calculate_per_class_metrics(
        self,
        confusion: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Рассчитать Precision, Recall и F1 отдельно для каждого класса."""

        rows: list[dict[str, Any]] = []
        for label in CLASS_LABELS:
            true_positive = int(confusion.loc[label, label])
            predicted_count = int(confusion[label].sum())
            support = int(confusion.loc[label].sum())
            false_positive = predicted_count - true_positive
            false_negative = support - true_positive
            precision = self._safe_divide(true_positive, predicted_count)
            recall = self._safe_divide(true_positive, support)
            f1 = self._safe_divide(
                2.0 * precision * recall,
                precision + recall,
            )
            rows.append(
                {
                    "class_label": label,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": support,
                    "predicted_count": predicted_count,
                    "true_positive": true_positive,
                    "false_positive": false_positive,
                    "false_negative": false_negative,
                }
            )
        return rows

    def _calculate_general_metrics(
        self,
        confusion: pd.DataFrame,
        per_class: Sequence[Mapping[str, Any]],
    ) -> dict[str, float]:
        """Рассчитать общие метрики классификации."""

        matrix = confusion.to_numpy(dtype=float)
        total = float(matrix.sum())
        accuracy = self._safe_divide(float(np.trace(matrix)), total)
        balanced_accuracy = float(
            np.mean([float(row["recall"]) for row in per_class])
        )
        macro_f1 = float(np.mean([float(row["f1"]) for row in per_class]))
        weighted_f1 = self._safe_divide(
            sum(float(row["f1"]) * int(row["support"]) for row in per_class),
            total,
        )
        return {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }

    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """Выполнить деление с нулевым результатом при нулевом знаменателе."""

        if denominator == 0.0:
            return 0.0
        return float(numerator / denominator)

    def _build_markdown_report(self, report: Mapping[str, Any]) -> str:
        """Сформировать человекочитаемый отчет этапа 7."""

        status = "выполнен" if report["passed"] else "не выполнен"
        metrics = report["metrics"]
        critical = report["critical_errors"]
        lines = [
            "# Проверка классификации уровней качества",
            "",
            "## Итоговый статус",
            "",
            f"Расчетный этап **{status}**.",
            "",
            f"- этап: {report['stage']};",
            f"- сценариев: {report['row_count']};",
            "- классы: `low`, `medium`, `high`;",
            (
                "- пороги: `low < {low}`, `{low} <= medium < {high}`, "
                "`high >= {high}`."
            ).format(
                low=report["thresholds"]["low_max"],
                high=report["thresholds"]["high_min"],
            ),
            "",
            "## Общие метрики",
            "",
            "| Метрика | Значение |",
            "|---|---:|",
            f"| Accuracy | {metrics['accuracy']:.10f} |",
            (
                "| Balanced Accuracy | "
                f"{metrics['balanced_accuracy']:.10f} |"
            ),
            f"| Macro F1 | {metrics['macro_f1']:.10f} |",
            f"| Weighted F1 | {metrics['weighted_f1']:.10f} |",
            "",
            "## Метрики по классам",
            "",
            (
                "| Класс | Precision | Recall | F1 | Support | "
                "Прогнозов |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in report["per_class_metrics"]:
            lines.append(
                "| {class_label} | {precision:.10f} | {recall:.10f} | "
                "{f1:.10f} | {support} | {predicted_count} |".format(**row)
            )

        lines.extend(
            [
                "",
                "## Матрица ошибок",
                "",
                "Строки соответствуют фактическим классам, столбцы — прогнозным.",
                "",
                "| Факт / прогноз | low | medium | high |",
                "|---|---:|---:|---:|",
            ]
        )
        for factual in CLASS_LABELS:
            row = report["confusion_matrix"][factual]
            lines.append(
                f"| {factual} | {row['low']} | {row['medium']} | {row['high']} |"
            )

        lines.extend(
            [
                "",
                "## Критические ошибки",
                "",
                f"- `low -> high`: {critical['low_to_high']};",
                f"- `high -> low`: {critical['high_to_low']};",
                f"- всего критических ошибок: {critical['total']}.",
                "",
                "## Распределение классов",
                "",
                "| Класс | Прогноз | Факт |",
                "|---|---:|---:|",
            ]
        )
        for label in CLASS_LABELS:
            lines.append(
                f"| {label} | "
                f"{report['predicted_class_distribution'][label]} | "
                f"{report['factual_class_distribution'][label]} |"
            )

        lines.extend(
            [
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
            raise ClassificationValidationError(
                "В конфигурации отсутствуют ключи объединения."
            )
        return tuple(keys)

    def _expected_row_count(self) -> int:
        """Вернуть ожидаемое число сценариев."""

        return int(self.config.merge.expected_row_count)


__all__ = [
    "CLASS_LABELS",
    "ClassificationValidationError",
    "ClassificationValidationResult",
    "ClassificationValidator",
]
