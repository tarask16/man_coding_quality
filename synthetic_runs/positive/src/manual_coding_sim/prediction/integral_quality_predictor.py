"""Расчет интегральной априорной оценки качества ``Q_pred``.

Модуль агрегирует шесть частных прогнозных критериев главы 5 в один
интегральный показатель. Расчет не использует фактические наблюдения,
целевые значения качества или результаты последующей верификации: входом
является только таблица ``q_pred_components.csv`` этапа 7 и веса частных
критериев из конфигурации главы 5.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from manual_coding_sim.prediction.chapter5_config import (
    Chapter5QualityWeights,
    QUALITY_CRITERIA,
)

SERVICE_COLUMNS = ("scenario_id", "protocol_id", "run_id", "alternative_id")
MERGE_COLUMNS = ("scenario_id", "protocol_id")


@dataclass(frozen=True)
class IntegralQualityCriterionReport:
    """Отчет по вкладу одного частного критерия в ``Q_pred``."""

    criterion_name: str
    weight: float
    prediction_column: str
    contribution_column: str
    prediction_min: float
    prediction_max: float
    prediction_mean: float
    contribution_min: float
    contribution_max: float
    contribution_mean: float


@dataclass(frozen=True)
class IntegralQualityPredictionReport:
    """Сводный отчет расчета интегрального показателя качества."""

    row_count: int
    criteria: tuple[str, ...]
    weights: dict[str, float]
    weight_sum: float
    merge_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    q_pred_min: float
    q_pred_max: float
    q_pred_mean: float
    q_pred_std: float
    criterion_reports: tuple[IntegralQualityCriterionReport, ...]

    def to_dict(self) -> dict[str, object]:
        """Преобразовать отчет в JSON-совместимый словарь."""

        payload = asdict(self)
        payload["criterion_reports"] = [
            asdict(item) for item in self.criterion_reports
        ]
        return payload


@dataclass(frozen=True)
class IntegralQualityPredictionResult:
    """Результат расчета интегрального прогнозного показателя качества."""

    q_pred: pd.DataFrame
    report: IntegralQualityPredictionReport


class IntegralQualityPredictionError(ValueError):
    """Ошибка расчета интегрального показателя качества главы 5."""


class IntegralQualityPredictor:
    """Агрегирует частные прогнозные критерии в ``Q_pred``."""

    def __init__(
        self,
        quality_weights: Chapter5QualityWeights | Mapping[str, float],
        *,
        criteria: Sequence[str] = QUALITY_CRITERIA,
        service_columns: Sequence[str] = SERVICE_COLUMNS,
        merge_columns: Sequence[str] = MERGE_COLUMNS,
        latent_column: str = "q_latent",
    ) -> None:
        """Создать калькулятор интегральной оценки.

        ``quality_weights`` может быть объектом ``Chapter5QualityWeights`` или
        словарем вида ``{"q_acc": 0.16, ...}``. Веса должны покрывать все
        шесть частных критериев и суммироваться в единицу.
        """

        self.criteria = tuple(criteria)
        self.service_columns = tuple(service_columns)
        self.merge_columns = tuple(merge_columns)
        self.latent_column = latent_column
        self.weights = self._coerce_quality_weights(quality_weights)
        self._validate_weights()

    def predict(self, q_pred_components: pd.DataFrame) -> IntegralQualityPredictionResult:
        """Рассчитать ``Q_pred`` по таблице частных прогнозных критериев."""

        if q_pred_components.empty:
            msg = "Таблица частных прогнозных критериев пуста."
            raise IntegralQualityPredictionError(msg)

        self._require_columns(q_pred_components, self.merge_columns, "компонентах качества")
        self._reject_duplicate_keys(q_pred_components)
        prediction_columns = tuple(f"{criterion}_pred" for criterion in self.criteria)
        self._require_columns(q_pred_components, prediction_columns, "компонентах качества")

        result = self._build_base_table(q_pred_components)
        criterion_reports: list[IntegralQualityCriterionReport] = []
        q_pred = pd.Series(0.0, index=result.index, dtype="float64")

        for criterion_name in self.criteria:
            prediction_column = f"{criterion_name}_pred"
            contribution_column = f"{criterion_name}_contribution"
            weight_column = f"{criterion_name}_weight"
            prediction = pd.to_numeric(
                q_pred_components[prediction_column],
                errors="coerce",
            )
            if prediction.isna().any():
                msg = f"В колонке {prediction_column} найдены пропуски."
                raise IntegralQualityPredictionError(msg)
            if not prediction.between(0.0, 1.0).all():
                msg = f"Значения {prediction_column} должны быть в диапазоне [0, 1]."
                raise IntegralQualityPredictionError(msg)

            weight = float(self.weights[criterion_name])
            contribution = prediction * weight
            result[prediction_column] = prediction
            result[weight_column] = weight
            result[contribution_column] = contribution
            q_pred = q_pred + contribution
            criterion_reports.append(
                IntegralQualityCriterionReport(
                    criterion_name=criterion_name,
                    weight=weight,
                    prediction_column=prediction_column,
                    contribution_column=contribution_column,
                    prediction_min=float(prediction.min()),
                    prediction_max=float(prediction.max()),
                    prediction_mean=float(prediction.mean()),
                    contribution_min=float(contribution.min()),
                    contribution_max=float(contribution.max()),
                    contribution_mean=float(contribution.mean()),
                )
            )

        result["q_pred"] = q_pred.clip(0.0, 1.0)
        self._require_q_pred_range(result)
        report = IntegralQualityPredictionReport(
            row_count=int(result.shape[0]),
            criteria=self.criteria,
            weights=dict(self.weights),
            weight_sum=float(sum(self.weights.values())),
            merge_columns=self.merge_columns,
            output_columns=tuple(str(column) for column in result.columns),
            q_pred_min=float(result["q_pred"].min()),
            q_pred_max=float(result["q_pred"].max()),
            q_pred_mean=float(result["q_pred"].mean()),
            q_pred_std=float(result["q_pred"].std(ddof=0)),
            criterion_reports=tuple(criterion_reports),
        )
        return IntegralQualityPredictionResult(q_pred=result, report=report)

    def save_outputs(
        self,
        result: IntegralQualityPredictionResult,
        *,
        q_pred_path: Path,
        report_path: Path | None = None,
    ) -> None:
        """Сохранить таблицу ``q_pred.csv`` и необязательный JSON-отчет."""

        q_pred_path.parent.mkdir(parents=True, exist_ok=True)
        result.q_pred.to_csv(q_pred_path, index=False)
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(result.report.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _build_base_table(self, q_pred_components: pd.DataFrame) -> pd.DataFrame:
        """Собрать базовую таблицу результата по идентификаторам сценариев."""

        base_columns = [
            column for column in self.service_columns if column in q_pred_components.columns
        ]
        result = q_pred_components[base_columns].copy()
        if self.latent_column in q_pred_components.columns:
            result[self.latent_column] = q_pred_components[self.latent_column]
        return result

    def _validate_weights(self) -> None:
        """Проверить полноту и нормировку весов частных критериев."""

        missing = [criterion for criterion in self.criteria if criterion not in self.weights]
        if missing:
            msg = "В весах интегрального показателя отсутствуют критерии: " f"{missing}."
            raise IntegralQualityPredictionError(msg)
        unknown = [criterion for criterion in self.weights if criterion not in self.criteria]
        if unknown:
            msg = "В весах интегрального показателя найдены лишние критерии: " f"{unknown}."
            raise IntegralQualityPredictionError(msg)
        for criterion_name, weight in self.weights.items():
            if not 0.0 <= float(weight) <= 1.0:
                msg = (
                    "Вес частного критерия должен быть в диапазоне [0, 1]: "
                    f"{criterion_name}={weight}."
                )
                raise IntegralQualityPredictionError(msg)
        weight_sum = float(sum(self.weights.values()))
        if abs(weight_sum - 1.0) > 1e-6:
            msg = f"Сумма весов частных критериев должна быть равна 1: {weight_sum}."
            raise IntegralQualityPredictionError(msg)

    def _require_columns(
        self,
        data_frame: pd.DataFrame,
        columns: Sequence[str],
        source_name: str,
    ) -> None:
        """Проверить наличие обязательных колонок."""

        missing = [column for column in columns if column not in data_frame.columns]
        if missing:
            msg = f"В {source_name} отсутствуют обязательные колонки: {missing}."
            raise IntegralQualityPredictionError(msg)

    def _reject_duplicate_keys(self, q_pred_components: pd.DataFrame) -> None:
        """Запретить дубли ключей объединения."""

        duplicated = q_pred_components.duplicated(list(self.merge_columns))
        if duplicated.any():
            examples = q_pred_components.loc[duplicated, list(self.merge_columns)].head(5)
            msg = (
                "В таблице компонентов качества найдены дубли ключей: "
                f"{examples.to_dict(orient='records')}."
            )
            raise IntegralQualityPredictionError(msg)

    def _require_q_pred_range(self, result: pd.DataFrame) -> None:
        """Проверить допустимый диапазон интегральной оценки."""

        if result["q_pred"].isna().any():
            msg = "В интегральном показателе q_pred найдены пропуски."
            raise IntegralQualityPredictionError(msg)
        if not result["q_pred"].between(0.0, 1.0).all():
            msg = "Интегральный показатель q_pred должен находиться в диапазоне [0, 1]."
            raise IntegralQualityPredictionError(msg)

    def _coerce_quality_weights(
        self,
        quality_weights: Chapter5QualityWeights | Mapping[str, float],
    ) -> dict[str, float]:
        """Преобразовать веса критериев к простому словарю."""

        if isinstance(quality_weights, Chapter5QualityWeights):
            return {name: float(value) for name, value in quality_weights.weights.items()}
        if isinstance(quality_weights, Mapping):
            return {str(name): float(value) for name, value in quality_weights.items()}
        msg = "Веса частных критериев заданы в неподдерживаемом формате."
        raise IntegralQualityPredictionError(msg)
