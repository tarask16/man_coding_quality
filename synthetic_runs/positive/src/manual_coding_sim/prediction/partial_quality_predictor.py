"""Расчет частных прогнозных критериев качества главы 5.

Модуль формирует таблицу ``q_pred_components.csv``. Для каждого частного
критерия качества рассчитываются две составляющие:

* наблюдаемая априорная составляющая ``B_j(X_prior)`` по нормированным
  признакам ``*_norm``;
* латентная составляющая ``L_j(theta_prior)`` по ранее рассчитанной
  компоненте ``q_latent``.

Итоговый частный прогноз строится как выпуклая комбинация этих составляющих.
Все консольные сообщения формируются в ``chapter5_runner.py``; данный модуль
возвращает структурированный результат и JSON-совместимый отчет.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from manual_coding_sim.prediction.chapter5_config import (
    Chapter5FeatureCriterionWeights,
    Chapter5FeatureWeights,
    QUALITY_CRITERIA,
)

SERVICE_COLUMNS = ("scenario_id", "protocol_id", "run_id", "alternative_id")
MERGE_COLUMNS = ("scenario_id", "protocol_id")


@dataclass(frozen=True)
class PartialQualityCriterionReport:
    """Отчет по одному частному прогнозному критерию."""

    criterion_name: str
    observed_weight: float
    latent_weight: float
    used_features: dict[str, float]
    feature_component_min: float
    feature_component_max: float
    latent_component_min: float
    latent_component_max: float
    prediction_min: float
    prediction_max: float
    prediction_mean: float


@dataclass(frozen=True)
class PartialQualityPredictionReport:
    """Сводный отчет расчета частных прогнозных критериев."""

    row_count: int
    criteria: tuple[str, ...]
    merge_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    criterion_reports: tuple[PartialQualityCriterionReport, ...]

    def to_dict(self) -> dict[str, object]:
        """Преобразовать отчет в JSON-совместимый словарь."""

        payload = asdict(self)
        payload["criterion_reports"] = [
            asdict(item) for item in self.criterion_reports
        ]
        return payload


@dataclass(frozen=True)
class PartialQualityPredictionResult:
    """Результат расчета частных прогнозных критериев."""

    components: pd.DataFrame
    report: PartialQualityPredictionReport


class PartialQualityPredictionError(ValueError):
    """Ошибка расчета частных прогнозных критериев главы 5."""


class PartialQualityPredictor:
    """Рассчитывает прогнозы ``q_acc_pred`` ... ``q_fit_pred``."""

    def __init__(
        self,
        feature_weights: Chapter5FeatureWeights | Mapping[str, object],
        *,
        criteria: Sequence[str] = QUALITY_CRITERIA,
        service_columns: Sequence[str] = SERVICE_COLUMNS,
        merge_columns: Sequence[str] = MERGE_COLUMNS,
        latent_column: str = "q_latent",
    ) -> None:
        """Создать калькулятор частных прогнозных критериев.

        ``feature_weights`` может быть объектом ``Chapter5FeatureWeights`` или
        словарем совместимой структуры. Веса признаков внутри каждого критерия
        должны быть нормированы на уровне конфигурационного слоя.
        """

        self.criteria = tuple(criteria)
        self.service_columns = tuple(service_columns)
        self.merge_columns = tuple(merge_columns)
        self.latent_column = latent_column
        self.feature_weights = self._coerce_feature_weights(feature_weights)
        self._validate_feature_weights()

    def predict(
        self,
        normalized_prior_features: pd.DataFrame,
        latent_quality: pd.DataFrame,
    ) -> PartialQualityPredictionResult:
        """Рассчитать частные прогнозные критерии качества."""

        if normalized_prior_features.empty:
            msg = "Таблица нормированных априорных признаков пуста."
            raise PartialQualityPredictionError(msg)
        if latent_quality.empty:
            msg = "Таблица латентной компоненты качества пуста."
            raise PartialQualityPredictionError(msg)

        self._require_columns(normalized_prior_features, self.merge_columns, "нормированных признаках")
        self._require_columns(latent_quality, (*self.merge_columns, self.latent_column), "латентной компоненте")
        self._reject_duplicate_keys(normalized_prior_features, "нормированных признаков")
        self._reject_duplicate_keys(latent_quality, "латентной компоненты")

        result = self._build_base_table(normalized_prior_features, latent_quality)
        criterion_reports: list[PartialQualityCriterionReport] = []

        for criterion_name in self.criteria:
            criterion_config = self.feature_weights[criterion_name]
            report = self._calculate_criterion(
                result,
                normalized_prior_features,
                criterion_name,
                criterion_config,
            )
            criterion_reports.append(report)

        self._require_prediction_range(result)
        report = PartialQualityPredictionReport(
            row_count=int(result.shape[0]),
            criteria=self.criteria,
            merge_columns=self.merge_columns,
            output_columns=tuple(str(column) for column in result.columns),
            criterion_reports=tuple(criterion_reports),
        )
        return PartialQualityPredictionResult(components=result, report=report)

    def save_outputs(
        self,
        result: PartialQualityPredictionResult,
        *,
        components_path: Path,
        report_path: Path | None = None,
    ) -> None:
        """Сохранить таблицу компонентов и необязательный JSON-отчет."""

        components_path.parent.mkdir(parents=True, exist_ok=True)
        result.components.to_csv(components_path, index=False)
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(result.report.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _calculate_criterion(
        self,
        result: pd.DataFrame,
        normalized_prior_features: pd.DataFrame,
        criterion_name: str,
        criterion_config: Chapter5FeatureCriterionWeights,
    ) -> PartialQualityCriterionReport:
        """Рассчитать один частный критерий и добавить колонки в результат."""

        feature_component_column = f"{criterion_name}_feature_component"
        latent_component_column = f"{criterion_name}_latent_component"
        prediction_column = f"{criterion_name}_pred"
        observed_weight_column = f"{criterion_name}_observed_weight"
        latent_weight_column = f"{criterion_name}_latent_weight"

        feature_component = pd.Series(0.0, index=result.index, dtype="float64")
        for feature_name, feature_weight in criterion_config.features.items():
            normalized_column = f"{feature_name}_norm"
            if normalized_column not in normalized_prior_features.columns:
                msg = (
                    "В нормированной таблице отсутствует признак для критерия "
                    f"{criterion_name}: {normalized_column}."
                )
                raise PartialQualityPredictionError(msg)
            feature_values = pd.to_numeric(
                normalized_prior_features[normalized_column],
                errors="coerce",
            )
            if feature_values.isna().any():
                msg = f"В признаке {normalized_column} найдены пропуски."
                raise PartialQualityPredictionError(msg)
            feature_component = feature_component + feature_values * float(feature_weight)

        observed_weight = float(criterion_config.observed_weight)
        latent_weight = 1.0 - observed_weight
        latent_component = pd.to_numeric(result[self.latent_column], errors="coerce")
        prediction = observed_weight * feature_component + latent_weight * latent_component
        prediction = prediction.clip(0.0, 1.0)

        result[feature_component_column] = feature_component.clip(0.0, 1.0)
        result[latent_component_column] = latent_component.clip(0.0, 1.0)
        result[observed_weight_column] = observed_weight
        result[latent_weight_column] = latent_weight
        result[prediction_column] = prediction

        return PartialQualityCriterionReport(
            criterion_name=criterion_name,
            observed_weight=observed_weight,
            latent_weight=latent_weight,
            used_features=dict(criterion_config.features),
            feature_component_min=float(result[feature_component_column].min()),
            feature_component_max=float(result[feature_component_column].max()),
            latent_component_min=float(result[latent_component_column].min()),
            latent_component_max=float(result[latent_component_column].max()),
            prediction_min=float(result[prediction_column].min()),
            prediction_max=float(result[prediction_column].max()),
            prediction_mean=float(result[prediction_column].mean()),
        )

    def _build_base_table(
        self,
        normalized_prior_features: pd.DataFrame,
        latent_quality: pd.DataFrame,
    ) -> pd.DataFrame:
        """Собрать базовую таблицу результата по идентификаторам сценариев."""

        base_columns = [
            column for column in self.service_columns if column in normalized_prior_features.columns
        ]
        base = normalized_prior_features[base_columns].copy()
        latent_columns = [*self.merge_columns, self.latent_column]
        merged = base.merge(
            latent_quality[latent_columns],
            on=list(self.merge_columns),
            how="left",
            validate="one_to_one",
        )
        if merged[self.latent_column].isna().any():
            missing_count = int(merged[self.latent_column].isna().sum())
            msg = (
                "Не для всех сценариев найдена латентная компонента качества. "
                f"Пропущено строк: {missing_count}."
            )
            raise PartialQualityPredictionError(msg)
        return merged

    def _validate_feature_weights(self) -> None:
        """Проверить, что настройки покрывают все критерии."""

        missing = [criterion for criterion in self.criteria if criterion not in self.feature_weights]
        if missing:
            joined = ", ".join(missing)
            msg = f"Не заданы веса признаков для частных критериев: {joined}."
            raise PartialQualityPredictionError(msg)
        for criterion_name, criterion_config in self.feature_weights.items():
            if not 0.0 <= float(criterion_config.observed_weight) <= 1.0:
                msg = (
                    "Вес наблюдаемой части критерия должен быть в диапазоне [0, 1]: "
                    f"{criterion_name}."
                )
                raise PartialQualityPredictionError(msg)
            feature_weight_sum = sum(float(value) for value in criterion_config.features.values())
            if abs(feature_weight_sum - 1.0) > 1e-6:
                msg = (
                    "Сумма весов признаков частного критерия должна быть равна 1: "
                    f"{criterion_name}."
                )
                raise PartialQualityPredictionError(msg)

    def _coerce_feature_weights(
        self,
        feature_weights: Chapter5FeatureWeights | Mapping[str, object],
    ) -> dict[str, Chapter5FeatureCriterionWeights]:
        """Привести настройки весов к внутреннему формату."""

        if isinstance(feature_weights, Chapter5FeatureWeights):
            return dict(feature_weights.criteria)
        result: dict[str, Chapter5FeatureCriterionWeights] = {}
        for criterion_name, raw_config in dict(feature_weights).items():
            if isinstance(raw_config, Chapter5FeatureCriterionWeights):
                result[str(criterion_name)] = raw_config
                continue
            raw_mapping = dict(raw_config)  # type: ignore[arg-type]
            result[str(criterion_name)] = Chapter5FeatureCriterionWeights(
                observed_weight=float(raw_mapping.get("observed_weight", 0.5)),
                features={
                    str(name): float(value)
                    for name, value in dict(raw_mapping.get("features", {})).items()
                },
            )
        return result

    def _require_columns(
        self,
        dataframe: pd.DataFrame,
        columns: Sequence[str],
        source_name: str,
    ) -> None:
        """Проверить наличие обязательных колонок."""

        missing = [column for column in columns if column not in dataframe.columns]
        if missing:
            joined = ", ".join(missing)
            msg = f"В таблице {source_name} отсутствуют обязательные колонки: {joined}."
            raise PartialQualityPredictionError(msg)

    def _reject_duplicate_keys(self, dataframe: pd.DataFrame, source_name: str) -> None:
        """Запретить дубли ключей объединения."""

        duplicate_mask = dataframe.duplicated(list(self.merge_columns), keep=False)
        if duplicate_mask.any():
            examples = dataframe.loc[duplicate_mask, list(self.merge_columns)].head(3).to_dict("records")
            msg = (
                "Обнаружены дубли ключей при расчете частных критериев. "
                f"Таблица: {source_name}. Примеры: {examples}."
            )
            raise PartialQualityPredictionError(msg)

    def _require_prediction_range(self, result: pd.DataFrame) -> None:
        """Проверить диапазон всех прогнозных критериев."""

        prediction_columns = [column for column in result.columns if column.endswith("_pred")]
        if not prediction_columns:
            msg = "Не сформировано ни одного частного прогнозного критерия."
            raise PartialQualityPredictionError(msg)
        values = result[prediction_columns]
        if values.isna().any().any():
            msg = "В частных прогнозных критериях найдены пропуски."
            raise PartialQualityPredictionError(msg)
        if ((values < 0.0) | (values > 1.0)).any().any():
            msg = "Частные прогнозные критерии вышли за диапазон [0, 1]."
            raise PartialQualityPredictionError(msg)
