"""Нормировка априорных признаков главы 5.

Модуль преобразует числовые признаки ``X_prior`` к диапазону ``[0, 1]``.
Направление нормировки берется из словаря априорных признаков: признаки, для
которых меньшее значение соответствует лучшему качеству, инвертируются после
min-max преобразования. Нечисловые априорные признаки не используются в
расчетной нормированной матрице, но фиксируются в отчете.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pandas.api.types import is_numeric_dtype

from manual_coding_sim.prediction.chapter5_config import Chapter5PriorFeatureDictionary

SERVICE_COLUMNS = ("scenario_id", "protocol_id", "run_id", "alternative_id")
ALLOWED_DIRECTIONS = {"higher_is_better", "lower_is_better", "neutral"}


@dataclass(frozen=True)
class PriorFeatureNormalizationItem:
    """Отчет по нормировке одного априорного признака."""

    feature_name: str
    direction: str
    source_min: float | None
    source_max: float | None
    missing_count: int
    out_of_range_count: int
    is_constant: bool
    is_numeric: bool
    normalized_column: str | None
    status: str


@dataclass(frozen=True)
class PriorFeatureNormalizationReport:
    """Сводный отчет нормировки априорных признаков."""

    row_count: int
    input_column_count: int
    normalized_feature_count: int
    skipped_feature_count: int
    missing_dictionary_features: tuple[str, ...] = ()
    unknown_input_features: tuple[str, ...] = ()
    non_numeric_features: tuple[str, ...] = ()
    constant_features: tuple[str, ...] = ()
    items: tuple[PriorFeatureNormalizationItem, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Преобразовать отчет в JSON-совместимый словарь."""

        payload = asdict(self)
        payload["items"] = [asdict(item) for item in self.items]
        return payload


@dataclass(frozen=True)
class PriorFeatureNormalizationResult:
    """Результат нормировки априорных признаков."""

    normalized_features: pd.DataFrame
    report: PriorFeatureNormalizationReport


class PriorFeatureNormalizationError(ValueError):
    """Ошибка нормировки априорных признаков главы 5."""


class PriorFeatureNormalizer:
    """Нормирует числовые признаки ``X_prior`` к диапазону ``[0, 1]``."""

    def __init__(
        self,
        feature_dictionary: Chapter5PriorFeatureDictionary | dict[str, dict[str, Any]],
        *,
        service_columns: Iterable[str] = SERVICE_COLUMNS,
        constant_fill_value: float = 1.0,
    ) -> None:
        """Создать нормировщик по словарю априорных признаков.

        ``constant_fill_value`` используется для признаков, имеющих одно значение
        во всем корпусе. В текущем исследовательском контуре такое значение
        трактуется как отсутствие различающего риска, поэтому по умолчанию оно
        получает нормированную оценку 1.0.
        """

        if isinstance(feature_dictionary, Chapter5PriorFeatureDictionary):
            self.feature_dictionary = feature_dictionary.features
        else:
            self.feature_dictionary = feature_dictionary
        self.service_columns = tuple(service_columns)
        self.constant_fill_value = float(constant_fill_value)
        self._validate_dictionary()

    def normalize(self, prior_features: pd.DataFrame) -> PriorFeatureNormalizationResult:
        """Нормировать априорные признаки и вернуть таблицу с отчетом."""

        if prior_features.empty:
            msg = "Таблица априорных признаков пуста, нормировка невозможна."
            raise PriorFeatureNormalizationError(msg)
        self._require_service_columns(prior_features)

        normalized = prior_features[
            [column for column in self.service_columns if column in prior_features.columns]
        ].copy()
        input_prior_columns = tuple(
            column for column in prior_features.columns if column.startswith("prior_")
        )
        dictionary_prior_columns = tuple(self.feature_dictionary.keys())
        unknown_input_features = tuple(
            column for column in input_prior_columns if column not in self.feature_dictionary
        )
        missing_dictionary_features = tuple(
            column for column in dictionary_prior_columns if column not in prior_features.columns
        )

        items: list[PriorFeatureNormalizationItem] = []
        non_numeric_features: list[str] = []
        constant_features: list[str] = []

        for feature_name in input_prior_columns:
            meta = self.feature_dictionary.get(feature_name)
            if meta is None:
                items.append(
                    PriorFeatureNormalizationItem(
                        feature_name=feature_name,
                        direction="unknown",
                        source_min=None,
                        source_max=None,
                        missing_count=int(prior_features[feature_name].isna().sum()),
                        out_of_range_count=0,
                        is_constant=False,
                        is_numeric=False,
                        normalized_column=None,
                        status="unknown_feature_skipped",
                    )
                )
                continue

            direction = str(meta.get("direction", ""))
            if not is_numeric_dtype(prior_features[feature_name]):
                non_numeric_features.append(feature_name)
                items.append(
                    PriorFeatureNormalizationItem(
                        feature_name=feature_name,
                        direction=direction,
                        source_min=None,
                        source_max=None,
                        missing_count=int(prior_features[feature_name].isna().sum()),
                        out_of_range_count=0,
                        is_constant=False,
                        is_numeric=False,
                        normalized_column=None,
                        status="non_numeric_skipped",
                    )
                )
                continue

            normalized_column = f"{feature_name}_norm"
            values = pd.to_numeric(prior_features[feature_name], errors="coerce")
            normalized_values, item = self._normalize_numeric_series(
                feature_name=feature_name,
                values=values,
                direction=direction,
                normalized_column=normalized_column,
            )
            normalized[normalized_column] = normalized_values
            items.append(item)
            if item.is_constant:
                constant_features.append(feature_name)

        self._require_normalized_range(normalized)
        report = PriorFeatureNormalizationReport(
            row_count=int(prior_features.shape[0]),
            input_column_count=int(prior_features.shape[1]),
            normalized_feature_count=len(
                [item for item in items if item.status in {"normalized", "constant_filled"}]
            ),
            skipped_feature_count=len(
                [item for item in items if item.status.endswith("skipped")]
            ),
            missing_dictionary_features=missing_dictionary_features,
            unknown_input_features=unknown_input_features,
            non_numeric_features=tuple(non_numeric_features),
            constant_features=tuple(constant_features),
            items=tuple(items),
        )
        return PriorFeatureNormalizationResult(normalized_features=normalized, report=report)

    def save_outputs(
        self,
        result: PriorFeatureNormalizationResult,
        *,
        normalized_features_path: Path,
        report_path: Path,
    ) -> None:
        """Сохранить нормированную таблицу и JSON-отчет."""

        normalized_features_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        result.normalized_features.to_csv(normalized_features_path, index=False)
        report_path.write_text(
            json.dumps(result.report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _normalize_numeric_series(
        self,
        *,
        feature_name: str,
        values: pd.Series,
        direction: str,
        normalized_column: str,
    ) -> tuple[pd.Series, PriorFeatureNormalizationItem]:
        """Выполнить min-max нормировку одного числового признака."""

        source_min = float(values.min(skipna=True))
        source_max = float(values.max(skipna=True))
        missing_count = int(values.isna().sum())
        if pd.isna(source_min) or pd.isna(source_max):
            msg = f"Признак {feature_name} не содержит числовых значений для нормировки."
            raise PriorFeatureNormalizationError(msg)

        is_constant = source_max == source_min
        if is_constant:
            normalized_values = pd.Series(self.constant_fill_value, index=values.index)
            status = "constant_filled"
        else:
            normalized_values = (values - source_min) / (source_max - source_min)
            if direction == "lower_is_better":
                normalized_values = 1.0 - normalized_values
            normalized_values = normalized_values.fillna(self.constant_fill_value).clip(0.0, 1.0)
            status = "normalized"

        out_of_range_count = int(((normalized_values < 0.0) | (normalized_values > 1.0)).sum())
        return normalized_values, PriorFeatureNormalizationItem(
            feature_name=feature_name,
            direction=direction,
            source_min=source_min,
            source_max=source_max,
            missing_count=missing_count,
            out_of_range_count=out_of_range_count,
            is_constant=is_constant,
            is_numeric=True,
            normalized_column=normalized_column,
            status=status,
        )

    def _validate_dictionary(self) -> None:
        """Проверить корректность словаря признаков для нормировки."""

        if not self.feature_dictionary:
            msg = "Словарь априорных признаков для нормировки пуст."
            raise PriorFeatureNormalizationError(msg)
        for feature_name, meta in self.feature_dictionary.items():
            if not feature_name.startswith("prior_"):
                msg = f"В словаре нормировки найден неаприорный признак: {feature_name}."
                raise PriorFeatureNormalizationError(msg)
            direction = str(meta.get("direction", ""))
            if direction not in ALLOWED_DIRECTIONS:
                msg = f"Недопустимое направление нормировки признака {feature_name}: {direction}."
                raise PriorFeatureNormalizationError(msg)

    def _require_service_columns(self, prior_features: pd.DataFrame) -> None:
        """Проверить наличие обязательных идентификаторов сценария."""

        required_columns = ("scenario_id", "protocol_id")
        missing_columns = [column for column in required_columns if column not in prior_features.columns]
        if missing_columns:
            joined = ", ".join(missing_columns)
            msg = f"В таблице априорных признаков отсутствуют обязательные колонки: {joined}."
            raise PriorFeatureNormalizationError(msg)

    @staticmethod
    def _require_normalized_range(normalized_features: pd.DataFrame) -> None:
        """Проверить диапазон всех нормированных колонок."""

        normalized_columns = [
            column for column in normalized_features.columns if column.endswith("_norm")
        ]
        if not normalized_columns:
            msg = "После нормировки не сформировано ни одной колонки *_norm."
            raise PriorFeatureNormalizationError(msg)
        values = normalized_features[normalized_columns]
        bad_mask = (values < 0.0) | (values > 1.0)
        if bad_mask.any().any():
            bad_columns = values.columns[bad_mask.any(axis=0)].tolist()
            msg = f"Нормированные значения вышли за диапазон [0, 1]: {bad_columns}."
            raise PriorFeatureNormalizationError(msg)
