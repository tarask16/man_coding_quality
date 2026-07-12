"""Токенизация априорных признаков для LDA-корпуса главы 4.

Модуль преобразует табличное описание априорных признаков ``X_prior`` в
последовательности дискретных токенов. Эти токены далее используются как
аналог «слов» в LDA-модели латентных факторов качества.

Все правила токенизации должны быть воспроизводимыми: после обучения
токенизатора его состояние сохраняется в ``token_map.json`` и может быть
использовано для повторного построения корпуса.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, Mapping, Sequence

from manual_coding_sim.lda.config import LdaTokenizationConfig


DEFAULT_IDENTIFIER_COLUMNS = frozenset(
    {
        "id",
        "run_id",
        "protocol_id",
        "scenario_id",
        "alternative_id",
        "sample_id",
        "document_id",
        "created_at",
    }
)

_BINARY_TRUE_VALUES = frozenset({"1", "true", "yes", "y", "да", "истина"})
_BINARY_FALSE_VALUES = frozenset({"0", "false", "no", "n", "нет", "ложь"})
_MISSING_VALUES = frozenset({"", "nan", "none", "null", "na", "n/a"})
_LEVEL_LABELS = {
    2: ("low", "high"),
    3: ("low", "mid", "high"),
    4: ("very_low", "low", "high", "very_high"),
    5: ("very_low", "low", "mid", "high", "very_high"),
}


@dataclass(frozen=True)
class TokenizedFeature:
    """Один токен, полученный из одного признака одной строки корпуса."""

    feature_name: str
    token: str
    token_kind: str
    raw_value: str


@dataclass(frozen=True)
class FeatureTokenizationRule:
    """Воспроизводимое правило токенизации одного признака."""

    feature_name: str
    token_kind: str
    thresholds: tuple[float, ...] = ()
    categories: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Преобразовать правило в JSON-совместимый словарь."""

        return {
            "feature_name": self.feature_name,
            "token_kind": self.token_kind,
            "thresholds": list(self.thresholds),
            "categories": list(self.categories),
        }


class FeatureTokenizer:
    """Преобразует строки ``X_prior`` в наборы LDA-токенов.

    Токенизатор сначала обучается на всем наборе априорных признаков, чтобы
    зафиксировать типы колонок и пороги дискретизации числовых признаков.
    После этого он может стабильно преобразовывать каждую строку в токены.
    """

    def __init__(
        self,
        config: LdaTokenizationConfig | None = None,
        identifier_columns: Iterable[str] | None = None,
    ) -> None:
        """Создать токенизатор с заданными параметрами дискретизации."""

        self.config = config or LdaTokenizationConfig()
        self.identifier_columns = set(identifier_columns or DEFAULT_IDENTIFIER_COLUMNS)
        self.rules: dict[str, FeatureTokenizationRule] = {}
        self.feature_columns: tuple[str, ...] = ()

    def fit(self, rows: Sequence[Mapping[str, object]]) -> "FeatureTokenizer":
        """Зафиксировать правила токенизации по строкам априорных признаков."""

        self.config.validate()
        if not rows:
            msg = "Нельзя обучить токенизатор на пустом наборе строк."
            raise ValueError(msg)

        columns = tuple(rows[0].keys())
        self.feature_columns = tuple(
            column for column in columns if column not in self.identifier_columns
        )
        if not self.feature_columns:
            msg = "В данных не найдено ни одной колонки априорных признаков."
            raise ValueError(msg)

        rules: dict[str, FeatureTokenizationRule] = {}
        for column in self.feature_columns:
            values = [row.get(column) for row in rows]
            rules[column] = self._build_rule(column, values)
        self.rules = rules
        return self

    def transform_row(self, row: Mapping[str, object]) -> list[TokenizedFeature]:
        """Преобразовать одну строку априорных признаков в список токенов."""

        if not self.rules:
            msg = "Токенизатор должен быть обучен методом fit перед transform_row."
            raise RuntimeError(msg)

        tokens: list[TokenizedFeature] = []
        for column in self.feature_columns:
            rule = self.rules[column]
            raw_value = row.get(column)
            token = self._tokenize_value(rule, raw_value)
            tokens.append(
                TokenizedFeature(
                    feature_name=column,
                    token=token,
                    token_kind=rule.token_kind,
                    raw_value=self._stringify_value(raw_value),
                )
            )
        return tokens

    def transform(self, rows: Sequence[Mapping[str, object]]) -> list[list[TokenizedFeature]]:
        """Преобразовать набор строк в список документов-токенов."""

        return [self.transform_row(row) for row in rows]

    def to_token_map(self) -> dict[str, object]:
        """Вернуть воспроизводимое описание правил токенизации."""

        return {
            "numeric_strategy": self.config.numeric_strategy,
            "numeric_bins": self.config.numeric_bins,
            "identifier_columns": sorted(self.identifier_columns),
            "feature_columns": list(self.feature_columns),
            "rules": [self.rules[column].to_dict() for column in self.feature_columns],
        }

    def _build_rule(
        self,
        column: str,
        values: Sequence[object],
    ) -> FeatureTokenizationRule:
        """Построить правило токенизации для одной колонки."""

        non_missing = [value for value in values if not self._is_missing(value)]
        if not non_missing:
            return FeatureTokenizationRule(
                feature_name=column,
                token_kind="missing_only",
            )

        if self._is_binary_series(non_missing):
            return FeatureTokenizationRule(
                feature_name=column,
                token_kind="binary",
            )

        numeric_values = self._collect_numeric_values(non_missing)
        if len(numeric_values) == len(non_missing):
            thresholds = self._build_numeric_thresholds(numeric_values)
            return FeatureTokenizationRule(
                feature_name=column,
                token_kind="numeric",
                thresholds=thresholds,
            )

        categories = tuple(
            sorted({_normalize_token_part(self._stringify_value(value)) for value in non_missing})
        )
        return FeatureTokenizationRule(
            feature_name=column,
            token_kind="categorical",
            categories=categories,
        )

    def _tokenize_value(
        self,
        rule: FeatureTokenizationRule,
        value: object,
    ) -> str:
        """Преобразовать значение признака в строковый токен."""

        safe_feature = _normalize_token_part(rule.feature_name)
        if self._is_missing(value):
            return f"{safe_feature}__missing"
        if rule.token_kind == "missing_only":
            return f"{safe_feature}__missing"
        if rule.token_kind == "binary":
            suffix = "present" if self._to_binary_bool(value) else "absent"
            return f"{safe_feature}__{suffix}"
        if rule.token_kind == "numeric":
            numeric_value = self._to_float(value)
            bin_index = self._numeric_bin_index(numeric_value, rule.thresholds)
            label = self._bin_label(bin_index)
            return f"{safe_feature}__level_{label}"

        safe_value = _normalize_token_part(self._stringify_value(value))
        return f"{safe_feature}__value_{safe_value}"

    def _build_numeric_thresholds(self, values: Sequence[float]) -> tuple[float, ...]:
        """Построить внутренние пороги дискретизации числового признака."""

        unique_values = sorted(set(values))
        if len(unique_values) <= 1:
            return ()

        if self.config.numeric_strategy == "uniform":
            return self._uniform_thresholds(unique_values)
        return self._quantile_thresholds(values)

    def _quantile_thresholds(self, values: Sequence[float]) -> tuple[float, ...]:
        """Построить квантильные пороги дискретизации."""

        sorted_values = sorted(values)
        thresholds: list[float] = []
        for boundary_index in range(1, self.config.numeric_bins):
            position = boundary_index * (len(sorted_values) - 1) / self.config.numeric_bins
            lower_index = int(math.floor(position))
            upper_index = int(math.ceil(position))
            if lower_index == upper_index:
                threshold = sorted_values[lower_index]
            else:
                fraction = position - lower_index
                threshold = sorted_values[lower_index] * (1 - fraction)
                threshold += sorted_values[upper_index] * fraction
            thresholds.append(float(threshold))
        return tuple(_unique_sorted_thresholds(thresholds))

    def _uniform_thresholds(self, values: Sequence[float]) -> tuple[float, ...]:
        """Построить равномерные пороги дискретизации."""

        min_value = min(values)
        max_value = max(values)
        if min_value == max_value:
            return ()
        step = (max_value - min_value) / self.config.numeric_bins
        thresholds = [min_value + step * i for i in range(1, self.config.numeric_bins)]
        return tuple(_unique_sorted_thresholds(thresholds))

    def _numeric_bin_index(self, value: float, thresholds: Sequence[float]) -> int:
        """Определить номер интервала для числового значения."""

        for index, threshold in enumerate(thresholds):
            if value <= threshold:
                return index
        return len(thresholds)

    def _bin_label(self, bin_index: int) -> str:
        """Вернуть читаемую метку интервала для числового токена."""

        labels = _LEVEL_LABELS.get(self.config.numeric_bins)
        if labels is not None and bin_index < len(labels):
            return labels[bin_index]
        return f"bin_{bin_index}"

    def _is_binary_series(self, values: Sequence[object]) -> bool:
        """Проверить, является ли колонка бинарной."""

        normalized = {self._stringify_value(value).strip().lower() for value in values}
        return bool(normalized) and normalized <= (_BINARY_TRUE_VALUES | _BINARY_FALSE_VALUES)

    def _to_binary_bool(self, value: object) -> bool:
        """Преобразовать бинарное значение в логический признак."""

        normalized = self._stringify_value(value).strip().lower()
        return normalized in _BINARY_TRUE_VALUES

    def _collect_numeric_values(self, values: Sequence[object]) -> list[float]:
        """Собрать значения, которые можно представить как числа."""

        numeric_values: list[float] = []
        for value in values:
            try:
                numeric_values.append(self._to_float(value))
            except ValueError:
                return []
        return numeric_values

    def _to_float(self, value: object) -> float:
        """Преобразовать значение CSV в число с плавающей точкой."""

        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, int | float):
            numeric_value = float(value)
        else:
            numeric_value = float(str(value).replace(",", "."))
        if math.isnan(numeric_value):
            msg = "NaN не может быть числовым значением признака."
            raise ValueError(msg)
        return numeric_value

    def _is_missing(self, value: object) -> bool:
        """Проверить, является ли значение пропуском."""

        if value is None:
            return True
        if isinstance(value, float) and math.isnan(value):
            return True
        normalized = self._stringify_value(value).strip().lower()
        return normalized in _MISSING_VALUES

    def _stringify_value(self, value: object) -> str:
        """Получить стабильное строковое представление значения."""

        if value is None:
            return ""
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)


def _normalize_token_part(value: str) -> str:
    """Нормализовать часть токена до безопасного идентификатора."""

    normalized = value.strip().lower()
    normalized = re.sub(r"[^0-9a-zA-Zа-яА-ЯёЁ]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "empty"


def _unique_sorted_thresholds(values: Iterable[float]) -> list[float]:
    """Удалить дублирующиеся пороги без нарушения числового порядка."""

    result: list[float] = []
    for value in sorted(values):
        if not result or not math.isclose(value, result[-1]):
            result.append(value)
    return result
