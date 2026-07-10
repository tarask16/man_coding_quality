"""Конфигурация программного блока главы 5.

Модуль описывает пути, веса, направления факторов и параметры неопределенности
для построения априорной интегральной оценки качества. На этапе 2 здесь
реализована загрузка YAML-конфигураций и проверка их внутренней согласованности;
расчет ``Q_pred`` будет добавлен последующими этапами.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from manual_coding_sim.prediction.paths import (
    DEFAULT_CHAPTER5_REPORTS_DIR,
    DEFAULT_PREDICTION_REPORT_JSON_PATH,
    DEFAULT_PREDICTION_REPORT_MD_PATH,
    DEFAULT_PRIOR_FEATURES_PATH,
    DEFAULT_Q_PRED_COMPONENTS_PATH,
    DEFAULT_Q_PRED_PATH,
    DEFAULT_THETA_PRIOR_PATH,
    DEFAULT_TOPIC_INTERPRETATION_PATH,
)

QUALITY_CRITERIA = ("q_acc", "q_time", "q_effort", "q_res", "q_rep", "q_fit")
THETA_COLUMNS = ("theta_0", "theta_1", "theta_2")


@dataclass(frozen=True)
class Chapter5InputPaths:
    """Пути к входным артефактам априорного прогноза главы 5."""

    prior_features_path: Path = DEFAULT_PRIOR_FEATURES_PATH
    theta_prior_path: Path = DEFAULT_THETA_PRIOR_PATH
    topic_interpretation_path: Path = DEFAULT_TOPIC_INTERPRETATION_PATH


@dataclass(frozen=True)
class Chapter5OutputPaths:
    """Пути к выходным артефактам априорного прогноза главы 5."""

    reports_dir: Path = DEFAULT_CHAPTER5_REPORTS_DIR
    q_pred_path: Path = DEFAULT_Q_PRED_PATH
    q_pred_components_path: Path = DEFAULT_Q_PRED_COMPONENTS_PATH
    report_json_path: Path = DEFAULT_PREDICTION_REPORT_JSON_PATH
    report_md_path: Path = DEFAULT_PREDICTION_REPORT_MD_PATH


@dataclass(frozen=True)
class Chapter5ConfigFilePaths:
    """Пути к YAML-файлам с параметрами расчета главы 5."""

    quality_weights_path: Path = Path("configs/chapter5_quality_weights.yaml")
    feature_weights_path: Path = Path("configs/chapter5_feature_weights.yaml")
    factor_directions_path: Path = Path("configs/chapter5_factor_directions.yaml")
    uncertainty_path: Path = Path("configs/chapter5_uncertainty.yaml")
    decision_thresholds_path: Path = Path("configs/chapter5_decision_thresholds.yaml")
    prior_feature_dictionary_path: Path = Path("data/schema/prior_feature_dictionary.yaml")


@dataclass(frozen=True)
class Chapter5QualityWeights:
    """Веса частных критериев в интегральной оценке качества."""

    weights: dict[str, float] = field(
        default_factory=lambda: {criterion: 1.0 / len(QUALITY_CRITERIA) for criterion in QUALITY_CRITERIA}
    )

    def validate(self) -> None:
        """Проверить полноту и нормировку весов частных критериев."""

        _require_keys(self.weights, QUALITY_CRITERIA, "весов частных критериев")
        _validate_weight_sum(self.weights, "Сумма весов частных критериев")


@dataclass(frozen=True)
class Chapter5FactorDirections:
    """Направленность влияния латентных факторов на прогноз качества."""

    directions: dict[str, float] = field(
        default_factory=lambda: {"theta_0": -1.0, "theta_1": -1.0, "theta_2": 1.0}
    )

    def validate(self) -> None:
        """Проверить наличие направлений для всех компонентов theta."""

        _require_keys(self.directions, THETA_COLUMNS, "направлений латентных факторов")
        for name, value in self.directions.items():
            if not -1.0 <= value <= 1.0 or value == 0:
                msg = (
                    "Направление латентного фактора должно быть ненулевым "
                    f"числом в диапазоне [-1, 1]: {name}={value}."
                )
                raise ValueError(msg)


@dataclass(frozen=True)
class Chapter5FeatureCriterionWeights:
    """Веса априорных признаков для одного частного критерия."""

    observed_weight: float
    features: dict[str, float]

    def validate(self, criterion_name: str) -> None:
        """Проверить веса признаков внутри частного критерия."""

        if not 0.0 <= self.observed_weight <= 1.0:
            msg = (
                "Вес наблюдаемой части критерия должен быть в диапазоне [0, 1]: "
                f"{criterion_name}."
            )
            raise ValueError(msg)
        if not self.features:
            msg = f"Для критерия {criterion_name} должен быть задан хотя бы один признак."
            raise ValueError(msg)
        _validate_weight_sum(
            self.features,
            f"Сумма весов признаков критерия {criterion_name}",
        )


@dataclass(frozen=True)
class Chapter5FeatureWeights:
    """Набор весов априорных признаков для всех частных критериев."""

    criteria: dict[str, Chapter5FeatureCriterionWeights] = field(default_factory=dict)

    def validate(self) -> None:
        """Проверить наличие и корректность весов признаков."""

        _require_keys(self.criteria, QUALITY_CRITERIA, "настроек признаков частных критериев")
        for criterion_name, criterion_weights in self.criteria.items():
            criterion_weights.validate(criterion_name)


@dataclass(frozen=True)
class Chapter5UncertaintyConfig:
    """Параметры расчета неопределенности прогноза."""

    delta: float = 0.15
    mean_stability: float = 0.84885
    min_stability: float = 0.808418
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "theta_entropy": 0.5,
            "lda_stability": 0.3,
            "input_quality": 0.2,
        }
    )

    def validate(self) -> None:
        """Проверить параметры неопределенности."""

        if self.delta < 0:
            msg = "Коэффициент delta для интервала неопределенности не может быть отрицательным."
            raise ValueError(msg)
        if not 0.0 <= self.mean_stability <= 1.0:
            msg = "mean_stability должен находиться в диапазоне [0, 1]."
            raise ValueError(msg)
        if not 0.0 <= self.min_stability <= 1.0:
            msg = "min_stability должен находиться в диапазоне [0, 1]."
            raise ValueError(msg)
        _require_keys(
            self.weights,
            ("theta_entropy", "lda_stability", "input_quality"),
            "весов неопределенности",
        )
        _validate_weight_sum(self.weights, "Сумма весов неопределенности")


@dataclass(frozen=True)
class Chapter5DecisionThresholds:
    """Пороги классов качества по ``Q_pred``."""

    low_max: float = 0.45
    high_min: float = 0.70
    class_labels: dict[str, str] = field(
        default_factory=lambda: {
            "low": "low_quality",
            "medium": "medium_quality",
            "high": "high_quality",
        }
    )

    def validate(self) -> None:
        """Проверить корректность порогов классификации качества."""

        if not 0.0 <= self.low_max < self.high_min <= 1.0:
            msg = "Пороги качества должны удовлетворять условию 0 <= low_max < high_min <= 1."
            raise ValueError(msg)
        _require_keys(self.class_labels, ("low", "medium", "high"), "меток классов качества")


@dataclass(frozen=True)
class Chapter5PriorFeatureDictionary:
    """Словарь априорных признаков и направлений нормировки."""

    features: dict[str, dict[str, Any]] = field(default_factory=dict)

    def validate(self) -> None:
        """Проверить, что словарь содержит только допустимые априорные признаки."""

        if not self.features:
            msg = "Словарь априорных признаков главы 5 не должен быть пустым."
            raise ValueError(msg)
        allowed_directions = {"higher_is_better", "lower_is_better", "neutral"}
        for name, meta in self.features.items():
            if not name.startswith("prior_"):
                msg = f"В словаре априорных признаков найден недопустимый признак: {name}."
                raise ValueError(msg)
            role = str(meta.get("role", ""))
            if role != "prior":
                msg = f"Признак {name} должен иметь роль prior."
                raise ValueError(msg)
            direction = str(meta.get("direction", ""))
            if direction not in allowed_directions:
                msg = (
                    "Направление нормировки признака должно быть higher_is_better, "
                    f"lower_is_better или neutral: {name}."
                )
                raise ValueError(msg)


@dataclass(frozen=True)
class Chapter5PredictionConfig:
    """Единая конфигурация главы 5."""

    inputs: Chapter5InputPaths = field(default_factory=Chapter5InputPaths)
    outputs: Chapter5OutputPaths = field(default_factory=Chapter5OutputPaths)
    config_files: Chapter5ConfigFilePaths = field(default_factory=Chapter5ConfigFilePaths)
    quality_weights: Chapter5QualityWeights = field(default_factory=Chapter5QualityWeights)
    feature_weights: Chapter5FeatureWeights = field(default_factory=Chapter5FeatureWeights)
    factor_directions: Chapter5FactorDirections = field(default_factory=Chapter5FactorDirections)
    uncertainty: Chapter5UncertaintyConfig = field(default_factory=Chapter5UncertaintyConfig)
    decision_thresholds: Chapter5DecisionThresholds = field(default_factory=Chapter5DecisionThresholds)
    prior_feature_dictionary: Chapter5PriorFeatureDictionary = field(
        default_factory=Chapter5PriorFeatureDictionary
    )
    expected_topic_count: int = 3
    forbid_fact_features: bool = True
    forbid_quality_targets: bool = True
    overwrite: bool = True

    def validate(self) -> None:
        """Проверить базовую корректность конфигурации главы 5."""

        if self.expected_topic_count < 2:
            msg = "Число латентных факторов главы 5 должно быть не меньше 2."
            raise ValueError(msg)
        if not str(self.outputs.reports_dir):
            msg = "Каталог отчетов главы 5 не должен быть пустым."
            raise ValueError(msg)
        self.quality_weights.validate()
        self.feature_weights.validate()
        self.factor_directions.validate()
        self.uncertainty.validate()
        self.decision_thresholds.validate()
        self.prior_feature_dictionary.validate()


@dataclass(frozen=True)
class Chapter5ConfigLoadResult:
    """Результат загрузки YAML-конфигурации главы 5."""

    config_path: Path
    project_root: Path
    config: Chapter5PredictionConfig


class Chapter5ConfigLoader:
    """Загружает основной YAML и связанные конфигурации главы 5."""

    def load(
        self,
        config_path: str | Path = "configs/chapter5.yaml",
        project_root: str | Path = ".",
    ) -> Chapter5ConfigLoadResult:
        """Прочитать YAML-конфигурации и вернуть готовый объект настроек."""

        root = Path(project_root)
        path = _resolve_from_root(root, Path(config_path))
        if not path.exists():
            msg = f"Файл конфигурации главы 5 не найден: {path}"
            raise FileNotFoundError(msg)

        payload = _read_yaml_mapping(path)
        config = self._build_config(payload, root)
        config.validate()
        return Chapter5ConfigLoadResult(config_path=path, project_root=root, config=config)

    def _build_config(
        self,
        payload: dict[str, Any],
        project_root: Path,
    ) -> Chapter5PredictionConfig:
        """Собрать объект ``Chapter5PredictionConfig`` из YAML-словарей."""

        input_section = _section(payload, "input", "inputs")
        output_section = _section(payload, "output", "outputs")
        config_files_section = _section(payload, "config_files")
        model_section = _section(payload, "model")
        safety_section = _section(payload, "safety")
        runner_section = _section(payload, "runner")

        config_files = Chapter5ConfigFilePaths(
            quality_weights_path=_path(
                config_files_section.get("quality_weights"),
                "configs/chapter5_quality_weights.yaml",
            ),
            feature_weights_path=_path(
                config_files_section.get("feature_weights"),
                "configs/chapter5_feature_weights.yaml",
            ),
            factor_directions_path=_path(
                config_files_section.get("factor_directions"),
                "configs/chapter5_factor_directions.yaml",
            ),
            uncertainty_path=_path(
                config_files_section.get("uncertainty"),
                "configs/chapter5_uncertainty.yaml",
            ),
            decision_thresholds_path=_path(
                config_files_section.get("decision_thresholds"),
                "configs/chapter5_decision_thresholds.yaml",
            ),
            prior_feature_dictionary_path=_path(
                config_files_section.get("prior_feature_dictionary"),
                "data/schema/prior_feature_dictionary.yaml",
            ),
        )

        return Chapter5PredictionConfig(
            inputs=Chapter5InputPaths(
                prior_features_path=_path(input_section.get("prior_features"), DEFAULT_PRIOR_FEATURES_PATH),
                theta_prior_path=_path(input_section.get("theta_prior"), DEFAULT_THETA_PRIOR_PATH),
                topic_interpretation_path=_path(
                    input_section.get("topic_interpretation"),
                    DEFAULT_TOPIC_INTERPRETATION_PATH,
                ),
            ),
            outputs=Chapter5OutputPaths(
                reports_dir=_path(output_section.get("reports_dir"), DEFAULT_CHAPTER5_REPORTS_DIR),
                q_pred_path=_path(output_section.get("q_pred"), DEFAULT_Q_PRED_PATH),
                q_pred_components_path=_path(
                    output_section.get("q_pred_components"),
                    DEFAULT_Q_PRED_COMPONENTS_PATH,
                ),
                report_json_path=_path(
                    output_section.get("report_json"),
                    DEFAULT_PREDICTION_REPORT_JSON_PATH,
                ),
                report_md_path=_path(
                    output_section.get("report_md"),
                    DEFAULT_PREDICTION_REPORT_MD_PATH,
                ),
            ),
            config_files=config_files,
            quality_weights=_load_quality_weights(project_root, config_files.quality_weights_path),
            feature_weights=_load_feature_weights(project_root, config_files.feature_weights_path),
            factor_directions=_load_factor_directions(project_root, config_files.factor_directions_path),
            uncertainty=_load_uncertainty(project_root, config_files.uncertainty_path),
            decision_thresholds=_load_decision_thresholds(
                project_root,
                config_files.decision_thresholds_path,
            ),
            prior_feature_dictionary=_load_prior_feature_dictionary(
                project_root,
                config_files.prior_feature_dictionary_path,
            ),
            expected_topic_count=_positive_int(
                model_section.get("expected_topic_count", 3),
                "expected_topic_count",
            ),
            forbid_fact_features=_bool(safety_section.get("forbid_fact_features", True)),
            forbid_quality_targets=_bool(safety_section.get("forbid_quality_targets", True)),
            overwrite=_bool(runner_section.get("overwrite", True)),
        )


def load_chapter5_prediction_config(
    config_path: str | Path = "configs/chapter5.yaml",
    project_root: str | Path = ".",
) -> Chapter5PredictionConfig:
    """Загрузить конфигурацию главы 5 и вернуть только объект настроек."""

    return Chapter5ConfigLoader().load(config_path=config_path, project_root=project_root).config


def _load_quality_weights(project_root: Path, path: Path) -> Chapter5QualityWeights:
    payload = _read_yaml_mapping(_resolve_from_root(project_root, path))
    return Chapter5QualityWeights(weights=_float_dict(_section(payload, "quality_weights")))


def _load_factor_directions(project_root: Path, path: Path) -> Chapter5FactorDirections:
    payload = _read_yaml_mapping(_resolve_from_root(project_root, path))
    return Chapter5FactorDirections(directions=_float_dict(_section(payload, "factor_directions")))


def _load_feature_weights(project_root: Path, path: Path) -> Chapter5FeatureWeights:
    payload = _read_yaml_mapping(_resolve_from_root(project_root, path))
    section = _section(payload, "feature_weights")
    criteria: dict[str, Chapter5FeatureCriterionWeights] = {}
    for criterion_name, raw_config in section.items():
        if not isinstance(raw_config, dict):
            msg = f"Настройка критерия {criterion_name} должна быть словарем."
            raise ValueError(msg)
        criteria[criterion_name] = Chapter5FeatureCriterionWeights(
            observed_weight=_float(raw_config.get("observed_weight", 0.5), "observed_weight"),
            features=_float_dict(_section(raw_config, "features")),
        )
    return Chapter5FeatureWeights(criteria=criteria)


def _load_uncertainty(project_root: Path, path: Path) -> Chapter5UncertaintyConfig:
    payload = _read_yaml_mapping(_resolve_from_root(project_root, path))
    section = _section(payload, "uncertainty")
    return Chapter5UncertaintyConfig(
        delta=_float(section.get("delta", 0.15), "delta"),
        mean_stability=_float(section.get("mean_stability", 0.84885), "mean_stability"),
        min_stability=_float(section.get("min_stability", 0.808418), "min_stability"),
        weights=_float_dict(_section(section, "uncertainty_weights")),
    )


def _load_decision_thresholds(project_root: Path, path: Path) -> Chapter5DecisionThresholds:
    payload = _read_yaml_mapping(_resolve_from_root(project_root, path))
    section = _section(payload, "decision_thresholds")
    labels = _section(section, "class_labels")
    return Chapter5DecisionThresholds(
        low_max=_float(section.get("low_max", 0.45), "low_max"),
        high_min=_float(section.get("high_min", 0.70), "high_min"),
        class_labels={str(key): str(value) for key, value in labels.items()},
    )


def _load_prior_feature_dictionary(project_root: Path, path: Path) -> Chapter5PriorFeatureDictionary:
    payload = _read_yaml_mapping(_resolve_from_root(project_root, path))
    return Chapter5PriorFeatureDictionary(features=_section(payload, "features"))


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Прочитать YAML-файл и убедиться, что корневой элемент является словарем."""

    if not path.exists():
        msg = f"Файл конфигурации главы 5 не найден: {path}"
        raise FileNotFoundError(msg)
    with path.open("r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj) or {}
    if not isinstance(payload, dict):
        msg = f"Корневой элемент YAML-файла должен быть словарем: {path}"
        raise ValueError(msg)
    return payload


def _section(payload: dict[str, Any], *names: str) -> dict[str, Any]:
    """Вернуть словарную секцию YAML по одному из допустимых имен."""

    for name in names:
        value = payload.get(name)
        if value is None:
            continue
        if not isinstance(value, dict):
            msg = f"Секция '{name}' должна быть словарем."
            raise ValueError(msg)
        return value
    joined = ", ".join(names)
    msg = f"В YAML-конфигурации отсутствует обязательная секция: {joined}."
    raise ValueError(msg)


def _path(value: Any, default: str | Path) -> Path:
    """Преобразовать значение YAML в путь."""

    return Path(default if value is None else str(value))


def _resolve_from_root(project_root: Path, path: Path) -> Path:
    """Вернуть абсолютный или относительный к корню проекта путь."""

    if path.is_absolute():
        return path
    return project_root / path


def _float(value: Any, name: str) -> float:
    """Преобразовать значение к float с русскоязычной ошибкой."""

    try:
        return float(value)
    except (TypeError, ValueError) as error:
        msg = f"Параметр {name} должен быть числом."
        raise ValueError(msg) from error


def _positive_int(value: Any, name: str) -> int:
    """Преобразовать значение к положительному int."""

    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        msg = f"Параметр {name} должен быть целым числом."
        raise ValueError(msg) from error
    if result < 1:
        msg = f"Параметр {name} должен быть положительным целым числом."
        raise ValueError(msg)
    return result


def _bool(value: Any) -> bool:
    """Преобразовать значение YAML к bool."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "да"}:
            return True
        if lowered in {"false", "0", "no", "нет"}:
            return False
    return bool(value)


def _float_dict(payload: dict[str, Any]) -> dict[str, float]:
    """Преобразовать словарь YAML-значений к словарю float."""

    return {str(key): _float(value, str(key)) for key, value in payload.items()}


def _require_keys(mapping: dict[str, Any], required_keys: tuple[str, ...], name: str) -> None:
    """Проверить наличие обязательных ключей в словаре."""

    missing = [key for key in required_keys if key not in mapping]
    if missing:
        joined = ", ".join(missing)
        msg = f"В конфигурации {name} отсутствуют обязательные ключи: {joined}."
        raise ValueError(msg)


def _validate_weight_sum(weights: dict[str, float], label: str, tolerance: float = 1e-6) -> None:
    """Проверить неотрицательность и сумму весов."""

    negative_keys = [key for key, value in weights.items() if value < 0]
    if negative_keys:
        joined = ", ".join(negative_keys)
        msg = f"{label}: веса не могут быть отрицательными: {joined}."
        raise ValueError(msg)
    total = sum(weights.values())
    if abs(total - 1.0) > tolerance:
        msg = f"{label} должна быть равна 1, текущее значение: {total:.6f}."
        raise ValueError(msg)
