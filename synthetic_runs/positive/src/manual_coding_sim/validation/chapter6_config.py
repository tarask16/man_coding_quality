"""Конфигурационный слой экспериментальной проверки главы 6.

Модуль загружает единый YAML-файл главы 6 и проверяет неизменяемые
методические параметры: ключи объединения, пороги классов качества и
параметры bootstrap-анализа. Фактические данные на этом этапе не читаются.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from manual_coding_sim.validation.paths import (
    DEFAULT_ACCEPTANCE_REPORT_JSON_PATH,
    DEFAULT_ACCEPTANCE_REPORT_MD_PATH,
    DEFAULT_BASELINE_COMPARISON_PATH,
    DEFAULT_BASELINE_PREDICTIONS_PATH,
    DEFAULT_BOOTSTRAP_CONFIDENCE_INTERVALS_PATH,
    DEFAULT_BOOTSTRAP_MODEL_DIFFERENCES_PATH,
    DEFAULT_CHAPTER5_ACCEPTANCE_REPORT_PATH,
    DEFAULT_CHAPTER5_PREDICTION_REPORT_PATH,
    DEFAULT_CHAPTER6_FIGURES_DIR,
    DEFAULT_CHAPTER6_REPORTS_DIR,
    DEFAULT_CLASSIFICATION_REPORT_JSON_PATH,
    DEFAULT_CLASSIFICATION_REPORT_MD_PATH,
    DEFAULT_CONFUSION_MATRIX_PATH,
    DEFAULT_ERROR_GROUP_ANALYSIS_PATH,
    DEFAULT_FACT_FEATURES_PATH,
    DEFAULT_FINAL_REPORT_JSON_PATH,
    DEFAULT_FINAL_REPORT_MD_PATH,
    DEFAULT_INPUT_VALIDATION_REPORT_JSON_PATH,
    DEFAULT_INPUT_VALIDATION_REPORT_MD_PATH,
    DEFAULT_INTEGRAL_PREDICTION_ERRORS_PATH,
    DEFAULT_INTEGRAL_QUALITY_CONSISTENCY_PATH,
    DEFAULT_INTEGRAL_QUALITY_REPORT_JSON_PATH,
    DEFAULT_INTEGRAL_QUALITY_REPORT_MD_PATH,
    DEFAULT_INTERVAL_COVERAGE_DETAILS_PATH,
    DEFAULT_INTERVAL_COVERAGE_REPORT_JSON_PATH,
    DEFAULT_INTERVAL_COVERAGE_REPORT_MD_PATH,
    DEFAULT_LATENT_QUALITY_COMPONENT_PATH,
    DEFAULT_NORMALIZED_PRIOR_FEATURES_PATH,
    DEFAULT_PARTIAL_CRITERIA_VALIDATION_PATH,
    DEFAULT_PIPELINE_RUN_REPORT_PATH,
    DEFAULT_PREDICTION_UNCERTAINTY_PATH,
    DEFAULT_Q_PRED_COMPONENTS_PATH,
    DEFAULT_Q_PRED_PATH,
    DEFAULT_QUALITY_TARGETS_PATH,
    DEFAULT_THETA_PRIOR_PATH,
    DEFAULT_TOP_PREDICTION_ERRORS_PATH,
    DEFAULT_VALIDATION_DATASET_PATH,
    DEFAULT_VALIDATION_METRICS_JSON_PATH,
    DEFAULT_VALIDATION_METRICS_MD_PATH,
    resolve_project_path,
)

EXPECTED_MERGE_KEYS = ("scenario_id", "protocol_id")
QUALITY_CLASS_NAMES = ("low", "medium", "high")
ALLOWED_MERGE_VALIDATIONS = {"one_to_one"}


class Chapter6ConfigError(ValueError):
    """Ошибка структуры или значений конфигурации главы 6."""


@dataclass(frozen=True)
class Chapter6InputPaths:
    """Пути к зафиксированным прогнозным и фактическим артефактам."""

    q_pred_path: Path = DEFAULT_Q_PRED_PATH
    q_pred_components_path: Path = DEFAULT_Q_PRED_COMPONENTS_PATH
    prediction_uncertainty_path: Path = DEFAULT_PREDICTION_UNCERTAINTY_PATH
    chapter5_prediction_report_path: Path = DEFAULT_CHAPTER5_PREDICTION_REPORT_PATH
    chapter5_acceptance_report_path: Path = DEFAULT_CHAPTER5_ACCEPTANCE_REPORT_PATH
    normalized_prior_features_path: Path = DEFAULT_NORMALIZED_PRIOR_FEATURES_PATH
    latent_quality_component_path: Path = DEFAULT_LATENT_QUALITY_COMPONENT_PATH
    theta_prior_path: Path = DEFAULT_THETA_PRIOR_PATH
    quality_targets_path: Path = DEFAULT_QUALITY_TARGETS_PATH
    fact_features_path: Path = DEFAULT_FACT_FEATURES_PATH


@dataclass(frozen=True)
class Chapter6OutputPaths:
    """Пути к планируемым отчетам и таблицам главы 6."""

    reports_dir: Path = DEFAULT_CHAPTER6_REPORTS_DIR
    figures_dir: Path = DEFAULT_CHAPTER6_FIGURES_DIR
    input_validation_report_json_path: Path = DEFAULT_INPUT_VALIDATION_REPORT_JSON_PATH
    input_validation_report_md_path: Path = DEFAULT_INPUT_VALIDATION_REPORT_MD_PATH
    validation_dataset_path: Path = DEFAULT_VALIDATION_DATASET_PATH
    integral_quality_consistency_path: Path = DEFAULT_INTEGRAL_QUALITY_CONSISTENCY_PATH
    integral_quality_report_json_path: Path = DEFAULT_INTEGRAL_QUALITY_REPORT_JSON_PATH
    integral_quality_report_md_path: Path = DEFAULT_INTEGRAL_QUALITY_REPORT_MD_PATH
    validation_metrics_json_path: Path = DEFAULT_VALIDATION_METRICS_JSON_PATH
    validation_metrics_md_path: Path = DEFAULT_VALIDATION_METRICS_MD_PATH
    integral_prediction_errors_path: Path = DEFAULT_INTEGRAL_PREDICTION_ERRORS_PATH
    partial_criteria_validation_path: Path = DEFAULT_PARTIAL_CRITERIA_VALIDATION_PATH
    classification_report_json_path: Path = DEFAULT_CLASSIFICATION_REPORT_JSON_PATH
    classification_report_md_path: Path = DEFAULT_CLASSIFICATION_REPORT_MD_PATH
    confusion_matrix_path: Path = DEFAULT_CONFUSION_MATRIX_PATH
    interval_coverage_details_path: Path = DEFAULT_INTERVAL_COVERAGE_DETAILS_PATH
    interval_coverage_report_json_path: Path = DEFAULT_INTERVAL_COVERAGE_REPORT_JSON_PATH
    interval_coverage_report_md_path: Path = DEFAULT_INTERVAL_COVERAGE_REPORT_MD_PATH
    baseline_predictions_path: Path = DEFAULT_BASELINE_PREDICTIONS_PATH
    baseline_comparison_path: Path = DEFAULT_BASELINE_COMPARISON_PATH
    bootstrap_confidence_intervals_path: Path = (
        DEFAULT_BOOTSTRAP_CONFIDENCE_INTERVALS_PATH
    )
    bootstrap_model_differences_path: Path = DEFAULT_BOOTSTRAP_MODEL_DIFFERENCES_PATH
    top_prediction_errors_path: Path = DEFAULT_TOP_PREDICTION_ERRORS_PATH
    error_group_analysis_path: Path = DEFAULT_ERROR_GROUP_ANALYSIS_PATH
    final_report_json_path: Path = DEFAULT_FINAL_REPORT_JSON_PATH
    final_report_md_path: Path = DEFAULT_FINAL_REPORT_MD_PATH
    pipeline_run_report_path: Path = DEFAULT_PIPELINE_RUN_REPORT_PATH
    acceptance_report_json_path: Path = DEFAULT_ACCEPTANCE_REPORT_JSON_PATH
    acceptance_report_md_path: Path = DEFAULT_ACCEPTANCE_REPORT_MD_PATH


@dataclass(frozen=True)
class Chapter6MergeConfig:
    """Правила объединения таблиц проверочного контура."""

    key_columns: tuple[str, ...] = EXPECTED_MERGE_KEYS
    validation: str = "one_to_one"
    expected_row_count: int = 150

    def validate(self) -> None:
        """Проверить ключи, режим объединения и ожидаемое число строк."""

        if self.key_columns != EXPECTED_MERGE_KEYS:
            msg = (
                "Ключи объединения главы 6 должны быть заданы в порядке "
                f"{EXPECTED_MERGE_KEYS}, получено {self.key_columns}."
            )
            raise Chapter6ConfigError(msg)
        if len(set(self.key_columns)) != len(self.key_columns):
            raise Chapter6ConfigError("Ключи объединения не должны повторяться.")
        if self.validation not in ALLOWED_MERGE_VALIDATIONS:
            msg = "Для главы 6 разрешен только режим объединения one_to_one."
            raise Chapter6ConfigError(msg)
        if self.expected_row_count <= 0:
            raise Chapter6ConfigError("Ожидаемое число строк должно быть положительным.")


@dataclass(frozen=True)
class Chapter6DecisionThresholds:
    """Пороговые значения трех классов фактического и прогнозного качества."""

    low_max: float = 0.45
    high_min: float = 0.70
    class_labels: dict[str, str] = field(
        default_factory=lambda: {
            "low": "low",
            "medium": "medium",
            "high": "high",
        }
    )

    def validate(self) -> None:
        """Проверить порядок порогов и полноту меток классов."""

        if not 0.0 <= self.low_max < self.high_min <= 1.0:
            msg = "Пороги должны удовлетворять условию 0 <= low_max < high_min <= 1."
            raise Chapter6ConfigError(msg)
        if set(self.class_labels) != set(QUALITY_CLASS_NAMES):
            msg = f"Метки классов должны быть заданы для {QUALITY_CLASS_NAMES}."
            raise Chapter6ConfigError(msg)
        labels = tuple(self.class_labels[name] for name in QUALITY_CLASS_NAMES)
        if any(not label.strip() for label in labels):
            raise Chapter6ConfigError("Метки классов не должны быть пустыми.")
        if len(set(labels)) != len(labels):
            raise Chapter6ConfigError("Метки классов должны быть уникальными.")


@dataclass(frozen=True)
class Chapter6BootstrapConfig:
    """Параметры воспроизводимого bootstrap-анализа по сценариям."""

    resamples: int = 1000
    confidence_level: float = 0.95
    random_seed: int = 42
    sampling_unit: str = "scenario_id"

    def validate(self, merge_keys: tuple[str, ...]) -> None:
        """Проверить число повторов, доверительный уровень и единицу выборки."""

        if type(self.resamples) is not int or self.resamples < 100:
            raise Chapter6ConfigError(
                "Число bootstrap-повторов должно быть целым и не меньше 100."
            )
        if not 0.0 < self.confidence_level < 1.0:
            raise Chapter6ConfigError("Доверительный уровень должен находиться между 0 и 1.")
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise Chapter6ConfigError("random_seed должен быть неотрицательным целым числом.")
        if self.sampling_unit not in merge_keys:
            msg = "Единица bootstrap-выборки должна входить в ключи объединения."
            raise Chapter6ConfigError(msg)


@dataclass(frozen=True)
class Chapter6RunnerConfig:
    """Служебные параметры CLI и сохранения артефактов."""

    overwrite: bool = True


@dataclass(frozen=True)
class Chapter6ValidationConfig:
    """Единая проверенная конфигурация программного контура главы 6."""

    inputs: Chapter6InputPaths = field(default_factory=Chapter6InputPaths)
    outputs: Chapter6OutputPaths = field(default_factory=Chapter6OutputPaths)
    merge: Chapter6MergeConfig = field(default_factory=Chapter6MergeConfig)
    decision_thresholds: Chapter6DecisionThresholds = field(
        default_factory=Chapter6DecisionThresholds
    )
    bootstrap: Chapter6BootstrapConfig = field(default_factory=Chapter6BootstrapConfig)
    runner: Chapter6RunnerConfig = field(default_factory=Chapter6RunnerConfig)

    def validate(self) -> None:
        """Проверить все методические параметры главы 6."""

        self.merge.validate()
        self.decision_thresholds.validate()
        self.bootstrap.validate(self.merge.key_columns)

    def to_dict(self) -> dict[str, Any]:
        """Преобразовать конфигурацию в структуру, пригодную для JSON-вывода."""

        return _paths_to_strings(asdict(self))

    def resolved_paths(self, project_root: Path) -> dict[str, dict[str, str]]:
        """Вернуть абсолютные пути входных и выходных артефактов."""

        return {
            "input": {
                name: str(resolve_project_path(project_root, value))
                for name, value in vars(self.inputs).items()
            },
            "output": {
                name: str(resolve_project_path(project_root, value))
                for name, value in vars(self.outputs).items()
            },
        }


class Chapter6ConfigLoader:
    """Загрузчик YAML-конфигурации главы 6."""

    def __init__(self, project_root: Path | str = Path(".")) -> None:
        """Сохранить корень проекта для разрешения пути к YAML-файлу."""

        self.project_root = Path(project_root)

    def load(self, config_path: Path | str) -> Chapter6ValidationConfig:
        """Загрузить, преобразовать и проверить конфигурацию главы 6."""

        path = resolve_project_path(self.project_root, Path(config_path))
        if not path.exists():
            raise FileNotFoundError(f"Файл конфигурации главы 6 не найден: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise Chapter6ConfigError(
                f"Не удалось разобрать YAML-конфигурацию главы 6: {path}"
            ) from error
        if not isinstance(raw, Mapping):
            raise Chapter6ConfigError("Корень YAML-конфигурации должен быть отображением.")

        config = Chapter6ValidationConfig(
            inputs=_build_input_paths(_section(raw, "input")),
            outputs=_build_output_paths(_section(raw, "output")),
            merge=_build_merge_config(_section(raw, "merge")),
            decision_thresholds=_build_thresholds(
                _section(raw, "decision_thresholds")
            ),
            bootstrap=_build_bootstrap(_section(raw, "bootstrap")),
            runner=_build_runner(_section(raw, "runner")),
        )
        config.validate()
        return config


def load_chapter6_validation_config(
    config_path: Path | str = Path("configs/chapter6.yaml"),
    project_root: Path | str = Path("."),
) -> Chapter6ValidationConfig:
    """Загрузить конфигурацию главы 6 из стандартного или указанного файла."""

    return Chapter6ConfigLoader(project_root=project_root).load(config_path)


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Получить секцию YAML и проверить ее тип."""

    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise Chapter6ConfigError(f"Секция {name} должна быть отображением.")
    return value


def _build_input_paths(raw: Mapping[str, Any]) -> Chapter6InputPaths:
    """Построить пути входных артефактов из YAML-секции input."""

    defaults = Chapter6InputPaths()
    return Chapter6InputPaths(
        **{
            name: Path(raw.get(_yaml_name(name), default))
            for name, default in vars(defaults).items()
        }
    )


def _build_output_paths(raw: Mapping[str, Any]) -> Chapter6OutputPaths:
    """Построить пути выходных артефактов из YAML-секции output."""

    defaults = Chapter6OutputPaths()
    return Chapter6OutputPaths(
        **{
            name: Path(raw.get(_yaml_name(name), default))
            for name, default in vars(defaults).items()
        }
    )


def _build_merge_config(raw: Mapping[str, Any]) -> Chapter6MergeConfig:
    """Построить параметры проверяемого объединения таблиц."""

    key_columns = raw.get("key_columns", list(EXPECTED_MERGE_KEYS))
    if not isinstance(key_columns, (list, tuple)):
        raise Chapter6ConfigError("merge.key_columns должен быть списком строк.")
    if any(not isinstance(item, str) for item in key_columns):
        raise Chapter6ConfigError("Все ключи объединения должны быть строками.")
    return Chapter6MergeConfig(
        key_columns=tuple(key_columns),
        validation=str(raw.get("validation", "one_to_one")),
        expected_row_count=_as_int(raw.get("expected_row_count", 150), "expected_row_count"),
    )


def _build_thresholds(raw: Mapping[str, Any]) -> Chapter6DecisionThresholds:
    """Построить пороги и метки классов качества."""

    class_labels_raw = raw.get(
        "class_labels", {"low": "low", "medium": "medium", "high": "high"}
    )
    if not isinstance(class_labels_raw, Mapping):
        raise Chapter6ConfigError("decision_thresholds.class_labels должен быть отображением.")
    return Chapter6DecisionThresholds(
        low_max=_as_float(raw.get("low_max", 0.45), "low_max"),
        high_min=_as_float(raw.get("high_min", 0.70), "high_min"),
        class_labels={str(key): str(value) for key, value in class_labels_raw.items()},
    )


def _build_bootstrap(raw: Mapping[str, Any]) -> Chapter6BootstrapConfig:
    """Построить параметры bootstrap-анализа."""

    return Chapter6BootstrapConfig(
        resamples=_as_int(raw.get("resamples", 1000), "resamples"),
        confidence_level=_as_float(
            raw.get("confidence_level", 0.95), "confidence_level"
        ),
        random_seed=_as_int(raw.get("random_seed", 42), "random_seed"),
        sampling_unit=str(raw.get("sampling_unit", "scenario_id")),
    )


def _build_runner(raw: Mapping[str, Any]) -> Chapter6RunnerConfig:
    """Построить служебные параметры запуска."""

    overwrite = raw.get("overwrite", True)
    if not isinstance(overwrite, bool):
        raise Chapter6ConfigError("runner.overwrite должен быть логическим значением.")
    return Chapter6RunnerConfig(overwrite=overwrite)


def _yaml_name(field_name: str) -> str:
    """Преобразовать имя поля dataclass в имя параметра YAML."""

    return field_name.removesuffix("_path")


def _as_int(value: Any, name: str) -> int:
    """Преобразовать значение в целое без неявного принятия логического типа."""

    if isinstance(value, bool):
        raise Chapter6ConfigError(f"Параметр {name} должен быть целым числом.")
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise Chapter6ConfigError(f"Параметр {name} должен быть целым числом.") from error
    if isinstance(value, float) and not value.is_integer():
        raise Chapter6ConfigError(f"Параметр {name} должен быть целым числом.")
    return converted


def _as_float(value: Any, name: str) -> float:
    """Преобразовать значение в число с плавающей точкой."""

    if isinstance(value, bool):
        raise Chapter6ConfigError(f"Параметр {name} должен быть числом.")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise Chapter6ConfigError(f"Параметр {name} должен быть числом.") from error


def _paths_to_strings(value: Any) -> Any:
    """Рекурсивно преобразовать объекты Path в строки."""

    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {key: _paths_to_strings(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_paths_to_strings(item) for item in value]
    if isinstance(value, list):
        return [_paths_to_strings(item) for item in value]
    return value
