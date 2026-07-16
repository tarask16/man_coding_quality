"""Загрузка и проверка конфигурации изолированного расширения."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExtensionMetadata:
    """Метаданные самостоятельного пакета расширения."""

    name: str
    package_name: str
    version: str
    random_seed: int

    def validate(self) -> None:
        """Проверить обязательные метаданные расширения."""
        if not self.name:
            raise ValueError("Не задано имя расширения.")
        if self.package_name != "manual_coding_sim_decoding":
            raise ValueError(
                "Имя пакета расширения должно быть manual_coding_sim_decoding."
            )
        if not self.version:
            raise ValueError("Не задана версия расширения.")
        if self.random_seed < 0:
            raise ValueError(
                "Зерно генератора расширения не должно быть отрицательным."
            )


@dataclass(frozen=True)
class BaseContractConfig:
    """Контракт подключения к неизменяемому базовому пакету."""

    package_name: str
    baseline_manifest: str
    required_symbols: dict[str, tuple[str, ...]]

    def validate(self) -> None:
        """Проверить структуру контракта базового пакета."""
        if self.package_name != "manual_coding_sim":
            raise ValueError("Базовым пакетом должен быть manual_coding_sim.")
        if not self.baseline_manifest:
            raise ValueError("Не указан baseline-манифест SHA-256.")
        if not self.required_symbols:
            raise ValueError("Не задан перечень обязательных публичных символов.")
        for module_name, symbols in self.required_symbols.items():
            if not module_name:
                raise ValueError("Имя модуля в контракте не должно быть пустым.")
            if not symbols:
                raise ValueError(
                    f"Для модуля {module_name} не заданы обязательные символы."
                )


@dataclass(frozen=True)
class ExtensionPathConfig:
    """Относительные пути изолированного расширения."""

    reports_dir: str
    data_dir: str
    manifests_dir: str

    def validate(self) -> None:
        """Проверить, что артефакты направлены в папку расширения."""
        values = (self.reports_dir, self.data_dir, self.manifests_dir)
        required_prefix = "extensions/decoding_simulation/"
        for value in values:
            normalized = value.replace("\\", "/")
            if not normalized.startswith(required_prefix):
                raise ValueError(
                    "Пути расширения должны находиться внутри "
                    "extensions/decoding_simulation/."
                )


@dataclass(frozen=True)
class MaterialEncodingConfig:
    """Правила формирования материального кодированного сообщения C."""

    encoded_message_prefix: str = "C"
    substitution_prefix: str = "ERR_SUB"
    reference_prefix: str = "ERR_REF"
    service_marker_prefix: str = "ERR_SERVICE"
    unknown_error_prefix: str = "ERR_UNKNOWN"
    position_shift: int = 1

    def validate(self) -> None:
        """Проверить идентификаторы и параметры материального кодирования."""
        prefixes = {
            "encoded_message_prefix": self.encoded_message_prefix,
            "substitution_prefix": self.substitution_prefix,
            "reference_prefix": self.reference_prefix,
            "service_marker_prefix": self.service_marker_prefix,
            "unknown_error_prefix": self.unknown_error_prefix,
        }
        for field_name, value in prefixes.items():
            if not value or not value.strip():
                raise ValueError(f"Поле {field_name} не должно быть пустым.")
            if any(character.isspace() for character in value):
                raise ValueError(f"Поле {field_name} не должно содержать пробелы.")
        if self.position_shift <= 0:
            raise ValueError("position_shift должен быть положительным.")


@dataclass(frozen=True)
class DecodingOperationRule:
    """Правило соответствия прямой и обратной абстрактных операций."""

    encoding_operation_type: str
    decoding_operation_type: str
    nominal_time: float
    complexity: float
    reference_required: bool = False

    def validate(self) -> None:
        """Проверить параметры одной обратной операции."""
        if not self.encoding_operation_type:
            raise ValueError("Не задан тип прямой операции кодирования.")
        if not self.decoding_operation_type:
            raise ValueError("Не задан тип обратной операции декодирования.")
        if self.nominal_time <= 0:
            raise ValueError("Нормативное время декодирования должно быть положительным.")
        if not 0.0 <= self.complexity <= 1.0:
            raise ValueError(
                "Сложность обратной операции должна находиться в диапазоне [0; 1]."
            )


@dataclass(frozen=True)
class FormalDecodingConfig:
    """Конфигурация нормативной обратной процедуры ``D_h(C)``."""

    decoding_procedure_id: str = "D_001"
    token_prefix: str = "TOK"
    token_position_width: int = 4
    fail_on_unknown_token: bool = False
    service_values: tuple[str, ...] = ("SEP", "CTRL", "END")
    operation_rules: tuple[DecodingOperationRule, ...] = field(
        default_factory=lambda: (
            DecodingOperationRule(
                encoding_operation_type="abstract_substitution",
                decoding_operation_type="abstract_inverse_substitution",
                nominal_time=1.10,
                complexity=0.40,
            ),
            DecodingOperationRule(
                encoding_operation_type="abstract_numeric_mapping",
                decoding_operation_type="abstract_inverse_numeric_mapping",
                nominal_time=1.30,
                complexity=0.50,
                reference_required=True,
            ),
            DecodingOperationRule(
                encoding_operation_type="abstract_service_marking",
                decoding_operation_type="abstract_inverse_service_interpretation",
                nominal_time=1.55,
                complexity=0.60,
                reference_required=True,
            ),
            DecodingOperationRule(
                encoding_operation_type="abstract_copying",
                decoding_operation_type="abstract_inverse_copying",
                nominal_time=0.90,
                complexity=0.30,
            ),
        )
    )
    unresolved_rule: DecodingOperationRule = field(
        default_factory=lambda: DecodingOperationRule(
            encoding_operation_type="__unresolved__",
            decoding_operation_type="abstract_unresolved_inverse_operation",
            nominal_time=1.80,
            complexity=0.80,
            reference_required=True,
        )
    )

    def validate(self) -> None:
        """Проверить правила и синтаксические параметры D_h."""
        if not self.decoding_procedure_id:
            raise ValueError("Не задан decoding_procedure_id.")
        if not self.token_prefix or any(
            character.isspace() for character in self.token_prefix
        ):
            raise ValueError("token_prefix должен быть непустым и без пробелов.")
        if self.token_position_width <= 0:
            raise ValueError("token_position_width должен быть положительным.")
        if not self.service_values:
            raise ValueError("Набор служебных значений не должен быть пустым.")
        if not self.operation_rules:
            raise ValueError("Набор обратных операций не должен быть пустым.")

        seen_operations: set[str] = set()
        for rule in self.operation_rules:
            rule.validate()
            if rule.encoding_operation_type in seen_operations:
                raise ValueError(
                    "Для прямой операции задано несколько обратных правил: "
                    f"{rule.encoding_operation_type}."
                )
            seen_operations.add(rule.encoding_operation_type)
        self.unresolved_rule.validate()


@dataclass(frozen=True)
class DecodingOperatorProfile:
    """Априорный профиль отдельного декодирующего оператора O_d."""

    operator_id: str = "OD_001"
    preparation_level: float = 0.75
    experience_level: float = 0.65
    base_attention: float = 0.85
    fatigue_rate: float = 0.015
    reference_skill: float = 0.70
    interpretation_skill: float = 0.72
    work_rate: float = 1.00
    min_attention: float = 0.25

    def validate(self) -> None:
        """Проверить параметры профиля декодирующего оператора."""
        if not self.operator_id:
            raise ValueError("Не задан operator_id декодирующего оператора.")
        for field_name, value in (
            ("preparation_level", self.preparation_level),
            ("experience_level", self.experience_level),
            ("base_attention", self.base_attention),
            ("fatigue_rate", self.fatigue_rate),
            ("reference_skill", self.reference_skill),
            ("interpretation_skill", self.interpretation_skill),
            ("min_attention", self.min_attention),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Поле decoding_operator.profile.{field_name} "
                    "должно находиться в диапазоне [0; 1]."
                )
        if self.work_rate <= 0:
            raise ValueError("Скорость работы декодирующего оператора должна быть положительной.")
        if self.min_attention > self.base_attention:
            raise ValueError(
                "Минимальное внимание декодирующего оператора не должно "
                "превышать базовое внимание."
            )


@dataclass(frozen=True)
class DecodingOperatorConfig:
    """Конфигурация детерминированной модели декодирующего оператора."""

    profile: DecodingOperatorProfile = field(default_factory=DecodingOperatorProfile)
    reference_penalty: float = 0.10
    complexity_weight: float = 0.40
    fatigue_time_weight: float = 0.45
    unresolved_token_penalty: float = 0.35

    def validate(self) -> None:
        """Проверить параметры модели O_d без генерации ошибок."""
        self.profile.validate()
        for field_name, value in (
            ("reference_penalty", self.reference_penalty),
            ("complexity_weight", self.complexity_weight),
            ("fatigue_time_weight", self.fatigue_time_weight),
            ("unresolved_token_penalty", self.unresolved_token_penalty),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Поле decoding_operator.{field_name} "
                    "должно находиться в диапазоне [0; 1]."
                )


@dataclass(frozen=True)
class DecodingConditionProfile:
    """Априорный профиль условий выполнения декодирования U_d."""

    condition_id: str = "UD_001"
    time_limit_seconds: float | None = None
    noise_level: float = 0.20
    instruction_access: float = 0.90
    workload_level: float = 0.30
    interruption_rate: float = 0.10
    lighting_quality: float = 0.90

    def validate(self) -> None:
        """Проверить профиль условий декодирования."""
        if not self.condition_id:
            raise ValueError("Не задан condition_id условий декодирования.")
        if self.time_limit_seconds is not None and self.time_limit_seconds <= 0:
            raise ValueError("Лимит времени декодирования должен быть положительным.")
        for field_name, value in (
            ("noise_level", self.noise_level),
            ("instruction_access", self.instruction_access),
            ("workload_level", self.workload_level),
            ("interruption_rate", self.interruption_rate),
            ("lighting_quality", self.lighting_quality),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Поле decoding_conditions.profile.{field_name} "
                    "должно находиться в диапазоне [0; 1]."
                )


@dataclass(frozen=True)
class DecodingConditionConfig:
    """Конфигурация детерминированной модели условий декодирования."""

    profile: DecodingConditionProfile = field(default_factory=DecodingConditionProfile)
    noise_weight: float = 0.25
    workload_weight: float = 0.30
    instruction_weight: float = 0.20
    interruption_weight: float = 0.15
    lighting_weight: float = 0.10
    time_pressure_weight: float = 0.30
    time_expansion_weight: float = 0.45
    environment_attention_weight: float = 0.35
    time_attention_weight: float = 0.30
    ambiguity_time_weight: float = 0.35
    ambiguity_attention_weight: float = 0.35

    def validate(self) -> None:
        """Проверить параметры модели U_d без генерации ошибок."""
        self.profile.validate()
        for field_name, value in (
            ("noise_weight", self.noise_weight),
            ("workload_weight", self.workload_weight),
            ("instruction_weight", self.instruction_weight),
            ("interruption_weight", self.interruption_weight),
            ("lighting_weight", self.lighting_weight),
            ("time_pressure_weight", self.time_pressure_weight),
            ("time_expansion_weight", self.time_expansion_weight),
            ("environment_attention_weight", self.environment_attention_weight),
            ("time_attention_weight", self.time_attention_weight),
            ("ambiguity_time_weight", self.ambiguity_time_weight),
            ("ambiguity_attention_weight", self.ambiguity_attention_weight),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Поле decoding_conditions.{field_name} "
                    "должно находиться в диапазоне [0; 1]."
                )


@dataclass(frozen=True)
class DecodingExtensionConfig:
    """Полная конфигурация изолированного расширения декодирования."""

    extension: ExtensionMetadata
    base_contract: BaseContractConfig
    paths: ExtensionPathConfig
    material_encoding: MaterialEncodingConfig = field(
        default_factory=MaterialEncodingConfig
    )
    formal_decoding: FormalDecodingConfig = field(
        default_factory=FormalDecodingConfig
    )
    decoding_operator: DecodingOperatorConfig = field(
        default_factory=DecodingOperatorConfig
    )
    decoding_conditions: DecodingConditionConfig = field(
        default_factory=DecodingConditionConfig
    )

    def validate(self) -> None:
        """Проверить все разделы конфигурации."""
        self.extension.validate()
        self.base_contract.validate()
        self.paths.validate()
        self.material_encoding.validate()
        self.formal_decoding.validate()
        self.decoding_operator.validate()
        self.decoding_conditions.validate()


def _require_mapping(value: Any, section_name: str) -> dict[str, Any]:
    """Вернуть YAML-раздел как словарь или завершить проверку ошибкой."""
    if not isinstance(value, dict):
        raise ValueError(f"Раздел {section_name} должен быть YAML-словарем.")
    return value


def _parse_required_symbols(value: Any) -> dict[str, tuple[str, ...]]:
    """Преобразовать описание обязательных символов в неизменяемые кортежи."""
    source = _require_mapping(value, "base_contract.required_symbols")
    parsed: dict[str, tuple[str, ...]] = {}
    for module_name, symbols in source.items():
        if not isinstance(module_name, str):
            raise ValueError("Имена модулей контракта должны быть строками.")
        if not isinstance(symbols, list) or not all(
            isinstance(symbol, str) and symbol for symbol in symbols
        ):
            raise ValueError(
                f"Символы модуля {module_name} должны быть непустым списком строк."
            )
        parsed[module_name] = tuple(symbols)
    return parsed


def _parse_operation_rule(raw: Any, section_name: str) -> DecodingOperationRule:
    """Преобразовать YAML-словарь в правило обратной операции."""
    source = _require_mapping(raw, section_name)
    return DecodingOperationRule(
        encoding_operation_type=str(source.get("encoding_operation_type", "")),
        decoding_operation_type=str(source.get("decoding_operation_type", "")),
        nominal_time=_require_positive_number(
            source.get("nominal_time"),
            f"{section_name}.nominal_time",
        ),
        complexity=_require_unit_number(
            source.get("complexity"),
            f"{section_name}.complexity",
        ),
        reference_required=bool(source.get("reference_required", False)),
    )


def _parse_formal_decoding(raw: dict[str, Any]) -> FormalDecodingConfig:
    """Загрузить раздел формальной обратной процедуры."""
    default_config = FormalDecodingConfig()
    rules_raw = raw.get("operation_rules")
    if rules_raw is None:
        rules = default_config.operation_rules
    else:
        if not isinstance(rules_raw, list):
            raise ValueError("formal_decoding.operation_rules должен быть списком.")
        rules = tuple(
            _parse_operation_rule(item, f"formal_decoding.operation_rules[{index}]")
            for index, item in enumerate(rules_raw)
        )

    unresolved_raw = raw.get("unresolved_rule")
    unresolved_rule = (
        default_config.unresolved_rule
        if unresolved_raw is None
        else _parse_operation_rule(
            unresolved_raw,
            "formal_decoding.unresolved_rule",
        )
    )
    service_values_raw = raw.get("service_values", list(default_config.service_values))
    if not isinstance(service_values_raw, list) or not all(
        isinstance(value, str) and value for value in service_values_raw
    ):
        raise ValueError("formal_decoding.service_values должен быть списком строк.")

    return FormalDecodingConfig(
        decoding_procedure_id=str(
            raw.get("decoding_procedure_id", default_config.decoding_procedure_id)
        ),
        token_prefix=str(raw.get("token_prefix", default_config.token_prefix)),
        token_position_width=_require_positive_int(
            raw.get("token_position_width", default_config.token_position_width),
            "formal_decoding.token_position_width",
        ),
        fail_on_unknown_token=bool(
            raw.get("fail_on_unknown_token", default_config.fail_on_unknown_token)
        ),
        service_values=tuple(service_values_raw),
        operation_rules=rules,
        unresolved_rule=unresolved_rule,
    )


def _parse_decoding_operator(raw: dict[str, Any]) -> DecodingOperatorConfig:
    """Загрузить отдельный профиль и параметры декодирующего оператора."""
    default = DecodingOperatorConfig()
    profile_raw = _require_mapping(raw.get("profile", {}), "decoding_operator.profile")
    default_profile = default.profile
    profile = DecodingOperatorProfile(
        operator_id=str(profile_raw.get("operator_id", default_profile.operator_id)),
        preparation_level=_require_unit_number(
            profile_raw.get("preparation_level", default_profile.preparation_level),
            "decoding_operator.profile.preparation_level",
        ),
        experience_level=_require_unit_number(
            profile_raw.get("experience_level", default_profile.experience_level),
            "decoding_operator.profile.experience_level",
        ),
        base_attention=_require_unit_number(
            profile_raw.get("base_attention", default_profile.base_attention),
            "decoding_operator.profile.base_attention",
        ),
        fatigue_rate=_require_unit_number(
            profile_raw.get("fatigue_rate", default_profile.fatigue_rate),
            "decoding_operator.profile.fatigue_rate",
        ),
        reference_skill=_require_unit_number(
            profile_raw.get("reference_skill", default_profile.reference_skill),
            "decoding_operator.profile.reference_skill",
        ),
        interpretation_skill=_require_unit_number(
            profile_raw.get("interpretation_skill", default_profile.interpretation_skill),
            "decoding_operator.profile.interpretation_skill",
        ),
        work_rate=_require_positive_number(
            profile_raw.get("work_rate", default_profile.work_rate),
            "decoding_operator.profile.work_rate",
        ),
        min_attention=_require_unit_number(
            profile_raw.get("min_attention", default_profile.min_attention),
            "decoding_operator.profile.min_attention",
        ),
    )
    return DecodingOperatorConfig(
        profile=profile,
        reference_penalty=_require_unit_number(
            raw.get("reference_penalty", default.reference_penalty),
            "decoding_operator.reference_penalty",
        ),
        complexity_weight=_require_unit_number(
            raw.get("complexity_weight", default.complexity_weight),
            "decoding_operator.complexity_weight",
        ),
        fatigue_time_weight=_require_unit_number(
            raw.get("fatigue_time_weight", default.fatigue_time_weight),
            "decoding_operator.fatigue_time_weight",
        ),
        unresolved_token_penalty=_require_unit_number(
            raw.get("unresolved_token_penalty", default.unresolved_token_penalty),
            "decoding_operator.unresolved_token_penalty",
        ),
    )


def _parse_optional_positive_number(value: Any, field_name: str) -> float | None:
    """Загрузить необязательное положительное число."""
    if value is None:
        return None
    return _require_positive_number(value, field_name)


def _parse_decoding_conditions(raw: dict[str, Any]) -> DecodingConditionConfig:
    """Загрузить отдельный профиль и параметры условий декодирования."""
    default = DecodingConditionConfig()
    profile_raw = _require_mapping(raw.get("profile", {}), "decoding_conditions.profile")
    default_profile = default.profile
    profile = DecodingConditionProfile(
        condition_id=str(profile_raw.get("condition_id", default_profile.condition_id)),
        time_limit_seconds=_parse_optional_positive_number(
            profile_raw.get("time_limit_seconds", default_profile.time_limit_seconds),
            "decoding_conditions.profile.time_limit_seconds",
        ),
        noise_level=_require_unit_number(
            profile_raw.get("noise_level", default_profile.noise_level),
            "decoding_conditions.profile.noise_level",
        ),
        instruction_access=_require_unit_number(
            profile_raw.get("instruction_access", default_profile.instruction_access),
            "decoding_conditions.profile.instruction_access",
        ),
        workload_level=_require_unit_number(
            profile_raw.get("workload_level", default_profile.workload_level),
            "decoding_conditions.profile.workload_level",
        ),
        interruption_rate=_require_unit_number(
            profile_raw.get("interruption_rate", default_profile.interruption_rate),
            "decoding_conditions.profile.interruption_rate",
        ),
        lighting_quality=_require_unit_number(
            profile_raw.get("lighting_quality", default_profile.lighting_quality),
            "decoding_conditions.profile.lighting_quality",
        ),
    )
    values: dict[str, float] = {}
    for name in (
        "noise_weight",
        "workload_weight",
        "instruction_weight",
        "interruption_weight",
        "lighting_weight",
        "time_pressure_weight",
        "time_expansion_weight",
        "environment_attention_weight",
        "time_attention_weight",
        "ambiguity_time_weight",
        "ambiguity_attention_weight",
    ):
        values[name] = _require_unit_number(
            raw.get(name, getattr(default, name)),
            f"decoding_conditions.{name}",
        )
    return DecodingConditionConfig(profile=profile, **values)


def load_decoding_extension_config(
    config_path: str | Path,
) -> DecodingExtensionConfig:
    """Загрузить YAML-конфигурацию изолированного расширения."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл конфигурации расширения не найден: {path}")

    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    root = _require_mapping(raw, "корневой")
    extension_raw = _require_mapping(root.get("extension"), "extension")
    contract_raw = _require_mapping(root.get("base_contract"), "base_contract")
    paths_raw = _require_mapping(root.get("paths"), "paths")
    material_raw = _require_mapping(root.get("material_encoding", {}), "material_encoding")
    formal_raw = _require_mapping(root.get("formal_decoding", {}), "formal_decoding")
    operator_raw = _require_mapping(root.get("decoding_operator", {}), "decoding_operator")
    conditions_raw = _require_mapping(
        root.get("decoding_conditions", {}),
        "decoding_conditions",
    )

    config = DecodingExtensionConfig(
        extension=ExtensionMetadata(
            name=str(extension_raw.get("name", "")),
            package_name=str(extension_raw.get("package_name", "")),
            version=str(extension_raw.get("version", "")),
            random_seed=_require_non_negative_int(
                extension_raw.get("random_seed"),
                "extension.random_seed",
            ),
        ),
        base_contract=BaseContractConfig(
            package_name=str(contract_raw.get("package_name", "")),
            baseline_manifest=str(contract_raw.get("baseline_manifest", "")),
            required_symbols=_parse_required_symbols(
                contract_raw.get("required_symbols")
            ),
        ),
        paths=ExtensionPathConfig(
            reports_dir=str(paths_raw.get("reports_dir", "")),
            data_dir=str(paths_raw.get("data_dir", "")),
            manifests_dir=str(paths_raw.get("manifests_dir", "")),
        ),
        material_encoding=MaterialEncodingConfig(
            encoded_message_prefix=str(
                material_raw.get("encoded_message_prefix", "C")
            ),
            substitution_prefix=str(
                material_raw.get("substitution_prefix", "ERR_SUB")
            ),
            reference_prefix=str(material_raw.get("reference_prefix", "ERR_REF")),
            service_marker_prefix=str(
                material_raw.get("service_marker_prefix", "ERR_SERVICE")
            ),
            unknown_error_prefix=str(
                material_raw.get("unknown_error_prefix", "ERR_UNKNOWN")
            ),
            position_shift=_require_positive_int(
                material_raw.get("position_shift", 1),
                "material_encoding.position_shift",
            ),
        ),
        formal_decoding=_parse_formal_decoding(formal_raw),
        decoding_operator=_parse_decoding_operator(operator_raw),
        decoding_conditions=_parse_decoding_conditions(conditions_raw),
    )
    config.validate()
    return config


def _require_non_negative_int(value: Any, field_name: str) -> int:
    """Проверить целочисленное неотрицательное поле конфигурации."""
    if not isinstance(value, int):
        raise ValueError(f"Поле {field_name} должно быть целым числом.")
    if value < 0:
        raise ValueError(f"Поле {field_name} не должно быть отрицательным.")
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    """Проверить целочисленное положительное поле конфигурации."""
    if not isinstance(value, int):
        raise ValueError(f"Поле {field_name} должно быть целым числом.")
    if value <= 0:
        raise ValueError(f"Поле {field_name} должно быть положительным.")
    return value


def _require_positive_number(value: Any, field_name: str) -> float:
    """Проверить положительное числовое поле."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Поле {field_name} должно быть числом.")
    result = float(value)
    if result <= 0:
        raise ValueError(f"Поле {field_name} должно быть положительным.")
    return result


def _require_unit_number(value: Any, field_name: str) -> float:
    """Проверить числовое поле диапазона [0; 1]."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Поле {field_name} должно быть числом.")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"Поле {field_name} должно находиться в диапазоне [0; 1].")
    return result
