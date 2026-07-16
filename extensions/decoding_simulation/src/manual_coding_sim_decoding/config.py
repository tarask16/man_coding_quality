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
    """Правила формирования материального кодированного сообщения C.

    Конфигурация задает только абстрактное представление ошибок. Она не
    раскрывает конкретную схему кодирования и не изменяет базовые модели.
    """

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
class DecodingExtensionConfig:
    """Полная конфигурация изолированного расширения декодирования."""

    extension: ExtensionMetadata
    base_contract: BaseContractConfig
    paths: ExtensionPathConfig
    material_encoding: MaterialEncodingConfig = field(
        default_factory=MaterialEncodingConfig
    )

    def validate(self) -> None:
        """Проверить все разделы конфигурации."""
        self.extension.validate()
        self.base_contract.validate()
        self.paths.validate()
        self.material_encoding.validate()


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
    material_raw_value = root.get("material_encoding", {})
    material_raw = _require_mapping(material_raw_value, "material_encoding")

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
