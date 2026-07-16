"""Контракт безопасного подключения к базовому симулятору."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path

from manual_coding_sim_decoding.config import DecodingExtensionConfig
from manual_coding_sim_decoding.paths import DecodingExtensionPaths


@dataclass(frozen=True)
class BaseContractCheckResult:
    """Результат проверки публичных интерфейсов и baseline-хэшей."""

    base_package_importable: bool
    imported_package_file: str | None
    missing_modules: tuple[str, ...]
    missing_symbols: dict[str, tuple[str, ...]]
    baseline_manifest_found: bool
    missing_baseline_files: tuple[str, ...]
    changed_baseline_files: tuple[str, ...]

    @property
    def is_compatible(self) -> bool:
        """Вернуть итоговый статус совместимости расширения с базой."""
        return (
            self.base_package_importable
            and not self.missing_modules
            and not self.missing_symbols
            and self.baseline_manifest_found
            and not self.missing_baseline_files
            and not self.changed_baseline_files
        )

    def to_dict(self) -> dict[str, object]:
        """Преобразовать результат проверки в сериализуемый словарь."""
        return {
            "is_compatible": self.is_compatible,
            "base_package_importable": self.base_package_importable,
            "imported_package_file": self.imported_package_file,
            "missing_modules": list(self.missing_modules),
            "missing_symbols": {
                module: list(symbols)
                for module, symbols in self.missing_symbols.items()
            },
            "baseline_manifest_found": self.baseline_manifest_found,
            "missing_baseline_files": list(self.missing_baseline_files),
            "changed_baseline_files": list(self.changed_baseline_files),
        }


class ManualCodingSimAdapterContract:
    """Проверяемая граница между расширением и базовым пакетом."""

    def __init__(
        self,
        project_root: str | Path,
        config: DecodingExtensionConfig,
    ) -> None:
        """Сохранить конфигурацию без изменения базовых объектов."""
        self.config = config
        self.paths = DecodingExtensionPaths.from_config(project_root, config)

    def check(self) -> BaseContractCheckResult:
        """Проверить импорты, публичные символы и неизменность baseline."""
        package_importable, package_file = self._check_base_package()
        missing_modules, missing_symbols = self._check_required_symbols()
        (
            manifest_found,
            missing_files,
            changed_files,
        ) = self._check_baseline_hashes()

        return BaseContractCheckResult(
            base_package_importable=package_importable,
            imported_package_file=package_file,
            missing_modules=tuple(sorted(missing_modules)),
            missing_symbols={
                module: tuple(sorted(symbols))
                for module, symbols in sorted(missing_symbols.items())
            },
            baseline_manifest_found=manifest_found,
            missing_baseline_files=tuple(sorted(missing_files)),
            changed_baseline_files=tuple(sorted(changed_files)),
        )

    def _check_base_package(self) -> tuple[bool, str | None]:
        """Проверить отдельный импорт базового пакета."""
        try:
            module = importlib.import_module(self.config.base_contract.package_name)
        except ModuleNotFoundError:
            return False, None
        module_file = getattr(module, "__file__", None)
        return True, str(module_file) if module_file else None

    def _check_required_symbols(
        self,
    ) -> tuple[list[str], dict[str, list[str]]]:
        """Проверить только зафиксированные публичные символы адаптера."""
        missing_modules: list[str] = []
        missing_symbols: dict[str, list[str]] = {}
        package_name = self.config.base_contract.package_name

        for relative_module, symbols in self.config.base_contract.required_symbols.items():
            module_name = f"{package_name}.{relative_module}"
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                missing_modules.append(module_name)
                continue

            absent = [symbol for symbol in symbols if not hasattr(module, symbol)]
            if absent:
                missing_symbols[module_name] = absent

        return missing_modules, missing_symbols

    def _check_baseline_hashes(self) -> tuple[bool, list[str], list[str]]:
        """Сопоставить текущие файлы базы с манифестом этапа 0."""
        manifest_path = self.paths.baseline_manifest
        if not manifest_path.exists():
            return False, [], []

        with manifest_path.open("r", encoding="utf-8") as stream:
            baseline = json.load(stream)
        if not isinstance(baseline, dict):
            raise ValueError("Baseline-манифест должен содержать JSON-словарь.")

        missing_files: list[str] = []
        changed_files: list[str] = []
        for relative_path, expected_hash in baseline.items():
            path = self.paths.project_root / str(relative_path)
            if not path.exists():
                missing_files.append(str(relative_path))
                continue
            actual_hash = _sha256(path)
            if actual_hash != expected_hash:
                changed_files.append(str(relative_path))

        return True, missing_files, changed_files


def _sha256(path: Path) -> str:
    """Рассчитать SHA-256 файла без загрузки целиком в память."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
