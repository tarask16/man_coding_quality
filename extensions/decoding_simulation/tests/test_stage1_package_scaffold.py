"""Тесты этапа 1: каркас изолированного расширения."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXTENSION_SRC = PROJECT_ROOT / "extensions/decoding_simulation/src"
BASE_SRC = PROJECT_ROOT / "src"
for source_path in (EXTENSION_SRC, BASE_SRC):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from manual_coding_sim_decoding import __version__
from manual_coding_sim_decoding.base_adapter import ManualCodingSimAdapterContract
from manual_coding_sim_decoding.config import load_decoding_extension_config
from manual_coding_sim_decoding.paths import DecodingExtensionPaths
from manual_coding_sim_decoding.runner import main


CONFIG_PATH = (
    PROJECT_ROOT
    / "extensions/decoding_simulation/configs/decoding_stage1.yaml"
)


def test_stage1_package_imports_with_isolated_name() -> None:
    """Проверить отдельное имя пакета и версию каркаса."""
    import manual_coding_sim_decoding

    assert manual_coding_sim_decoding.__name__ == "manual_coding_sim_decoding"
    assert __version__ == "0.1.0"
    assert "extensions/decoding_simulation" in str(
        Path(manual_coding_sim_decoding.__file__).resolve()
    ).replace("\\", "/")


def test_stage1_configuration_resolves_only_extension_paths() -> None:
    """Проверить конфигурацию и изоляцию каталогов вывода."""
    config = load_decoding_extension_config(CONFIG_PATH)
    paths = DecodingExtensionPaths.from_config(PROJECT_ROOT, config)

    assert config.extension.package_name == "manual_coding_sim_decoding"
    assert config.base_contract.package_name == "manual_coding_sim"
    assert paths.extension_root == (
        PROJECT_ROOT / "extensions/decoding_simulation"
    ).resolve()
    for output_path in (paths.reports_dir, paths.data_dir, paths.manifests_dir):
        assert paths.extension_root in output_path.resolve().parents


def test_stage1_base_contract_and_baseline_hashes_are_compatible() -> None:
    """Проверить публичные символы базы и неизменность baseline-файлов."""
    config = load_decoding_extension_config(CONFIG_PATH)
    result = ManualCodingSimAdapterContract(PROJECT_ROOT, config).check()

    assert result.base_package_importable is True
    assert result.missing_modules == ()
    assert result.missing_symbols == {}
    assert result.baseline_manifest_found is True
    assert result.missing_baseline_files == ()
    assert result.changed_baseline_files == ()
    assert result.is_compatible is True


def test_stage1_cli_shows_config_and_checks_contract(capsys: pytest.CaptureFixture[str]) -> None:
    """Проверить CLI этапа без записи в базовый проект."""
    exit_code = main(
        [
            "--project-root",
            str(PROJECT_ROOT),
            "--config",
            str(CONFIG_PATH),
            "--show-config",
            "--check-base-contract",
            "--json",
        ]
    )
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "Конфигурация расширения успешно загружена." in captured
    json_start = captured.index("{")
    result = json.loads(captured[json_start:])
    assert result["is_compatible"] is True
    assert result["changed_baseline_files"] == []
