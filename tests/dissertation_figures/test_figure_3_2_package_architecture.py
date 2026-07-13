"""Тесты генератора рисунка 3.2."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from manual_coding_sim.dissertation_figures.figure_3_2_package_architecture import (
    COMPONENTS,
    CONNECTIONS,
    EXPECTED_COMPONENT_CODES,
    FILE_STEM,
    OUTPUT_DIR,
    REQUIRED_OUTPUT_GROUPS,
    generate,
    main,
    validate_package_architecture,
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG из заголовка без дополнительных библиотек."""

    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def test_all_required_components_are_present_and_ordered() -> None:
    """Архитектура должна содержать все компоненты из спецификации главы 3."""

    validate_package_architecture()

    assert tuple(component.code for component in COMPONENTS) == EXPECTED_COMPONENT_CODES
    assert tuple(component.module_name for component in COMPONENTS) == (
        "runner",
        "message",
        "procedure",
        "operator",
        "condition",
        "error",
        "control",
        "protocol",
        "features",
        "quality",
        "dataset",
    )


def test_domain_components_are_connected_to_runner_and_protocol() -> None:
    """Runner должен управлять предметными модулями, а они — питать protocol."""

    pairs = {(item.source, item.target) for item in CONNECTIONS}
    for code in ("message", "procedure", "operator", "condition", "error", "control"):
        assert ("runner", code) in pairs
        assert (code, "protocol") in pairs


def test_analytics_and_dataset_flow_is_complete() -> None:
    """Протокол должен формировать признаки и качество, поступающие в dataset."""

    pairs = {(item.source, item.target) for item in CONNECTIONS}
    assert ("protocol", "features") in pairs
    assert ("protocol", "quality") in pairs
    assert ("features", "dataset") in pairs
    assert ("quality", "dataset") in pairs


def test_components_have_three_responsibilities_and_outputs() -> None:
    """Каждый компонент должен иметь компактное трёхпунктовое описание и выход."""

    assert all(len(component.responsibility) == 3 for component in COMPONENTS)
    assert all(component.output.strip() for component in COMPONENTS)
    assert len(REQUIRED_OUTPUT_GROUPS) == 5


def test_validation_rejects_missing_protocol_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Проверка должна отклонять архитектуру без связи control → protocol."""

    import manual_coding_sim.dissertation_figures.figure_3_2_package_architecture as module

    modified = tuple(
        item
        for item in module.CONNECTIONS
        if not (item.source == "control" and item.target == "protocol")
    )
    monkeypatch.setattr(module, "CONNECTIONS", modified)

    with pytest.raises(ValueError, match="control должен передавать данные в protocol"):
        module.validate_package_architecture()


def test_generate_creates_png_and_svg(tmp_path: Path) -> None:
    """Генератор должен создать PNG и SVG в каталоге рисунков главы 3."""

    result = generate(project_root=tmp_path, dpi=300)

    expected_dir = tmp_path / OUTPUT_DIR
    assert result.png_path == expected_dir / f"{FILE_STEM}.png"
    assert result.svg_path == expected_dir / f"{FILE_STEM}.svg"
    assert result.png_path.is_file()
    assert result.svg_path.is_file()
    assert result.png_path.stat().st_size > 180_000
    assert result.svg_path.stat().st_size > 22_000


def test_png_has_dissertation_ready_resolution(tmp_path: Path) -> None:
    """PNG должен иметь разрешение, достаточное для печати в A4."""

    result = generate(project_root=tmp_path, dpi=300)
    width, height = _read_png_dimensions(result.png_path)

    assert width >= 4300
    assert height >= 2400


def test_svg_contains_components_outputs_and_methodical_separation(
    tmp_path: Path,
) -> None:
    """SVG должен сохранять модули, выходные файлы и разделение данных."""

    result = generate(project_root=tmp_path, dpi=300)
    svg = result.svg_path.read_text(encoding="utf-8")

    for text in (
        "runner",
        "message",
        "procedure",
        "operator",
        "condition",
        "error",
        "control",
        "protocol",
        "features",
        "quality",
        "dataset",
        "X_prior",
        "X_fact",
        "Y_fact",
        "prior_features.csv",
        "quality_targets.csv",
        "random_seed",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg or "<rect" in svg


def test_cli_reports_both_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI должен успешно завершиться и вывести пути к двум форматам."""

    exit_code = main(["--project-root", str(tmp_path), "--dpi", "300"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Рисунок 3.2 успешно сформирован" in output
    assert ".png" in output
    assert ".svg" in output
