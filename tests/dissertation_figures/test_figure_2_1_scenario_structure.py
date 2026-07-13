"""Тесты генератора рисунка 2.1."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from manual_coding_sim.dissertation_figures.figure_2_1_scenario_structure import (
    EXPECTED_FEATURE_SETS,
    EXPECTED_SYMBOLS,
    FILE_STEM,
    OUTPUT_DIR,
    SCENARIO_COMPONENTS,
    generate,
    main,
    validate_scenario_structure,
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG из заголовка без внешних библиотек."""

    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def test_scenario_contains_five_components_in_required_order() -> None:
    """Схема должна содержать S, O, U, G и K в фиксированном порядке."""

    validate_scenario_structure()

    assert len(SCENARIO_COMPONENTS) == 5
    assert tuple(component.symbol for component in SCENARIO_COMPONENTS) == EXPECTED_SYMBOLS
    assert (
        tuple(component.feature_set for component in SCENARIO_COMPONENTS)
        == EXPECTED_FEATURE_SETS
    )


def test_every_component_has_apriori_parameters() -> None:
    """Для каждого компонента должны быть заданы параметры и набор признаков."""

    for component in SCENARIO_COMPONENTS:
        assert len(component.parameters) >= 4
        assert len(set(component.parameters)) == len(component.parameters)
        assert component.feature_set.startswith("X_")
        assert component.feature_summary.strip()


def test_validation_rejects_missing_component(monkeypatch: pytest.MonkeyPatch) -> None:
    """Проверка должна отклонять сценарий без одного обязательного компонента."""

    import manual_coding_sim.dissertation_figures.figure_2_1_scenario_structure as module

    monkeypatch.setattr(module, "SCENARIO_COMPONENTS", module.SCENARIO_COMPONENTS[:-1])

    with pytest.raises(ValueError, match="ровно пять"):
        module.validate_scenario_structure()


def test_generate_creates_png_and_svg(tmp_path: Path) -> None:
    """Генератор должен создать два обязательных формата рисунка."""

    result = generate(project_root=tmp_path, dpi=300)

    expected_dir = tmp_path / OUTPUT_DIR
    assert result.png_path == expected_dir / f"{FILE_STEM}.png"
    assert result.svg_path == expected_dir / f"{FILE_STEM}.svg"
    assert result.png_path.is_file()
    assert result.svg_path.is_file()
    assert result.png_path.stat().st_size > 120_000
    assert result.svg_path.stat().st_size > 18_000


def test_png_has_dissertation_ready_resolution(tmp_path: Path) -> None:
    """PNG должен иметь достаточное разрешение для A4-документа."""

    result = generate(project_root=tmp_path, dpi=300)
    width, height = _read_png_dimensions(result.png_path)

    assert width >= 4000
    assert height >= 2200


def test_svg_contains_components_and_apriori_union(tmp_path: Path) -> None:
    """SVG должен сохранять все компоненты и формулу X_prior."""

    result = generate(project_root=tmp_path, dpi=300)
    svg = result.svg_path.read_text(encoding="utf-8")

    for text in (
        "Средство",
        "кодирования",
        "Оператор",
        "Условия",
        "применения",
        "Класс",
        "сообщений",
        "Контрольные",
        "процедуры",
        "X_prior",
        "X_S",
        "X_O",
        "X_U",
        "X_G",
        "X_K",
        "X_fact",
        "Y_fact",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg or "<rect" in svg


def test_cli_reports_both_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI должен завершаться успешно и выводить пути PNG и SVG."""

    exit_code = main(["--project-root", str(tmp_path), "--dpi", "300"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Рисунок 2.1 успешно сформирован" in output
    assert ".png" in output
    assert ".svg" in output
