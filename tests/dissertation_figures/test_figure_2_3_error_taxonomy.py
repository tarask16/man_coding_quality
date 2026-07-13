"""Тесты генератора рисунка 2.3."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from manual_coding_sim.dissertation_figures.figure_2_3_error_taxonomy import (
    ERROR_TYPES,
    EXPECTED_ERROR_CODES,
    EXPECTED_GROUPS,
    EXPECTED_MODIFIER_CODES,
    FILE_STEM,
    MODIFIER_GROUPS,
    OUTPUT_DIR,
    generate,
    main,
    validate_taxonomy,
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG из заголовка без внешних библиотек."""

    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def test_taxonomy_contains_six_required_error_types() -> None:
    """Классификация должна содержать шесть обязательных типов ошибок."""

    validate_taxonomy()

    assert tuple(item.code for item in ERROR_TYPES) == EXPECTED_ERROR_CODES
    assert len(ERROR_TYPES) == 6


def test_error_types_are_split_into_two_methodical_groups() -> None:
    """Типы ошибок должны быть разделены на элементные и процедурные."""

    groups = tuple(dict.fromkeys(item.group for item in ERROR_TYPES))

    assert groups == EXPECTED_GROUPS
    assert sum(item.group == "Элементные ошибки" for item in ERROR_TYPES) == 4
    assert sum(item.group == "Процедурные ошибки" for item in ERROR_TYPES) == 2


def test_modifier_groups_cover_all_scenario_components() -> None:
    """Модификаторы должны охватывать процедуру, оператора, условия и контроль."""

    assert tuple(group.code for group in MODIFIER_GROUPS) == EXPECTED_MODIFIER_CODES
    assert all(len(group.factors) >= 3 for group in MODIFIER_GROUPS)

    titles = {group.title for group in MODIFIER_GROUPS}
    assert "Процедура и сообщение" in titles
    assert "Оператор" in titles
    assert "Условия применения" in titles
    assert "Организация контроля" in titles


def test_validation_rejects_missing_control_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Проверка должна отклонять классификацию без ошибки контроля."""

    import manual_coding_sim.dissertation_figures.figure_2_3_error_taxonomy as module

    monkeypatch.setattr(
        module,
        "ERROR_TYPES",
        tuple(item for item in module.ERROR_TYPES if item.code != "E_ctrl"),
    )

    with pytest.raises(ValueError, match="ошибку контроля"):
        module.validate_taxonomy()


def test_generate_creates_png_and_svg(tmp_path: Path) -> None:
    """Генератор должен создать оба обязательных формата рисунка."""

    result = generate(project_root=tmp_path, dpi=300)

    expected_dir = tmp_path / OUTPUT_DIR
    assert result.png_path == expected_dir / f"{FILE_STEM}.png"
    assert result.svg_path == expected_dir / f"{FILE_STEM}.svg"
    assert result.png_path.is_file()
    assert result.svg_path.is_file()
    assert result.png_path.stat().st_size > 170_000
    assert result.svg_path.stat().st_size > 25_000


def test_png_has_dissertation_ready_resolution(tmp_path: Path) -> None:
    """PNG должен иметь разрешение, достаточное для печатного документа."""

    result = generate(project_root=tmp_path, dpi=300)
    width, height = _read_png_dimensions(result.png_path)

    assert width >= 4200
    assert height >= 2400


def test_svg_contains_error_types_modifiers_and_formula(tmp_path: Path) -> None:
    """SVG должен сохранять классификацию, модификаторы и формулу."""

    result = generate(project_root=tmp_path, dpi=300)
    svg = result.svg_path.read_text(encoding="utf-8")

    for text in (
        "Замена",
        "Пропуск",
        "Вставка",
        "Перестановка",
        "Ошибочный выбор",
        "правила",
        "Ошибка контроля",
        "Элементные ошибки",
        "Процедурные ошибки",
        "Процедура и сообщение",
        "Оператор",
        "Условия применения",
        "Организация контроля",
        "p(E_j | A_i)",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg or "<rect" in svg


def test_cli_reports_both_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI должен завершаться успешно и выводить пути PNG и SVG."""

    exit_code = main(["--project-root", str(tmp_path), "--dpi", "300"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Рисунок 2.3 успешно сформирован" in output
    assert ".png" in output
    assert ".svg" in output
