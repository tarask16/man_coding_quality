"""Тесты генератора рисунка 1.1."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from manual_coding_sim.dissertation_figures.figure_1_1_quality_factors import (
    FILE_STEM,
    OUTPUT_DIR,
    generate,
    main,
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать ширину и высоту PNG из заголовка без внешних библиотек."""

    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def test_generate_creates_png_and_svg(tmp_path: Path) -> None:
    """Генератор должен создать оба обязательных формата рисунка."""

    result = generate(project_root=tmp_path, dpi=300)

    expected_dir = tmp_path / OUTPUT_DIR
    assert result.png_path == expected_dir / f"{FILE_STEM}.png"
    assert result.svg_path == expected_dir / f"{FILE_STEM}.svg"
    assert result.png_path.is_file()
    assert result.svg_path.is_file()
    assert result.png_path.stat().st_size > 50_000
    assert result.svg_path.stat().st_size > 10_000


def test_png_has_dissertation_ready_resolution(tmp_path: Path) -> None:
    """PNG должен иметь разрешение, достаточное для вставки в A4-документ."""

    result = generate(project_root=tmp_path, dpi=300)
    width, height = _read_png_dimensions(result.png_path)

    assert width >= 3000
    assert height >= 1600


def test_svg_contains_text_and_vector_elements(tmp_path: Path) -> None:
    """SVG должен оставаться редактируемым векторным изображением с текстом."""

    result = generate(project_root=tmp_path, dpi=300)
    svg = result.svg_path.read_text(encoding="utf-8")

    assert "<svg" in svg
    assert "Сценарий применения" in svg
    assert "Интегральное качество" in svg
    assert "<path" in svg
    assert "<text" in svg


def test_cli_reports_both_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI должен завершаться успешно и выводить пути PNG и SVG."""

    exit_code = main(["--project-root", str(tmp_path), "--dpi", "300"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Рисунок 1.1 успешно сформирован" in output
    assert ".png" in output
    assert ".svg" in output
