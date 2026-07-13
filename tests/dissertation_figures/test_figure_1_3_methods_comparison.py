"""Тесты генератора рисунка 1.3."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from manual_coding_sim.dissertation_figures.figure_1_3_methods_comparison import (
    COMPARISON_MATRIX,
    CRITERIA,
    FILE_STEM,
    LEVEL_LABELS,
    METHODS,
    OUTPUT_DIR,
    generate,
    main,
    validate_comparison_matrix,
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать ширину и высоту PNG из заголовка без внешних библиотек."""

    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def test_matrix_matches_methods_and_criteria() -> None:
    """Матрица должна иметь строку для каждого метода и столбец для критерия."""

    validate_comparison_matrix()

    assert COMPARISON_MATRIX.shape == (len(METHODS), len(CRITERIA))
    assert len(METHODS) == 5
    assert len(CRITERIA) == 6
    assert set(np.unique(COMPARISON_MATRIX)).issubset({0, 1, 2, 3})


def test_matrix_contains_expected_methodological_profile() -> None:
    """Ключевые аналитические различия подходов должны быть зафиксированы."""

    expert_row = METHODS.index("Экспертный")
    simulation_row = METHODS.index("Имитационный")
    latent_row = METHODS.index("Латентно-вероятностный")
    hidden_structure_column = CRITERIA.index("Выявление скрытой\nструктуры")
    process_column = CRITERIA.index("Учет процессной\nдинамики")

    assert COMPARISON_MATRIX[expert_row, hidden_structure_column] == 0
    assert COMPARISON_MATRIX[simulation_row, process_column] == 3
    assert COMPARISON_MATRIX[latent_row, hidden_structure_column] == 3
    assert LEVEL_LABELS[3] == "высокая"


def test_validation_rejects_invalid_matrix() -> None:
    """Проверка должна отклонять неверный размер и значения вне шкалы."""

    with pytest.raises(ValueError):
        validate_comparison_matrix(np.zeros((2, 2), dtype=int))

    invalid_values = COMPARISON_MATRIX.copy()
    invalid_values[0, 0] = 4
    with pytest.raises(ValueError):
        validate_comparison_matrix(invalid_values)


def test_generate_creates_png_and_svg(tmp_path: Path) -> None:
    """Генератор должен создать оба обязательных формата рисунка."""

    result = generate(project_root=tmp_path, dpi=300)

    expected_dir = tmp_path / OUTPUT_DIR
    assert result.png_path == expected_dir / f"{FILE_STEM}.png"
    assert result.svg_path == expected_dir / f"{FILE_STEM}.svg"
    assert result.png_path.is_file()
    assert result.svg_path.is_file()
    assert result.png_path.stat().st_size > 100_000
    assert result.svg_path.stat().st_size > 25_000


def test_png_has_dissertation_ready_resolution(tmp_path: Path) -> None:
    """PNG должен иметь разрешение, достаточное для вставки в A4-документ."""

    result = generate(project_root=tmp_path, dpi=300)
    width, height = _read_png_dimensions(result.png_path)

    assert width >= 3200
    assert height >= 1800


def test_svg_contains_all_methods_and_qualitative_note(tmp_path: Path) -> None:
    """SVG должен сохранять подписи методов, критериев и методическое ограничение."""

    result = generate(project_root=tmp_path, dpi=300)
    svg = result.svg_path.read_text(encoding="utf-8")

    assert "<svg" in svg
    assert "Экспертный" in svg
    assert "Многокритериальный" in svg
    assert "Латентно-вероятностный" in svg
    assert "Учет процессной" in svg
    assert "Выявление скрытой" in svg
    assert "аналитическое качественное сопоставление" in svg
    assert "не является результатом измерения точности" in svg
    assert "<text" in svg
    assert "<path" in svg or "<rect" in svg


def test_cli_reports_both_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI должен завершаться успешно и выводить пути PNG и SVG."""

    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--dpi",
            "300",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Рисунок 1.3 успешно сформирован" in output
    assert ".png" in output
    assert ".svg" in output
