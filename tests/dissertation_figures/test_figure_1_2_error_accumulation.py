"""Тесты генератора рисунка 1.2."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from manual_coding_sim.dissertation_figures.figure_1_2_error_accumulation import (
    FILE_STEM,
    OUTPUT_DIR,
    build_series,
    calculate_error_free_probability,
    calculate_half_probability_operation,
    generate,
    main,
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать ширину и высоту PNG из заголовка без внешних библиотек."""

    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def test_probability_formula_matches_analytical_values() -> None:
    """Расчет должен соответствовать формуле независимых операций."""

    operations = np.array([0, 1, 10], dtype=int)
    values = calculate_error_free_probability(operations, 0.01)

    assert values[0] == pytest.approx(1.0)
    assert values[1] == pytest.approx(0.99)
    assert values[2] == pytest.approx(0.99**10)
    assert np.all(np.diff(values) <= 0.0)


def test_half_probability_operation_is_correct() -> None:
    """Порог P₀ <= 0,5 должен определяться без смещения на один шаг."""

    operation_count = calculate_half_probability_operation(0.01)

    assert operation_count == 69
    assert 0.99**operation_count <= 0.5
    assert 0.99 ** (operation_count - 1) > 0.5


def test_build_series_covers_requested_range() -> None:
    """Числовые ряды должны включать нулевую и максимальную операцию."""

    operations, series = build_series(
        max_operations=120,
        error_probabilities=(0.01, 0.02),
    )

    assert operations[0] == 0
    assert operations[-1] == 120
    assert len(operations) == 121
    assert set(series) == {0.01, 0.02}
    assert all(len(values) == len(operations) for values in series.values())


def test_generate_creates_png_and_svg(tmp_path: Path) -> None:
    """Генератор должен создать оба обязательных формата рисунка."""

    result = generate(project_root=tmp_path, dpi=300, max_operations=200)

    expected_dir = tmp_path / OUTPUT_DIR
    assert result.png_path == expected_dir / f"{FILE_STEM}.png"
    assert result.svg_path == expected_dir / f"{FILE_STEM}.svg"
    assert result.png_path.is_file()
    assert result.svg_path.is_file()
    assert result.png_path.stat().st_size > 100_000
    assert result.svg_path.stat().st_size > 25_000


def test_png_has_dissertation_ready_resolution(tmp_path: Path) -> None:
    """PNG должен иметь разрешение, достаточное для вставки в A4-документ."""

    result = generate(project_root=tmp_path, dpi=300, max_operations=200)
    width, height = _read_png_dimensions(result.png_path)

    assert width >= 3000
    assert height >= 1700


def test_svg_contains_labels_and_vector_curves(tmp_path: Path) -> None:
    """SVG должен содержать подписи, формулу и редактируемые кривые."""

    result = generate(project_root=tmp_path, dpi=300, max_operations=200)
    svg = result.svg_path.read_text(encoding="utf-8")

    assert "<svg" in svg
    assert "Накопление вероятности ошибки" in svg
    assert "Число последовательно выполняемых операций" in svg
    assert "P₀(n) = (1 − p)ⁿ" in svg
    assert "p = 0.010" in svg
    assert "<path" in svg
    assert "<text" in svg


def test_cli_reports_both_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI должен завершаться успешно и выводить пути PNG и SVG."""

    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--dpi",
            "300",
            "--max-operations",
            "160",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Рисунок 1.2 успешно сформирован" in output
    assert ".png" in output
    assert ".svg" in output
