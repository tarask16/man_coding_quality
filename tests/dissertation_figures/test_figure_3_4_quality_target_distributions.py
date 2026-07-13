"""Тесты генератора рисунка 3.4."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from manual_coding_sim.dissertation_figures.figure_3_4_quality_target_distributions import (
    DEFAULT_INPUT_PATH,
    EXPECTED_SCENARIO_COUNT,
    FILE_STEM,
    OUTPUT_DIR,
    REQUIRED_COLUMNS,
    TARGET_SPECS,
    generate,
    load_quality_targets,
    main,
    summarize_quality_targets,
)


def _write_test_data(path: Path, *, rows: int = EXPECTED_SCENARIO_COUNT) -> Path:
    """Создать воспроизводимый CSV с семью показателями качества."""

    index = np.arange(rows, dtype=float)
    phase = index / max(rows - 1, 1)
    frame = pd.DataFrame(
        {
            "q_acc": np.clip(0.46 + 0.42 * phase + 0.05 * np.sin(index), 0, 1),
            "q_time": np.clip(0.50 + 0.34 * phase + 0.04 * np.cos(index / 3), 0, 1),
            "q_effort": np.clip(0.30 + 0.49 * phase + 0.05 * np.sin(index / 4), 0, 1),
            "q_res": np.where((index.astype(int) % 3) == 0, 1.0, 0.35),
            "q_rep": np.clip(0.35 + 0.55 * phase + 0.08 * np.sin(index / 5), 0, 1),
            "q_fit": np.clip(0.70 + 0.22 * phase + 0.025 * np.cos(index / 7), 0, 1),
        }
    )
    frame["integral_quality"] = frame.mean(axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG из заголовка без Pillow."""

    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def test_target_spec_contains_seven_quality_indicators() -> None:
    """Спецификация должна содержать шесть частных и один интегральный показатель."""

    assert len(TARGET_SPECS) == 7
    assert REQUIRED_COLUMNS == (
        "q_acc",
        "q_time",
        "q_effort",
        "q_res",
        "q_rep",
        "q_fit",
        "integral_quality",
    )
    assert DEFAULT_INPUT_PATH == Path("data/processed/quality_targets.csv")


def test_loader_reads_valid_data_and_summary(tmp_path: Path) -> None:
    """Загрузчик должен прочитать корректные данные и рассчитать статистики."""

    path = _write_test_data(tmp_path / "quality_targets.csv")
    frame = load_quality_targets(path)
    summary = summarize_quality_targets(frame)

    assert frame.shape == (EXPECTED_SCENARIO_COUNT, 7)
    assert tuple(summary.index) == REQUIRED_COLUMNS
    assert summary.loc["q_res", "min"] == pytest.approx(0.35)
    assert summary.loc["q_res", "max"] == pytest.approx(1.0)
    assert 0.0 <= summary.loc["integral_quality", "mean"] <= 1.0


def test_loader_rejects_missing_column(tmp_path: Path) -> None:
    """Загрузчик должен отклонить CSV без обязательного показателя."""

    path = _write_test_data(tmp_path / "quality_targets.csv")
    frame = pd.read_csv(path).drop(columns=["q_rep"])
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="q_rep"):
        load_quality_targets(path)


def test_loader_rejects_value_outside_unit_interval(tmp_path: Path) -> None:
    """Все показатели качества должны находиться в диапазоне [0; 1]."""

    path = _write_test_data(tmp_path / "quality_targets.csv")
    frame = pd.read_csv(path)
    frame.loc[0, "q_time"] = 1.1
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match=r"диапазоне \[0; 1\]"):
        load_quality_targets(path)


def test_loader_rejects_non_numeric_value(tmp_path: Path) -> None:
    """Нечисловое значение должно приводить к контролируемой ошибке."""

    path = _write_test_data(tmp_path / "quality_targets.csv")
    frame = pd.read_csv(path)
    frame["q_fit"] = frame["q_fit"].astype(object)
    frame.loc[0, "q_fit"] = "нет данных"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="q_fit"):
        load_quality_targets(path)


def test_generate_creates_png_and_svg(tmp_path: Path) -> None:
    """Генератор должен создать оба формата в каталоге главы 3."""

    source = _write_test_data(tmp_path / DEFAULT_INPUT_PATH)
    result = generate(project_root=tmp_path, input_path=source, dpi=300)

    expected_dir = tmp_path / OUTPUT_DIR
    assert result.png_path == expected_dir / f"{FILE_STEM}.png"
    assert result.svg_path == expected_dir / f"{FILE_STEM}.svg"
    assert result.png_path.is_file()
    assert result.svg_path.is_file()
    assert result.png_path.stat().st_size > 120_000
    assert result.svg_path.stat().st_size > 35_000


def test_png_has_dissertation_ready_resolution(tmp_path: Path) -> None:
    """PNG должен иметь разрешение, достаточное для полноширинной вставки."""

    source = _write_test_data(tmp_path / DEFAULT_INPUT_PATH)
    result = generate(project_root=tmp_path, input_path=source, dpi=300)
    width, height = _read_png_dimensions(result.png_path)

    assert width >= 3800
    assert height >= 2100


def test_svg_contains_all_targets_and_methodical_note(tmp_path: Path) -> None:
    """SVG должен содержать семь показателей и ограничение априорного контура."""

    source = _write_test_data(tmp_path / DEFAULT_INPUT_PATH)
    result = generate(project_root=tmp_path, input_path=source, dpi=300)
    svg = result.svg_path.read_text(encoding="utf-8")

    for text in (
        "Точность восстановления",
        "Временная эффективность",
        "Трудоёмкость",
        "Результативность контроля",
        "Повторяемость результата",
        "Соответствие условиям",
        "Интегральное качество",
        "N = 150",
        "не входят в априорный контур",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg


def test_cli_reports_both_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI должен завершиться успешно и вывести пути к PNG и SVG."""

    source = _write_test_data(tmp_path / DEFAULT_INPUT_PATH)
    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--input",
            str(source),
            "--dpi",
            "300",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Рисунок 3.4 успешно сформирован" in output
    assert ".png" in output
    assert ".svg" in output
