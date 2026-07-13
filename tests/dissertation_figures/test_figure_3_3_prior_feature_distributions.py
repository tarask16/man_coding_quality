"""Тесты генератора рисунка 3.3."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from manual_coding_sim.dissertation_figures.figure_3_3_prior_feature_distributions import (
    DEFAULT_INPUT_PATH,
    EXPECTED_SCENARIO_COUNT,
    FEATURE_SPECS,
    FILE_STEM,
    OUTPUT_DIR,
    REQUIRED_COLUMNS,
    generate,
    load_prior_features,
    main,
    summarize_prior_features,
)


def _write_test_data(path: Path, *, rows: int = EXPECTED_SCENARIO_COUNT) -> Path:
    """Создать воспроизводимый CSV с шестью обязательными признаками."""

    index = np.arange(rows)
    frame = pd.DataFrame(
        {
            "prior_mean_complexity": index % 5 + 1,
            "prior_mean_message_criticality": (index // 5) % 5 + 1,
            "prior_operator_total_estimated_time": 20.0 + index * 0.48,
            "prior_condition_time_pressure": index % 4,
            "prior_operator_attention": (index // 3) % 5 + 1,
            "prior_expected_error_probability": 0.04 + index * 0.0017,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG из заголовка без Pillow."""

    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def test_feature_spec_contains_six_required_prior_features() -> None:
    """Спецификация должна содержать шесть признаков из реестра рисунков."""

    assert len(FEATURE_SPECS) == 6
    assert REQUIRED_COLUMNS == (
        "prior_mean_complexity",
        "prior_mean_message_criticality",
        "prior_operator_total_estimated_time",
        "prior_condition_time_pressure",
        "prior_operator_attention",
        "prior_expected_error_probability",
    )
    assert DEFAULT_INPUT_PATH == Path("data/processed/prior_features.csv")


def test_loader_reads_valid_data_and_summary(tmp_path: Path) -> None:
    """Загрузчик должен прочитать корректные данные и рассчитать статистики."""

    path = _write_test_data(tmp_path / "prior_features.csv")
    frame = load_prior_features(path)
    summary = summarize_prior_features(frame)

    assert frame.shape == (EXPECTED_SCENARIO_COUNT, 6)
    assert tuple(summary.index) == REQUIRED_COLUMNS
    assert summary.loc["prior_mean_complexity", "min"] == 1.0
    assert summary.loc["prior_mean_complexity", "max"] == 5.0
    assert summary.loc["prior_condition_time_pressure", "unique"] == 4.0


def test_loader_rejects_missing_column(tmp_path: Path) -> None:
    """Загрузчик должен отклонить CSV без обязательного признака."""

    path = _write_test_data(tmp_path / "prior_features.csv")
    frame = pd.read_csv(path).drop(columns=["prior_operator_attention"])
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="prior_operator_attention"):
        load_prior_features(path)


def test_loader_rejects_invalid_probability(tmp_path: Path) -> None:
    """Ожидаемая вероятность ошибки должна оставаться в диапазоне [0; 1]."""

    path = _write_test_data(tmp_path / "prior_features.csv")
    frame = pd.read_csv(path)
    frame.loc[0, "prior_expected_error_probability"] = 1.2
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="выходит выше допустимой границы"):
        load_prior_features(path)


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

    assert width >= 3900
    assert height >= 2300


def test_svg_contains_all_panels_and_methodical_note(tmp_path: Path) -> None:
    """SVG должен содержать подписи шести панелей и методическое ограничение."""

    source = _write_test_data(tmp_path / DEFAULT_INPUT_PATH)
    result = generate(project_root=tmp_path, input_path=source, dpi=300)
    svg = result.svg_path.read_text(encoding="utf-8")

    for text in (
        "Средняя сложность сообщения",
        "Средняя критичность сообщения",
        "Расчётное время оператора",
        "Давление времени",
        "Априорный уровень внимания",
        "Ожидаемая ошибкоопасность",
        "N = 150",
        "фактические ошибки",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg or "<rect" in svg


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
    assert "Рисунок 3.3 успешно сформирован" in output
    assert ".png" in output
    assert ".svg" in output
