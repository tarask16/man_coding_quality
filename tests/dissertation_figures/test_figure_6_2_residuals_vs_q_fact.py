"""Локальные тесты генератора рисунка 6.2."""

from __future__ import annotations

import csv
import struct
from pathlib import Path

import numpy as np
import pytest

from manual_coding_sim.dissertation_figures.figure_6_2_residuals_vs_q_fact import (
    DEFAULT_Q_FACT_PATH,
    DEFAULT_Q_PRED_PATH,
    FILE_STEM,
    FactValue,
    PredictionValue,
    ResidualPoint,
    calculate_summary,
    gaussian_smooth,
    generate,
    load_q_fact,
    load_q_pred,
    merge_residual_points,
    validate_points,
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG из заголовка IHDR."""

    with path.open("rb") as stream:
        stream.read(16)
        return struct.unpack(">II", stream.read(8))


def _reference_values() -> tuple[tuple[PredictionValue, ...], tuple[FactValue, ...]]:
    """Сформировать согласованный набор с отрицательными и положительными остатками."""

    q_pred = (0.20, 0.36, 0.51, 0.69, 0.86)
    q_fact = (0.30, 0.42, 0.55, 0.67, 0.82)
    predictions = tuple(
        PredictionValue(f"scn_{index:04d}", f"prt_{index:04d}", value)
        for index, value in enumerate(q_pred)
    )
    facts = tuple(
        FactValue(f"scn_{index:04d}", f"prt_{index:04d}", value)
        for index, value in enumerate(q_fact)
    )
    return predictions, facts


def _write_reference_inputs(project_root: Path) -> tuple[Path, Path]:
    """Записать тестовые q_pred.csv и quality_targets.csv."""

    predictions, facts = _reference_values()
    q_pred_path = project_root / DEFAULT_Q_PRED_PATH
    q_pred_path.parent.mkdir(parents=True, exist_ok=True)
    with q_pred_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("scenario_id", "protocol_id", "q_pred"),
        )
        writer.writeheader()
        for row in predictions:
            writer.writerow(
                {
                    "scenario_id": row.scenario_id,
                    "protocol_id": row.protocol_id,
                    "q_pred": row.q_pred,
                }
            )

    q_fact_path = project_root / DEFAULT_Q_FACT_PATH
    q_fact_path.parent.mkdir(parents=True, exist_ok=True)
    with q_fact_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("scenario_id", "protocol_id", "integral_quality"),
        )
        writer.writeheader()
        for row in facts:
            writer.writerow(
                {
                    "scenario_id": row.scenario_id,
                    "protocol_id": row.protocol_id,
                    "integral_quality": row.q_fact,
                }
            )
    return q_pred_path, q_fact_path


def test_default_paths_match_chapter_artifacts() -> None:
    """Генератор должен использовать артефакты глав 5 и 6."""

    assert DEFAULT_Q_PRED_PATH.as_posix() == "reports/chapter5/q_pred.csv"
    assert DEFAULT_Q_FACT_PATH.as_posix() == "data/processed/quality_targets.csv"
    assert FILE_STEM == "residuals_vs_q_fact"


def test_loaders_read_reference_inputs(tmp_path: Path) -> None:
    """Загрузчики должны прочитать прогнозные и фактические значения."""

    q_pred_path, q_fact_path = _write_reference_inputs(tmp_path)
    assert len(load_q_pred(q_pred_path)) == 5
    assert len(load_q_fact(q_fact_path)) == 5


def test_loader_rejects_out_of_range_value(tmp_path: Path) -> None:
    """Показатели качества вне [0; 1] должны отклоняться."""

    q_pred_path, _ = _write_reference_inputs(tmp_path)
    text = q_pred_path.read_text(encoding="utf-8")
    q_pred_path.write_text(text.replace("0.2", "1.2", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="диапазона"):
        load_q_pred(q_pred_path)


def test_merge_rejects_mismatched_key_sets() -> None:
    """Объединение должно отклонять несовпадающие составные ключи."""

    predictions, facts = _reference_values()
    altered = FactValue("other", "other", facts[0].q_fact)
    with pytest.raises(ValueError, match="Наборы ключей"):
        merge_residual_points(predictions, (altered, *facts[1:]))


def test_validate_points_rejects_constant_q_fact() -> None:
    """Постоянный Q_fact не подходит для графика зависимости остатков."""

    points = tuple(
        ResidualPoint(f"scn_{index}", f"prt_{index}", 0.3 + index * 0.1, 0.5)
        for index in range(3)
    )
    with pytest.raises(ValueError, match="постоянного ряда"):
        validate_points(points)


def test_summary_calculates_bias_and_direction_counts() -> None:
    """Сводка должна корректно рассчитывать Bias и направления ошибок."""

    predictions, facts = _reference_values()
    points = merge_residual_points(predictions, facts)
    summary = calculate_summary(points)
    expected = np.asarray([point.residual for point in points])
    assert summary.count == 5
    assert summary.bias == pytest.approx(float(np.mean(expected)))
    assert summary.underestimation_count == 3
    assert summary.overestimation_count == 2
    assert summary.zero_count == 0


def test_summary_reports_extreme_residuals() -> None:
    """Сводка должна сохранять минимум, максимум и их сценарии."""

    predictions, facts = _reference_values()
    summary = calculate_summary(merge_residual_points(predictions, facts))
    assert summary.residual_min == pytest.approx(-0.10)
    assert summary.residual_max == pytest.approx(0.04)
    assert summary.minimum_scenario == "scn_0000"
    assert summary.maximum_scenario == "scn_0004"


def test_gaussian_smooth_returns_finite_ordered_curve() -> None:
    """Сглаживание должно возвращать конечную кривую на упорядоченной сетке."""

    q_fact = np.asarray([0.2, 0.4, 0.6, 0.8])
    residuals = np.asarray([-0.2, -0.1, 0.0, 0.1])
    x_grid, smooth = gaussian_smooth(q_fact, residuals, bandwidth=0.15)
    assert len(x_grid) == len(smooth) == 180
    assert np.all(np.diff(x_grid) > 0.0)
    assert np.all(np.isfinite(smooth))
    assert smooth[-1] > smooth[0]


def test_png_and_svg_are_dissertation_ready(tmp_path: Path) -> None:
    """PNG должен быть крупным, а SVG — содержать редактируемые подписи."""

    _write_reference_inputs(tmp_path)
    result = generate(project_root=tmp_path, dpi=300)
    width, height = _read_png_dimensions(result.png_path)
    svg = result.svg_path.read_text(encoding="utf-8")

    assert result.png_path.name == f"{FILE_STEM}.png"
    assert result.svg_path.name == f"{FILE_STEM}.svg"
    assert width >= 4200
    assert height >= 2200
    for text in (
        "Остатки интегрального прогноза относительно Q_fact",
        "нулевая линия e = 0",
        "сглаженная тенденция",
        "Центр распределения",
        "Диапазон и квартильная структура",
        "систематическое занижение",
        "остаток e = Q_pred − Q_fact",
        "не является абсолютной калибровкой",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg


def test_cli_main_generates_files(tmp_path: Path) -> None:
    """CLI должен создавать PNG и SVG в каталоге главы 6."""

    from manual_coding_sim.dissertation_figures.figure_6_2_residuals_vs_q_fact import (
        main,
    )

    _write_reference_inputs(tmp_path)
    exit_code = main(["--project-root", str(tmp_path), "--dpi", "300"])
    assert exit_code == 0
    output_dir = tmp_path / "reports" / "chapter6" / "figures"
    assert (output_dir / f"{FILE_STEM}.png").is_file()
    assert (output_dir / f"{FILE_STEM}.svg").is_file()
