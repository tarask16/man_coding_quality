"""Локальные тесты генератора рисунка 6.3."""

from __future__ import annotations

import csv
import struct
from pathlib import Path

import numpy as np
import pytest

from manual_coding_sim.dissertation_figures.figure_6_2_residuals_vs_q_fact import (
    DEFAULT_Q_FACT_PATH,
    DEFAULT_Q_PRED_PATH,
    FactValue,
    PredictionValue,
    merge_residual_points,
)
from manual_coding_sim.dissertation_figures.figure_6_3_absolute_error_distribution import (
    DEFAULT_THRESHOLDS,
    FILE_STEM,
    calculate_absolute_errors,
    calculate_summary,
    empirical_cdf,
    generate,
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG из заголовка IHDR."""

    with path.open("rb") as stream:
        stream.read(16)
        return struct.unpack(">II", stream.read(8))


def _reference_values() -> tuple[tuple[PredictionValue, ...], tuple[FactValue, ...]]:
    """Сформировать набор с различными абсолютными ошибками."""

    q_pred = (0.20, 0.35, 0.48, 0.72, 0.91)
    q_fact = (0.21, 0.42, 0.60, 0.68, 0.70)
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
    """Записать согласованные q_pred.csv и quality_targets.csv."""

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


def test_default_configuration_matches_stage_29() -> None:
    """Генератор должен использовать артефакты глав 5 и 6."""

    assert DEFAULT_Q_PRED_PATH.as_posix() == "reports/chapter5/q_pred.csv"
    assert DEFAULT_Q_FACT_PATH.as_posix() == "data/processed/quality_targets.csv"
    assert FILE_STEM == "absolute_error_distribution"
    assert DEFAULT_THRESHOLDS == (0.05, 0.10, 0.15, 0.20, 0.30)


def test_absolute_errors_are_nonnegative_and_correct() -> None:
    """Абсолютные ошибки должны совпадать с модулями остатков."""

    predictions, facts = _reference_values()
    points = merge_residual_points(predictions, facts)
    errors = calculate_absolute_errors(points)
    assert np.all(errors >= 0.0)
    assert errors == pytest.approx(np.asarray([0.01, 0.07, 0.12, 0.04, 0.21]))


def test_absolute_errors_require_at_least_three_points() -> None:
    """Для распределения требуется не менее трёх наблюдений."""

    predictions, facts = _reference_values()
    points = merge_residual_points(predictions[:2], facts[:2])
    with pytest.raises(ValueError, match="не менее трёх"):
        calculate_absolute_errors(points)


def test_summary_calculates_mae_median_rmse_and_maximum() -> None:
    """Сводка должна рассчитывать основные показатели абсолютной ошибки."""

    predictions, facts = _reference_values()
    points = merge_residual_points(predictions, facts)
    summary = calculate_summary(points)
    residuals = np.asarray([point.residual for point in points])
    errors = np.abs(residuals)
    assert summary.count == 5
    assert summary.mean == pytest.approx(float(np.mean(errors)))
    assert summary.median == pytest.approx(float(np.median(errors)))
    assert summary.rmse == pytest.approx(float(np.sqrt(np.mean(residuals**2))))
    assert summary.maximum == pytest.approx(0.21)
    assert summary.maximum_scenario == "scn_0004"


def test_summary_reports_threshold_coverage() -> None:
    """Сводка должна рассчитывать численность сценариев под порогами."""

    predictions, facts = _reference_values()
    summary = calculate_summary(merge_residual_points(predictions, facts))
    by_threshold = {threshold: (count, share) for threshold, count, share in summary.threshold_counts}
    assert by_threshold[0.05] == pytest.approx((2, 0.4))
    assert by_threshold[0.10] == pytest.approx((3, 0.6))
    assert by_threshold[0.15] == pytest.approx((4, 0.8))
    assert by_threshold[0.30] == pytest.approx((5, 1.0))


def test_summary_rejects_invalid_threshold() -> None:
    """Порог вне диапазона (0; 1] должен отклоняться."""

    predictions, facts = _reference_values()
    with pytest.raises(ValueError, match="диапазону"):
        calculate_summary(
            merge_residual_points(predictions, facts),
            thresholds=(0.0, 0.1),
        )


def test_empirical_cdf_is_sorted_and_monotonic() -> None:
    """ЭФР должна иметь упорядоченные значения и возрастающие вероятности."""

    x_values, probabilities = empirical_cdf([0.2, 0.05, 0.1, 0.1])
    assert np.all(np.diff(x_values) >= 0.0)
    assert np.all(np.diff(probabilities) > 0.0)
    assert probabilities[-1] == pytest.approx(1.0)


def test_empirical_cdf_rejects_empty_series() -> None:
    """Пустой ряд не подходит для построения ЭФР."""

    with pytest.raises(ValueError, match="непустой"):
        empirical_cdf([])


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
        "Распределение абсолютной ошибки интегрального прогноза",
        "Гистограмма абсолютной ошибки",
        "Эмпирическая функция распределения",
        "Покрытие фиксированных порогов ошибки",
        "MAE",
        "Median |e|",
        "Max |e|",
        "Максимальная абсолютная ошибка",
        "не подтверждает абсолютную калибровку",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg


def test_cli_main_generates_files(tmp_path: Path) -> None:
    """CLI должен создавать PNG и SVG в каталоге главы 6."""

    from manual_coding_sim.dissertation_figures.figure_6_3_absolute_error_distribution import (
        main,
    )

    _write_reference_inputs(tmp_path)
    exit_code = main(["--project-root", str(tmp_path), "--dpi", "300"])
    assert exit_code == 0
    output_dir = tmp_path / "reports" / "chapter6" / "figures"
    assert (output_dir / f"{FILE_STEM}.png").is_file()
    assert (output_dir / f"{FILE_STEM}.svg").is_file()
