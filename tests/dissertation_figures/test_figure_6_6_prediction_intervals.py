"""Локальные тесты генератора рисунка 6.6."""

from __future__ import annotations

import csv
import struct
from pathlib import Path

import pytest

from manual_coding_sim.dissertation_figures.figure_6_6_prediction_intervals import (
    DEFAULT_INTERVALS_PATH,
    DEFAULT_Q_FACT_PATH,
    FILE_STEM,
    STATUS_ORDER,
    PredictionIntervalPoint,
    calculate_interval_coverage_summary,
    generate,
    load_prediction_interval_points,
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG из заголовка IHDR."""

    with path.open("rb") as stream:
        stream.read(16)
        return struct.unpack(">II", stream.read(8))



def _write_reference_inputs(project_root: Path) -> tuple[Path, Path]:
    """Создать интервалы с покрытием и промахами в обоих направлениях."""

    intervals_path = project_root / DEFAULT_INTERVALS_PATH
    intervals_path.parent.mkdir(parents=True, exist_ok=True)
    fact_path = project_root / DEFAULT_Q_FACT_PATH
    fact_path.parent.mkdir(parents=True, exist_ok=True)
    rows = (
        (0.20, 0.15, 0.25, 0.20, 0.10),  # покрыто
        (0.35, 0.30, 0.40, 0.52, 0.20),  # факт выше интервала
        (0.55, 0.50, 0.60, 0.44, 0.30),  # факт ниже интервала
        (0.72, 0.66, 0.78, 0.75, 0.25),  # покрыто
        (0.82, 0.76, 0.88, 0.93, 0.35),  # факт выше интервала
        (0.64, 0.58, 0.70, 0.60, 0.28),  # покрыто
    )
    with intervals_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "scenario_id",
                "protocol_id",
                "q_pred",
                "q_pred_lower",
                "q_pred_upper",
                "uncertainty_score",
            ],
        )
        writer.writeheader()
        for index, (q_pred, lower, upper, _, uncertainty) in enumerate(rows):
            writer.writerow(
                {
                    "scenario_id": f"scn_{index:04d}",
                    "protocol_id": f"prt_{index:04d}",
                    "q_pred": q_pred,
                    "q_pred_lower": lower,
                    "q_pred_upper": upper,
                    "uncertainty_score": uncertainty,
                }
            )
    with fact_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["scenario_id", "protocol_id", "integral_quality"],
        )
        writer.writeheader()
        for index, (_, _, _, q_fact, _) in enumerate(rows):
            writer.writerow(
                {
                    "scenario_id": f"scn_{index:04d}",
                    "protocol_id": f"prt_{index:04d}",
                    "integral_quality": q_fact,
                }
            )
    return intervals_path, fact_path



def test_default_configuration_matches_stage_32() -> None:
    """Генератор должен использовать диагностические интервалы главы 5."""

    assert DEFAULT_INTERVALS_PATH.as_posix() == "reports/chapter5/prediction_uncertainty.csv"
    assert DEFAULT_Q_FACT_PATH.as_posix() == "data/processed/quality_targets.csv"
    assert FILE_STEM == "prediction_intervals"
    assert STATUS_ORDER == ("covered", "miss_above", "miss_below")



def test_loader_returns_aligned_interval_points(tmp_path: Path) -> None:
    """Загрузчик должен согласовать интервалы и фактические значения."""

    intervals_path, fact_path = _write_reference_inputs(tmp_path)
    points = load_prediction_interval_points(intervals_path, fact_path)
    assert len(points) == 6
    assert points[0].status == "covered"
    assert points[1].status == "miss_above"
    assert points[2].status == "miss_below"
    assert points[1].distance_to_interval == pytest.approx(0.12)
    assert points[2].signed_miss_distance == pytest.approx(-0.06)



def test_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    """Объединение one-to-one должно отклонять дублирующий ключ."""

    intervals_path, fact_path = _write_reference_inputs(tmp_path)
    with intervals_path.open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(
            ["scn_0000", "prt_0000", 0.2, 0.1, 0.3, 0.2]
        )
    with pytest.raises(ValueError, match="дублирующий ключ"):
        load_prediction_interval_points(intervals_path, fact_path)



def test_loader_rejects_mismatched_key_sets(tmp_path: Path) -> None:
    """Наборы интервальных и фактических ключей должны совпадать."""

    intervals_path, fact_path = _write_reference_inputs(tmp_path)
    rows = list(csv.DictReader(fact_path.open("r", encoding="utf-8")))[:-1]
    with fact_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["scenario_id", "protocol_id", "integral_quality"],
        )
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Наборы ключей"):
        load_prediction_interval_points(intervals_path, fact_path)



def test_loader_rejects_invalid_interval_bounds(tmp_path: Path) -> None:
    """Нижняя граница не должна превышать верхнюю."""

    intervals_path, fact_path = _write_reference_inputs(tmp_path)
    rows = list(csv.DictReader(intervals_path.open("r", encoding="utf-8")))
    rows[0]["q_pred_lower"] = "0.30"
    rows[0]["q_pred_upper"] = "0.10"
    with intervals_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="нижняя граница"):
        load_prediction_interval_points(intervals_path, fact_path)



def test_loader_requires_prediction_inside_interval(tmp_path: Path) -> None:
    """Центральный прогноз должен находиться между границами."""

    intervals_path, fact_path = _write_reference_inputs(tmp_path)
    rows = list(csv.DictReader(intervals_path.open("r", encoding="utf-8")))
    rows[0]["q_pred"] = "0.40"
    with intervals_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="q_pred должен находиться"):
        load_prediction_interval_points(intervals_path, fact_path)



def test_summary_calculates_coverage_and_miss_directions(tmp_path: Path) -> None:
    """Сводка должна учитывать покрытие и оба направления промаха."""

    intervals_path, fact_path = _write_reference_inputs(tmp_path)
    points = load_prediction_interval_points(intervals_path, fact_path)
    summary = calculate_interval_coverage_summary(points)
    assert summary.total == 6
    assert summary.covered_count == 3
    assert summary.miss_above_count == 2
    assert summary.miss_below_count == 1
    assert summary.coverage_rate == pytest.approx(0.5)
    assert summary.mean_interval_width == pytest.approx(0.11)
    assert summary.max_distance_scenario == "scn_0001"



def test_summary_requires_at_least_three_points() -> None:
    """Для анализа покрытия требуется не менее трёх наблюдений."""

    points = (
        PredictionIntervalPoint("a", "a", 0.2, 0.2, 0.1, 0.3, 0.1),
        PredictionIntervalPoint("b", "b", 0.5, 0.6, 0.4, 0.55, 0.2),
    )
    with pytest.raises(ValueError, match="не менее трёх"):
        calculate_interval_coverage_summary(points)



def test_png_and_svg_are_dissertation_ready(tmp_path: Path) -> None:
    """PNG должен быть крупным, а SVG — содержать редактируемые подписи."""

    _write_reference_inputs(tmp_path)
    result = generate(project_root=tmp_path, dpi=300)
    width, height = _read_png_dimensions(result.png_path)
    svg = result.svg_path.read_text(encoding="utf-8")

    assert result.png_path.name == f"{FILE_STEM}.png"
    assert result.svg_path.name == f"{FILE_STEM}.svg"
    assert width >= 5000
    assert height >= 2200
    for text in (
        "Покрытие фактического качества диагностическими интервалами",
        "Интервалы прогнозов, упорядоченные по фактическому качеству",
        "Промахи выше и ниже интервала",
        "Исходы покрытия",
        "Диагностика интервалов",
        "Coverage rate",
        "Промах выше",
        "Промах ниже",
        "не должна трактоваться как подтверждённый доверительный уровень",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg



def test_cli_main_generates_files(tmp_path: Path) -> None:
    """CLI должен создавать PNG и SVG в каталоге главы 6."""

    from manual_coding_sim.dissertation_figures.figure_6_6_prediction_intervals import main

    _write_reference_inputs(tmp_path)
    exit_code = main(["--project-root", str(tmp_path), "--dpi", "300"])
    assert exit_code == 0
    output_dir = tmp_path / "reports" / "chapter6" / "figures"
    assert (output_dir / f"{FILE_STEM}.png").is_file()
    assert (output_dir / f"{FILE_STEM}.svg").is_file()
