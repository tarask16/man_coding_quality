"""Локальные тесты генератора рисунка 6.5."""

from __future__ import annotations

import csv
import struct
from pathlib import Path

import pytest

from manual_coding_sim.dissertation_figures.figure_6_5_confusion_matrix import (
    CLASS_ORDER,
    DEFAULT_Q_FACT_PATH,
    DEFAULT_Q_PRED_PATH,
    FILE_STEM,
    HIGH_THRESHOLD,
    LOW_THRESHOLD,
    QualityClassPoint,
    calculate_confusion_summary,
    classify_quality,
    generate,
    load_quality_class_points,
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG из заголовка IHDR."""

    with path.open("rb") as stream:
        stream.read(16)
        return struct.unpack(">II", stream.read(8))


def _write_reference_inputs(project_root: Path) -> tuple[Path, Path]:
    """Создать согласованные значения для матрицы без критических ошибок."""

    prediction_path = project_root / DEFAULT_Q_PRED_PATH
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    fact_path = project_root / DEFAULT_Q_FACT_PATH
    fact_path.parent.mkdir(parents=True, exist_ok=True)

    values = (
        (0.20, 0.25),  # low -> low
        (0.30, 0.50),  # medium -> low
        (0.55, 0.52),  # medium -> medium
        (0.65, 0.76),  # high -> medium
        (0.82, 0.88),  # high -> high
        (0.74, 0.80),  # high -> high
    )
    with prediction_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["scenario_id", "protocol_id", "q_pred"],
        )
        writer.writeheader()
        for index, (q_pred, _) in enumerate(values):
            writer.writerow(
                {
                    "scenario_id": f"scn_{index:04d}",
                    "protocol_id": f"prt_{index:04d}",
                    "q_pred": q_pred,
                }
            )
    with fact_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["scenario_id", "protocol_id", "integral_quality"],
        )
        writer.writeheader()
        for index, (_, q_fact) in enumerate(values):
            writer.writerow(
                {
                    "scenario_id": f"scn_{index:04d}",
                    "protocol_id": f"prt_{index:04d}",
                    "integral_quality": q_fact,
                }
            )
    return prediction_path, fact_path


def test_default_configuration_matches_stage_31() -> None:
    """Генератор должен использовать выходы глав 5 и 6."""

    assert DEFAULT_Q_PRED_PATH.as_posix() == "reports/chapter5/q_pred.csv"
    assert DEFAULT_Q_FACT_PATH.as_posix() == "data/processed/quality_targets.csv"
    assert FILE_STEM == "confusion_matrix"
    assert CLASS_ORDER == ("low", "medium", "high")
    assert LOW_THRESHOLD == pytest.approx(0.45)
    assert HIGH_THRESHOLD == pytest.approx(0.70)


def test_classification_respects_threshold_boundaries() -> None:
    """Границы 0,45 и 0,70 должны относиться к старшему классу."""

    assert classify_quality(0.0) == "low"
    assert classify_quality(0.449999) == "low"
    assert classify_quality(0.45) == "medium"
    assert classify_quality(0.699999) == "medium"
    assert classify_quality(0.70) == "high"
    assert classify_quality(1.0) == "high"


def test_loader_returns_aligned_class_points(tmp_path: Path) -> None:
    """Загрузчик должен согласовать ключи и сформировать классы."""

    prediction_path, fact_path = _write_reference_inputs(tmp_path)
    points = load_quality_class_points(prediction_path, fact_path)
    assert len(points) == 6
    assert points[0].predicted_class == "low"
    assert points[1].actual_class == "medium"
    assert points[-1].predicted_class == "high"


def test_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    """Объединение one-to-one должно отклонять дубликаты."""

    prediction_path, fact_path = _write_reference_inputs(tmp_path)
    with prediction_path.open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(["scn_0000", "prt_0000", 0.5])
    with pytest.raises(ValueError, match="дублирующий ключ"):
        load_quality_class_points(prediction_path, fact_path)


def test_loader_rejects_mismatched_key_sets(tmp_path: Path) -> None:
    """Наборы прогнозных и фактических ключей должны совпадать."""

    prediction_path, fact_path = _write_reference_inputs(tmp_path)
    rows = list(csv.DictReader(fact_path.open("r", encoding="utf-8")))[:-1]
    with fact_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["scenario_id", "protocol_id", "integral_quality"],
        )
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Наборы ключей"):
        load_quality_class_points(prediction_path, fact_path)


def test_summary_builds_absolute_matrix_without_critical_errors(tmp_path: Path) -> None:
    """Матрица должна содержать абсолютные числа и нулевые критические переходы."""

    prediction_path, fact_path = _write_reference_inputs(tmp_path)
    summary = calculate_confusion_summary(
        load_quality_class_points(prediction_path, fact_path)
    )
    assert summary.matrix == ((1, 0, 0), (1, 1, 0), (0, 1, 2))
    assert summary.total == 6
    assert summary.correct == 4
    assert summary.adjacent_errors == 2
    assert summary.critical_errors == 0
    assert summary.accuracy == pytest.approx(4 / 6)


def test_summary_detects_both_critical_directions() -> None:
    """Переходы low -> high и high -> low должны учитываться отдельно."""

    points = (
        QualityClassPoint("a", "a", 0.8, 0.2, "high", "low"),
        QualityClassPoint("b", "b", 0.2, 0.8, "low", "high"),
        QualityClassPoint("c", "c", 0.5, 0.5, "medium", "medium"),
    )
    summary = calculate_confusion_summary(points)
    assert summary.critical_low_to_high == 1
    assert summary.critical_high_to_low == 1
    assert summary.critical_errors == 2


def test_summary_requires_at_least_three_points() -> None:
    """Для матрицы ошибок требуется не менее трёх наблюдений."""

    points = (
        QualityClassPoint("a", "a", 0.2, 0.2, "low", "low"),
        QualityClassPoint("b", "b", 0.5, 0.5, "medium", "medium"),
    )
    with pytest.raises(ValueError, match="не менее трёх"):
        calculate_confusion_summary(points)


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
        "Матрица ошибок классов интегрального качества",
        "Матрица ошибок в абсолютных значениях",
        "Фактический класс качества",
        "Прогнозный класс качества",
        "Критические переходы",
        "low → high",
        "high → low",
        "Критические ошибки отсутствуют",
        "не заменяет анализ непрерывных ошибок",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg


def test_cli_main_generates_files(tmp_path: Path) -> None:
    """CLI должен создавать PNG и SVG в каталоге главы 6."""

    from manual_coding_sim.dissertation_figures.figure_6_5_confusion_matrix import main

    _write_reference_inputs(tmp_path)
    exit_code = main(["--project-root", str(tmp_path), "--dpi", "300"])
    assert exit_code == 0
    output_dir = tmp_path / "reports" / "chapter6" / "figures"
    assert (output_dir / f"{FILE_STEM}.png").is_file()
    assert (output_dir / f"{FILE_STEM}.svg").is_file()
