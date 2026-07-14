"""Локальные тесты генератора рисунка 6.1."""

from __future__ import annotations

import csv
import struct
from pathlib import Path

import pytest

from manual_coding_sim.dissertation_figures.figure_6_1_q_pred_vs_q_fact import (
    DEFAULT_Q_FACT_PATH,
    DEFAULT_Q_PRED_PATH,
    FILE_STEM,
    FactValue,
    PredictionValue,
    ValidationPoint,
    calculate_summary,
    generate,
    load_q_fact,
    load_q_pred,
    merge_validation_points,
    validate_points,
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG из заголовка IHDR."""

    with path.open("rb") as stream:
        stream.read(16)
        return struct.unpack(">II", stream.read(8))


def _reference_values() -> tuple[tuple[PredictionValue, ...], tuple[FactValue, ...]]:
    """Сформировать согласованный тестовый набор из пяти сценариев."""

    q_pred = (0.20, 0.35, 0.50, 0.65, 0.80)
    q_fact = (0.30, 0.42, 0.55, 0.69, 0.83)
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
    """Генератор должен использовать зафиксированные артефакты глав 5 и 6."""

    assert DEFAULT_Q_PRED_PATH.as_posix() == "reports/chapter5/q_pred.csv"
    assert DEFAULT_Q_FACT_PATH.as_posix() == "data/processed/quality_targets.csv"
    assert FILE_STEM == "q_pred_vs_q_fact"


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


def test_loader_rejects_duplicate_key(tmp_path: Path) -> None:
    """Повтор составного ключа должен отклоняться."""

    q_pred_path, _ = _write_reference_inputs(tmp_path)
    rows = q_pred_path.read_text(encoding="utf-8").splitlines()
    q_pred_path.write_text("\n".join(rows + [rows[1]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="должны быть уникальными"):
        load_q_pred(q_pred_path)


def test_merge_rejects_mismatched_key_sets() -> None:
    """Объединение должно выполняться только для совпадающих наборов ключей."""

    predictions, facts = _reference_values()
    altered = FactValue("other", "other", facts[0].q_fact)
    with pytest.raises(ValueError, match="Наборы ключей"):
        merge_validation_points(predictions, (altered, *facts[1:]))


def test_validate_points_rejects_constant_series() -> None:
    """Постоянный ряд не позволяет вычислить корреляцию."""

    points = tuple(
        ValidationPoint(f"scn_{index}", f"prt_{index}", 0.5, 0.2 + index * 0.1)
        for index in range(3)
    )
    with pytest.raises(ValueError, match="постоянного ряда"):
        validate_points(points)


def test_summary_calculates_expected_bias_and_counts() -> None:
    """Сводка должна корректно рассчитывать смещение и направления ошибок."""

    predictions, facts = _reference_values()
    summary = calculate_summary(merge_validation_points(predictions, facts))
    expected_bias = sum(p.q_pred - f.q_fact for p, f in zip(predictions, facts)) / 5
    assert summary.count == 5
    assert summary.bias == pytest.approx(expected_bias)
    assert summary.underestimation_count == 5
    assert summary.overestimation_count == 0
    assert summary.mae == pytest.approx(-expected_bias)


def test_summary_reports_strong_monotonic_association() -> None:
    """Монотонно согласованный набор должен иметь высокую корреляцию."""

    predictions, facts = _reference_values()
    summary = calculate_summary(merge_validation_points(predictions, facts))
    assert summary.pearson > 0.99
    assert summary.spearman == pytest.approx(1.0)
    assert summary.regression_slope > 0.0


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
        "Сопоставление прогнозного и фактического интегрального качества",
        "идеальное соответствие y = x",
        "Pearson",
        "Spearman",
        "Bias",
        "Систематическое смещение",
        "систематическое занижение",
        "отсутствие абсолютной калибровки",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg


def test_cli_main_generates_files(tmp_path: Path) -> None:
    """CLI должен создавать PNG и SVG в каталоге главы 6."""

    from manual_coding_sim.dissertation_figures.figure_6_1_q_pred_vs_q_fact import (
        main,
    )

    _write_reference_inputs(tmp_path)
    exit_code = main(["--project-root", str(tmp_path), "--dpi", "300"])
    assert exit_code == 0
    output_dir = tmp_path / "reports" / "chapter6" / "figures"
    assert (output_dir / f"{FILE_STEM}.png").is_file()
    assert (output_dir / f"{FILE_STEM}.svg").is_file()
