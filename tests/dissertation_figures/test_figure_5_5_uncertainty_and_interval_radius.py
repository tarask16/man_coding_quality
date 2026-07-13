"""Локальные тесты генератора рисунка 5.5."""

from __future__ import annotations

import csv
import json
import struct
from pathlib import Path

import pytest

from manual_coding_sim.dissertation_figures.figure_5_5_uncertainty_and_interval_radius import (
    DEFAULT_DELTA,
    DEFAULT_INPUT_PATH,
    DEFAULT_REPORT_PATH,
    DEFAULT_WEIGHTS,
    FILE_STEM,
    UncertaintyMetadata,
    UncertaintyRow,
    calculate_summary,
    generate,
    infer_delta,
    load_metadata,
    load_uncertainty,
    validate_metadata,
    validate_proportionality,
    validate_uncertainty_rows,
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG из заголовка IHDR."""

    with path.open("rb") as stream:
        stream.read(16)
        return struct.unpack(">II", stream.read(8))


def _reference_rows(delta: float = DEFAULT_DELTA) -> tuple[UncertaintyRow, ...]:
    """Сформировать согласованный набор интервальных оценок."""

    uncertainties = (0.10, 0.18, 0.25, 0.32, 0.41, 0.53)
    q_pred_values = (0.80, 0.70, 0.62, 0.50, 0.38, 0.25)
    rows: list[UncertaintyRow] = []
    for index, (uncertainty, q_pred) in enumerate(
        zip(uncertainties, q_pred_values, strict=True),
        start=1,
    ):
        radius = delta * uncertainty
        rows.append(
            UncertaintyRow(
                scenario_id=f"scn_{index:04d}",
                protocol_id=f"prt_{index:04d}",
                q_pred=q_pred,
                uncertainty_score=uncertainty,
                interval_radius=radius,
                q_pred_lower=max(0.0, q_pred - radius),
                q_pred_upper=min(1.0, q_pred + radius),
            )
        )
    return tuple(rows)


def _write_reference_inputs(project_root: Path) -> tuple[Path, Path]:
    """Записать тестовые CSV и JSON в стандартные каталоги."""

    csv_path = project_root / DEFAULT_INPUT_PATH
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "scenario_id",
                "protocol_id",
                "q_pred",
                "uncertainty_score",
                "interval_radius",
                "q_pred_lower",
                "q_pred_upper",
            ),
        )
        writer.writeheader()
        for row in _reference_rows():
            writer.writerow(
                {
                    "scenario_id": row.scenario_id,
                    "protocol_id": row.protocol_id,
                    "q_pred": row.q_pred,
                    "uncertainty_score": row.uncertainty_score,
                    "interval_radius": row.interval_radius,
                    "q_pred_lower": row.q_pred_lower,
                    "q_pred_upper": row.q_pred_upper,
                }
            )

    report_path = project_root / DEFAULT_REPORT_PATH
    report_path.write_text(
        json.dumps(
            {
                "delta": DEFAULT_DELTA,
                "weights": DEFAULT_WEIGHTS,
                "mean_stability": 0.84885,
                "input_missing_share": 0.0,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return csv_path, report_path


def test_default_formula_parameters_match_chapter5() -> None:
    """Коэффициент и веса должны совпадать с конфигурацией главы 5."""

    assert DEFAULT_DELTA == pytest.approx(0.15)
    assert DEFAULT_WEIGHTS == {
        "theta_entropy": 0.50,
        "lda_stability": 0.30,
        "input_quality": 0.20,
    }
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_load_uncertainty_reads_reference_csv(tmp_path: Path) -> None:
    """Загрузчик должен прочитать все обязательные поля."""

    csv_path, _ = _write_reference_inputs(tmp_path)
    rows = load_uncertainty(csv_path)
    assert len(rows) == 6
    assert rows[0].uncertainty_score == pytest.approx(0.10)
    assert rows[-1].interval_radius == pytest.approx(0.0795)


def test_infer_delta_recovers_proportionality_coefficient() -> None:
    """По согласованным данным должен восстанавливаться delta = 0,15."""

    assert infer_delta(_reference_rows()) == pytest.approx(DEFAULT_DELTA)


def test_validate_rejects_nonproportional_radius() -> None:
    """Нарушение формулы r = delta · U должно отклоняться."""

    rows = list(_reference_rows())
    source = rows[-1]
    rows[-1] = UncertaintyRow(
        source.scenario_id,
        source.protocol_id,
        source.q_pred,
        source.uncertainty_score,
        source.interval_radius + 0.01,
        source.q_pred_lower,
        source.q_pred_upper,
    )
    with pytest.raises(ValueError, match="постоянной долей"):
        validate_proportionality(tuple(rows), expected_delta=DEFAULT_DELTA)


def test_validate_rejects_duplicate_key() -> None:
    """Дублирование пары ключей должно отклоняться."""

    row = _reference_rows()[0]
    with pytest.raises(ValueError, match="должны быть уникальными"):
        validate_uncertainty_rows((row, row))


def test_validate_rejects_q_pred_outside_interval() -> None:
    """Центральный прогноз должен находиться внутри интервала."""

    with pytest.raises(ValueError, match="внутри собственного интервала"):
        validate_uncertainty_rows(
            (
                UncertaintyRow(
                    "scn",
                    "prt",
                    0.8,
                    0.2,
                    0.03,
                    0.60,
                    0.70,
                ),
            )
        )


def test_metadata_requires_normalized_weights() -> None:
    """Сумма весов источников неопределённости должна быть равна единице."""

    metadata = UncertaintyMetadata(
        delta=0.15,
        weights={
            "theta_entropy": 0.5,
            "lda_stability": 0.3,
            "input_quality": 0.3,
        },
        mean_stability=None,
        input_missing_share=None,
    )
    with pytest.raises(ValueError, match="равна единице"):
        validate_metadata(metadata)


def test_summary_preserves_exact_linear_relation() -> None:
    """Сводка должна отражать единичную корреляцию, заданную алгоритмом."""

    summary = calculate_summary(_reference_rows(), delta=DEFAULT_DELTA)
    assert summary.count == 6
    assert summary.delta == pytest.approx(0.15)
    assert summary.pearson == pytest.approx(1.0)
    assert summary.radius.mean == pytest.approx(
        DEFAULT_DELTA * summary.uncertainty.mean
    )


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
        "Неопределённость прогноза и радиус диагностического интервала",
        "interval_radius = δ · uncertainty_score",
        "Маргинальное распределение uncertainty_score",
        "Линейная зависимость радиуса",
        "Распределение",
        "Параметры расчёта",
        "Pearson(U, r)",
        "не доказывает абсолютную калибровку",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg


def test_cli_main_generates_files(tmp_path: Path) -> None:
    """CLI должен создавать PNG и SVG в стандартном каталоге."""

    from manual_coding_sim.dissertation_figures.figure_5_5_uncertainty_and_interval_radius import (
        main,
    )

    _write_reference_inputs(tmp_path)
    exit_code = main(["--project-root", str(tmp_path), "--dpi", "300"])
    assert exit_code == 0
    output_dir = tmp_path / "reports" / "chapter5" / "figures"
    assert (output_dir / f"{FILE_STEM}.png").is_file()
    assert (output_dir / f"{FILE_STEM}.svg").is_file()
