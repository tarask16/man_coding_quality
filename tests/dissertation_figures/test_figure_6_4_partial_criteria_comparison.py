"""Локальные тесты генератора рисунка 6.4."""

from __future__ import annotations

import csv
import struct
from pathlib import Path

import numpy as np
import pytest

from manual_coding_sim.dissertation_figures.figure_6_4_partial_criteria_comparison import (
    CRITERIA,
    DEFAULT_Q_COMPONENTS_PATH,
    DEFAULT_Q_FACT_PATH,
    FILE_STEM,
    CriterionPair,
    calculate_all_metrics,
    calculate_metrics,
    generate,
    load_criterion_pairs,
    spearman_correlation,
    summarize_metrics,
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG из заголовка IHDR."""

    with path.open("rb") as stream:
        stream.read(16)
        return struct.unpack(">II", stream.read(8))


def _write_reference_inputs(project_root: Path) -> tuple[Path, Path]:
    """Записать согласованные прогнозные и фактические критерии."""

    components_path = project_root / DEFAULT_Q_COMPONENTS_PATH
    components_path.parent.mkdir(parents=True, exist_ok=True)
    facts_path = project_root / DEFAULT_Q_FACT_PATH
    facts_path.parent.mkdir(parents=True, exist_ok=True)

    base_fact = np.asarray([0.20, 0.35, 0.50, 0.65, 0.80], dtype=float)
    offsets = {
        "acc": -0.05,
        "time": -0.03,
        "effort": 0.02,
        "res": -0.08,
        "rep": 0.04,
        "fit": -0.10,
    }
    with components_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "scenario_id",
            "protocol_id",
            *(f"q_{code}_pred" for code, _ in CRITERIA),
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, fact_value in enumerate(base_fact):
            row = {
                "scenario_id": f"scn_{index:04d}",
                "protocol_id": f"prt_{index:04d}",
            }
            for code, _ in CRITERIA:
                row[f"q_{code}_pred"] = min(max(fact_value + offsets[code], 0.0), 1.0)
            writer.writerow(row)

    with facts_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "scenario_id",
            "protocol_id",
            *(f"q_{code}" for code, _ in CRITERIA),
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, fact_value in enumerate(base_fact):
            row = {
                "scenario_id": f"scn_{index:04d}",
                "protocol_id": f"prt_{index:04d}",
            }
            for code, _ in CRITERIA:
                row[f"q_{code}"] = fact_value
            writer.writerow(row)
    return components_path, facts_path


def test_default_configuration_matches_stage_30() -> None:
    """Генератор должен использовать артефакты глав 5 и 6."""

    assert DEFAULT_Q_COMPONENTS_PATH.as_posix() == "reports/chapter5/q_pred_components.csv"
    assert DEFAULT_Q_FACT_PATH.as_posix() == "data/processed/quality_targets.csv"
    assert FILE_STEM == "partial_criteria_comparison"
    assert [code for code, _ in CRITERIA] == ["acc", "time", "effort", "res", "rep", "fit"]


def test_loader_returns_six_aligned_pairs(tmp_path: Path) -> None:
    """Загрузчик должен вернуть шесть согласованных рядов одинаковой длины."""

    components, facts = _write_reference_inputs(tmp_path)
    pairs = load_criterion_pairs(components, facts)
    assert len(pairs) == 6
    assert all(len(pair.q_pred) == 5 for pair in pairs)
    assert all(len(pair.q_fact) == 5 for pair in pairs)


def test_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    """Режим one-to-one должен отклонять дублирующиеся ключи."""

    components, facts = _write_reference_inputs(tmp_path)
    with components.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scn_0000", "prt_0000", *([0.5] * 6)])
    with pytest.raises(ValueError, match="дублирующий ключ"):
        load_criterion_pairs(components, facts)


def test_loader_rejects_missing_criterion_column(tmp_path: Path) -> None:
    """Отсутствующая колонка частного критерия должна выявляться явно."""

    components, facts = _write_reference_inputs(tmp_path)
    rows = list(csv.DictReader(components.open("r", encoding="utf-8")))
    with components.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [name for name in rows[0] if name != "q_fit_pred"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row.pop("q_fit_pred")
            writer.writerow(row)
    with pytest.raises(ValueError, match="q_fit_pred"):
        load_criterion_pairs(components, facts)


def test_calculate_metrics_matches_reference_values() -> None:
    """MAE, RMSE, Bias и Spearman должны рассчитываться корректно."""

    pair = CriterionPair(
        code="acc",
        label="q_acc — точность",
        q_pred=(0.10, 0.30, 0.55, 0.70),
        q_fact=(0.20, 0.35, 0.50, 0.80),
    )
    metrics = calculate_metrics(pair)
    residuals = np.asarray([-0.10, -0.05, 0.05, -0.10])
    assert metrics.mae == pytest.approx(float(np.mean(np.abs(residuals))))
    assert metrics.rmse == pytest.approx(float(np.sqrt(np.mean(residuals**2))))
    assert metrics.bias == pytest.approx(float(np.mean(residuals)))
    assert metrics.spearman == pytest.approx(1.0)


def test_spearman_handles_tied_ranks() -> None:
    """Средние ранги должны обеспечивать корректную работу при совпадениях."""

    value = spearman_correlation([1, 1, 2, 3], [2, 2, 3, 4])
    assert value == pytest.approx(1.0)


def test_all_metrics_requires_fixed_criterion_order(tmp_path: Path) -> None:
    """Критерии должны анализироваться в закреплённом порядке."""

    components, facts = _write_reference_inputs(tmp_path)
    pairs = load_criterion_pairs(components, facts)
    with pytest.raises(ValueError, match="порядку"):
        calculate_all_metrics(tuple(reversed(pairs)))


def test_summary_finds_best_and_worst_criteria(tmp_path: Path) -> None:
    """Сводка должна определять критерии с минимальным и максимальным MAE."""

    components, facts = _write_reference_inputs(tmp_path)
    metrics = calculate_all_metrics(load_criterion_pairs(components, facts))
    summary = summarize_metrics(metrics)
    assert summary.count == 5
    assert summary.best_mae_code == "effort"
    assert summary.worst_mae_code == "fit"
    assert summary.mean_spearman == pytest.approx(1.0)


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
        "Сравнение прогнозных и фактических частных критериев",
        "Средняя абсолютная ошибка (MAE)",
        "Среднеквадратическая ошибка (RMSE)",
        "Систематическое смещение (Bias)",
        "Ранговая корреляция Spearman",
        "q_acc — точность",
        "q_effort — трудоёмкость",
        "q_fit — соответствие",
        "не подтверждает абсолютную калибровку",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg


def test_cli_main_generates_files(tmp_path: Path) -> None:
    """CLI должен создавать PNG и SVG в каталоге главы 6."""

    from manual_coding_sim.dissertation_figures.figure_6_4_partial_criteria_comparison import (
        main,
    )

    _write_reference_inputs(tmp_path)
    exit_code = main(["--project-root", str(tmp_path), "--dpi", "300"])
    assert exit_code == 0
    output_dir = tmp_path / "reports" / "chapter6" / "figures"
    assert (output_dir / f"{FILE_STEM}.png").is_file()
    assert (output_dir / f"{FILE_STEM}.svg").is_file()
