"""Локальные тесты генератора рисунка 4.3."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from manual_coding_sim.dissertation_figures.figure_4_3_k_selection_metrics import (
    DEFAULT_REPORT_PATH,
    FILE_STEM,
    OUTPUT_DIR,
    KSelectionRecord,
    build_argument_parser,
    build_figure,
    generate,
    load_k_selection_report,
    summarize_k_selection,
    validate_k_selection_records,
)


def _records() -> tuple[KSelectionRecord, ...]:
    """Вернуть компактный корректный набор кандидатных моделей."""

    return (
        KSelectionRecord(3, 94.8, -0.53, 0.967, 1.0, True),
        KSelectionRecord(4, 96.4, -0.58, 0.925, 0.60, False),
        KSelectionRecord(5, 95.2, -0.56, 0.880, 0.66, False),
    )


def _write_report(path: Path, records: tuple[KSelectionRecord, ...]) -> None:
    """Записать тестовый CSV-отчёт выбора K."""

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "k",
                "perplexity",
                "mean_coherence",
                "topic_diversity",
                "selection_score",
                "is_recommended",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.k,
                    record.perplexity,
                    record.mean_coherence,
                    record.topic_diversity,
                    record.selection_score,
                    str(record.is_recommended).lower(),
                ]
            )


def test_constants_match_roadmap() -> None:
    """Пути и имя файла должны соответствовать реестру рисунков."""

    assert OUTPUT_DIR == Path("reports/chapter4/figures")
    assert FILE_STEM == "figure_4_3_k_selection_metrics"
    assert DEFAULT_REPORT_PATH == Path("reports/chapter4/k_selection_report.csv")


def test_load_k_selection_report(tmp_path: Path) -> None:
    """Загрузчик должен восстанавливать все обязательные метрики."""

    report_path = tmp_path / "k_selection_report.csv"
    _write_report(report_path, _records())
    loaded = load_k_selection_report(report_path)

    assert loaded == _records()


def test_validate_rejects_duplicate_k() -> None:
    """Повторяющиеся значения K должны отклоняться."""

    records = _records() + (
        KSelectionRecord(5, 97.0, -0.60, 0.84, 0.20, False),
    )
    with pytest.raises(ValueError, match="уникальными"):
        validate_k_selection_records(records)


def test_validate_requires_single_recommended_model() -> None:
    """В отчёте должна быть ровно одна рекомендованная модель."""

    records = tuple(
        KSelectionRecord(
            record.k,
            record.perplexity,
            record.mean_coherence,
            record.topic_diversity,
            record.selection_score,
            False,
        )
        for record in _records()
    )
    with pytest.raises(ValueError, match="ровно одна"):
        validate_k_selection_records(records)


def test_validate_recommended_model_has_maximum_score() -> None:
    """Рекомендованная модель должна иметь максимальный selection_score."""

    records = (
        KSelectionRecord(3, 94.8, -0.53, 0.967, 0.70, True),
        KSelectionRecord(4, 96.4, -0.58, 0.925, 0.90, False),
    )
    with pytest.raises(ValueError, match="максимальный"):
        validate_k_selection_records(records)


def test_validate_metric_ranges() -> None:
    """Нормированные метрики должны оставаться в диапазоне [0; 1]."""

    records = (
        KSelectionRecord(3, 94.8, -0.53, 1.2, 1.0, True),
        KSelectionRecord(4, 96.4, -0.58, 0.925, 0.6, False),
    )
    with pytest.raises(ValueError, match="Topic diversity"):
        validate_k_selection_records(records)


def test_summary_returns_selected_k() -> None:
    """Сводка должна содержать метрики рекомендованного K."""

    summary = summarize_k_selection(_records())

    assert summary.selected_k == 3
    assert summary.candidate_count == 3
    assert summary.selected_score == pytest.approx(1.0)


def test_build_figure_contains_four_metric_panels() -> None:
    """Рисунок должен содержать четыре основные панели метрик."""

    figure = build_figure(_records())
    try:
        panel_titles = [axis.get_title() for axis in figure.axes if axis.get_title()]
        assert len(panel_titles) == 4
        assert any("Perplexity" in title for title in panel_titles)
        assert any("согласованность" in title for title in panel_titles)
        assert any("Разнообразие" in title for title in panel_titles)
        assert any("интегральный" in title for title in panel_titles)
    finally:
        plt.close(figure)


def test_generate_creates_png_and_svg(tmp_path: Path) -> None:
    """Генератор должен формировать оба обязательных формата."""

    report_path = tmp_path / "k_selection_report.csv"
    _write_report(report_path, _records())

    result = generate(
        project_root=tmp_path,
        report_path=report_path,
        dpi=150,
    )

    assert result.png_path.is_file()
    assert result.svg_path.is_file()
    svg_text = result.svg_path.read_text(encoding="utf-8")
    assert "Рисунок 4.3" in svg_text
    assert "selection score" in svg_text.lower()
    assert "K = 3" in svg_text


def test_cli_parser_accepts_input_and_dpi() -> None:
    """CLI должен принимать путь к отчёту и разрешение PNG."""

    parser = build_argument_parser()
    args = parser.parse_args(
        [
            "--project-root",
            ".",
            "--input",
            "reports/chapter4/k_selection_report.csv",
            "--dpi",
            "300",
        ]
    )

    assert args.input == Path("reports/chapter4/k_selection_report.csv")
    assert args.dpi == 300
