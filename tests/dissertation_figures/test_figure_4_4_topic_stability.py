"""Локальные тесты генератора рисунка 4.4."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from manual_coding_sim.dissertation_figures.figure_4_4_topic_stability import (
    DEFAULT_REPORT_PATH,
    FILE_STEM,
    OUTPUT_DIR,
    PairwiseStabilityRecord,
    TopicStabilityRecord,
    TopicStabilityReport,
    build_argument_parser,
    build_figure,
    build_similarity_matrix,
    generate,
    identify_sensitive_pairs,
    load_topic_stability_report,
    validate_topic_stability_report,
)


def _report() -> TopicStabilityReport:
    """Вернуть компактный корректный отчёт устойчивости."""

    return TopicStabilityReport(
        mean_stability=0.84885,
        min_topic_mean_stability=0.80842,
        random_states_count=4,
        topics=(
            TopicStabilityRecord(0, 0.80842, 0.67623, 3),
            TopicStabilityRecord(1, 0.86881, 0.80326, 3),
            TopicStabilityRecord(2, 0.86932, 0.80122, 3),
        ),
        pairs=(
            PairwiseStabilityRecord(11, 42, 0.89691, 0.84549),
            PairwiseStabilityRecord(11, 77, 0.86292, 0.80326),
            PairwiseStabilityRecord(11, 101, 0.80677, 0.67623),
            PairwiseStabilityRecord(42, 77, 0.81754, 0.74499),
            PairwiseStabilityRecord(42, 101, 0.85200, 0.75752),
            PairwiseStabilityRecord(77, 101, 0.84775, 0.64479),
        ),
    )


def _write_report(path: Path, report: TopicStabilityReport) -> None:
    """Записать тестовый CSV-отчёт устойчивости."""

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "row_type",
                "topic_id",
                "left_random_state",
                "right_random_state",
                "mean_similarity",
                "min_similarity",
                "matched_runs_count",
            ]
        )
        writer.writerow(
            [
                "summary",
                "",
                "",
                "",
                report.mean_stability,
                report.min_topic_mean_stability,
                report.random_states_count,
            ]
        )
        for topic in report.topics:
            writer.writerow(
                [
                    "topic",
                    topic.topic_id,
                    11,
                    "",
                    topic.mean_similarity,
                    topic.min_similarity,
                    topic.matched_runs_count,
                ]
            )
        for pair in report.pairs:
            writer.writerow(
                [
                    "pairwise_run",
                    "",
                    pair.left_random_state,
                    pair.right_random_state,
                    pair.mean_similarity,
                    pair.min_similarity,
                    "",
                ]
            )


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG без внешних библиотек обработки изображений."""

    payload = path.read_bytes()
    return int.from_bytes(payload[16:20], "big"), int.from_bytes(payload[20:24], "big")


def test_constants_match_roadmap() -> None:
    """Пути и имя файла должны соответствовать реестру рисунков."""

    assert OUTPUT_DIR == Path("reports/chapter4/figures")
    assert FILE_STEM == "figure_4_4_topic_stability"
    assert DEFAULT_REPORT_PATH == Path("reports/chapter4/topic_stability_report.csv")


def test_load_topic_stability_report(tmp_path: Path) -> None:
    """Загрузчик должен восстанавливать темы, пары и сводные метрики."""

    report_path = tmp_path / "topic_stability_report.csv"
    expected = _report()
    _write_report(report_path, expected)

    loaded = load_topic_stability_report(report_path)

    assert loaded == expected


def test_validate_requires_all_pairwise_combinations() -> None:
    """Неполный набор попарных сравнений должен отклоняться."""

    report = _report()
    incomplete = TopicStabilityReport(
        report.mean_stability,
        report.min_topic_mean_stability,
        report.random_states_count,
        report.topics,
        report.pairs[:-1],
    )
    with pytest.raises(ValueError, match="все попарные"):
        validate_topic_stability_report(incomplete)


def test_validate_rejects_similarity_outside_unit_interval() -> None:
    """Коэффициенты сходства должны принадлежать диапазону [0; 1]."""

    report = _report()
    invalid = TopicStabilityReport(
        1.2,
        report.min_topic_mean_stability,
        report.random_states_count,
        report.topics,
        report.pairs,
    )
    with pytest.raises(ValueError, match="диапазону"):
        validate_topic_stability_report(invalid)


def test_build_similarity_matrix_is_symmetric() -> None:
    """Матрицы попарных сходств должны быть симметричными с единичной диагональю."""

    matrix = build_similarity_matrix(_report())

    assert matrix.random_states == (11, 42, 77, 101)
    assert np.allclose(matrix.mean_similarity, matrix.mean_similarity.T)
    assert np.allclose(matrix.min_similarity, matrix.min_similarity.T)
    assert np.allclose(np.diag(matrix.mean_similarity), 1.0)


def test_identify_sensitive_pairs_uses_minimum_similarity() -> None:
    """Чувствительными должны считаться пары с min similarity ниже порога."""

    pairs = identify_sensitive_pairs(_report(), threshold=0.70)

    assert [(item.left_random_state, item.right_random_state) for item in pairs] == [
        (77, 101),
        (11, 101),
    ]


def test_build_figure_contains_heatmap_and_topic_intervals() -> None:
    """Рисунок должен содержать тепловую карту и интервальную панель."""

    figure = build_figure(_report())
    try:
        titles = [axis.get_title() for axis in figure.axes if axis.get_title()]
        assert any("Попарное сходство" in title for title in titles)
        assert any("Интервалы устойчивости" in title for title in titles)
        assert len(figure.axes) >= 3  # две панели и шкала цвета
    finally:
        plt.close(figure)


def test_generate_creates_png_and_svg(tmp_path: Path) -> None:
    """Генератор должен формировать PNG и SVG с обязательными подписями."""

    report_path = tmp_path / "topic_stability_report.csv"
    _write_report(report_path, _report())

    result = generate(project_root=tmp_path, report_path=report_path, dpi=150)

    assert result.png_path.is_file()
    assert result.svg_path.is_file()
    svg = result.svg_path.read_text(encoding="utf-8")
    for text in (
        "Рисунок 4.4",
        "Попарное сходство запусков",
        "Интервалы устойчивости по факторам",
        "Чувствительные пары seed",
        "seed 77 ↔ 101",
    ):
        assert text in svg


def test_png_and_svg_are_dissertation_ready(tmp_path: Path) -> None:
    """PNG должен быть крупным, а SVG — содержать редактируемые подписи."""

    report_path = tmp_path / "topic_stability_report.csv"
    _write_report(report_path, _report())
    result = generate(project_root=tmp_path, report_path=report_path, dpi=300)
    width, height = _read_png_dimensions(result.png_path)
    svg = result.svg_path.read_text(encoding="utf-8")

    assert width >= 4000
    assert height >= 2200
    assert "font-family" in svg
    assert "mean 0,897" in svg
    assert "min 0,645" in svg


def test_cli_parser_accepts_input_and_dpi() -> None:
    """CLI должен принимать путь к отчёту и разрешение PNG."""

    parser = build_argument_parser()
    args = parser.parse_args(
        [
            "--project-root",
            ".",
            "--input",
            "reports/chapter4/topic_stability_report.csv",
            "--dpi",
            "300",
        ]
    )

    assert args.input == Path("reports/chapter4/topic_stability_report.csv")
    assert args.dpi == 300
