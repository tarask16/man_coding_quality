"""Генерация рисунка 4.4 с оценкой устойчивости латентных факторов LDA_prior."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

from manual_coding_sim.dissertation_figures.common import (
    FigureExportResult,
    configure_dissertation_style,
    export_figure,
)

OUTPUT_DIR = Path("reports/chapter4/figures")
FILE_STEM = "figure_4_4_topic_stability"
DEFAULT_REPORT_PATH = Path("reports/chapter4/topic_stability_report.csv")

TOPIC_LABELS = {
    0: "Тема 0 — процедурная\nтрудоёмкость",
    1: "Тема 1 — операционный\nриск",
    2: "Тема 2 — благоприятные\nусловия",
}


@dataclass(frozen=True, slots=True)
class TopicStabilityRecord:
    """Средняя и минимальная устойчивость одного латентного фактора."""

    topic_id: int
    mean_similarity: float
    min_similarity: float
    matched_runs_count: int


@dataclass(frozen=True, slots=True)
class PairwiseStabilityRecord:
    """Попарная устойчивость двух запусков с различными random_state."""

    left_random_state: int
    right_random_state: int
    mean_similarity: float
    min_similarity: float


@dataclass(frozen=True, slots=True)
class TopicStabilityReport:
    """Сводный набор метрик устойчивости LDA_prior."""

    mean_stability: float
    min_topic_mean_stability: float
    random_states_count: int
    topics: tuple[TopicStabilityRecord, ...]
    pairs: tuple[PairwiseStabilityRecord, ...]


@dataclass(frozen=True, slots=True)
class StabilityMatrix:
    """Матрицы средних и минимальных сходств для набора random_state."""

    random_states: tuple[int, ...]
    mean_similarity: np.ndarray
    min_similarity: np.ndarray


def _parse_optional_int(value: str | None) -> int | None:
    """Преобразовать непустое значение CSV в целое число."""

    if value is None or not value.strip():
        return None
    return int(float(value))


def _parse_optional_float(value: str | None) -> float | None:
    """Преобразовать непустое значение CSV в вещественное число."""

    if value is None or not value.strip():
        return None
    return float(value)


def load_topic_stability_report(path: str | Path) -> TopicStabilityReport:
    """Загрузить сводку устойчивости тем из CSV-отчёта главы 4."""

    report_path = Path(path)
    if not report_path.is_file():
        raise FileNotFoundError(f"Не найден отчёт устойчивости тем: {report_path}")

    required_columns = {
        "row_type",
        "topic_id",
        "left_random_state",
        "right_random_state",
        "mean_similarity",
        "min_similarity",
        "matched_runs_count",
    }
    summary: tuple[float, float, int] | None = None
    topics: list[TopicStabilityRecord] = []
    pairs: list[PairwiseStabilityRecord] = []

    with report_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("CSV-отчёт не содержит строки заголовка.")
        missing = required_columns.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "В отчёте отсутствуют обязательные колонки: "
                + ", ".join(sorted(missing))
            )

        for row_number, row in enumerate(reader, start=2):
            row_type = (row.get("row_type") or "").strip()
            try:
                mean_similarity = _parse_optional_float(row.get("mean_similarity"))
                min_similarity = _parse_optional_float(row.get("min_similarity"))
                if mean_similarity is None or min_similarity is None:
                    raise ValueError("не заданы метрики сходства")

                if row_type == "summary":
                    run_count = _parse_optional_int(row.get("matched_runs_count"))
                    if run_count is None:
                        raise ValueError("не задано число запусков")
                    summary = (mean_similarity, min_similarity, run_count)
                elif row_type == "topic":
                    topic_id = _parse_optional_int(row.get("topic_id"))
                    matched_runs = _parse_optional_int(row.get("matched_runs_count"))
                    if topic_id is None or matched_runs is None:
                        raise ValueError("не заданы параметры темы")
                    topics.append(
                        TopicStabilityRecord(
                            topic_id=topic_id,
                            mean_similarity=mean_similarity,
                            min_similarity=min_similarity,
                            matched_runs_count=matched_runs,
                        )
                    )
                elif row_type == "pairwise_run":
                    left_state = _parse_optional_int(row.get("left_random_state"))
                    right_state = _parse_optional_int(row.get("right_random_state"))
                    if left_state is None or right_state is None:
                        raise ValueError("не заданы random_state пары")
                    pairs.append(
                        PairwiseStabilityRecord(
                            left_random_state=left_state,
                            right_random_state=right_state,
                            mean_similarity=mean_similarity,
                            min_similarity=min_similarity,
                        )
                    )
                else:
                    raise ValueError(f"неизвестный тип строки {row_type!r}")
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Некорректная строка {row_number} отчёта устойчивости тем: {error}."
                ) from error

    if summary is None:
        raise ValueError("В отчёте отсутствует строка summary.")

    report = TopicStabilityReport(
        mean_stability=summary[0],
        min_topic_mean_stability=summary[1],
        random_states_count=summary[2],
        topics=tuple(sorted(topics, key=lambda item: item.topic_id)),
        pairs=tuple(
            sorted(
                pairs,
                key=lambda item: (item.left_random_state, item.right_random_state),
            )
        ),
    )
    validate_topic_stability_report(report)
    return report


def validate_topic_stability_report(report: TopicStabilityReport) -> None:
    """Проверить полноту и внутреннюю согласованность отчёта устойчивости."""

    if report.random_states_count < 3:
        raise ValueError("Для анализа устойчивости требуется не менее трёх запусков.")
    if not report.topics:
        raise ValueError("Отчёт должен содержать устойчивость по темам.")
    if not report.pairs:
        raise ValueError("Отчёт должен содержать попарную устойчивость запусков.")

    topic_ids = [record.topic_id for record in report.topics]
    if len(topic_ids) != len(set(topic_ids)):
        raise ValueError("Идентификаторы тем должны быть уникальными.")
    if topic_ids != list(range(len(topic_ids))):
        raise ValueError("Идентификаторы тем должны образовывать последовательность от нуля.")

    pair_keys: set[tuple[int, int]] = set()
    random_states: set[int] = set()
    for pair in report.pairs:
        if pair.left_random_state >= pair.right_random_state:
            raise ValueError("Пары random_state должны быть записаны по возрастанию.")
        key = (pair.left_random_state, pair.right_random_state)
        if key in pair_keys:
            raise ValueError("Пары random_state не должны повторяться.")
        pair_keys.add(key)
        random_states.update(key)

    if len(random_states) != report.random_states_count:
        raise ValueError("Число уникальных random_state не совпадает со сводкой.")
    expected_pair_count = report.random_states_count * (report.random_states_count - 1) // 2
    if len(report.pairs) != expected_pair_count:
        raise ValueError("Отчёт должен содержать все попарные сочетания запусков.")

    numeric_values = [
        report.mean_stability,
        report.min_topic_mean_stability,
        *[record.mean_similarity for record in report.topics],
        *[record.min_similarity for record in report.topics],
        *[record.mean_similarity for record in report.pairs],
        *[record.min_similarity for record in report.pairs],
    ]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in numeric_values):
        raise ValueError("Все коэффициенты сходства должны принадлежать диапазону [0; 1].")

    for topic in report.topics:
        if topic.min_similarity > topic.mean_similarity:
            raise ValueError("Минимальное сходство темы не может превышать среднее.")
    for pair in report.pairs:
        if pair.min_similarity > pair.mean_similarity:
            raise ValueError("Минимальное сходство пары не может превышать среднее.")


def build_similarity_matrix(report: TopicStabilityReport) -> StabilityMatrix:
    """Построить симметричные матрицы средних и минимальных сходств запусков."""

    validate_topic_stability_report(report)
    states = tuple(
        sorted(
            {
                state
                for pair in report.pairs
                for state in (pair.left_random_state, pair.right_random_state)
            }
        )
    )
    index = {state: position for position, state in enumerate(states)}
    mean_matrix = np.eye(len(states), dtype=float)
    min_matrix = np.eye(len(states), dtype=float)

    for pair in report.pairs:
        left = index[pair.left_random_state]
        right = index[pair.right_random_state]
        mean_matrix[left, right] = pair.mean_similarity
        mean_matrix[right, left] = pair.mean_similarity
        min_matrix[left, right] = pair.min_similarity
        min_matrix[right, left] = pair.min_similarity

    return StabilityMatrix(
        random_states=states,
        mean_similarity=mean_matrix,
        min_similarity=min_matrix,
    )


def identify_sensitive_pairs(
    report: TopicStabilityReport,
    *,
    threshold: float = 0.70,
) -> tuple[PairwiseStabilityRecord, ...]:
    """Вернуть пары запусков с минимальным сходством ниже заданного порога."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Порог чувствительности должен принадлежать диапазону [0; 1].")
    return tuple(
        pair
        for pair in sorted(report.pairs, key=lambda item: item.min_similarity)
        if pair.min_similarity < threshold
    )


def _add_summary_box(
    figure: plt.Figure,
    *,
    x: float,
    y: float,
    width: float,
    title: str,
    value: str,
    note: str,
) -> None:
    """Добавить компактный сводный блок в нижней части рисунка."""

    patch = FancyBboxPatch(
        (x, y),
        width,
        0.102,
        transform=figure.transFigure,
        boxstyle="round,pad=0.009,rounding_size=0.012",
        linewidth=0.9,
        edgecolor="#B7C4CA",
        facecolor="#F7FAFB",
        zorder=2,
    )
    figure.add_artist(patch)
    figure.text(x + 0.014, y + 0.073, title, fontsize=9.1, fontweight="bold", color="#314A56")
    figure.text(x + 0.014, y + 0.039, value, fontsize=16.0, fontweight="bold", color="#204F6A")
    figure.text(x + width - 0.012, y + 0.040, note, fontsize=8.0, color="#5A6B73", ha="right")


def build_figure(report: TopicStabilityReport) -> plt.Figure:
    """Построить составной рисунок устойчивости латентных факторов."""

    configure_dissertation_style()
    validate_topic_stability_report(report)
    matrix = build_similarity_matrix(report)
    sensitive_pairs = identify_sensitive_pairs(report)

    figure = plt.figure(figsize=(16.6, 9.4), constrained_layout=False)
    grid = figure.add_gridspec(
        1,
        2,
        left=0.065,
        right=0.965,
        top=0.805,
        bottom=0.265,
        width_ratios=(1.12, 0.88),
        wspace=0.23,
    )
    heatmap_axis = figure.add_subplot(grid[0, 0])
    interval_axis = figure.add_subplot(grid[0, 1])

    figure.suptitle(
        "Рисунок 4.4. Устойчивость латентных факторов к начальному состоянию модели",
        fontsize=16.2,
        fontweight="bold",
        y=0.965,
    )
    figure.text(
        0.5,
        0.916,
        "Косинусное сходство распределений topic–word для K = 3 и random_state = 11, 42, 77, 101",
        ha="center",
        fontsize=10.4,
        color="#4C5E66",
    )

    image = heatmap_axis.imshow(
        matrix.mean_similarity,
        cmap="YlGnBu",
        vmin=0.78,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    ticks = np.arange(len(matrix.random_states))
    labels = [f"seed {state}" for state in matrix.random_states]
    heatmap_axis.set_xticks(ticks, labels=labels)
    heatmap_axis.set_yticks(ticks, labels=labels)
    heatmap_axis.set_title(
        "а) Попарное сходство запусков",
        fontsize=11.2,
        fontweight="bold",
        pad=12,
    )
    heatmap_axis.set_xlabel("Правый random_state")
    heatmap_axis.set_ylabel("Левый random_state")

    for row in range(len(matrix.random_states)):
        for column in range(len(matrix.random_states)):
            mean_value = matrix.mean_similarity[row, column]
            min_value = matrix.min_similarity[row, column]
            if row == column:
                text = "1,000\nэталон"
            else:
                text = f"mean {mean_value:.3f}\nmin {min_value:.3f}"
            text_color = "white" if mean_value >= 0.87 else "#18333F"
            heatmap_axis.text(
                column,
                row,
                text.replace(".", ","),
                ha="center",
                va="center",
                fontsize=8.2,
                fontweight="bold" if row != column else "normal",
                color=text_color,
            )

            if row < column and min_value < 0.70:
                heatmap_axis.add_patch(
                    Rectangle(
                        (column - 0.48, row - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="#B8362D",
                        linewidth=2.4,
                    )
                )
                heatmap_axis.add_patch(
                    Rectangle(
                        (row - 0.48, column - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="#B8362D",
                        linewidth=2.4,
                    )
                )

    colorbar = figure.colorbar(image, ax=heatmap_axis, fraction=0.046, pad=0.035)
    colorbar.set_label("Среднее косинусное сходство")

    topics = list(report.topics)
    y_positions = np.arange(len(topics))[::-1]
    for y, topic in zip(y_positions, topics, strict=True):
        interval_axis.hlines(
            y,
            topic.min_similarity,
            topic.mean_similarity,
            color="#7897A4",
            linewidth=4.2,
            zorder=1,
        )
        interval_axis.scatter(
            [topic.min_similarity],
            [y],
            marker="D",
            s=68,
            facecolor="#C75A48",
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        interval_axis.scatter(
            [topic.mean_similarity],
            [y],
            marker="o",
            s=88,
            facecolor="#2F7390",
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        interval_axis.text(
            topic.min_similarity - 0.012,
            y - 0.18,
            f"min {topic.min_similarity:.3f}".replace(".", ","),
            ha="right",
            va="center",
            fontsize=8.0,
            color="#8B3D34",
        )
        interval_axis.text(
            topic.mean_similarity + 0.012,
            y + 0.18,
            f"mean {topic.mean_similarity:.3f}".replace(".", ","),
            ha="left",
            va="center",
            fontsize=8.0,
            color="#225971",
        )

    interval_axis.axvspan(0.80, 1.0, facecolor="#E9F4EC", alpha=0.75, zorder=0)
    interval_axis.axvline(0.80, color="#6E9F78", linewidth=1.1, linestyle="--")
    interval_axis.set_yticks(y_positions, labels=["" for _ in topics])
    for y, topic in zip(y_positions, topics, strict=True):
        interval_axis.text(
            0.623,
            y + 0.27,
            TOPIC_LABELS.get(topic.topic_id, f"Тема {topic.topic_id}"),
            ha="left",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color="#243B46",
        )
    interval_axis.set_xlim(0.62, 0.92)
    interval_axis.set_ylim(-0.55, len(topics) - 0.20)
    interval_axis.set_xlabel("Косинусное сходство")
    interval_axis.set_title(
        "б) Интервалы устойчивости по факторам",
        fontsize=11.2,
        fontweight="bold",
        pad=12,
    )
    interval_axis.grid(axis="x", linewidth=0.7, alpha=0.28)
    interval_axis.spines[["top", "right", "left"]].set_visible(False)
    interval_axis.tick_params(axis="y", length=0)
    interval_axis.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#2F7390", markeredgecolor="white", markersize=8, label="Среднее сходство"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="#C75A48", markeredgecolor="white", markersize=7, label="Минимальное сходство"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=8.4,
    )

    minimum_pair = min(report.pairs, key=lambda item: item.min_similarity)
    _add_summary_box(
        figure,
        x=0.075,
        y=0.112,
        width=0.255,
        title="Средняя устойчивость модели",
        value=f"{report.mean_stability:.4f}".replace(".", ","),
        note="mean stability",
    )
    _add_summary_box(
        figure,
        x=0.372,
        y=0.112,
        width=0.255,
        title="Минимальная средняя по теме",
        value=f"{report.min_topic_mean_stability:.4f}".replace(".", ","),
        note="тема 0",
    )
    _add_summary_box(
        figure,
        x=0.669,
        y=0.112,
        width=0.255,
        title="Минимальное парное совпадение",
        value=f"{minimum_pair.min_similarity:.4f}".replace(".", ","),
        note=f"seed {minimum_pair.left_random_state} ↔ {minimum_pair.right_random_state}",
    )

    if sensitive_pairs:
        pairs_text = "; ".join(
            f"seed {pair.left_random_state} ↔ {pair.right_random_state}: min = {pair.min_similarity:.4f}".replace(".", ",")
            for pair in sensitive_pairs
        )
    else:
        pairs_text = "пары с min similarity < 0,70 отсутствуют"

    figure.text(
        0.075,
        0.072,
        "Чувствительные пары seed: " + pairs_text,
        fontsize=9.2,
        fontweight="bold",
        color="#8A332B",
    )
    figure.text(
        0.075,
        0.038,
        "Интерпретация: структура факторов в целом устойчива, однако отдельные темы чувствительны к начальному состоянию модели; сходство не подтверждает внешнюю валидность факторов.",
        fontsize=8.8,
        color="#4E5F67",
    )

    return figure


def generate(
    *,
    project_root: str | Path,
    report_path: str | Path | None = None,
    dpi: int = 300,
) -> FigureExportResult:
    """Сформировать рисунок 4.4 в PNG и SVG."""

    root = Path(project_root)
    input_path = Path(report_path) if report_path is not None else root / DEFAULT_REPORT_PATH
    report = load_topic_stability_report(input_path)
    figure = build_figure(report)
    return export_figure(
        figure,
        project_root=root,
        relative_output_dir=OUTPUT_DIR,
        file_stem=FILE_STEM,
        dpi=dpi,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Создать CLI-парсер генератора рисунка 4.4."""

    parser = argparse.ArgumentParser(
        description="Сформировать рисунок 4.4 с оценкой устойчивости латентных факторов.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Корень проекта manual_coding_sim.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Путь к topic_stability_report.csv; по умолчанию используется отчёт главы 4.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Разрешение PNG, не ниже 150 dpi.",
    )
    return parser


def main() -> None:
    """Выполнить CLI генератора рисунка 4.4."""

    args = build_argument_parser().parse_args()
    result = generate(
        project_root=args.project_root,
        report_path=args.input,
        dpi=args.dpi,
    )
    print("Рисунок 4.4 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")


if __name__ == "__main__":
    main()
