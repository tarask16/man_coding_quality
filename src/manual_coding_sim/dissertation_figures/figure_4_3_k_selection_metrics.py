"""Генерация рисунка 4.3 с метриками выбора числа латентных факторов K."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from manual_coding_sim.dissertation_figures.common import (
    FigureExportResult,
    configure_dissertation_style,
    export_figure,
)

OUTPUT_DIR = Path("reports/chapter4/figures")
FILE_STEM = "figure_4_3_k_selection_metrics"
DEFAULT_REPORT_PATH = Path("reports/chapter4/k_selection_report.csv")


@dataclass(frozen=True, slots=True)
class KSelectionRecord:
    """Метрики одной кандидатной LDA-модели с заданным числом факторов."""

    k: int
    perplexity: float
    mean_coherence: float
    topic_diversity: float
    selection_score: float
    is_recommended: bool


@dataclass(frozen=True, slots=True)
class KSelectionSummary:
    """Сводка по выбранному числу факторов и его метрикам."""

    selected_k: int
    selected_perplexity: float
    selected_mean_coherence: float
    selected_topic_diversity: float
    selected_score: float
    candidate_count: int


def _parse_bool(value: str) -> bool:
    """Преобразовать текстовое представление логического значения."""

    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "да"}:
        return True
    if normalized in {"false", "0", "no", "нет"}:
        return False
    raise ValueError(f"Некорректное логическое значение: {value!r}")


def load_k_selection_report(path: str | Path) -> tuple[KSelectionRecord, ...]:
    """Загрузить метрики выбора K из CSV-отчёта главы 4."""

    report_path = Path(path)
    if not report_path.is_file():
        raise FileNotFoundError(f"Не найден отчёт выбора K: {report_path}")

    required_columns = {
        "k",
        "perplexity",
        "mean_coherence",
        "topic_diversity",
        "selection_score",
        "is_recommended",
    }
    records: list[KSelectionRecord] = []
    with report_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("CSV-отчёт не содержит строки заголовка.")
        missing = required_columns.difference(reader.fieldnames)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"В CSV-отчёте отсутствуют обязательные колонки: {missing_text}")

        for row_number, row in enumerate(reader, start=2):
            try:
                records.append(
                    KSelectionRecord(
                        k=int(row["k"]),
                        perplexity=float(row["perplexity"]),
                        mean_coherence=float(row["mean_coherence"]),
                        topic_diversity=float(row["topic_diversity"]),
                        selection_score=float(row["selection_score"]),
                        is_recommended=_parse_bool(row["is_recommended"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Некорректная строка {row_number} в отчёте выбора K."
                ) from error

    return tuple(records)


def validate_k_selection_records(records: Sequence[KSelectionRecord]) -> None:
    """Проверить полноту и внутреннюю согласованность метрик выбора K."""

    if len(records) < 2:
        raise ValueError("Для выбора K требуется не менее двух кандидатных моделей.")

    k_values = [record.k for record in records]
    if any(k < 2 for k in k_values):
        raise ValueError("Число факторов K должно быть не меньше двух.")
    if len(set(k_values)) != len(k_values):
        raise ValueError("Значения K должны быть уникальными.")
    if k_values != sorted(k_values):
        raise ValueError("Строки отчёта должны быть упорядочены по возрастанию K.")

    for record in records:
        numeric_values = (
            record.perplexity,
            record.mean_coherence,
            record.topic_diversity,
            record.selection_score,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("Метрики выбора K должны быть конечными числами.")
        if record.perplexity <= 0.0:
            raise ValueError("Perplexity должна быть положительной.")
        if not 0.0 <= record.topic_diversity <= 1.0:
            raise ValueError("Topic diversity должна принадлежать диапазону [0; 1].")
        if not 0.0 <= record.selection_score <= 1.0:
            raise ValueError("Selection score должен принадлежать диапазону [0; 1].")

    recommended = [record for record in records if record.is_recommended]
    if len(recommended) != 1:
        raise ValueError("В отчёте должна быть ровно одна рекомендованная модель.")

    selected = recommended[0]
    maximum_score = max(record.selection_score for record in records)
    if not math.isclose(selected.selection_score, maximum_score, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("Рекомендованная модель должна иметь максимальный selection_score.")


def summarize_k_selection(records: Sequence[KSelectionRecord]) -> KSelectionSummary:
    """Сформировать сводку по рекомендованной кандидатной модели."""

    validate_k_selection_records(records)
    selected = next(record for record in records if record.is_recommended)
    return KSelectionSummary(
        selected_k=selected.k,
        selected_perplexity=selected.perplexity,
        selected_mean_coherence=selected.mean_coherence,
        selected_topic_diversity=selected.topic_diversity,
        selected_score=selected.selection_score,
        candidate_count=len(records),
    )


def _add_metric_panel(
    axis: plt.Axes,
    *,
    k_values: np.ndarray,
    values: np.ndarray,
    selected_k: int,
    title: str,
    ylabel: str,
    direction_note: str,
    value_formatter: Callable[[float], str],
    color: str,
    ylim: tuple[float, float] | None = None,
    selected_offset: tuple[int, int] = (16, 18),
    direction_y: float = 0.055,
) -> None:
    """Оформить одну панель метрики и выделить рекомендованное K."""

    selected_index = int(np.where(k_values == selected_k)[0][0])
    selected_value = float(values[selected_index])

    axis.axvspan(
        selected_k - 0.32,
        selected_k + 0.32,
        facecolor="#FCE9B9",
        alpha=0.78,
        zorder=0,
    )
    axis.plot(
        k_values,
        values,
        marker="o",
        markersize=6.5,
        linewidth=2.0,
        color=color,
        markerfacecolor="white",
        markeredgewidth=1.6,
        zorder=3,
    )
    axis.scatter(
        [selected_k],
        [selected_value],
        s=112,
        marker="*",
        facecolor="#B66B18",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
    axis.annotate(
        f"K = {selected_k}\n{value_formatter(selected_value)}",
        xy=(selected_k, selected_value),
        xytext=selected_offset,
        textcoords="offset points",
        ha="left" if selected_offset[0] >= 0 else "right",
        va="bottom" if selected_offset[1] >= 0 else "top",
        fontsize=8.2,
        fontweight="bold",
        color="#70400E",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "#FFF8E7",
            "edgecolor": "#D6A65A",
            "linewidth": 0.8,
        },
        arrowprops={
            "arrowstyle": "-",
            "color": "#B98334",
            "linewidth": 0.85,
        },
        zorder=6,
    )

    for k, value in zip(k_values, values, strict=True):
        if int(k) == selected_k:
            continue
        axis.text(
            float(k),
            float(value),
            value_formatter(float(value)),
            ha="center",
            va="bottom",
            fontsize=7.0,
            color="#536976",
            transform=axis.transData,
        )

    axis.set_title(title, fontsize=10.6, fontweight="bold", pad=10)
    axis.set_xlabel("Число латентных факторов K")
    axis.set_ylabel(ylabel)
    axis.set_xticks(k_values)
    axis.grid(axis="y", linewidth=0.7, alpha=0.28)
    axis.spines[["top", "right"]].set_visible(False)
    if ylim is not None:
        axis.set_ylim(*ylim)

    axis.text(
        0.985,
        direction_y,
        direction_note,
        ha="right",
        va="bottom",
        fontsize=7.6,
        fontweight="bold",
        color="#526771",
        transform=axis.transAxes,
    )


def _add_selection_card(
    figure: plt.Figure,
    *,
    summary: KSelectionSummary,
) -> None:
    """Добавить сводную карточку выбранной модели под панелями."""

    card_axis = figure.add_axes([0.205, 0.035, 0.59, 0.095])
    card_axis.axis("off")
    box = FancyBboxPatch(
        (0.0, 0.0),
        1.0,
        1.0,
        boxstyle="round,pad=0.016,rounding_size=0.025",
        facecolor="#F4F8FA",
        edgecolor="#7893A0",
        linewidth=1.1,
        transform=card_axis.transAxes,
    )
    card_axis.add_patch(box)
    card_axis.text(
        0.025,
        0.67,
        f"Рекомендовано: K = {summary.selected_k}",
        ha="left",
        va="center",
        fontsize=10.1,
        fontweight="bold",
        color="#203846",
        transform=card_axis.transAxes,
    )
    card_axis.text(
        0.025,
        0.28,
        (
            f"perplexity = {summary.selected_perplexity:.4f}; "
            f"mean coherence = {summary.selected_mean_coherence:.4f}; "
            f"topic diversity = {summary.selected_topic_diversity:.4f}; "
            f"selection score = {summary.selected_score:.3f}"
        ),
        ha="left",
        va="center",
        fontsize=8.6,
        color="#405863",
        transform=card_axis.transAxes,
    )


def build_figure(records: Sequence[KSelectionRecord]) -> plt.Figure:
    """Построить комбинированный рисунок метрик выбора числа факторов K."""

    configure_dissertation_style()
    summary = summarize_k_selection(records)

    k_values = np.asarray([record.k for record in records], dtype=int)
    perplexity = np.asarray([record.perplexity for record in records], dtype=float)
    coherence = np.asarray([record.mean_coherence for record in records], dtype=float)
    diversity = np.asarray([record.topic_diversity for record in records], dtype=float)
    score = np.asarray([record.selection_score for record in records], dtype=float)

    figure, axes = plt.subplots(2, 2, figsize=(15.4, 9.2), sharex=True)
    figure.subplots_adjust(
        left=0.075,
        right=0.965,
        top=0.845,
        bottom=0.185,
        hspace=0.38,
        wspace=0.24,
    )

    figure.suptitle(
        "Рисунок 4.3 — Сопоставление метрик при выборе числа латентных факторов",
        fontsize=14.1,
        fontweight="bold",
        y=0.958,
        color="#1D2B35",
    )
    figure.text(
        0.5,
        0.915,
        (
            f"Проверенный диапазон: K = {k_values.min()}…{k_values.max()}; "
            f"кандидатных моделей: {summary.candidate_count}. "
            "Золотистая область и звезда обозначают рекомендованное значение."
        ),
        ha="center",
        va="center",
        fontsize=9.2,
        color="#536976",
    )

    _add_metric_panel(
        axes[0, 0],
        k_values=k_values,
        values=perplexity,
        selected_k=summary.selected_k,
        title="а) Perplexity модели",
        ylabel="Perplexity",
        direction_note="меньше — лучше",
        value_formatter=lambda value: f"{value:.2f}",
        color="#2D6F8B",
        ylim=(perplexity.min() - 0.8, perplexity.max() + 1.3),
    )
    _add_metric_panel(
        axes[0, 1],
        k_values=k_values,
        values=coherence,
        selected_k=summary.selected_k,
        title="б) Средняя согласованность тем",
        ylabel="Mean coherence",
        direction_note="выше — лучше",
        value_formatter=lambda value: f"{value:.3f}",
        color="#607D3B",
        ylim=(coherence.min() - 0.018, coherence.max() + 0.025),
    )
    _add_metric_panel(
        axes[1, 0],
        k_values=k_values,
        values=diversity,
        selected_k=summary.selected_k,
        title="в) Разнообразие тем",
        ylabel="Topic diversity",
        direction_note="выше — лучше",
        value_formatter=lambda value: f"{value:.3f}",
        color="#7A5A9A",
        ylim=(max(0.0, diversity.min() - 0.035), min(1.02, diversity.max() + 0.035)),
    )
    _add_metric_panel(
        axes[1, 1],
        k_values=k_values,
        values=score,
        selected_k=summary.selected_k,
        title="г) Нормированный интегральный критерий",
        ylabel="Selection score",
        direction_note="выше — лучше",
        value_formatter=lambda value: f"{value:.3f}",
        color="#A65A44",
        ylim=(-0.05, 1.09),
        selected_offset=(16, -18),
        direction_y=0.91,
    )
    axes[1, 1].fill_between(
        k_values,
        0.0,
        score,
        color="#D9A08F",
        alpha=0.18,
        zorder=1,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#B66B18",
            markeredgecolor="white",
            markersize=12,
            label=f"рекомендовано K = {summary.selected_k}",
        ),
        Line2D(
            [0],
            [0],
            color="#7893A0",
            marker="o",
            markerfacecolor="white",
            linewidth=1.7,
            label="кандидатная модель",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )

    _add_selection_card(figure, summary=summary)
    figure.text(
        0.5,
        0.012,
        (
            "Примечание. Отрицательные значения coherence допустимы для логарифмической "
            "меры совместной встречаемости; интерпретируется относительное сравнение кандидатов. "
            "Selection score объединяет нормированные метрики и не является отдельной оценкой качества кодирования."
        ),
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="#536976",
    )
    return figure


def generate(
    project_root: str | Path = ".",
    *,
    report_path: str | Path | None = None,
    dpi: int = 300,
) -> FigureExportResult:
    """Сформировать рисунок 4.3 в форматах PNG и SVG."""

    root = Path(project_root).resolve()
    resolved_report = (
        Path(report_path).resolve()
        if report_path is not None
        else root / DEFAULT_REPORT_PATH
    )
    records = load_k_selection_report(resolved_report)
    figure = build_figure(records)
    return export_figure(
        figure,
        project_root=root,
        relative_output_dir=OUTPUT_DIR,
        file_stem=FILE_STEM,
        dpi=dpi,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Создать парсер аргументов командной строки."""

    parser = argparse.ArgumentParser(
        description="Генерация рисунка 4.3 с метриками выбора числа факторов K.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Корень проекта с отчётом главы 4 и каталогом reports.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Путь к k_selection_report.csv. По умолчанию используется "
            "reports/chapter4/k_selection_report.csv."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Разрешение PNG. Рекомендуемое значение — 300 dpi.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить CLI-генерацию рисунка 4.3."""

    args = build_argument_parser().parse_args(argv)
    result = generate(
        project_root=args.project_root,
        report_path=args.input,
        dpi=args.dpi,
    )
    print("Рисунок 4.3 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
