"""Генерация рисунка 1.1 о факторах качества сценария применения."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from manual_coding_sim.dissertation_figures.common import (
    FigureExportResult,
    configure_dissertation_style,
    export_figure,
)

OUTPUT_DIR = Path("reports/dissertation_figures/chapter1")
FILE_STEM = "figure_1_1_quality_factors"


def _add_box(
    axis: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = "#31465A",
    fontsize: float = 10.5,
    linewidth: float = 1.35,
) -> FancyBboxPatch:
    """Добавить скругленный блок и центрированную подпись."""

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=3,
    )
    axis.add_patch(box)
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#17212B",
        linespacing=1.18,
        zorder=4,
    )
    return box


def _add_arrow(
    axis: plt.Axes,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#516779",
    linewidth: float = 1.35,
    mutation_scale: float = 13,
) -> None:
    """Добавить направленную связь между элементами схемы."""

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        shrinkA=1.5,
        shrinkB=1.5,
        connectionstyle="arc3,rad=0.0",
        zorder=2,
    )
    axis.add_patch(arrow)


def build_figure() -> plt.Figure:
    """Построить структурную схему факторов и показателей качества."""

    configure_dissertation_style()
    figure, axis = plt.subplots(figsize=(13.5, 7.4))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    axis.text(
        0.5,
        0.955,
        "Факторы качества сценария применения ручного средства кодирования",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#17212B",
    )

    source_x = 0.035
    source_width = 0.215
    source_height = 0.115
    source_y = [0.77, 0.615, 0.46, 0.305, 0.15]
    source_labels = [
        "Средство кодирования S\nправила, таблицы,\nинструкция",
        "Оператор O\nподготовка, внимание,\nутомление",
        "Условия U\nвремя, помехи,\nдоступность инструкции",
        "Сообщения G\nдлина, сложность,\nкритичность",
        "Контроль K\nпроверка, обнаружение,\nисправление",
    ]
    source_colors = ["#E8F1F8", "#EAF3EE", "#F6F0E5", "#F1ECF7", "#F8ECEC"]

    for y, label, color in zip(source_y, source_labels, source_colors, strict=True):
        _add_box(
            axis,
            x=source_x,
            y=y,
            width=source_width,
            height=source_height,
            text=label,
            facecolor=color,
            fontsize=9.0,
        )

    process_x = 0.325
    process_y = 0.345
    process_width = 0.255
    process_height = 0.32
    _add_box(
        axis,
        x=process_x,
        y=process_y,
        width=process_width,
        height=process_height,
        text=(
            "Сценарий применения\n"
            "A = {S, O, U, G, K}\n\n"
            "ручное кодирование → контроль\n"
            "→ ручное декодирование"
        ),
        facecolor="#DCE8F2",
        edgecolor="#244A67",
        fontsize=11.2,
        linewidth=1.7,
    )

    for y in source_y:
        _add_arrow(
            axis,
            start=(source_x + source_width, y + source_height / 2),
            end=(process_x, process_y + process_height / 2),
        )

    metric_group_x = 0.65
    metric_group_y = 0.245
    metric_group_width = 0.325
    metric_group_height = 0.59
    metric_group = FancyBboxPatch(
        (metric_group_x, metric_group_y),
        metric_group_width,
        metric_group_height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.4,
        edgecolor="#6A7E8E",
        facecolor="#FBFCFD",
        linestyle="--",
        zorder=1,
    )
    axis.add_patch(metric_group)
    axis.text(
        metric_group_x + metric_group_width / 2,
        metric_group_y + metric_group_height - 0.035,
        "Частные показатели качества",
        ha="center",
        va="center",
        fontsize=11.2,
        fontweight="bold",
        color="#31465A",
        zorder=4,
    )

    metric_x = [0.675, 0.825]
    metric_y = [0.62, 0.455, 0.29]
    metric_width = 0.13
    metric_height = 0.105
    metric_labels = [
        ("q_acc\nточность", metric_x[0], metric_y[0]),
        ("q_time\nвременная\nэффективность", metric_x[1], metric_y[0]),
        ("q_effort\nтрудоемкость", metric_x[0], metric_y[1]),
        ("q_res\nустойчивость", metric_x[1], metric_y[1]),
        ("q_rep\nповторяемость", metric_x[0], metric_y[2]),
        ("q_fit\nпригодность\nусловиям", metric_x[1], metric_y[2]),
    ]

    for label, x, y in metric_labels:
        _add_box(
            axis,
            x=x,
            y=y,
            width=metric_width,
            height=metric_height,
            text=label,
            facecolor="#F1F4F7",
            edgecolor="#536B7C",
            fontsize=9.1,
        )

    _add_arrow(
        axis,
        start=(process_x + process_width, process_y + process_height / 2),
        end=(metric_group_x, metric_group_y + metric_group_height / 2),
        color="#526C7E",
        linewidth=1.6,
        mutation_scale=14,
    )

    integral_x = 0.70
    integral_y = 0.075
    integral_width = 0.225
    integral_height = 0.105
    _add_box(
        axis,
        x=integral_x,
        y=integral_y,
        width=integral_width,
        height=integral_height,
        text="Интегральное качество\nсценария Q(A)",
        facecolor="#DDEEDB",
        edgecolor="#3B6B3D",
        fontsize=10.8,
        linewidth=1.6,
    )
    _add_arrow(
        axis,
        start=(metric_group_x + metric_group_width / 2, metric_group_y),
        end=(integral_x + integral_width / 2, integral_y + integral_height),
        color="#55765A",
        linewidth=1.5,
        mutation_scale=13,
    )

    axis.text(
        process_x + process_width / 2,
        0.285,
        "Совместное влияние факторов формирует\nнаблюдаемые результаты сценария",
        ha="center",
        va="top",
        fontsize=9.2,
        color="#435566",
        style="italic",
    )

    return figure


def generate(
    *, project_root: Path, dpi: int = 300
) -> FigureExportResult:
    """Сформировать рисунок 1.1 в двух обязательных форматах."""

    figure = build_figure()
    return export_figure(
        figure,
        project_root=project_root,
        relative_output_dir=OUTPUT_DIR,
        file_stem=FILE_STEM,
        dpi=dpi,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Создать анализатор аргументов командной строки."""

    parser = argparse.ArgumentParser(
        description=(
            "Сформировать рисунок 1.1 о факторах качества сценария "
            "в форматах PNG и SVG."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Корень проекта manual_coding_sim.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Разрешение растрового PNG-файла.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить CLI-генерацию рисунка и вывести пути к артефактам."""

    args = build_argument_parser().parse_args(argv)
    result = generate(project_root=args.project_root, dpi=args.dpi)
    print("Рисунок 1.1 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
