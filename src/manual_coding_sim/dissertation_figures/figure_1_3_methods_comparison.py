"""Генерация рисунка 1.3 со сравнением подходов к оценке качества."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from manual_coding_sim.dissertation_figures.common import (
    FigureExportResult,
    configure_dissertation_style,
    export_figure,
)

OUTPUT_DIR = Path("reports/dissertation_figures/chapter1")
FILE_STEM = "figure_1_3_methods_comparison"

METHODS = (
    "Экспертный",
    "Балльный",
    "Многокритериальный",
    "Имитационный",
    "Латентно-вероятностный",
)

CRITERIA = (
    "Учет процессной\nдинамики",
    "Учет человеческого\nфактора",
    "Воспроизводимость\nрезультата",
    "Выявление скрытой\nструктуры",
    "Внешняя\nпроверяемость",
    "Простота начального\nприменения",
)

# Шкала: 0 — отсутствует или выражено слабо; 1 — ограниченно;
# 2 — умеренно; 3 — выражено сильно.
COMPARISON_MATRIX = np.array(
    [
        [0, 1, 0, 0, 0, 3],  # экспертный
        [0, 1, 1, 0, 1, 3],  # балльный
        [1, 1, 2, 0, 2, 2],  # многокритериальный
        [3, 3, 3, 1, 3, 1],  # имитационный
        [2, 2, 3, 3, 3, 1],  # латентно-вероятностный
    ],
    dtype=int,
)

LEVEL_LABELS = {
    0: "низкая",
    1: "ограниченная",
    2: "средняя",
    3: "высокая",
}

LEVEL_SHORT_LABELS = {
    0: "Низкая",
    1: "Ограниченная",
    2: "Средняя",
    3: "Высокая",
}


def validate_comparison_matrix(matrix: np.ndarray = COMPARISON_MATRIX) -> None:
    """Проверить размерность и допустимость значений сравнительной матрицы."""

    expected_shape = (len(METHODS), len(CRITERIA))
    if matrix.shape != expected_shape:
        raise ValueError(
            "Размер сравнительной матрицы должен соответствовать числу методов "
            "и критериев."
        )
    if not np.issubdtype(matrix.dtype, np.integer):
        raise ValueError("Значения сравнительной матрицы должны быть целыми.")
    if np.any(matrix < 0) or np.any(matrix > 3):
        raise ValueError("Значения сравнительной матрицы должны находиться в [0; 3].")


def build_figure() -> plt.Figure:
    """Построить качественную матрицу сопоставления методических подходов."""

    validate_comparison_matrix()
    configure_dissertation_style()

    colors = ("#F2F4F5", "#D6E4EE", "#8DB8D3", "#2F6F99")
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    figure, axis = plt.subplots(figsize=(14.8, 9.4))
    figure.subplots_adjust(left=0.22, right=0.96, top=0.83, bottom=0.31)

    image = axis.imshow(
        COMPARISON_MATRIX,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )

    axis.set_xticks(np.arange(len(CRITERIA)), labels=CRITERIA)
    axis.set_yticks(np.arange(len(METHODS)), labels=METHODS)
    axis.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)
    axis.tick_params(axis="x", labelsize=9.8, pad=8)
    axis.tick_params(axis="y", labelsize=10.4, pad=8)

    axis.set_xticks(np.arange(-0.5, len(CRITERIA), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(METHODS), 1), minor=True)
    axis.grid(which="minor", color="white", linestyle="-", linewidth=2.2)
    axis.tick_params(which="minor", bottom=False, left=False)

    for row_index in range(COMPARISON_MATRIX.shape[0]):
        for column_index in range(COMPARISON_MATRIX.shape[1]):
            value = int(COMPARISON_MATRIX[row_index, column_index])
            text_color = "white" if value == 3 else "#21313D"
            axis.text(
                column_index,
                row_index,
                LEVEL_SHORT_LABELS[value],
                ha="center",
                va="center",
                fontsize=9.4,
                color=text_color,
                fontweight="bold" if value >= 2 else "normal",
            )

    axis.set_title(
        "Качественное сопоставление подходов к априорной оценке качества",
        fontsize=14,
        fontweight="bold",
        pad=48,
    )

    colorbar_axis = figure.add_axes([0.36, 0.175, 0.46, 0.032])
    colorbar = figure.colorbar(
        image,
        cax=colorbar_axis,
        orientation="horizontal",
        ticks=[0, 1, 2, 3],
    )
    colorbar.ax.set_xticklabels(
        ["Низкая", "Ограниченная", "Средняя", "Высокая"],
        fontsize=9.2,
    )
    colorbar.set_label("Степень выраженности свойства", fontsize=10.2, labelpad=8)
    colorbar.outline.set_edgecolor("#6D7A84")
    colorbar.outline.set_linewidth(0.8)

    figure.text(
        0.22,
        0.115,
        (
            "Примечание — матрица отражает аналитическое качественное сопоставление "
            "свойств подходов и не является результатом измерения точности моделей."
        ),
        ha="left",
        va="center",
        fontsize=9.4,
        color="#435566",
        style="italic",
    )

    figure.text(
        0.59,
        0.055,
        (
            "Имитационный и латентно-вероятностный подходы взаимно дополняют друг друга: "
            "первый воспроизводит процесс, второй выявляет скрытую структуру признаков."
        ),
        ha="center",
        va="center",
        fontsize=9.2,
        color="#334B5D",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "#F7F9FA",
            "edgecolor": "#AAB4BC",
            "alpha": 0.98,
        },
    )

    for spine in axis.spines.values():
        spine.set_visible(False)

    return figure


def generate(*, project_root: Path, dpi: int = 300) -> FigureExportResult:
    """Сформировать рисунок 1.3 в форматах PNG и SVG."""

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
            "Сформировать рисунок 1.3 со сравнением подходов к оценке качества "
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
    print("Рисунок 1.3 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
