"""Генерация рисунка 1.2 о накоплении ошибок в ручной процедуре."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from manual_coding_sim.dissertation_figures.common import (
    FigureExportResult,
    configure_dissertation_style,
    export_figure,
)

OUTPUT_DIR = Path("reports/dissertation_figures/chapter1")
FILE_STEM = "figure_1_2_error_accumulation"
DEFAULT_ERROR_PROBABILITIES = (0.005, 0.01, 0.02, 0.05)
DEFAULT_MAX_OPERATIONS = 200


def calculate_error_free_probability(
    operation_count: np.ndarray,
    step_error_probability: float,
) -> np.ndarray:
    """Рассчитать вероятность безошибочного выполнения всей процедуры."""

    if not 0.0 <= step_error_probability < 1.0:
        raise ValueError("Вероятность ошибки на операции должна принадлежать [0; 1).")
    if np.any(operation_count < 0):
        raise ValueError("Число операций не может быть отрицательным.")

    return np.power(1.0 - step_error_probability, operation_count)


def calculate_half_probability_operation(step_error_probability: float) -> int:
    """Найти первое число операций, при котором вероятность не превышает 0,5."""

    if not 0.0 < step_error_probability < 1.0:
        raise ValueError("Для расчета порога вероятность ошибки должна быть в (0; 1).")

    return math.ceil(math.log(0.5) / math.log(1.0 - step_error_probability))


def build_series(
    *,
    max_operations: int = DEFAULT_MAX_OPERATIONS,
    error_probabilities: Sequence[float] = DEFAULT_ERROR_PROBABILITIES,
) -> tuple[np.ndarray, Mapping[float, np.ndarray]]:
    """Сформировать числовые ряды для всех отображаемых вероятностей ошибки."""

    if max_operations < 1:
        raise ValueError("Максимальное число операций должно быть положительным.")
    if not error_probabilities:
        raise ValueError("Необходимо задать хотя бы одну вероятность ошибки.")

    operation_count = np.arange(0, max_operations + 1, dtype=int)
    series = {
        probability: calculate_error_free_probability(operation_count, probability)
        for probability in error_probabilities
    }
    return operation_count, series


def build_figure(
    *,
    max_operations: int = DEFAULT_MAX_OPERATIONS,
    error_probabilities: Sequence[float] = DEFAULT_ERROR_PROBABILITIES,
) -> plt.Figure:
    """Построить график зависимости надежности процедуры от числа операций."""

    configure_dissertation_style()
    operation_count, series = build_series(
        max_operations=max_operations,
        error_probabilities=error_probabilities,
    )

    figure, axis = plt.subplots(figsize=(13.5, 7.4))
    figure.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.15)

    line_colors = ("#275D8C", "#36875A", "#C07A1D", "#A43F3F", "#6E55A2")
    for index, probability in enumerate(error_probabilities):
        values = series[probability]
        axis.plot(
            operation_count,
            values,
            linewidth=2.2,
            color=line_colors[index % len(line_colors)],
            label=f"p = {probability:.3f} ({probability * 100:.1f} %)",
        )

        half_operation = calculate_half_probability_operation(probability)
        if half_operation <= max_operations:
            half_value = float((1.0 - probability) ** half_operation)
            axis.scatter(
                [half_operation],
                [half_value],
                s=35,
                color=line_colors[index % len(line_colors)],
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )
            axis.annotate(
                f"n = {half_operation}",
                xy=(half_operation, half_value),
                xytext=(5, -15 if index % 2 == 0 else 8),
                textcoords="offset points",
                fontsize=8.5,
                color=line_colors[index % len(line_colors)],
            )

    axis.axhline(
        0.5,
        color="#606A73",
        linewidth=1.1,
        linestyle="--",
        alpha=0.9,
        label="Уровень P₀ = 0,5",
    )

    axis.set_xlim(0, max_operations)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Число последовательно выполняемых операций n")
    axis.set_ylabel("Вероятность безошибочного выполнения P₀(n)")
    axis.set_title(
        "Накопление вероятности ошибки при увеличении числа операций",
        pad=16,
        fontweight="bold",
    )

    axis.grid(True, which="major", linewidth=0.7, alpha=0.28)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.96,
        title="Ошибка на одной операции",
    )

    axis.text(
        0.015,
        0.065,
        "Модель независимых операций:  P₀(n) = (1 − p)ⁿ",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.2,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#AAB4BC",
            "alpha": 0.94,
        },
    )

    axis.text(
        0.5,
        -0.12,
        (
            "Даже малая вероятность ошибки отдельной операции приводит к быстрому "
            "снижению надежности длинной ручной процедуры."
        ),
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=9.2,
        color="#435566",
        style="italic",
    )

    return figure


def generate(
    *,
    project_root: Path,
    dpi: int = 300,
    max_operations: int = DEFAULT_MAX_OPERATIONS,
) -> FigureExportResult:
    """Сформировать рисунок 1.2 в форматах PNG и SVG."""

    figure = build_figure(max_operations=max_operations)
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
            "Сформировать рисунок 1.2 о накоплении ошибок ручной процедуры "
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
    parser.add_argument(
        "--max-operations",
        type=int,
        default=DEFAULT_MAX_OPERATIONS,
        help="Максимальное число операций по горизонтальной оси.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить CLI-генерацию рисунка и вывести пути к артефактам."""

    args = build_argument_parser().parse_args(argv)
    result = generate(
        project_root=args.project_root,
        dpi=args.dpi,
        max_operations=args.max_operations,
    )
    print("Рисунок 1.2 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
