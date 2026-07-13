"""Генерация рисунка 1.4 с общей логикой разработанного метода."""

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
FILE_STEM = "figure_1_4_method_logic"

PIPELINE_STEPS: tuple[str, ...] = (
    "Формальная модель\nсценария A = {S, O, U, G, K}",
    "Компьютерное\nмоделирование процесса",
    "Априорные признаки\nX_prior",
    "Латентный профиль\nLDA_prior → θ_prior",
    "Сравнительный индекс\nQ_pred",
    "Внешняя проверка\nпо Q_fact",
)

STEP_DETAILS: tuple[str, ...] = (
    "структура средства, оператора,\nусловий, сообщений и контроля",
    "протоколы, ошибки, время\nи контрольные события",
    "только сведения, доступные\nдо выполнения процедуры",
    "интерпретируемые факторы\nскрытой структуры признаков",
    "ранжирование и сравнительная\nоценка сценариев",
    "MAE, RMSE, Bias, ранговые\nи классификационные метрики",
)

VALIDATION_INPUTS: tuple[str, ...] = (
    "quality_targets.csv",
    "fact_features.csv",
)


def validate_method_logic() -> None:
    """Проверить полноту и порядок этапов общей методической схемы."""

    if len(PIPELINE_STEPS) != 6:
        raise ValueError("Методическая схема должна содержать шесть этапов.")
    if len(STEP_DETAILS) != len(PIPELINE_STEPS):
        raise ValueError("Для каждого этапа должно быть задано пояснение.")
    if "X_prior" not in PIPELINE_STEPS[2]:
        raise ValueError("Третий этап должен формировать априорные признаки X_prior.")
    if "LDA_prior" not in PIPELINE_STEPS[3]:
        raise ValueError("Четвертый этап должен содержать модель LDA_prior.")
    if "Q_pred" not in PIPELINE_STEPS[4]:
        raise ValueError("Пятый этап должен формировать индекс Q_pred.")
    if "Q_fact" not in PIPELINE_STEPS[5]:
        raise ValueError("Шестой этап должен выполнять внешнюю проверку по Q_fact.")


def _add_box(
    axis: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    detail: str,
    facecolor: str,
    edgecolor: str,
) -> None:
    """Добавить основной блок этапа с заголовком и пояснением."""

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.45,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=3,
    )
    axis.add_patch(box)
    axis.text(
        x + width / 2,
        y + height * 0.65,
        title,
        ha="center",
        va="center",
        fontsize=10.1,
        fontweight="bold",
        color="#17212B",
        linespacing=1.15,
        zorder=4,
    )
    axis.text(
        x + width / 2,
        y + height * 0.27,
        detail,
        ha="center",
        va="center",
        fontsize=8.25,
        color="#435566",
        linespacing=1.16,
        zorder=4,
    )


def _add_arrow(
    axis: plt.Axes,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#526C7E",
    linestyle: str = "-",
    linewidth: float = 1.5,
    mutation_scale: float = 13,
) -> None:
    """Добавить направленную связь между блоками схемы."""

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        shrinkA=1.5,
        shrinkB=1.5,
        connectionstyle="arc3,rad=0.0",
        zorder=2,
    )
    axis.add_patch(arrow)


def build_figure() -> plt.Figure:
    """Построить блок-схему общей логики научно-методического аппарата."""

    validate_method_logic()
    configure_dissertation_style()

    figure, axis = plt.subplots(figsize=(14.5, 8.2))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    axis.text(
        0.5,
        0.955,
        "Общая логика априорного сравнительного оценивания качества",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#17212B",
    )
    axis.text(
        0.5,
        0.908,
        "Прогнозный контур строится без использования фактических результатов сценария",
        ha="center",
        va="center",
        fontsize=10.2,
        color="#435566",
        style="italic",
    )

    box_width = 0.245
    box_height = 0.205
    top_y = 0.61
    bottom_y = 0.275
    columns = (0.065, 0.3775, 0.69)

    positions = (
        (columns[0], top_y),
        (columns[1], top_y),
        (columns[2], top_y),
        (columns[2], bottom_y),
        (columns[1], bottom_y),
        (columns[0], bottom_y),
    )
    facecolors = (
        "#E8F1F8",
        "#EAF3EE",
        "#F6F0E5",
        "#F1ECF7",
        "#DDEEDB",
        "#F8ECEC",
    )
    edgecolors = (
        "#315D7B",
        "#3F6B53",
        "#8A6A32",
        "#6B527F",
        "#3B6B3D",
        "#8A4D4D",
    )

    for index, ((x, y), title, detail, facecolor, edgecolor) in enumerate(
        zip(
            positions,
            PIPELINE_STEPS,
            STEP_DETAILS,
            facecolors,
            edgecolors,
            strict=True,
        ),
        start=1,
    ):
        _add_box(
            axis,
            x=x,
            y=y,
            width=box_width,
            height=box_height,
            title=title,
            detail=detail,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )
        axis.text(
            x + 0.018,
            y + box_height - 0.018,
            str(index),
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color="white",
            bbox={
                "boxstyle": "circle,pad=0.32",
                "facecolor": edgecolor,
                "edgecolor": edgecolor,
            },
            zorder=5,
        )

    # Верхний ряд читается слева направо.
    _add_arrow(
        axis,
        start=(columns[0] + box_width, top_y + box_height / 2),
        end=(columns[1], top_y + box_height / 2),
    )
    _add_arrow(
        axis,
        start=(columns[1] + box_width, top_y + box_height / 2),
        end=(columns[2], top_y + box_height / 2),
    )

    # После формирования X_prior поток переходит вниз к LDA_prior.
    _add_arrow(
        axis,
        start=(columns[2] + box_width / 2, top_y),
        end=(columns[2] + box_width / 2, bottom_y + box_height),
    )

    # Нижний ряд продолжает последовательность справа налево.
    _add_arrow(
        axis,
        start=(columns[2], bottom_y + box_height / 2),
        end=(columns[1] + box_width, bottom_y + box_height / 2),
    )
    _add_arrow(
        axis,
        start=(columns[1], bottom_y + box_height / 2),
        end=(columns[0] + box_width, bottom_y + box_height / 2),
    )

    phase_specs = (
        (
            columns[0],
            top_y + box_height + 0.028,
            box_width * 2 + (columns[1] - columns[0] - box_width),
            "Формализация и воспроизводимое моделирование",
            "#EEF4F8",
        ),
        (
            columns[2],
            top_y + box_height + 0.028,
            box_width,
            "Априорные данные",
            "#FAF4E8",
        ),
        (
            columns[1],
            bottom_y - 0.078,
            box_width * 2 + (columns[2] - columns[1] - box_width),
            "Априорный прогнозный контур",
            "#F4F0F8",
        ),
        (
            columns[0],
            bottom_y - 0.078,
            box_width,
            "Проверочный контур",
            "#FAF0F0",
        ),
    )
    for x, y, width, label, color in phase_specs:
        phase_box = FancyBboxPatch(
            (x, y),
            width,
            0.045,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            linewidth=0.9,
            edgecolor="#8B99A5",
            facecolor=color,
            zorder=1,
        )
        axis.add_patch(phase_box)
        axis.text(
            x + width / 2,
            y + 0.0225,
            label,
            ha="center",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color="#334B5D",
        )

    axis.text(
        0.5,
        0.535,
        "Запрещено использовать фактические признаки и целевые показатели для построения или корректировки Q_pred",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#8A3F3F",
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.42",
            "facecolor": "#FFF7F7",
            "edgecolor": "#C68A8A",
            "alpha": 0.98,
        },
    )

    data_x = columns[0] + 0.012
    data_y = 0.025
    data_width = box_width - 0.024
    data_height = 0.105
    data_box = FancyBboxPatch(
        (data_x, data_y),
        data_width,
        data_height,
        boxstyle="round,pad=0.01,rounding_size=0.016",
        linewidth=1.25,
        edgecolor="#8A4D4D",
        facecolor="#FFF7F7",
        linestyle="--",
        zorder=3,
    )
    axis.add_patch(data_box)
    axis.text(
        data_x + data_width / 2,
        data_y + data_height * 0.68,
        "Фактические проверочные данные",
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color="#6B3D3D",
    )
    axis.text(
        data_x + data_width / 2,
        data_y + data_height * 0.31,
        "quality_targets.csv  •  fact_features.csv",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#6B3D3D",
    )
    _add_arrow(
        axis,
        start=(data_x + data_width / 2, data_y + data_height),
        end=(columns[0] + box_width / 2, bottom_y),
        color="#9A5B5B",
        linestyle="--",
        linewidth=1.5,
    )

    result_x = 0.35
    result_y = 0.025
    result_width = 0.585
    result_height = 0.105
    result_box = FancyBboxPatch(
        (result_x, result_y),
        result_width,
        result_height,
        boxstyle="round,pad=0.01,rounding_size=0.016",
        linewidth=1.15,
        edgecolor="#AAB4BC",
        facecolor="#F7F9FA",
        zorder=3,
    )
    axis.add_patch(result_box)
    axis.text(
        result_x + result_width / 2,
        result_y + result_height / 2,
        (
            "Результат: интерпретируемый априорный индекс относительного качества,\n"
            "предназначенный для сравнения и ранжирования сценариев.\n"
            "Абсолютная точность оценивается во внешнем проверочном контуре."
        ),
        ha="center",
        va="center",
        fontsize=8.85,
        color="#334B5D",
        linespacing=1.18,
        zorder=4,
    )

    return figure


def generate(*, project_root: Path, dpi: int = 300) -> FigureExportResult:
    """Сформировать рисунок 1.4 в форматах PNG и SVG."""

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
            "Сформировать рисунок 1.4 с общей логикой разработанного метода "
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
    print("Рисунок 1.4 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
