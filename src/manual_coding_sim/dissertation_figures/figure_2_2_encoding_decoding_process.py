"""Генерация рисунка 2.2 с процессом ручного кодирования и декодирования."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from manual_coding_sim.dissertation_figures.common import (
    FigureExportResult,
    configure_dissertation_style,
    export_figure,
)

OUTPUT_DIR = Path("reports/dissertation_figures/chapter2")
FILE_STEM = "figure_2_2_encoding_decoding_process"


@dataclass(frozen=True, slots=True)
class ProcessNode:
    """Узел основной последовательности преобразования сообщения."""

    symbol: str
    title: str
    description: str


@dataclass(frozen=True, slots=True)
class ErrorChannel:
    """Канал возникновения ошибки, связанный с этапом процесса."""

    title: str
    details: tuple[str, ...]
    target_symbol: str


@dataclass(frozen=True, slots=True)
class ControlLoop:
    """Контур контроля, обнаружения и исправления ошибки."""

    symbol: str
    stage_symbol: str
    title: str
    actions: tuple[str, ...]


PROCESS_NODES: tuple[ProcessNode, ...] = (
    ProcessNode(
        symbol="M",
        title="Исходное\nсообщение",
        description="элементы, структура,\nкритичность",
    ),
    ProcessNode(
        symbol="E_h",
        title="Ручное\nкодирование",
        description="восприятие → выбор правила →\nпреобразование → запись",
    ),
    ProcessNode(
        symbol="C",
        title="Кодированное\nпредставление",
        description="результат прямого\nпреобразования",
    ),
    ProcessNode(
        symbol="D_h",
        title="Ручное\nдекодирование",
        description="чтение → выбор правила →\nвосстановление → запись",
    ),
    ProcessNode(
        symbol="M′",
        title="Восстановленное\nсообщение",
        description="фактический результат\nпроцедуры",
    ),
)

ERROR_CHANNELS: tuple[ErrorChannel, ...] = (
    ErrorChannel(
        title="Ошибки восприятия",
        details=("неверное чтение", "пропуск элемента"),
        target_symbol="E_h",
    ),
    ErrorChannel(
        title="Ошибки применения правила",
        details=("неверный выбор", "нарушение порядка"),
        target_symbol="E_h",
    ),
    ErrorChannel(
        title="Ошибки фиксации результата",
        details=("замена / вставка", "ошибка записи"),
        target_symbol="C",
    ),
    ErrorChannel(
        title="Ошибки декодирования",
        details=("неверная интерпретация", "ошибка восстановления"),
        target_symbol="D_h",
    ),
)

CONTROL_LOOPS: tuple[ControlLoop, ...] = (
    ControlLoop(
        symbol="K_e",
        stage_symbol="E_h",
        title="Контроль кодирования",
        actions=("обнаружение", "проверка", "исправление"),
    ),
    ControlLoop(
        symbol="K_d",
        stage_symbol="D_h",
        title="Контроль декодирования",
        actions=("обнаружение", "проверка", "исправление"),
    ),
)

EXPECTED_NODE_SYMBOLS: tuple[str, ...] = ("M", "E_h", "C", "D_h", "M′")
EXPECTED_CONTROL_SYMBOLS: tuple[str, ...] = ("K_e", "K_d")


def validate_process_model() -> None:
    """Проверить полноту и методическую согласованность процессной схемы."""

    node_symbols = tuple(node.symbol for node in PROCESS_NODES)
    if node_symbols != EXPECTED_NODE_SYMBOLS:
        raise ValueError(
            "Основная последовательность должна иметь вид M → E_h → C → D_h → M′."
        )

    valid_targets = set(node_symbols)
    if len(ERROR_CHANNELS) < 4:
        raise ValueError("Должно быть задано не менее четырёх каналов ошибок.")

    for channel in ERROR_CHANNELS:
        if channel.target_symbol not in valid_targets:
            raise ValueError(
                f"Канал «{channel.title}» ссылается на неизвестный этап."
            )
        if len(channel.details) < 2:
            raise ValueError(
                f"Для канала «{channel.title}» недостаточно примеров ошибок."
            )

    control_symbols = tuple(loop.symbol for loop in CONTROL_LOOPS)
    if control_symbols != EXPECTED_CONTROL_SYMBOLS:
        raise ValueError("Должны быть заданы контуры K_e и K_d.")

    for loop in CONTROL_LOOPS:
        if loop.stage_symbol not in {"E_h", "D_h"}:
            raise ValueError(
                f"Контур {loop.symbol} должен относиться к кодированию или декодированию."
            )
        if tuple(loop.actions) != ("обнаружение", "проверка", "исправление"):
            raise ValueError(
                f"Контур {loop.symbol} должен включать обнаружение, проверку и исправление."
            )


def _add_arrow(
    axis: plt.Axes,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    linewidth: float = 1.5,
    linestyle: str = "-",
    mutation_scale: float = 14,
    connectionstyle: str = "arc3,rad=0.0",
    zorder: int = 2,
) -> None:
    """Добавить направленную связь между элементами схемы."""

    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            shrinkA=2.5,
            shrinkB=2.5,
            connectionstyle=connectionstyle,
            zorder=zorder,
        )
    )


def _add_node(
    axis: plt.Axes,
    *,
    node: ProcessNode,
    x: float,
    y: float,
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str,
) -> None:
    """Добавить узел основной процессной последовательности."""

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.6,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=4,
    )
    axis.add_patch(box)

    symbol_w = min(0.065, width * 0.45)
    symbol_h = 0.052
    symbol_x = x + (width - symbol_w) / 2
    symbol_y = y + height - 0.068
    symbol_box = FancyBboxPatch(
        (symbol_x, symbol_y),
        symbol_w,
        symbol_h,
        boxstyle="round,pad=0.005,rounding_size=0.012",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor="white",
        zorder=5,
    )
    axis.add_patch(symbol_box)
    axis.text(
        symbol_x + symbol_w / 2,
        symbol_y + symbol_h / 2,
        node.symbol,
        ha="center",
        va="center",
        fontsize=12.2,
        fontweight="bold",
        color=edgecolor,
        zorder=6,
    )

    axis.text(
        x + width / 2,
        y + height * 0.54,
        node.title,
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
        color="#17212B",
        linespacing=1.05,
        zorder=6,
    )
    axis.text(
        x + width / 2,
        y + height * 0.18,
        node.description,
        ha="center",
        va="center",
        fontsize=7.9,
        color="#405464",
        linespacing=1.10,
        zorder=6,
    )


def _add_error_card(
    axis: plt.Axes,
    *,
    channel: ErrorChannel,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    """Добавить карточку канала возникновения ошибки."""

    card = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.009,rounding_size=0.014",
        linewidth=1.2,
        edgecolor="#984B4B",
        facecolor="#F9EEEE",
        zorder=4,
    )
    axis.add_patch(card)
    axis.text(
        x + width / 2,
        y + height * 0.68,
        channel.title,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="#7A3838",
        zorder=5,
    )
    axis.text(
        x + width / 2,
        y + height * 0.27,
        " • ".join(channel.details),
        ha="center",
        va="center",
        fontsize=7.3,
        color="#6B4A4A",
        zorder=5,
    )


def _add_control_card(
    axis: plt.Axes,
    *,
    loop: ControlLoop,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    """Добавить карточку контура обнаружения и исправления ошибок."""

    card = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.01,rounding_size=0.016",
        linewidth=1.35,
        edgecolor="#376C52",
        facecolor="#EAF5EE",
        zorder=4,
    )
    axis.add_patch(card)
    axis.text(
        x + 0.038,
        y + height * 0.67,
        loop.symbol,
        ha="center",
        va="center",
        fontsize=11.0,
        fontweight="bold",
        color="#2E5C46",
        zorder=5,
    )
    axis.text(
        x + width * 0.60,
        y + height * 0.69,
        loop.title,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="#284C3B",
        zorder=5,
    )
    axis.text(
        x + width / 2,
        y + height * 0.28,
        " → ".join(loop.actions),
        ha="center",
        va="center",
        fontsize=7.8,
        color="#3E6652",
        zorder=5,
    )


def build_figure() -> plt.Figure:
    """Построить процессную схему ручного кодирования и декодирования."""

    validate_process_model()
    configure_dissertation_style()

    figure, axis = plt.subplots(figsize=(17.5, 9.4))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    axis.text(
        0.5,
        0.965,
        "Процесс ручного кодирования и декодирования с каналами ошибок и контроля",
        ha="center",
        va="center",
        fontsize=14.2,
        fontweight="bold",
        color="#17212B",
    )
    axis.text(
        0.5,
        0.925,
        "Основная последовательность M → E_h → C → D_h → M′ дополняется ошибками, задержками и корректирующими контурами",
        ha="center",
        va="center",
        fontsize=10.1,
        color="#435566",
        style="italic",
    )

    node_y = 0.48
    node_h = 0.205
    node_widths = (0.13, 0.175, 0.13, 0.175, 0.14)
    node_xs = (0.025, 0.195, 0.405, 0.575, 0.81)
    facecolors = ("#EEF3F7", "#EAF2F8", "#F7F1E8", "#EEF4EC", "#F1EDF7")
    edgecolors = ("#415B6D", "#2E5D7B", "#886A36", "#416B4D", "#69517D")

    for node, x, width, facecolor, edgecolor in zip(
        PROCESS_NODES,
        node_xs,
        node_widths,
        facecolors,
        edgecolors,
        strict=True,
    ):
        _add_node(
            axis,
            node=node,
            x=x,
            y=node_y,
            width=width,
            height=node_h,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )

    for index in range(len(PROCESS_NODES) - 1):
        start = (node_xs[index] + node_widths[index], node_y + node_h / 2)
        end = (node_xs[index + 1], node_y + node_h / 2)
        _add_arrow(
            axis,
            start=start,
            end=end,
            color="#425D70",
            linewidth=2.2,
            mutation_scale=17,
            zorder=3,
        )

    error_y = 0.735
    error_h = 0.12
    error_w = 0.205
    error_xs = (0.105, 0.32, 0.535, 0.75)
    target_centers = {
        "E_h": node_xs[1] + node_widths[1] / 2,
        "C": node_xs[2] + node_widths[2] / 2,
        "D_h": node_xs[3] + node_widths[3] / 2,
    }
    target_tops = {
        "E_h": node_y + node_h,
        "C": node_y + node_h,
        "D_h": node_y + node_h,
    }

    e_h_offsets = (-0.032, 0.032)
    e_h_index = 0
    for channel, x in zip(ERROR_CHANNELS, error_xs, strict=True):
        _add_error_card(
            axis,
            channel=channel,
            x=x,
            y=error_y,
            width=error_w,
            height=error_h,
        )
        target_x = target_centers[channel.target_symbol]
        if channel.target_symbol == "E_h":
            target_x += e_h_offsets[e_h_index]
            e_h_index += 1
        _add_arrow(
            axis,
            start=(x + error_w / 2, error_y),
            end=(target_x, target_tops[channel.target_symbol]),
            color="#A95555",
            linewidth=1.25,
            linestyle="--",
            mutation_scale=12,
            zorder=2,
        )

    control_y = 0.255
    control_h = 0.115
    control_w = 0.245
    control_positions = {
        "E_h": (0.16, control_y),
        "D_h": (0.59, control_y),
    }
    stage_centers = {
        "E_h": node_xs[1] + node_widths[1] / 2,
        "D_h": node_xs[3] + node_widths[3] / 2,
    }

    for loop in CONTROL_LOOPS:
        x, y = control_positions[loop.stage_symbol]
        _add_control_card(
            axis,
            loop=loop,
            x=x,
            y=y,
            width=control_w,
            height=control_h,
        )
        stage_x = stage_centers[loop.stage_symbol]
        _add_arrow(
            axis,
            start=(stage_x - 0.03, node_y),
            end=(x + control_w * 0.36, y + control_h),
            color="#3F765B",
            linewidth=1.35,
            mutation_scale=12,
            connectionstyle="arc3,rad=0.18",
            zorder=2,
        )
        _add_arrow(
            axis,
            start=(x + control_w * 0.68, y + control_h),
            end=(stage_x + 0.03, node_y),
            color="#3F765B",
            linewidth=1.35,
            mutation_scale=12,
            connectionstyle="arc3,rad=-0.18",
            zorder=2,
        )

    delay_specs = (
        ("Δt_e", "задержка кодирования", node_xs[1] + node_widths[1] / 2),
        ("Δt_c", "задержка фиксации / передачи", node_xs[2] + node_widths[2] / 2),
        ("Δt_d", "задержка декодирования", node_xs[3] + node_widths[3] / 2),
    )
    delay_y = 0.405
    for symbol, label, center_x in delay_specs:
        width = 0.15 if symbol != "Δt_c" else 0.18
        delay_box = FancyBboxPatch(
            (center_x - width / 2, delay_y),
            width,
            0.052,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            linewidth=1.0,
            edgecolor="#A0702E",
            facecolor="#FBF2E3",
            zorder=4,
        )
        axis.add_patch(delay_box)
        axis.text(
            center_x,
            delay_y + 0.034,
            symbol,
            ha="center",
            va="center",
            fontsize=9.4,
            fontweight="bold",
            color="#805820",
            zorder=5,
        )
        axis.text(
            center_x,
            delay_y + 0.012,
            label,
            ha="center",
            va="center",
            fontsize=6.7,
            color="#755C38",
            zorder=5,
        )

    quality_x = 0.33
    quality_y = 0.075
    quality_w = 0.34
    quality_h = 0.105
    quality_box = FancyBboxPatch(
        (quality_x, quality_y),
        quality_w,
        quality_h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.5,
        edgecolor="#4F536C",
        facecolor="#F0F1F7",
        zorder=4,
    )
    axis.add_patch(quality_box)
    axis.text(
        0.5,
        quality_y + quality_h * 0.67,
        "Фактический результат выполнения сценария",
        ha="center",
        va="center",
        fontsize=10.1,
        fontweight="bold",
        color="#3D415A",
        zorder=5,
    )
    axis.text(
        0.5,
        quality_y + quality_h * 0.30,
        "d(M, M′), число ошибок, остаточная ошибка, T = Δt_e + Δt_c + Δt_d",
        ha="center",
        va="center",
        fontsize=8.7,
        color="#52566B",
        zorder=5,
    )

    _add_arrow(
        axis,
        start=(node_xs[0] + node_widths[0] / 2, node_y),
        end=(quality_x + quality_w * 0.30, quality_y + quality_h),
        color="#676B82",
        linewidth=1.1,
        linestyle=":",
        mutation_scale=11,
        connectionstyle="arc3,rad=0.12",
        zorder=1,
    )
    _add_arrow(
        axis,
        start=(node_xs[4] + node_widths[4] / 2, node_y),
        end=(quality_x + quality_w * 0.70, quality_y + quality_h),
        color="#676B82",
        linewidth=1.1,
        linestyle=":",
        mutation_scale=11,
        connectionstyle="arc3,rad=-0.12",
        zorder=1,
    )

    axis.text(
        0.5,
        0.018,
        (
            "Красные пунктирные связи показывают места возникновения ошибок; "
            "зелёные контуры — обнаружение, проверку и возможное исправление; "
            "временные затраты входят в фактическую трудоёмкость процесса."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4B5862",
        fontweight="bold",
    )

    return figure


def generate(*, project_root: Path, dpi: int = 300) -> FigureExportResult:
    """Сформировать рисунок 2.2 в форматах PNG и SVG."""

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
            "Сформировать рисунок 2.2 с процессом ручного кодирования "
            "и декодирования в форматах PNG и SVG."
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
    print("Рисунок 2.2 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
