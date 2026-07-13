"""Генерация рисунка 2.1 со структурой сценария применения."""

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
FILE_STEM = "figure_2_1_scenario_structure"


@dataclass(frozen=True, slots=True)
class ScenarioComponent:
    """Описание компонента сценария и его априорных признаков."""

    symbol: str
    title: str
    parameters: tuple[str, ...]
    feature_set: str
    feature_summary: str


SCENARIO_COMPONENTS: tuple[ScenarioComponent, ...] = (
    ScenarioComponent(
        symbol="Sᵢ",
        title="Средство\nкодирования",
        parameters=(
            "правила и таблицы",
            "число операций",
            "ветвления и режимы",
            "сложность инструкции",
            "промежуточные записи",
        ),
        feature_set="X_S",
        feature_summary="структурная и процедурная\nсложность средства",
    ),
    ScenarioComponent(
        symbol="Oᵢ",
        title="Оператор",
        parameters=(
            "уровень подготовки",
            "опыт применения",
            "внимание и утомление",
            "ожидаемая скорость",
            "ошибкоопасность",
        ),
        feature_set="X_O",
        feature_summary="априорный профиль\nоператора",
    ),
    ScenarioComponent(
        symbol="Uᵢ",
        title="Условия\nприменения",
        parameters=(
            "доступное время",
            "внешние помехи",
            "доступность инструкции",
            "дополнительная нагрузка",
            "режим и стресс",
        ),
        feature_set="X_U",
        feature_summary="временные и средовые\nограничения",
    ),
    ScenarioComponent(
        symbol="Gᵢ",
        title="Класс\nсообщений",
        parameters=(
            "длина сообщения",
            "структура элементов",
            "критичность",
            "повторяемость",
            "разнообразие",
        ),
        feature_set="X_G",
        feature_summary="структурные свойства\nсообщений",
    ),
    ScenarioComponent(
        symbol="Kᵢ",
        title="Контрольные\nпроцедуры",
        parameters=(
            "самопроверка",
            "внешний контроль",
            "число проверок",
            "обнаружение ошибок",
            "временная стоимость",
        ),
        feature_set="X_K",
        feature_summary="ожидаемая интенсивность\nи эффективность контроля",
    ),
)

EXPECTED_SYMBOLS: tuple[str, ...] = ("Sᵢ", "Oᵢ", "Uᵢ", "Gᵢ", "Kᵢ")
EXPECTED_FEATURE_SETS: tuple[str, ...] = ("X_S", "X_O", "X_U", "X_G", "X_K")


def validate_scenario_structure() -> None:
    """Проверить полноту пятикомпонентного описания сценария."""

    if len(SCENARIO_COMPONENTS) != 5:
        raise ValueError("Сценарий должен содержать ровно пять компонентов.")

    symbols = tuple(component.symbol for component in SCENARIO_COMPONENTS)
    if symbols != EXPECTED_SYMBOLS:
        raise ValueError(
            "Компоненты должны следовать в порядке S_i, O_i, U_i, G_i, K_i."
        )

    feature_sets = tuple(component.feature_set for component in SCENARIO_COMPONENTS)
    if feature_sets != EXPECTED_FEATURE_SETS:
        raise ValueError(
            "Каждому компоненту должно соответствовать подмножество X_S…X_K."
        )

    for component in SCENARIO_COMPONENTS:
        if len(component.parameters) < 4:
            raise ValueError(
                f"Для компонента {component.symbol} задано недостаточно параметров."
            )
        if len(set(component.parameters)) != len(component.parameters):
            raise ValueError(
                f"Для компонента {component.symbol} обнаружены повторяющиеся параметры."
            )
        if not component.feature_summary.strip():
            raise ValueError(
                f"Для компонента {component.symbol} отсутствует описание признаков."
            )


def _add_arrow(
    axis: plt.Axes,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#607789",
    linewidth: float = 1.35,
    linestyle: str = "-",
    mutation_scale: float = 12,
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
            shrinkA=2.0,
            shrinkB=2.0,
            connectionstyle="arc3,rad=0.0",
            zorder=2,
        )
    )


def _add_component_card(
    axis: plt.Axes,
    *,
    component: ScenarioComponent,
    x: float,
    y: float,
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str,
) -> None:
    """Добавить карточку компонента сценария с параметрами и признаками."""

    card = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=3,
    )
    axis.add_patch(card)

    symbol_width = min(0.067, width * 0.40)
    symbol_height = 0.055
    symbol_x = x + (width - symbol_width) / 2
    symbol_y = y + height - 0.076
    symbol_box = FancyBboxPatch(
        (symbol_x, symbol_y),
        symbol_width,
        symbol_height,
        boxstyle="round,pad=0.006,rounding_size=0.014",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor="white",
        zorder=4,
    )
    axis.add_patch(symbol_box)
    axis.text(
        symbol_x + symbol_width / 2,
        symbol_y + symbol_height / 2,
        component.symbol,
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color=edgecolor,
        zorder=5,
    )

    axis.text(
        x + width / 2,
        y + height - 0.123,
        component.title,
        ha="center",
        va="top",
        fontsize=9.8,
        fontweight="bold",
        color="#17212B",
        linespacing=1.05,
        zorder=5,
    )

    parameter_text = "\n".join(f"• {item}" for item in component.parameters)
    axis.text(
        x + 0.018,
        y + height - 0.205,
        parameter_text,
        ha="left",
        va="top",
        fontsize=8.35,
        color="#354B5B",
        linespacing=1.28,
        zorder=5,
    )

    feature_height = 0.105
    feature_y = y + 0.018
    feature_box = FancyBboxPatch(
        (x + 0.014, feature_y),
        width - 0.028,
        feature_height,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor="white",
        alpha=0.96,
        zorder=4,
    )
    axis.add_patch(feature_box)
    axis.text(
        x + width / 2,
        feature_y + feature_height * 0.68,
        component.feature_set,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=edgecolor,
        zorder=5,
    )
    axis.text(
        x + width / 2,
        feature_y + feature_height * 0.30,
        component.feature_summary,
        ha="center",
        va="center",
        fontsize=7.35,
        color="#435566",
        linespacing=1.05,
        zorder=5,
    )


def build_figure() -> plt.Figure:
    """Построить структурную схему сценария и априорного описания."""

    validate_scenario_structure()
    configure_dissertation_style()

    figure, axis = plt.subplots(figsize=(17.4, 9.4))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    axis.text(
        0.5,
        0.965,
        "Структура сценария применения и формирование априорного описания",
        ha="center",
        va="center",
        fontsize=14.2,
        fontweight="bold",
        color="#17212B",
    )
    axis.text(
        0.5,
        0.925,
        "Сценарий Aᵢ = {Sᵢ, Oᵢ, Uᵢ, Gᵢ, Kᵢ} задаётся до фактического выполнения ручной процедуры",
        ha="center",
        va="center",
        fontsize=10.3,
        color="#435566",
        style="italic",
    )

    scenario_x = 0.31
    scenario_y = 0.815
    scenario_w = 0.38
    scenario_h = 0.07
    scenario_box = FancyBboxPatch(
        (scenario_x, scenario_y),
        scenario_w,
        scenario_h,
        boxstyle="round,pad=0.01,rounding_size=0.018",
        linewidth=1.55,
        edgecolor="#344E62",
        facecolor="#EEF3F6",
        zorder=4,
    )
    axis.add_patch(scenario_box)
    axis.text(
        scenario_x + scenario_w / 2,
        scenario_y + scenario_h / 2,
        "Сценарий применения  Aᵢ = {Sᵢ, Oᵢ, Uᵢ, Gᵢ, Kᵢ}",
        ha="center",
        va="center",
        fontsize=11.7,
        fontweight="bold",
        color="#263D4D",
        zorder=5,
    )

    left_margin = 0.025
    gap = 0.012
    card_width = (0.95 - 4 * gap) / 5
    card_height = 0.46
    card_y = 0.285
    xs = tuple(left_margin + i * (card_width + gap) for i in range(5))

    facecolors = ("#EAF2F8", "#EDF5EE", "#FAF3E6", "#F2EDF8", "#F8EEEE")
    edgecolors = ("#2E5D7B", "#3F6B50", "#8A6A32", "#6B527F", "#8A4D4D")

    branch_y = 0.785
    axis.plot(
        [xs[0] + card_width / 2, xs[-1] + card_width / 2],
        [branch_y, branch_y],
        color="#718696",
        linewidth=1.25,
        zorder=1,
    )
    _add_arrow(
        axis,
        start=(0.5, scenario_y),
        end=(0.5, branch_y),
        color="#536D80",
        linewidth=1.5,
    )

    for component, x, facecolor, edgecolor in zip(
        SCENARIO_COMPONENTS,
        xs,
        facecolors,
        edgecolors,
        strict=True,
    ):
        _add_arrow(
            axis,
            start=(x + card_width / 2, branch_y),
            end=(x + card_width / 2, card_y + card_height),
            color="#718696",
            linewidth=1.15,
        )
        _add_component_card(
            axis,
            component=component,
            x=x,
            y=card_y,
            width=card_width,
            height=card_height,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )

    merge_y = 0.225
    merge_left = xs[0] + card_width / 2
    merge_right = xs[-1] + card_width / 2
    axis.plot(
        [merge_left, merge_right],
        [merge_y, merge_y],
        color="#607789",
        linewidth=1.3,
        zorder=1,
    )
    for x in xs:
        _add_arrow(
            axis,
            start=(x + card_width / 2, card_y),
            end=(x + card_width / 2, merge_y),
            color="#607789",
            linewidth=1.15,
        )

    output_x = 0.22
    output_y = 0.055
    output_w = 0.56
    output_h = 0.115
    output_box = FancyBboxPatch(
        (output_x, output_y),
        output_w,
        output_h,
        boxstyle="round,pad=0.014,rounding_size=0.022",
        linewidth=1.6,
        edgecolor="#315E48",
        facecolor="#EAF4ED",
        zorder=4,
    )
    axis.add_patch(output_box)
    _add_arrow(
        axis,
        start=(0.5, merge_y),
        end=(0.5, output_y + output_h),
        color="#315E48",
        linewidth=1.7,
        mutation_scale=14,
    )
    axis.text(
        0.5,
        output_y + output_h * 0.67,
        "Априорное описание сценария",
        ha="center",
        va="center",
        fontsize=11.3,
        fontweight="bold",
        color="#284C3B",
        zorder=5,
    )
    axis.text(
        0.5,
        output_y + output_h * 0.31,
        "X_prior(Aᵢ) = X_S ∪ X_O ∪ X_U ∪ X_G ∪ X_K",
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color="#284C3B",
        zorder=5,
    )

    axis.text(
        0.5,
        0.018,
        (
            "В X_prior включаются только сведения, известные до кодирования и декодирования; "
            "фактические ошибки, время и результаты контроля относятся к X_fact и Y_fact."
        ),
        ha="center",
        va="bottom",
        fontsize=8.7,
        color="#6E3F3F",
        fontweight="bold",
    )

    return figure


def generate(*, project_root: Path, dpi: int = 300) -> FigureExportResult:
    """Сформировать рисунок 2.1 в форматах PNG и SVG."""

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
            "Сформировать рисунок 2.1 со структурой сценария применения "
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
    print("Рисунок 2.1 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
