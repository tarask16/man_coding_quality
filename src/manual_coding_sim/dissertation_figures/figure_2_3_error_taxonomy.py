"""Генерация рисунка 2.3 с классификацией ошибок ручного кодирования."""

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
FILE_STEM = "figure_2_3_error_taxonomy"


@dataclass(frozen=True, slots=True)
class ErrorType:
    """Тип ошибки в процессе ручного кодирования или декодирования."""

    code: str
    title: str
    description: str
    examples: tuple[str, ...]
    group: str


@dataclass(frozen=True, slots=True)
class ModifierGroup:
    """Группа факторов, изменяющих вероятность возникновения ошибок."""

    code: str
    title: str
    factors: tuple[str, ...]


ERROR_TYPES: tuple[ErrorType, ...] = (
    ErrorType(
        code="E_sub",
        title="Замена",
        description="Один элемент сообщения\nзаменён другим.",
        examples=("сходные символы", "неверное кодовое\nсоответствие"),
        group="Элементные ошибки",
    ),
    ErrorType(
        code="E_omit",
        title="Пропуск",
        description="Элемент или операция\nне выполнены.",
        examples=("пропуск символа", "пропуск шага\nинструкции"),
        group="Элементные ошибки",
    ),
    ErrorType(
        code="E_ins",
        title="Вставка",
        description="В результат добавлен\nлишний элемент.",
        examples=("дублирование", "лишняя запись"),
        group="Элементные ошибки",
    ),
    ErrorType(
        code="E_perm",
        title="Перестановка",
        description="Нарушен порядок элементов\nили действий.",
        examples=("смена порядка", "обмен соседних\nэлементов"),
        group="Элементные ошибки",
    ),
    ErrorType(
        code="E_rule",
        title="Ошибочный выбор\nправила",
        description="Применено неверное правило,\nрежим или таблица.",
        examples=("неверная ветвь", "ошибка обращения\nк таблице"),
        group="Процедурные ошибки",
    ),
    ErrorType(
        code="E_ctrl",
        title="Ошибка контроля",
        description="Ошибка не обнаружена либо\nисправлена неверно.",
        examples=("ложное подтверждение", "неверная коррекция"),
        group="Процедурные ошибки",
    ),
)

MODIFIER_GROUPS: tuple[ModifierGroup, ...] = (
    ModifierGroup(
        code="F_proc",
        title="Процедура и сообщение",
        factors=(
            "длина и критичность сообщения",
            "число операций и ветвлений",
            "сложность правил и таблиц",
        ),
    ),
    ModifierGroup(
        code="F_oper",
        title="Оператор",
        factors=(
            "подготовка и навык",
            "внимание и память",
            "утомление и ошибкоопасность",
        ),
    ),
    ModifierGroup(
        code="F_env",
        title="Условия применения",
        factors=(
            "дефицит времени",
            "шум и отвлекающие воздействия",
            "доступность инструкции",
        ),
    ),
    ModifierGroup(
        code="F_ctrl",
        title="Организация контроля",
        factors=(
            "интенсивность проверки",
            "вероятность обнаружения",
            "возможность исправления",
        ),
    ),
)

EXPECTED_ERROR_CODES: tuple[str, ...] = (
    "E_sub",
    "E_omit",
    "E_ins",
    "E_perm",
    "E_rule",
    "E_ctrl",
)
EXPECTED_GROUPS: tuple[str, ...] = ("Элементные ошибки", "Процедурные ошибки")
EXPECTED_MODIFIER_CODES: tuple[str, ...] = ("F_proc", "F_oper", "F_env", "F_ctrl")


def validate_taxonomy() -> None:
    """Проверить полноту классификации и состава факторов-модификаторов."""

    error_codes = tuple(item.code for item in ERROR_TYPES)
    if error_codes != EXPECTED_ERROR_CODES:
        raise ValueError(
            "Классификация должна содержать замену, пропуск, вставку, "
            "перестановку, ошибочный выбор правила и ошибку контроля."
        )

    groups = tuple(dict.fromkeys(item.group for item in ERROR_TYPES))
    if groups != EXPECTED_GROUPS:
        raise ValueError(
            "Типы ошибок должны быть разделены на элементные и процедурные."
        )

    for item in ERROR_TYPES:
        if len(item.examples) < 2:
            raise ValueError(
                f"Для типа ошибки «{item.title}» необходимо не менее двух примеров."
            )
        if not item.description.strip():
            raise ValueError(f"Для типа ошибки «{item.title}» отсутствует описание.")

    modifier_codes = tuple(item.code for item in MODIFIER_GROUPS)
    if modifier_codes != EXPECTED_MODIFIER_CODES:
        raise ValueError(
            "Должны быть заданы модификаторы процедуры, оператора, условий и контроля."
        )

    for group in MODIFIER_GROUPS:
        if len(group.factors) < 3:
            raise ValueError(
                f"Группа факторов «{group.title}» должна содержать не менее трёх факторов."
            )


def _add_arrow(
    axis: plt.Axes,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    linewidth: float = 1.3,
    linestyle: str = "-",
    mutation_scale: float = 13,
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


def _add_error_card(
    axis: plt.Axes,
    *,
    item: ErrorType,
    x: float,
    y: float,
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str,
) -> None:
    """Добавить карточку отдельного типа ошибки."""

    card = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        linewidth=1.35,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=4,
    )
    axis.add_patch(card)

    code_w = min(width * 0.34, 0.056)
    code_h = 0.044
    code_box = FancyBboxPatch(
        (x + 0.012, y + height - code_h - 0.012),
        code_w,
        code_h,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        linewidth=0.9,
        edgecolor=edgecolor,
        facecolor="white",
        zorder=5,
    )
    axis.add_patch(code_box)
    axis.text(
        x + 0.012 + code_w / 2,
        y + height - code_h / 2 - 0.012,
        item.code,
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color=edgecolor,
        zorder=6,
    )

    axis.text(
        x + width / 2,
        y + height * 0.69,
        item.title,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#17212B",
        wrap=True,
        zorder=6,
    )
    axis.text(
        x + width / 2,
        y + height * 0.40,
        item.description,
        ha="center",
        va="center",
        fontsize=6.9,
        color="#405464",
        wrap=True,
        linespacing=1.10,
        zorder=6,
    )
    axis.text(
        x + width / 2,
        y + height * 0.16,
        " • ".join(item.examples),
        ha="center",
        va="center",
        fontsize=6.1,
        color="#586874",
        wrap=True,
        zorder=6,
    )


def _add_modifier_card(
    axis: plt.Axes,
    *,
    group: ModifierGroup,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    """Добавить карточку группы факторов-модификаторов."""

    card = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.009,rounding_size=0.014",
        linewidth=1.15,
        edgecolor="#4F6678",
        facecolor="#EDF3F7",
        zorder=4,
    )
    axis.add_patch(card)
    axis.text(
        x + width / 2,
        y + height * 0.75,
        group.title,
        ha="center",
        va="center",
        fontsize=8.4,
        fontweight="bold",
        color="#304A5E",
        zorder=5,
    )
    axis.text(
        x + width / 2,
        y + height * 0.36,
        "\n".join(f"• {factor}" for factor in group.factors),
        ha="center",
        va="center",
        fontsize=6.8,
        color="#465A68",
        linespacing=1.15,
        zorder=5,
    )


def build_figure() -> plt.Figure:
    """Построить иерархическую схему классификации ошибок."""

    validate_taxonomy()
    configure_dissertation_style()

    figure, axis = plt.subplots(figsize=(18.6, 10.8))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    axis.text(
        0.5,
        0.972,
        "Классификация ошибок ручного кодирования и факторы-модификаторы",
        ha="center",
        va="top",
        fontsize=15.2,
        fontweight="bold",
        color="#17212B",
    )
    axis.text(
        0.5,
        0.932,
        (
            "Тип ошибки определяет форму наблюдаемого отклонения; факторы сценария "
            "изменяют вероятность её возникновения и обнаружения."
        ),
        ha="center",
        va="top",
        fontsize=9.2,
        color="#52616D",
    )

    root_x, root_y, root_w, root_h = 0.345, 0.822, 0.31, 0.075
    root = FancyBboxPatch(
        (root_x, root_y),
        root_w,
        root_h,
        boxstyle="round,pad=0.012,rounding_size=0.020",
        linewidth=1.65,
        edgecolor="#46506B",
        facecolor="#EEF0F8",
        zorder=5,
    )
    axis.add_patch(root)
    axis.text(
        0.5,
        root_y + root_h * 0.63,
        "Ошибки ручного кодирования и декодирования",
        ha="center",
        va="center",
        fontsize=10.8,
        fontweight="bold",
        color="#343B56",
        zorder=6,
    )
    axis.text(
        0.5,
        root_y + root_h * 0.25,
        "E = {E_sub, E_omit, E_ins, E_perm, E_rule, E_ctrl}",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#4A5067",
        zorder=6,
    )

    group_specs = (
        ("Элементные ошибки", 0.07, 0.742, 0.52, "#8C5A3C", "#F8EEE6"),
        ("Процедурные ошибки", 0.63, 0.742, 0.30, "#6C4F82", "#F2ECF7"),
    )
    for title, x, y, width, edgecolor, facecolor in group_specs:
        group_box = FancyBboxPatch(
            (x, y),
            width,
            0.052,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            linewidth=1.2,
            edgecolor=edgecolor,
            facecolor=facecolor,
            zorder=4,
        )
        axis.add_patch(group_box)
        axis.text(
            x + width / 2,
            y + 0.026,
            title,
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color=edgecolor,
            zorder=5,
        )
        _add_arrow(
            axis,
            start=(0.5, root_y),
            end=(x + width / 2, y + 0.052),
            color=edgecolor,
            linewidth=1.15,
            connectionstyle="arc3,rad=0.08" if x < 0.5 else "arc3,rad=-0.08",
            zorder=2,
        )

    card_y = 0.515
    card_h = 0.190
    card_w = 0.145
    card_xs = (0.035, 0.190, 0.345, 0.500, 0.665, 0.820)

    for index, (item, x) in enumerate(zip(ERROR_TYPES, card_xs, strict=True)):
        if item.group == "Элементные ошибки":
            facecolor, edgecolor = "#FBF1E9", "#9B6544"
            group_center = 0.33
        else:
            facecolor, edgecolor = "#F5EFF9", "#73558A"
            group_center = 0.78

        _add_error_card(
            axis,
            item=item,
            x=x,
            y=card_y,
            width=card_w,
            height=card_h,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )
        _add_arrow(
            axis,
            start=(group_center, 0.742),
            end=(x + card_w / 2, card_y + card_h),
            color=edgecolor,
            linewidth=1.0,
            mutation_scale=11,
            connectionstyle=f"arc3,rad={0.04 * (index - 2.5):.2f}",
            zorder=1,
        )

    modifier_title_y = 0.452
    modifier_title = FancyBboxPatch(
        (0.285, modifier_title_y),
        0.43,
        0.045,
        boxstyle="round,pad=0.005,rounding_size=0.012",
        linewidth=1.1,
        edgecolor="#456678",
        facecolor="#EAF2F6",
        zorder=4,
    )
    axis.add_patch(modifier_title)
    axis.text(
        0.5,
        modifier_title_y + 0.0225,
        "Факторы-модификаторы вероятности и обнаружения ошибок",
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color="#315465",
        zorder=5,
    )

    modifier_y = 0.225
    modifier_h = 0.185
    modifier_w = 0.215
    modifier_xs = (0.045, 0.285, 0.525, 0.765)
    for group, x in zip(MODIFIER_GROUPS, modifier_xs, strict=True):
        _add_modifier_card(
            axis,
            group=group,
            x=x,
            y=modifier_y,
            width=modifier_w,
            height=modifier_h,
        )
        _add_arrow(
            axis,
            start=(x + modifier_w / 2, modifier_y + modifier_h),
            end=(0.5, modifier_title_y),
            color="#57798A",
            linewidth=1.0,
            linestyle="--",
            mutation_scale=10,
            connectionstyle="arc3,rad=0.0",
            zorder=1,
        )

    _add_arrow(
        axis,
        start=(0.5, modifier_title_y + 0.045),
        end=(0.5, card_y),
        color="#4E7487",
        linewidth=1.3,
        linestyle="--",
        mutation_scale=12,
        zorder=2,
    )

    formula_x, formula_y, formula_w, formula_h = 0.215, 0.075, 0.57, 0.105
    formula_box = FancyBboxPatch(
        (formula_x, formula_y),
        formula_w,
        formula_h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.35,
        edgecolor="#4F536C",
        facecolor="#F1F2F7",
        zorder=4,
    )
    axis.add_patch(formula_box)
    axis.text(
        0.5,
        formula_y + formula_h * 0.68,
        "Вероятность ошибки в сценарии",
        ha="center",
        va="center",
        fontsize=9.7,
        fontweight="bold",
        color="#3D415A",
        zorder=5,
    )
    axis.text(
        0.5,
        formula_y + formula_h * 0.35,
        "p(E_j | A_i) = f_j(S_i, O_i, U_i, G_i, K_i)",
        ha="center",
        va="center",
        fontsize=10.1,
        color="#464B63",
        zorder=5,
    )
    axis.text(
        0.5,
        0.025,
        (
            "Классификация фиксирует наблюдаемую форму отклонения, но не предполагает "
            "причинной независимости типов ошибок: одна ранняя ошибка может порождать последующие."
        ),
        ha="center",
        va="bottom",
        fontsize=8.4,
        color="#4B5862",
        fontweight="bold",
    )

    return figure


def generate(*, project_root: Path, dpi: int = 300) -> FigureExportResult:
    """Сформировать рисунок 2.3 в форматах PNG и SVG."""

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
            "Сформировать рисунок 2.3 с иерархической классификацией ошибок "
            "и факторами-модификаторами в форматах PNG и SVG."
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
    print("Рисунок 2.3 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
