"""Генерация рисунка 2.5 о разделении данных и токенизации X_prior."""

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
FILE_STEM = "figure_2_5_information_sets_and_tokenization"


@dataclass(frozen=True, slots=True)
class InformationSet:
    """Информационное множество и его допустимость для априорной модели."""

    code: str
    symbol: str
    title: str
    examples: tuple[str, ...]
    allowed_for_prior: bool


@dataclass(frozen=True, slots=True)
class TransformationStep:
    """Этап преобразования априорного описания в документ корпуса."""

    code: str
    title: str
    description: tuple[str, ...]
    example: str


INFORMATION_SETS: tuple[InformationSet, ...] = (
    InformationSet(
        code="prior",
        symbol="X_prior",
        title="Априорные признаки",
        examples=(
            "структура средства и процедуры",
            "подготовка и внимание оператора",
            "условия, сообщения и контроль",
            "признаки prior_*",
        ),
        allowed_for_prior=True,
    ),
    InformationSet(
        code="fact",
        symbol="X_fact",
        title="Фактические признаки",
        examples=(
            "фактическое время выполнения",
            "ошибки, повторы и отказы",
            "результаты срабатывания контроля",
            "признаки fact_* и actual_*",
        ),
        allowed_for_prior=False,
    ),
    InformationSet(
        code="target",
        symbol="Y_fact",
        title="Фактические показатели",
        examples=(
            "q_acc, q_time, q_effort",
            "q_res, q_rep, q_fit",
            "integral_quality",
            "quality_class",
        ),
        allowed_for_prior=False,
    ),
)

TRANSFORMATION_STEPS: tuple[TransformationStep, ...] = (
    TransformationStep(
        code="discretization",
        title="Дискретизация",
        description=(
            "числовые признаки → интервалы",
            "категориальные признаки → состояния",
            "правила фиксируются заранее",
        ),
        example="quantile bins: low / mid / high",
    ),
    TransformationStep(
        code="tokenization",
        title="Токенизация",
        description=(
            "признак и состояние объединяются",
            "каждый токен сохраняет источник",
            "порядок элементов не используется",
        ),
        example="prior_attention__level_low",
    ),
    TransformationStep(
        code="document",
        title="Документ сценария d_i",
        description=(
            "набор токенов одного сценария",
            "частотное представление bag-of-words",
            "идентификаторы сценария сохраняются",
        ),
        example="d_i = {t_i1, t_i2, …, t_im}",
    ),
)

EXPECTED_SET_CODES: tuple[str, ...] = ("prior", "fact", "target")
EXPECTED_STEP_CODES: tuple[str, ...] = (
    "discretization",
    "tokenization",
    "document",
)
FORBIDDEN_SOURCE_CODES: tuple[str, ...] = ("fact", "target")
FORMULA_CORPUS = "D_prior = {d_i | d_i = Tokenize(Discretize(X_prior(A_i)))}"
LEAKAGE_RULE = "X_fact ∪ Y_fact ↛ d_i"


def validate_information_flow() -> None:
    """Проверить полноту множеств и отсутствие фактических входов в d_i."""

    set_codes = tuple(item.code for item in INFORMATION_SETS)
    if set_codes != EXPECTED_SET_CODES:
        raise ValueError(
            "Схема должна содержать X_prior, X_fact и Y_fact в заданном порядке."
        )

    allowed_codes = tuple(
        item.code for item in INFORMATION_SETS if item.allowed_for_prior
    )
    if allowed_codes != ("prior",):
        raise ValueError(
            "В априорный документ разрешено включать только признаки X_prior."
        )

    step_codes = tuple(step.code for step in TRANSFORMATION_STEPS)
    if step_codes != EXPECTED_STEP_CODES:
        raise ValueError(
            "Преобразование должно включать дискретизацию, токенизацию и документ."
        )

    for item in INFORMATION_SETS:
        if len(item.examples) < 3:
            raise ValueError(
                f"Для множества {item.symbol} недостаточно содержательных примеров."
            )

    for step in TRANSFORMATION_STEPS:
        if len(step.description) != 3 or not step.example.strip():
            raise ValueError(
                f"Этап «{step.title}» должен иметь три пояснения и пример."
            )


def _add_box(
    axis: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str,
    title_color: str = "#17212B",
    body_color: str = "#405464",
    title_size: float = 10.0,
    body_size: float = 7.8,
    linewidth: float = 1.4,
    zorder: int = 4,
) -> None:
    """Добавить скруглённый информационный блок."""

    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=zorder,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height * 0.77,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
        wrap=True,
        zorder=zorder + 1,
    )
    axis.text(
        x + width / 2,
        y + height * 0.39,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=body_color,
        wrap=True,
        linespacing=1.23,
        zorder=zorder + 1,
    )


def _add_arrow(
    axis: plt.Axes,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    linewidth: float = 1.6,
    linestyle: str = "-",
    connectionstyle: str = "arc3,rad=0.0",
    mutation_scale: float = 14,
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


def _add_forbidden_link(
    axis: plt.Axes,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    cross: tuple[float, float],
    label: str,
    rad: float,
) -> None:
    """Добавить перечёркнутую связь, запрещённую методикой."""

    color = "#B3261E"
    _add_arrow(
        axis,
        start=start,
        end=end,
        color=color,
        linewidth=1.45,
        linestyle="--",
        connectionstyle=f"arc3,rad={rad}",
        mutation_scale=12,
        zorder=1,
    )
    axis.plot(
        [cross[0] - 0.012, cross[0] + 0.012],
        [cross[1] - 0.018, cross[1] + 0.018],
        color=color,
        linewidth=2.6,
        solid_capstyle="round",
        zorder=5,
    )
    axis.plot(
        [cross[0] - 0.012, cross[0] + 0.012],
        [cross[1] + 0.018, cross[1] - 0.018],
        color=color,
        linewidth=2.6,
        solid_capstyle="round",
        zorder=5,
    )
    axis.text(
        cross[0],
        cross[1] - 0.045,
        label,
        ha="center",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color=color,
        zorder=6,
    )


def build_figure() -> plt.Figure:
    """Построить схему разделения данных и токенизации априорных признаков."""

    validate_information_flow()
    configure_dissertation_style()

    figure, axis = plt.subplots(figsize=(15.0, 8.7))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    figure.suptitle(
        "Разделение информационных множеств и формирование документа LDA_prior",
        fontsize=15.2,
        fontweight="bold",
        y=0.985,
        color="#17212B",
    )
    axis.text(
        0.5,
        0.946,
        "Фактические признаки и целевые показатели отделены от априорного контура",
        ha="center",
        va="center",
        fontsize=10.3,
        color="#526575",
    )

    # Верхний уровень: три информационных множества.
    set_x = {"prior": 0.035, "fact": 0.365, "target": 0.695}
    palette = {
        "prior": ("#EAF4FB", "#2F6690"),
        "fact": ("#F5F5F5", "#7B8791"),
        "target": ("#FFF2F0", "#B05A4A"),
    }
    for item in INFORMATION_SETS:
        facecolor, edgecolor = palette[item.code]
        status = (
            "РАЗРЕШЕНО для LDA_prior"
            if item.allowed_for_prior
            else "ТОЛЬКО для внешней проверки"
        )
        body = "\n".join(item.examples) + f"\n\n{status}"
        _add_box(
            axis,
            x=set_x[item.code],
            y=0.67,
            width=0.27,
            height=0.225,
            title=f"{item.symbol} — {item.title}",
            body=body,
            facecolor=facecolor,
            edgecolor=edgecolor,
            title_size=9.9,
            body_size=7.45,
        )

    # Разрешённый конвейер преобразования X_prior.
    axis.text(
        0.5,
        0.615,
        "Разрешённый путь формирования документа корпуса",
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color="#2F6690",
    )

    step_positions = {
        "discretization": (0.065, 0.35),
        "tokenization": (0.355, 0.35),
        "document": (0.645, 0.35),
    }
    step_colors = {
        "discretization": ("#EEF6FC", "#3D7EA6"),
        "tokenization": ("#EEF8F3", "#3C8D63"),
        "document": ("#F2EEF8", "#6A4C93"),
    }

    _add_arrow(
        axis,
        start=(0.17, 0.67),
        end=(0.17, 0.56),
        color="#2F6690",
        linewidth=1.9,
    )
    axis.text(
        0.182,
        0.61,
        "единственный допустимый вход",
        ha="left",
        va="center",
        fontsize=7.5,
        color="#2F6690",
    )

    for step in TRANSFORMATION_STEPS:
        x, y = step_positions[step.code]
        facecolor, edgecolor = step_colors[step.code]
        _add_box(
            axis,
            x=x,
            y=y,
            width=0.25,
            height=0.205,
            title=step.title,
            body="\n".join(step.description) + f"\n\nПример: {step.example}",
            facecolor=facecolor,
            edgecolor=edgecolor,
            title_size=10.0,
            body_size=7.25,
        )

    _add_arrow(
        axis,
        start=(0.315, 0.452),
        end=(0.355, 0.452),
        color="#456A7F",
        linewidth=1.8,
    )
    _add_arrow(
        axis,
        start=(0.605, 0.452),
        end=(0.645, 0.452),
        color="#456A7F",
        linewidth=1.8,
    )

    # Запрещённые связи фактических данных с документом.
    _add_forbidden_link(
        axis,
        start=(0.50, 0.67),
        end=(0.34, 0.53),
        cross=(0.43, 0.595),
        label="запрещено",
        rad=0.08,
    )
    _add_forbidden_link(
        axis,
        start=(0.83, 0.67),
        end=(0.60, 0.53),
        cross=(0.72, 0.595),
        label="запрещено",
        rad=0.10,
    )

    # Нижний уровень: аудит и корпус.
    _add_box(
        axis,
        x=0.065,
        y=0.095,
        width=0.35,
        height=0.175,
        title="LeakageGuard и аудит априорности",
        body=(
            "проверка имён колонок и источников\n"
            "запрет fact_*, actual_*, target_* и q_*\n"
            f"правило: {LEAKAGE_RULE}"
        ),
        facecolor="#FFF9E8",
        edgecolor="#A07B19",
        title_size=9.6,
        body_size=7.5,
    )
    _add_arrow(
        axis,
        start=(0.24, 0.27),
        end=(0.24, 0.35),
        color="#A07B19",
        linewidth=1.45,
        linestyle=":",
        mutation_scale=12,
    )

    _add_box(
        axis,
        x=0.47,
        y=0.095,
        width=0.465,
        height=0.175,
        title="Итоговый корпус априорных документов",
        body=(
            f"{FORMULA_CORPUS}\n"
            "каждый документ связан со scenario_id и protocol_id\n"
            "корпус D_prior используется для обучения только LDA_prior"
        ),
        facecolor="#F5F2FA",
        edgecolor="#6A4C93",
        title_size=9.6,
        body_size=7.35,
    )
    _add_arrow(
        axis,
        start=(0.77, 0.35),
        end=(0.77, 0.27),
        color="#6A4C93",
        linewidth=1.8,
    )

    axis.text(
        0.5,
        0.045,
        "Методическое следствие: внешняя проверка Q_pred по X_fact и Y_fact выполняется после построения априорного профиля и не изменяет d_i.",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#4B5F6C",
    )

    figure.subplots_adjust(left=0.012, right=0.988, top=0.935, bottom=0.035)
    return figure


def generate(*, project_root: Path, dpi: int = 300) -> FigureExportResult:
    """Сформировать рисунок 2.5 в обязательных форматах PNG и SVG."""

    figure = build_figure()
    return export_figure(
        figure,
        project_root=project_root,
        relative_output_dir=OUTPUT_DIR,
        file_stem=FILE_STEM,
        dpi=dpi,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Создать парсер аргументов командной строки."""

    parser = argparse.ArgumentParser(
        description=(
            "Сформировать рисунок 2.5 с разделением X_prior, X_fact, Y_fact "
            "и конвейером токенизации априорных признаков."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Корень проекта, относительно которого сохраняются отчёты.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Разрешение PNG. Значение должно быть не ниже 150 dpi.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить генерацию рисунка из командной строки."""

    args = build_argument_parser().parse_args(argv)
    result = generate(project_root=args.project_root, dpi=args.dpi)

    print("Рисунок 2.5 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
