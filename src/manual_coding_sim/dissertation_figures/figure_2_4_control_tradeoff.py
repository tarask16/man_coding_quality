"""Генерация рисунка 2.4 с причинно-функциональной схемой контроля."""

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
FILE_STEM = "figure_2_4_control_tradeoff"


@dataclass(frozen=True, slots=True)
class ControlBranch:
    """Ветвь влияния контрольных процедур на частный критерий качества."""

    code: str
    title: str
    mechanism: tuple[str, ...]
    criterion: str
    direction: str
    interpretation: str


CONTROL_BRANCHES: tuple[ControlBranch, ...] = (
    ControlBranch(
        code="accuracy",
        title="Снижение остаточной ошибки",
        mechanism=(
            "вероятность обнаружения P(D_j) ↑",
            "вероятность исправления P(C_j) ↑",
            "необнаруженные и неисправленные ошибки ↓",
        ),
        criterion="q_acc ↑; q_res ↑",
        direction="benefit",
        interpretation="положительный эффект контроля",
    ),
    ControlBranch(
        code="time",
        title="Дополнительное время",
        mechanism=(
            "число контрольных действий R_c ↑",
            "T_ctrl = Σ t_ctrl,r ↑",
            "риск превышения допустимого времени ↑",
        ),
        criterion="q_time ↓",
        direction="cost",
        interpretation="временная стоимость контроля",
    ),
    ControlBranch(
        code="effort",
        title="Дополнительная трудоёмкость",
        mechanism=(
            "проверки и повторные операции ↑",
            "операционная и когнитивная нагрузка ↑",
            "суммарная трудоёмкость L_ctrl ↑",
        ),
        criterion="q_effort ↓",
        direction="cost",
        interpretation="ресурсная стоимость контроля",
    ),
)

EXPECTED_CODES: tuple[str, ...] = ("accuracy", "time", "effort")
EXPECTED_DIRECTIONS: tuple[str, ...] = ("benefit", "cost", "cost")

FORMULA_RESIDUAL = "p_res,j = p_err,j · [1 − P(D_j) · P(C_j)]"
FORMULA_TIME = "T = T_base + Σ t_ctrl,r"
FORMULA_QUALITY = "Q(A) = Σ w_m · q_m(A)"


def validate_control_tradeoff() -> None:
    """Проверить полноту и направленность причинных ветвей схемы."""

    codes = tuple(branch.code for branch in CONTROL_BRANCHES)
    if codes != EXPECTED_CODES:
        raise ValueError(
            "Схема должна содержать ветви точности, времени и трудоёмкости."
        )

    directions = tuple(branch.direction for branch in CONTROL_BRANCHES)
    if directions != EXPECTED_DIRECTIONS:
        raise ValueError(
            "Контроль должен давать один положительный эффект и две ресурсные стоимости."
        )

    for branch in CONTROL_BRANCHES:
        if len(branch.mechanism) != 3:
            raise ValueError(
                f"Ветвь «{branch.title}» должна содержать три причинных звена."
            )
        if not branch.criterion.strip():
            raise ValueError(
                f"Для ветви «{branch.title}» не указан частный критерий качества."
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
    body_size: float = 8.0,
    linewidth: float = 1.35,
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
        y + height * 0.72,
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
        y + height * 0.35,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=body_color,
        wrap=True,
        linespacing=1.25,
        zorder=zorder + 1,
    )


def _add_arrow(
    axis: plt.Axes,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    linewidth: float = 1.5,
    linestyle: str = "-",
    connectionstyle: str = "arc3,rad=0.0",
    mutation_scale: float = 14,
    zorder: int = 2,
) -> None:
    """Добавить направленную причинную связь."""

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


def _branch_palette(direction: str) -> tuple[str, str, str]:
    """Вернуть цвета фона, границы и связи для ветви."""

    if direction == "benefit":
        return "#EAF6EE", "#2E7D4F", "#2E7D4F"
    return "#FFF3E7", "#B86322", "#B86322"


def build_figure() -> plt.Figure:
    """Построить причинно-функциональную схему влияния контроля."""

    validate_control_tradeoff()
    configure_dissertation_style()

    figure, axis = plt.subplots(figsize=(14.7, 8.55))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    figure.suptitle(
        "Влияние контрольных процедур на частные критерии и интегральное качество",
        fontsize=15.2,
        fontweight="bold",
        y=0.985,
        color="#17212B",
    )
    axis.text(
        0.5,
        0.945,
        "Контроль уменьшает остаточную ошибку, но требует дополнительного времени и труда",
        ha="center",
        va="center",
        fontsize=10.4,
        color="#526575",
    )

    # Исходные параметры сценария.
    _add_box(
        axis,
        x=0.018,
        y=0.665,
        width=0.185,
        height=0.205,
        title="Сценарий применения A_i",
        body=(
            "критичность сообщения\n"
            "допустимое время T_lim\n"
            "состояние оператора O_i\n"
            "условия U_i и веса w_m"
        ),
        facecolor="#EEF3F8",
        edgecolor="#496A85",
        title_size=10.2,
        body_size=8.0,
    )

    _add_box(
        axis,
        x=0.018,
        y=0.365,
        width=0.185,
        height=0.235,
        title="Контрольные процедуры K",
        body=(
            "самопроверка K_self\n"
            "внешний контроль K_ext\n"
            "повтор K_repeat\n"
            "проверка правил K_rule\n"
            "интенсивность и число R_c"
        ),
        facecolor="#E9F1FA",
        edgecolor="#2F6690",
        title_size=10.2,
        body_size=7.7,
    )

    _add_arrow(
        axis,
        start=(0.112, 0.665),
        end=(0.112, 0.605),
        color="#496A85",
        linewidth=1.3,
    )
    axis.text(
        0.122,
        0.635,
        "задаёт ограничения",
        ha="left",
        va="center",
        fontsize=7.6,
        color="#496A85",
    )

    # Ветви влияния контроля.
    y_positions = (0.705, 0.445, 0.185)
    for branch, y in zip(CONTROL_BRANCHES, y_positions, strict=True):
        facecolor, edgecolor, arrow_color = _branch_palette(branch.direction)

        _add_arrow(
            axis,
            start=(0.203, 0.482),
            end=(0.255, y + 0.075),
            color=arrow_color,
            linewidth=1.4,
            connectionstyle="arc3,rad=0.06" if y > 0.45 else "arc3,rad=-0.06",
        )

        _add_box(
            axis,
            x=0.255,
            y=y,
            width=0.205,
            height=0.15,
            title=branch.title,
            body="\n".join(branch.mechanism),
            facecolor=facecolor,
            edgecolor=edgecolor,
            title_size=9.6,
            body_size=7.15,
        )

        criterion_face = "#F1FAF4" if branch.direction == "benefit" else "#FFF8EF"
        _add_box(
            axis,
            x=0.515,
            y=y + 0.012,
            width=0.145,
            height=0.126,
            title="Частный результат",
            body=f"{branch.criterion}\n{branch.interpretation}",
            facecolor=criterion_face,
            edgecolor=edgecolor,
            title_size=8.8,
            body_size=7.25,
        )

        _add_arrow(
            axis,
            start=(0.46, y + 0.075),
            end=(0.515, y + 0.075),
            color=arrow_color,
            linewidth=1.55,
        )

        _add_arrow(
            axis,
            start=(0.66, y + 0.075),
            end=(0.735, 0.53),
            color=arrow_color,
            linewidth=1.5,
            connectionstyle=(
                "arc3,rad=-0.10" if y > 0.55 else "arc3,rad=0.10" if y < 0.3 else "arc3,rad=0.0"
            ),
        )

    # Интегральный результат.
    _add_box(
        axis,
        x=0.735,
        y=0.405,
        width=0.235,
        height=0.25,
        title="Интегральный результат",
        body=(
            f"{FORMULA_QUALITY}\n\n"
            "веса w_m задаются сценарием\n"
            "положительный вклад:\nточность и устойчивость\n"
            "отрицательный вклад:\nвремя и трудоёмкость"
        ),
        facecolor="#F2EEF8",
        edgecolor="#6A4C93",
        title_size=10.8,
        body_size=8.0,
    )

    # Формульные пояснения.
    _add_box(
        axis,
        x=0.255,
        y=0.035,
        width=0.405,
        height=0.105,
        title="Функциональные зависимости",
        body=f"{FORMULA_RESIDUAL}\n{FORMULA_TIME}",
        facecolor="#F7F9FB",
        edgecolor="#7B8C99",
        title_size=8.8,
        body_size=7.8,
    )

    _add_box(
        axis,
        x=0.735,
        y=0.115,
        width=0.235,
        height=0.205,
        title="Методический вывод",
        body=(
            "Наличие контроля не гарантирует\n"
            "монотонного роста Q(A).\n\n"
            "Рациональная интенсивность контроля\n"
            "определяется компромиссом между\n"
            "снижением ошибки и ресурсными затратами."
        ),
        facecolor="#FFFCEB",
        edgecolor="#A07B19",
        title_size=9.6,
        body_size=7.6,
    )

    figure.subplots_adjust(left=0.012, right=0.988, top=0.935, bottom=0.035)
    return figure


def generate(*, project_root: Path, dpi: int = 300) -> FigureExportResult:
    """Сформировать рисунок 2.4 в обязательных форматах PNG и SVG."""

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
            "Сформировать рисунок 2.4 с причинно-функциональной схемой "
            "влияния контрольных процедур."
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

    print("Рисунок 2.4 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
