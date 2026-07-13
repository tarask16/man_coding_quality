"""Генерация рисунка 4.1 с конвейером построения ``LDA_prior``."""

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

OUTPUT_DIR = Path("reports/chapter4/figures")
FILE_STEM = "figure_4_1_lda_pipeline"


@dataclass(frozen=True, slots=True)
class LdaPipelineStage:
    """Этап построения априорной LDA-модели."""

    code: str
    number: int
    title: str
    artifact: str
    details: tuple[str, ...]
    group: str


@dataclass(frozen=True, slots=True)
class PipelineConnection:
    """Направленная связь между элементами LDA-конвейера."""

    source: str
    target: str
    label: str
    kind: str = "main"


STAGES: tuple[LdaPipelineStage, ...] = (
    LdaPipelineStage(
        code="prior_features",
        number=1,
        title="Априорные признаки",
        artifact="prior_features.csv",
        details=(
            "150 сценариев",
            "только признаки prior_*",
            "scenario_id и protocol_id",
        ),
        group="input",
    ),
    LdaPipelineStage(
        code="discretization",
        number=2,
        title="Дискретизация",
        artifact="quantile / 3 bins",
        details=(
            "числовые признаки → интервалы",
            "категории сохраняются",
            "правила фиксируются",
        ),
        group="corpus",
    ),
    LdaPipelineStage(
        code="token_map",
        number=3,
        title="Токенизация",
        artifact="token_map.json",
        details=(
            "признак + уровень → токен",
            "единое кодирование корпуса",
            "воспроизводимая карта",
        ),
        group="corpus",
    ),
    LdaPipelineStage(
        code="corpus_prior",
        number=4,
        title="Априорный корпус",
        artifact="corpus_prior.csv",
        details=(
            "один документ на сценарий",
            "4800 вхождений токенов",
            "hash корпуса фиксируется",
        ),
        group="corpus",
    ),
    LdaPipelineStage(
        code="dictionary",
        number=5,
        title="Словарь и матрица",
        artifact="dictionary.json",
        details=(
            "df_min = 2",
            "df_max_ratio = 0,95",
            "X ∈ N^(150×96)",
        ),
        group="matrix",
    ),
    LdaPipelineStage(
        code="lda_prior",
        number=6,
        title="Обучение LDA_prior",
        artifact="lda_prior.joblib",
        details=(
            "K = 3 латентных фактора",
            "batch, max_iter = 100",
            "random_state = 11",
        ),
        group="model",
    ),
)

EXPECTED_STAGE_CODES: tuple[str, ...] = (
    "prior_features",
    "discretization",
    "token_map",
    "corpus_prior",
    "dictionary",
    "lda_prior",
)

CONNECTIONS: tuple[PipelineConnection, ...] = (
    PipelineConnection("prior_features", "discretization", "X_prior"),
    PipelineConnection("discretization", "token_map", "уровни"),
    PipelineConnection("token_map", "corpus_prior", "токены"),
    PipelineConnection("corpus_prior", "dictionary", "df-фильтр"),
    PipelineConnection("dictionary", "lda_prior", "X: 150×96"),
    PipelineConnection("lda_prior", "theta_prior", "распределения факторов по сценариям", kind="output"),
    PipelineConnection("lda_prior", "topic_word", "распределения токенов по факторам", kind="output"),
)

OUTPUT_ARTIFACTS: tuple[str, ...] = (
    "theta_prior.csv",
    "topic_word.csv",
    "lda_prior_metadata.json",
)

FORBIDDEN_INPUTS: tuple[str, ...] = (
    "fact_features.csv",
    "quality_targets.csv",
    "diagnostic_features.csv",
)

MODEL_FACTS: tuple[str, ...] = (
    "document_count = 150",
    "dictionary_token_count = 96",
    "selected_k = 3",
    "random_state = 11",
)


def validate_lda_pipeline(
    stages: Sequence[LdaPipelineStage] = STAGES,
    connections: Sequence[PipelineConnection] = CONNECTIONS,
) -> None:
    """Проверить полноту и методическую корректность LDA-конвейера."""

    codes = tuple(stage.code for stage in stages)
    if codes != EXPECTED_STAGE_CODES:
        raise ValueError("LDA-конвейер должен содержать шесть этапов в заданном порядке.")

    numbers = tuple(stage.number for stage in stages)
    if numbers != tuple(range(1, 7)):
        raise ValueError("Нумерация этапов должна быть последовательной от 1 до 6.")

    pairs = {(item.source, item.target) for item in connections}
    required_main = set(zip(EXPECTED_STAGE_CODES[:-1], EXPECTED_STAGE_CODES[1:]))
    missing_main = required_main - pairs
    if missing_main:
        missing_text = ", ".join(f"{a} → {b}" for a, b in sorted(missing_main))
        raise ValueError(f"Отсутствуют обязательные связи: {missing_text}.")

    required_outputs = {("lda_prior", "theta_prior"), ("lda_prior", "topic_word")}
    if not required_outputs.issubset(pairs):
        raise ValueError("LDA_prior должна формировать theta_prior и topic_word.")

    forbidden_nodes = {"fact_features", "quality_targets", "diagnostic_features"}
    for connection in connections:
        if connection.source in forbidden_nodes and connection.target in EXPECTED_STAGE_CODES:
            raise ValueError("Фактические и диагностические данные запрещены во входе LDA_prior.")

    for stage in stages:
        if len(stage.details) != 3:
            raise ValueError(f"Этап {stage.code} должен содержать три пояснения.")
        if not stage.artifact.strip():
            raise ValueError(f"Для этапа {stage.code} не указан артефакт.")


def _add_stage_box(
    axis: plt.Axes,
    *,
    stage: LdaPipelineStage,
    x: float,
    y: float,
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str,
) -> None:
    """Добавить основной блок этапа LDA-конвейера."""

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=4,
    )
    axis.add_patch(box)

    number_box = FancyBboxPatch(
        (x + 0.007, y + height - 0.055),
        0.027,
        0.040,
        boxstyle="round,pad=0.003,rounding_size=0.007",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor="white",
        zorder=5,
    )
    axis.add_patch(number_box)
    axis.text(
        x + 0.0205,
        y + height - 0.035,
        str(stage.number),
        ha="center",
        va="center",
        fontsize=8.9,
        fontweight="bold",
        color=edgecolor,
        zorder=6,
    )

    axis.text(
        x + width * 0.59,
        y + height - 0.040,
        stage.title,
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color="#17212B",
        zorder=6,
    )

    detail_y = y + height - 0.103
    for index, detail in enumerate(stage.details):
        axis.text(
            x + 0.012,
            detail_y - index * 0.040,
            f"• {detail}",
            ha="left",
            va="center",
            fontsize=7.05,
            color="#334A5A",
            zorder=6,
        )

    axis.plot(
        [x + 0.011, x + width - 0.011],
        [y + 0.049, y + 0.049],
        color=edgecolor,
        linewidth=0.75,
        alpha=0.55,
        zorder=5,
    )
    axis.text(
        x + width / 2,
        y + 0.024,
        stage.artifact,
        ha="center",
        va="center",
        fontsize=6.75,
        color=edgecolor,
        family="DejaVu Sans Mono",
        zorder=6,
    )


def _add_arrow(
    axis: plt.Axes,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    color: str = "#456A7D",
    linestyle: str = "-",
    label_offset: float = 0.027,
) -> None:
    """Добавить стрелку между блоками с компактной подписью."""

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.4,
        color=color,
        linestyle=linestyle,
        shrinkA=2,
        shrinkB=2,
        zorder=3,
    )
    axis.add_patch(arrow)
    axis.text(
        (start[0] + end[0]) / 2,
        (start[1] + end[1]) / 2 + label_offset,
        label,
        ha="center",
        va="center",
        fontsize=6.35,
        color=color,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7, "alpha": 0.95},
        zorder=7,
    )


def _add_output_box(
    axis: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    artifact: str,
    details: tuple[str, ...],
    edgecolor: str,
) -> None:
    """Добавить блок выходного распределения LDA-модели."""

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        linewidth=1.45,
        edgecolor=edgecolor,
        facecolor="#F3FAF5",
        zorder=4,
    )
    axis.add_patch(box)
    axis.text(
        x + width / 2,
        y + height - 0.037,
        title,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="#183D2A",
        zorder=6,
    )
    axis.text(
        x + width / 2,
        y + height - 0.079,
        artifact,
        ha="center",
        va="center",
        fontsize=7.15,
        family="DejaVu Sans Mono",
        color=edgecolor,
        zorder=6,
    )
    for index, detail in enumerate(details):
        axis.text(
            x + 0.017,
            y + height - 0.122 - index * 0.043,
            f"• {detail}",
            ha="left",
            va="center",
            fontsize=7.0,
            color="#334A5A",
            zorder=6,
        )


def build_figure() -> plt.Figure:
    """Построить блок-схему формирования корпуса и обучения ``LDA_prior``."""

    validate_lda_pipeline()
    configure_dissertation_style()

    figure, axis = plt.subplots(figsize=(19.0, 10.2))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    figure.suptitle(
        "Рисунок 4.1 — Конвейер построения априорной LDA-модели",
        fontsize=15.2,
        fontweight="bold",
        y=0.985,
    )
    axis.text(
        0.5,
        0.935,
        "От априорных признаков сценария к латентному профилю и распределениям токенов",
        ha="center",
        va="center",
        fontsize=10.0,
        color="#465B68",
    )

    band = FancyBboxPatch(
        (0.027, 0.535),
        0.946,
        0.341,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=1.0,
        edgecolor="#C3D1D8",
        facecolor="#FAFCFD",
        zorder=0,
    )
    axis.add_patch(band)
    axis.text(
        0.043,
        0.850,
        "Основной априорный вычислительный контур",
        ha="left",
        va="center",
        fontsize=8.4,
        fontweight="bold",
        color="#526976",
        zorder=2,
    )

    xs = (0.044, 0.199, 0.354, 0.509, 0.664, 0.819)
    y = 0.565
    width = 0.128
    height = 0.250
    colors = {
        "input": ("#EAF2FB", "#34699A"),
        "corpus": ("#F4F0FB", "#7557A6"),
        "matrix": ("#FFF4E8", "#A56623"),
        "model": ("#E9F6EF", "#2F7A50"),
    }

    positions: dict[str, tuple[float, float, float, float]] = {}
    for stage, x in zip(STAGES, xs, strict=True):
        facecolor, edgecolor = colors[stage.group]
        _add_stage_box(
            axis,
            stage=stage,
            x=x,
            y=y,
            width=width,
            height=height,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )
        positions[stage.code] = (x, y, width, height)

    main_connections = [item for item in CONNECTIONS if item.kind == "main"]
    for connection in main_connections:
        sx, sy, sw, sh = positions[connection.source]
        tx, ty, tw, th = positions[connection.target]
        _add_arrow(
            axis,
            start=(sx + sw + 0.004, sy + sh / 2),
            end=(tx - 0.004, ty + th / 2),
            label=connection.label,
        )

    theta_x, topic_x = 0.535, 0.742
    output_y = 0.218
    output_w = 0.185
    output_h = 0.238
    _add_output_box(
        axis,
        x=theta_x,
        y=output_y,
        width=output_w,
        height=output_h,
        title="Латентный профиль сценария",
        artifact="theta_prior.csv",
        details=(
            "θ_prior(A_i) = (θ_i0, θ_i1, θ_i2)",
            "сумма компонент равна 1",
            "150 строк по сценариям",
        ),
        edgecolor="#2F7A50",
    )
    _add_output_box(
        axis,
        x=topic_x,
        y=output_y,
        width=output_w,
        height=output_h,
        title="Структура латентных факторов",
        artifact="topic_word.csv",
        details=(
            "φ_k(token) для каждого фактора",
            "96 весов на фактор",
            "основа интерпретации тем",
        ),
        edgecolor="#2F7A50",
    )

    lda_x, lda_y, lda_w, lda_h = positions["lda_prior"]
    branch_y = 0.500
    axis.plot(
        [lda_x + lda_w / 2, lda_x + lda_w / 2],
        [lda_y - 0.006, branch_y],
        color="#2F7A50",
        linewidth=1.4,
        zorder=3,
    )
    axis.plot(
        [theta_x + output_w / 2, topic_x + output_w / 2],
        [branch_y, branch_y],
        color="#2F7A50",
        linewidth=1.4,
        zorder=3,
    )
    _add_arrow(
        axis,
        start=(theta_x + output_w / 2, branch_y),
        end=(theta_x + output_w / 2, output_y + output_h + 0.003),
        label="document × topic",
        color="#2F7A50",
        label_offset=0.020,
    )
    _add_arrow(
        axis,
        start=(topic_x + output_w / 2, branch_y),
        end=(topic_x + output_w / 2, output_y + output_h + 0.003),
        label="topic × token",
        color="#2F7A50",
        label_offset=0.020,
    )

    guard = FancyBboxPatch(
        (0.045, 0.218),
        0.425,
        0.238,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        linewidth=1.3,
        edgecolor="#A64646",
        facecolor="#FFF5F5",
        zorder=4,
    )
    axis.add_patch(guard)
    axis.text(
        0.2575,
        0.417,
        "LeakageGuard: только априорные данные",
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color="#8D3030",
        zorder=6,
    )
    axis.text(
        0.065,
        0.368,
        "Разрешено:",
        ha="left",
        va="center",
        fontsize=7.6,
        fontweight="bold",
        color="#2F7A50",
        zorder=6,
    )
    axis.text(
        0.132,
        0.368,
        "prior_features.csv",
        ha="left",
        va="center",
        fontsize=7.5,
        family="DejaVu Sans Mono",
        color="#2F7A50",
        zorder=6,
    )
    axis.text(
        0.065,
        0.319,
        "Запрещено:",
        ha="left",
        va="center",
        fontsize=7.6,
        fontweight="bold",
        color="#A64646",
        zorder=6,
    )
    axis.text(
        0.132,
        0.319,
        "fact_features.csv, quality_targets.csv",
        ha="left",
        va="center",
        fontsize=7.2,
        family="DejaVu Sans Mono",
        color="#A64646",
        zorder=6,
    )
    axis.text(
        0.065,
        0.268,
        "Диагностическая модель не изменяет параметры основной LDA_prior.",
        ha="left",
        va="center",
        fontsize=7.3,
        color="#5B4646",
        zorder=6,
    )

    footer = FancyBboxPatch(
        (0.045, 0.055),
        0.882,
        0.082,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        linewidth=1.0,
        edgecolor="#B9C8CF",
        facecolor="#F7F9FA",
        zorder=2,
    )
    axis.add_patch(footer)
    axis.text(
        0.486,
        0.119,
        "Результат этапа: воспроизводимая априорная модель с фиксированными корпусом, словарём, параметрами и хэшами",
        ha="center",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        color="#314A57",
        zorder=4,
    )
    axis.text(
        0.486,
        0.082,
        "θ_prior используется как латентное описание сценария; φ_k(token) — для содержательной интерпретации факторов",
        ha="center",
        va="center",
        fontsize=7.6,
        color="#526976",
        zorder=4,
    )

    axis.text(
        0.955,
        0.104,
        "N = 150\nV = 96\nK = 3",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color="#2F7A50",
        linespacing=1.45,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#F3FAF5",
            "edgecolor": "#7EAD90",
            "linewidth": 1.0,
        },
        zorder=6,
    )

    return figure


def generate(project_root: str | Path = ".", dpi: int = 300) -> FigureExportResult:
    """Сформировать рисунок 4.1 в форматах PNG и SVG."""

    figure = build_figure()
    return export_figure(
        figure,
        project_root=Path(project_root),
        relative_output_dir=OUTPUT_DIR,
        file_stem=FILE_STEM,
        dpi=dpi,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Создать парсер аргументов командной строки."""

    parser = argparse.ArgumentParser(
        description="Генерация рисунка 4.1 с конвейером построения LDA_prior.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Корень проекта, относительно которого сохраняются рисунки.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Разрешение PNG. Рекомендуемое значение — 300 dpi.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить CLI-генерацию рисунка 4.1."""

    args = build_argument_parser().parse_args(argv)
    result = generate(project_root=args.project_root, dpi=args.dpi)
    print("Рисунок 4.1 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
