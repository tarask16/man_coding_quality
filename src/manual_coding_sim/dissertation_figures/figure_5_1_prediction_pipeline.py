"""Генерация рисунка 5.1 с конвейером построения априорного прогноза."""

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

OUTPUT_DIR = Path("reports/chapter5/figures")
FILE_STEM = "figure_5_1_prediction_pipeline"


@dataclass(frozen=True, slots=True)
class PredictionStage:
    """Этап построения априорного интегрального прогноза."""

    code: str
    number: int
    title: str
    artifact: str
    details: tuple[str, ...]
    formula: str
    group: str


@dataclass(frozen=True, slots=True)
class PredictionConnection:
    """Направленная связь между этапами прогнозного конвейера."""

    source: str
    target: str
    label: str
    kind: str = "main"


STAGES: tuple[PredictionStage, ...] = (
    PredictionStage(
        code="inputs",
        number=1,
        title="Загрузка априорных входов",
        artifact="prior_features.csv + theta_prior.csv",
        details=(
            "150 сценариев",
            "ключи scenario_id, protocol_id",
            "X_prior и θ_prior согласованы",
        ),
        formula="X_prior(Aᵢ),  θ_prior(Aᵢ)",
        group="input",
    ),
    PredictionStage(
        code="leakage_guard",
        number=2,
        title="LeakageGuard",
        artifact="chapter5_leakage_report.json",
        details=(
            "разрешены только prior_* и θ₀–θ₂",
            "запрещённые колонки: 0",
            "is_safe = true",
        ),
        formula="X_fact ∪ Y_fact ↛ Q_pred",
        group="safety",
    ),
    PredictionStage(
        code="normalization",
        number=3,
        title="Направленная нормировка",
        artifact="normalized_prior_features.csv",
        details=(
            "min–max в диапазон [0; 1]",
            "lower_is_better инвертируется",
            "постоянные признаки → 1,0",
        ),
        formula="x_norm = N(x; direction)",
        group="transform",
    ),
    PredictionStage(
        code="latent_component",
        number=4,
        title="Латентная компонента",
        artifact="latent_quality_component.csv",
        details=(
            "d = (−1, −1, +1)",
            "процедурная нагрузка и риск снижают",
            "благоприятные условия повышают",
        ),
        formula="q_latent = (Σ dₖθₖ + 1) / 2",
        group="latent",
    ),
    PredictionStage(
        code="partial_criteria",
        number=5,
        title="Частные критерии",
        artifact="q_pred_components.csv",
        details=(
            "q_acc, q_time, q_effort",
            "q_res, q_rep, q_fit",
            "X_prior_norm + q_latent",
        ),
        formula="qⱼ_pred = αⱼBⱼ(X_norm) + (1−αⱼ)q_latent",
        group="prediction",
    ),
    PredictionStage(
        code="integral_quality",
        number=6,
        title="Интегральный индекс",
        artifact="q_pred.csv",
        details=(
            "6 частных критериев",
            "равные веса ≈ 1/6",
            "Q_pred ∈ [0; 1]",
        ),
        formula="Q_pred = Σ wⱼ qⱼ_pred",
        group="prediction",
    ),
    PredictionStage(
        code="uncertainty",
        number=7,
        title="Диагностика неопределённости",
        artifact="prediction_uncertainty.csv",
        details=(
            "энтропия θ, устойчивость LDA, входы",
            "δ = 0,15;  r = δ · uncertainty",
            "диагностический, не калиброванный интервал",
        ),
        formula="[Q_pred − r; Q_pred + r]",
        group="uncertainty",
    ),
    PredictionStage(
        code="acceptance",
        number=8,
        title="Финальная приёмка",
        artifact="chapter5_acceptance_report.json",
        details=(
            "18 обязательных проверок",
            "все выходы: 150 строк",
            "accepted = true",
        ),
        formula="accepted = ∧ checks",
        group="acceptance",
    ),
)

EXPECTED_STAGE_CODES: tuple[str, ...] = (
    "inputs",
    "leakage_guard",
    "normalization",
    "latent_component",
    "partial_criteria",
    "integral_quality",
    "uncertainty",
    "acceptance",
)

CONNECTIONS: tuple[PredictionConnection, ...] = (
    PredictionConnection("inputs", "leakage_guard", "входные таблицы"),
    PredictionConnection("leakage_guard", "normalization", "без утечки"),
    PredictionConnection("normalization", "latent_component", "X_prior_norm + θ"),
    PredictionConnection("latent_component", "partial_criteria", "q_latent"),
    PredictionConnection("partial_criteria", "integral_quality", "qⱼ_pred"),
    PredictionConnection("integral_quality", "uncertainty", "Q_pred"),
    PredictionConnection("uncertainty", "acceptance", "Q_pred и интервал"),
)

FORBIDDEN_INPUTS: tuple[str, ...] = (
    "fact_features.csv",
    "quality_targets.csv",
    "theta_diag.csv",
    "theta_full.csv",
)

OUTPUT_ARTIFACTS: tuple[str, ...] = (
    "normalized_prior_features.csv",
    "latent_quality_component.csv",
    "q_pred_components.csv",
    "q_pred.csv",
    "prediction_uncertainty.csv",
    "chapter5_acceptance_report.json",
)

ACCEPTANCE_CHECK_COUNT = 18
EXPECTED_ROW_COUNT = 150


def validate_prediction_pipeline(
    stages: Sequence[PredictionStage] = STAGES,
    connections: Sequence[PredictionConnection] = CONNECTIONS,
) -> None:
    """Проверить полноту и методическую корректность прогнозного конвейера."""

    codes = tuple(stage.code for stage in stages)
    if codes != EXPECTED_STAGE_CODES:
        raise ValueError("Прогнозный конвейер должен содержать восемь этапов в заданном порядке.")

    numbers = tuple(stage.number for stage in stages)
    if numbers != tuple(range(1, 9)):
        raise ValueError("Нумерация этапов должна быть последовательной от 1 до 8.")

    pairs = {(item.source, item.target) for item in connections}
    required_pairs = set(zip(EXPECTED_STAGE_CODES[:-1], EXPECTED_STAGE_CODES[1:]))
    missing = required_pairs - pairs
    if missing:
        joined = ", ".join(f"{source} → {target}" for source, target in sorted(missing))
        raise ValueError(f"Отсутствуют обязательные связи: {joined}.")

    forbidden_nodes = {"fact_features", "quality_targets", "theta_diag", "theta_full"}
    protected_targets = set(EXPECTED_STAGE_CODES[1:-1])
    for connection in connections:
        if connection.source in forbidden_nodes and connection.target in protected_targets:
            raise ValueError("Фактические или диагностические данные запрещены в Q_pred-конвейере.")

    for stage in stages:
        if len(stage.details) != 3:
            raise ValueError(f"Этап {stage.code} должен содержать три пояснения.")
        if not stage.artifact.strip() or not stage.formula.strip():
            raise ValueError(f"Для этапа {stage.code} не указаны артефакт или формула.")


def _add_stage_box(
    axis: plt.Axes,
    *,
    stage: PredictionStage,
    x: float,
    y: float,
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str,
) -> None:
    """Добавить блок этапа прогнозного конвейера."""

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.35,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=4,
    )
    axis.add_patch(box)

    number_box = FancyBboxPatch(
        (x + 0.006, y + height - 0.052),
        0.026,
        0.038,
        boxstyle="round,pad=0.003,rounding_size=0.006",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor="white",
        zorder=5,
    )
    axis.add_patch(number_box)
    axis.text(
        x + 0.019,
        y + height - 0.033,
        str(stage.number),
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=edgecolor,
        zorder=6,
    )

    axis.text(
        x + width * 0.58,
        y + height - 0.033,
        stage.title,
        ha="center",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color="#17212B",
        zorder=6,
    )

    axis.text(
        x + width / 2,
        y + height - 0.076,
        stage.formula,
        ha="center",
        va="center",
        fontsize=6.7,
        color=edgecolor,
        fontweight="bold",
        zorder=6,
    )

    detail_y = y + height - 0.116
    for index, detail in enumerate(stage.details):
        axis.text(
            x + 0.010,
            detail_y - index * 0.032,
            f"• {detail}",
            ha="left",
            va="center",
            fontsize=6.25,
            color="#334A5A",
            zorder=6,
        )

    axis.plot(
        [x + 0.009, x + width - 0.009],
        [y + 0.039, y + 0.039],
        color=edgecolor,
        linewidth=0.7,
        alpha=0.55,
        zorder=5,
    )
    axis.text(
        x + width / 2,
        y + 0.020,
        stage.artifact,
        ha="center",
        va="center",
        fontsize=6.0,
        color="#20303D",
        family="DejaVu Sans Mono",
        zorder=6,
    )


def _add_arrow(
    axis: plt.Axes,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    direction: str = "right",
) -> None:
    """Добавить стрелку основного прогнозного пути."""

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.25,
        color="#425A6B",
        connectionstyle="arc3,rad=0.0",
        zorder=3,
    )
    axis.add_patch(arrow)

    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2
    if direction == "right":
        axis.text(
            mid_x,
            mid_y + 0.018,
            label,
            ha="center",
            va="bottom",
            fontsize=6.1,
            color="#425A6B",
            zorder=6,
        )
    elif direction == "left":
        axis.text(
            mid_x,
            mid_y - 0.019,
            label,
            ha="center",
            va="top",
            fontsize=6.1,
            color="#425A6B",
            zorder=6,
        )
    else:
        axis.text(
            mid_x + 0.012,
            mid_y,
            label,
            ha="left",
            va="center",
            fontsize=6.1,
            color="#425A6B",
            zorder=6,
        )


def build_figure() -> plt.Figure:
    """Построить рисунок 5.1 без сохранения на диск."""

    validate_prediction_pipeline()
    configure_dissertation_style()

    figure, axis = plt.subplots(figsize=(18.2, 8.8))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    figure.suptitle(
        "Рисунок 5.1 — Конвейер построения априорной интегральной оценки качества",
        fontsize=15.5,
        fontweight="bold",
        y=0.976,
        color="#182733",
    )
    axis.text(
        0.5,
        0.925,
        "Только X_prior и θ_prior: фактические признаки и целевые показатели не участвуют в расчёте Q_pred",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#4A5E6D",
    )

    frame = FancyBboxPatch(
        (0.028, 0.155),
        0.944,
        0.716,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2,
        edgecolor="#6E8797",
        facecolor="#F8FBFD",
        zorder=0,
    )
    axis.add_patch(frame)
    axis.text(
        0.045,
        0.852,
        "АПРИОРНЫЙ РАСЧЁТНЫЙ КОНТУР",
        ha="left",
        va="center",
        fontsize=7.7,
        fontweight="bold",
        color="#527184",
        zorder=2,
    )

    colors = {
        "input": ("#EAF3FA", "#2D6D98"),
        "safety": ("#FFF2F1", "#B44743"),
        "transform": ("#EEF7F3", "#397C64"),
        "latent": ("#F3F0FA", "#6D57A5"),
        "prediction": ("#EEF4FB", "#3E6E9E"),
        "uncertainty": ("#FFF7E8", "#A87522"),
        "acceptance": ("#EEF8EC", "#4E8145"),
    }

    width = 0.214
    height = 0.247
    top_y = 0.575
    bottom_y = 0.245
    x_positions = (0.052, 0.282, 0.512, 0.742)

    positions: dict[str, tuple[float, float]] = {}
    top_stages = STAGES[:4]
    bottom_stages = tuple(reversed(STAGES[4:]))

    for stage, x in zip(top_stages, x_positions, strict=True):
        positions[stage.code] = (x, top_y)
        facecolor, edgecolor = colors[stage.group]
        _add_stage_box(
            axis,
            stage=stage,
            x=x,
            y=top_y,
            width=width,
            height=height,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )

    for stage, x in zip(bottom_stages, x_positions, strict=True):
        positions[stage.code] = (x, bottom_y)
        facecolor, edgecolor = colors[stage.group]
        _add_stage_box(
            axis,
            stage=stage,
            x=x,
            y=bottom_y,
            width=width,
            height=height,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )

    top_labels = ("входы", "без утечки", "X_norm")
    for index, label in enumerate(top_labels):
        start_x = x_positions[index] + width
        end_x = x_positions[index + 1]
        _add_arrow(
            axis,
            start=(start_x, top_y + height / 2),
            end=(end_x, top_y + height / 2),
            label=label,
            direction="right",
        )

    # Переход с верхнего ряда на нижний сохраняет непрерывность этапов 4 → 5.
    x_4 = positions["latent_component"][0] + width / 2
    x_5 = positions["partial_criteria"][0] + width / 2
    arrow_down = FancyArrowPatch(
        (x_4, top_y),
        (x_5, bottom_y + height),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.25,
        color="#425A6B",
        connectionstyle="arc3,rad=0.0",
        zorder=3,
    )
    axis.add_patch(arrow_down)
    axis.text(
        x_4 + 0.012,
        (top_y + bottom_y + height) / 2,
        "q_latent",
        ha="left",
        va="center",
        fontsize=6.1,
        color="#425A6B",
        zorder=6,
    )

    bottom_sequence = ("partial_criteria", "integral_quality", "uncertainty", "acceptance")
    bottom_labels = ("qⱼ_pred", "Q_pred", "Q_pred ± r")
    for source_code, target_code, label in zip(
        bottom_sequence[:-1],
        bottom_sequence[1:],
        bottom_labels,
        strict=True,
    ):
        source_x, _ = positions[source_code]
        target_x, _ = positions[target_code]
        _add_arrow(
            axis,
            start=(source_x, bottom_y + height / 2),
            end=(target_x + width, bottom_y + height / 2),
            label=label,
            direction="left",
        )

    forbidden_box = FancyBboxPatch(
        (0.046, 0.060),
        0.455,
        0.068,
        boxstyle="round,pad=0.009,rounding_size=0.011",
        linewidth=1.15,
        edgecolor="#B44743",
        facecolor="#FFF4F3",
        linestyle="--",
        zorder=2,
    )
    axis.add_patch(forbidden_box)
    axis.text(
        0.058,
        0.103,
        "ЗАПРЕЩЁННЫЕ ВХОДЫ",
        ha="left",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color="#A33E39",
        zorder=4,
    )
    axis.text(
        0.058,
        0.078,
        "fact_features.csv · quality_targets.csv · theta_diag.csv · theta_full.csv",
        ha="left",
        va="center",
        fontsize=6.7,
        color="#7F403C",
        family="DejaVu Sans Mono",
        zorder=4,
    )
    axis.text(
        0.487,
        0.094,
        "×",
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
        color="#B44743",
        zorder=5,
    )

    summary_box = FancyBboxPatch(
        (0.527, 0.060),
        0.427,
        0.068,
        boxstyle="round,pad=0.009,rounding_size=0.011",
        linewidth=1.1,
        edgecolor="#6B7E8B",
        facecolor="#F4F7F9",
        zorder=2,
    )
    axis.add_patch(summary_box)
    axis.text(
        0.540,
        0.103,
        "МЕТОДИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ",
        ha="left",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color="#4B6271",
        zorder=4,
    )
    axis.text(
        0.540,
        0.078,
        "Q_pred — априорный сравнительный индекс; интервал отражает диагностическую неопределённость, а не доказанную калибровку.",
        ha="left",
        va="center",
        fontsize=6.55,
        color="#526875",
        zorder=4,
    )

    return figure


def generate(*, project_root: Path, dpi: int = 300) -> FigureExportResult:
    """Сформировать рисунок 5.1 в PNG и SVG."""

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
        description="Сформировать рисунок 5.1 с конвейером априорного прогноза."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Корень проекта manual_coding_quality.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Разрешение PNG; по умолчанию 300 dpi.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить CLI генератора рисунка 5.1."""

    arguments = build_argument_parser().parse_args(argv)
    result = generate(project_root=arguments.project_root, dpi=arguments.dpi)
    print("Рисунок 5.1 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
