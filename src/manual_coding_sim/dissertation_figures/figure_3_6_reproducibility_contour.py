"""Генерация рисунка 3.6 с контуром воспроизводимости эксперимента."""

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

OUTPUT_DIR = Path("reports/chapter3/figures")
FILE_STEM = "figure_3_6_reproducibility_contour"


@dataclass(frozen=True, slots=True)
class ReproducibilityStage:
    """Этап вычислительного контура воспроизводимости."""

    code: str
    number: int
    title: str
    details: tuple[str, ...]
    evidence: str


@dataclass(frozen=True, slots=True)
class StageConnection:
    """Направленная связь между этапами контура."""

    source: str
    target: str
    label: str
    kind: str = "main"


STAGES: tuple[ReproducibilityStage, ...] = (
    ReproducibilityStage(
        code="configuration",
        number=1,
        title="Конфигурация и seed",
        details=(
            "YAML-параметры опыта",
            "фиксированный random_seed",
            "версии среды и кода",
        ),
        evidence="config + environment manifest",
    ),
    ReproducibilityStage(
        code="run",
        number=2,
        title="Управляемый запуск",
        details=(
            "единая точка входа runner",
            "журнал сценариев и протоколов",
            "детерминированный порядок",
        ),
        evidence="run_id + execution log",
    ),
    ReproducibilityStage(
        code="artifacts",
        number=3,
        title="CSV / JSON-артефакты",
        details=(
            "prior_features.csv",
            "fact_features.csv / targets",
            "protocols и метаданные",
        ),
        evidence="схема, размер и идентификаторы",
    ),
    ReproducibilityStage(
        code="checksums",
        number=4,
        title="Контрольные суммы",
        details=(
            "SHA-256 для каждого файла",
            "сравнение эталона и повтора",
            "фиксация расхождений",
        ),
        evidence="checksums.json",
    ),
    ReproducibilityStage(
        code="tests",
        number=5,
        title="Автоматические тесты",
        details=(
            "схема и диапазоны данных",
            "инварианты и связи таблиц",
            "регрессионная проверка",
        ),
        evidence="pytest report",
    ),
    ReproducibilityStage(
        code="report",
        number=6,
        title="Отчёт воспроизводимости",
        details=(
            "статусы gate и причины отказа",
            "версии, хэши и команды",
            "итог: passed / failed",
        ),
        evidence="reproducibility_report.json",
    ),
)

EXPECTED_STAGE_CODES: tuple[str, ...] = (
    "configuration",
    "run",
    "artifacts",
    "checksums",
    "tests",
    "report",
)

CONNECTIONS: tuple[StageConnection, ...] = (
    StageConnection("configuration", "run", "параметры и seed"),
    StageConnection("run", "artifacts", "результаты запуска"),
    StageConnection("artifacts", "checksums", "файлы и схема"),
    StageConnection("checksums", "tests", "результат сравнения"),
    StageConnection("tests", "report", "статусы проверок"),
    StageConnection(
        "configuration",
        "rerun",
        "те же параметры",
        kind="rerun",
    ),
    StageConnection(
        "rerun",
        "checksums",
        "повторный набор артефактов",
        kind="rerun",
    ),
)

REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "prior_features.csv",
    "fact_features.csv",
    "quality_targets.csv",
    "protocols.csv",
    "checksums.json",
    "reproducibility_report.json",
)

ACCEPTANCE_CRITERIA: tuple[str, ...] = (
    "совпадает конфигурация, seed и версия кода",
    "совпадают схема, размеры и идентификаторы таблиц",
    "совпадают SHA-256 детерминированных артефактов",
    "все модульные и регрессионные тесты завершены успешно",
)


def validate_reproducibility_contour(
    stages: Sequence[ReproducibilityStage] = STAGES,
    connections: Sequence[StageConnection] = CONNECTIONS,
) -> None:
    """Проверить полноту этапов и направленных связей контура."""

    codes = tuple(stage.code for stage in stages)
    if codes != EXPECTED_STAGE_CODES:
        raise ValueError("Контур должен содержать шесть этапов в заданном порядке.")

    numbers = tuple(stage.number for stage in stages)
    if numbers != tuple(range(1, 7)):
        raise ValueError("Нумерация этапов должна быть последовательной от 1 до 6.")

    pairs = {(item.source, item.target) for item in connections}
    required_pairs = set(zip(EXPECTED_STAGE_CODES[:-1], EXPECTED_STAGE_CODES[1:]))
    missing_pairs = required_pairs - pairs
    if missing_pairs:
        missing_text = ", ".join(f"{a} → {b}" for a, b in sorted(missing_pairs))
        raise ValueError(f"Отсутствуют обязательные связи: {missing_text}.")

    if ("configuration", "rerun") not in pairs or ("rerun", "checksums") not in pairs:
        raise ValueError("Должен быть показан независимый повторный запуск.")

    for stage in stages:
        if len(stage.details) != 3:
            raise ValueError(f"Этап {stage.code} должен содержать три пояснения.")
        if not stage.evidence.strip():
            raise ValueError(f"Для этапа {stage.code} не указан проверяемый артефакт.")


def _add_stage_box(
    axis: plt.Axes,
    *,
    stage: ReproducibilityStage,
    x: float,
    y: float,
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str,
) -> None:
    """Добавить блок этапа с номером, функциями и свидетельством."""

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        linewidth=1.45,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=4,
    )
    axis.add_patch(box)

    number_box = FancyBboxPatch(
        (x + 0.008, y + height - 0.057),
        0.028,
        0.042,
        boxstyle="round,pad=0.003,rounding_size=0.008",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor="white",
        zorder=5,
    )
    axis.add_patch(number_box)
    axis.text(
        x + 0.022,
        y + height - 0.036,
        str(stage.number),
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color=edgecolor,
        zorder=6,
    )

    axis.text(
        x + width * 0.62,
        y + height - 0.043,
        stage.title,
        ha="center",
        va="center",
        fontsize=7.6,
        fontweight="bold",
        color="#17212B",
        zorder=6,
    )

    detail_y = y + height - 0.105
    for index, detail in enumerate(stage.details):
        axis.text(
            x + 0.014,
            detail_y - index * 0.049,
            f"• {detail}",
            ha="left",
            va="center",
            fontsize=7.35,
            color="#334A5A",
            zorder=6,
        )

    axis.plot(
        [x + 0.012, x + width - 0.012],
        [y + 0.055, y + 0.055],
        color=edgecolor,
        linewidth=0.8,
        alpha=0.55,
        zorder=5,
    )
    axis.text(
        x + width / 2,
        y + 0.028,
        stage.evidence,
        ha="center",
        va="center",
        fontsize=6.95,
        color=edgecolor,
        family="DejaVu Sans Mono" if "." in stage.evidence or "+" in stage.evidence else "DejaVu Sans",
        zorder=6,
    )


def _add_arrow(
    axis: plt.Axes,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    color: str,
    linestyle: str = "-",
    label_y_offset: float = 0.028,
) -> None:
    """Добавить направленную стрелку и компактную подпись."""

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.45,
        color=color,
        linestyle=linestyle,
        connectionstyle="arc3,rad=0.0",
        shrinkA=2,
        shrinkB=2,
        zorder=3,
    )
    axis.add_patch(arrow)
    axis.text(
        (start[0] + end[0]) / 2,
        (start[1] + end[1]) / 2 + label_y_offset,
        label,
        ha="center",
        va="center",
        fontsize=6.7,
        color=color,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.94},
        zorder=7,
    )


def build_figure() -> plt.Figure:
    """Построить схему основного и повторного контура воспроизводимости."""

    validate_reproducibility_contour()
    configure_dissertation_style()

    figure, axis = plt.subplots(figsize=(18.8, 9.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    figure.suptitle(
        "Рисунок 3.6 — Контур воспроизводимости вычислительного эксперимента",
        fontsize=15.0,
        fontweight="bold",
        y=0.985,
    )
    axis.text(
        0.5,
        0.925,
        "Фиксированные входы → управляемое выполнение → проверяемые артефакты → автоматизированное подтверждение",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#3C4E5C",
    )

    positions = {
        "configuration": (0.025, 0.545),
        "run": (0.188, 0.545),
        "artifacts": (0.351, 0.545),
        "checksums": (0.514, 0.545),
        "tests": (0.677, 0.545),
        "report": (0.840, 0.545),
    }
    width = 0.135
    height = 0.285
    palette = (
        ("#EAF3FA", "#2E6387"),
        ("#EDF6EE", "#3B7046"),
        ("#FFF4DE", "#8A651E"),
        ("#F2EEFA", "#65498B"),
        ("#EAF6F4", "#2F716A"),
        ("#F8ECEC", "#8A4646"),
    )

    for stage, colors in zip(STAGES, palette):
        x, y = positions[stage.code]
        _add_stage_box(
            axis,
            stage=stage,
            x=x,
            y=y,
            width=width,
            height=height,
            facecolor=colors[0],
            edgecolor=colors[1],
        )

    for source, target, label in (
        ("configuration", "run", "параметры и seed"),
        ("run", "artifacts", "результаты"),
        ("artifacts", "checksums", "артефакты"),
        ("checksums", "tests", "сравнение"),
        ("tests", "report", "статусы gate"),
    ):
        source_x, source_y = positions[source]
        target_x, target_y = positions[target]
        _add_arrow(
            axis,
            start=(source_x + width, source_y + height / 2),
            end=(target_x, target_y + height / 2),
            label=label,
            color="#526979",
            label_y_offset=0.032,
        )

    rerun_box = FancyBboxPatch(
        (0.235, 0.315),
        0.355,
        0.125,
        boxstyle="round,pad=0.010,rounding_size=0.014",
        linewidth=1.35,
        edgecolor="#6B7680",
        facecolor="#F5F7F8",
        linestyle="--",
        zorder=4,
    )
    axis.add_patch(rerun_box)
    axis.text(
        0.4125,
        0.402,
        "Независимый повторный запуск",
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color="#2F3D47",
        zorder=6,
    )
    axis.text(
        0.4125,
        0.361,
        "чистое окружение · те же config и seed · новый run_id · повторный набор CSV / JSON",
        ha="center",
        va="center",
        fontsize=7.8,
        color="#4D5E69",
        zorder=6,
    )

    config_x, config_y = positions["configuration"]
    checksum_x, checksum_y = positions["checksums"]
    _add_arrow(
        axis,
        start=(config_x + width / 2, config_y),
        end=(0.27, 0.44),
        label="те же входы",
        color="#6B7680",
        linestyle="--",
        label_y_offset=-0.020,
    )
    _add_arrow(
        axis,
        start=(0.59, 0.377),
        end=(checksum_x + width / 2, checksum_y),
        label="повторные хэши",
        color="#6B7680",
        linestyle="--",
        label_y_offset=-0.018,
    )

    criteria_box = FancyBboxPatch(
        (0.025, 0.075),
        0.610,
        0.175,
        boxstyle="round,pad=0.010,rounding_size=0.014",
        linewidth=1.25,
        edgecolor="#3E627A",
        facecolor="#F2F7FA",
        zorder=3,
    )
    axis.add_patch(criteria_box)
    axis.text(
        0.045,
        0.218,
        "Критерии приёмки воспроизводимости",
        ha="left",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color="#264A62",
    )
    for index, criterion in enumerate(ACCEPTANCE_CRITERIA):
        axis.text(
            0.050 + (index % 2) * 0.300,
            0.171 - (index // 2) * 0.062,
            f"✓ {criterion}",
            ha="left",
            va="center",
            fontsize=7.8,
            color="#334A5A",
        )

    formula_box = FancyBboxPatch(
        (0.655, 0.075),
        0.320,
        0.175,
        boxstyle="round,pad=0.010,rounding_size=0.014",
        linewidth=1.25,
        edgecolor="#7A5A40",
        facecolor="#FBF6F0",
        zorder=3,
    )
    axis.add_patch(formula_box)
    axis.text(
        0.815,
        0.215,
        "Формальная фиксация",
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color="#69462D",
    )
    axis.text(
        0.815,
        0.165,
        r"$H_j=\operatorname{SHA\!\!-\!256}(file_j)$",
        ha="center",
        va="center",
        fontsize=11.0,
        color="#3A322B",
    )
    axis.text(
        0.815,
        0.119,
        r"$R=\mathbb{1}[schema \wedge hashes \wedge tests]$",
        ha="center",
        va="center",
        fontsize=10.2,
        color="#3A322B",
    )

    axis.text(
        0.5,
        0.025,
        "Методическое ограничение: воспроизводимость вычислительного результата не равнозначна его внешней валидности.",
        ha="center",
        va="center",
        fontsize=8.7,
        color="#5D3F3F",
        fontstyle="italic",
    )

    return figure


def generate(*, project_root: Path, dpi: int = 300) -> FigureExportResult:
    """Сформировать рисунок 3.6 в PNG и SVG."""

    figure = build_figure()
    return export_figure(
        figure,
        project_root=project_root,
        relative_output_dir=OUTPUT_DIR,
        file_stem=FILE_STEM,
        dpi=dpi,
    )


def build_parser() -> argparse.ArgumentParser:
    """Создать парсер аргументов командной строки."""

    parser = argparse.ArgumentParser(
        description="Сформировать рисунок 3.6 с контуром воспроизводимости."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Корень проекта, относительно которого создаются артефакты.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Разрешение PNG; значение по умолчанию — 300 dpi.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Запустить генерацию рисунка из командной строки."""

    args = build_parser().parse_args(argv)
    result = generate(project_root=args.project_root, dpi=args.dpi)
    print("Рисунок 3.6 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
