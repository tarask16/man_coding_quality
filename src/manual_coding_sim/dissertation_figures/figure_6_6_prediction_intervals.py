"""Генерация рисунка 6.6 с проверкой диагностических интервалов прогноза."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from manual_coding_sim.dissertation_figures.common import (
    FigureExportResult,
    configure_dissertation_style,
    export_figure,
)

OUTPUT_DIR = Path("reports/chapter6/figures")
FILE_STEM = "prediction_intervals"
DEFAULT_INTERVALS_PATH = Path("reports/chapter5/prediction_uncertainty.csv")
DEFAULT_Q_FACT_PATH = Path("data/processed/quality_targets.csv")
JOIN_KEYS = ("scenario_id", "protocol_id")
LOW_THRESHOLD = 0.45
HIGH_THRESHOLD = 0.70
STATUS_ORDER = ("covered", "miss_above", "miss_below")
STATUS_LABELS = {
    "covered": "Покрыто",
    "miss_above": "Промах выше",
    "miss_below": "Промах ниже",
}
STATUS_COLORS = {
    "covered": "#4c956c",
    "miss_above": "#c44e52",
    "miss_below": "#7a5195",
}


@dataclass(frozen=True, slots=True)
class PredictionIntervalPoint:
    """Фактическое значение и диагностический интервал одного сценария."""

    scenario_id: str
    protocol_id: str
    q_pred: float
    q_fact: float
    lower: float
    upper: float
    uncertainty_score: float

    @property
    def interval_width(self) -> float:
        """Вернуть ширину диагностического интервала."""

        return self.upper - self.lower

    @property
    def status(self) -> str:
        """Определить результат покрытия фактического значения."""

        if self.q_fact < self.lower:
            return "miss_below"
        if self.q_fact > self.upper:
            return "miss_above"
        return "covered"

    @property
    def distance_to_interval(self) -> float:
        """Вернуть неотрицательное расстояние факта до ближайшей границы."""

        if self.status == "miss_below":
            return self.lower - self.q_fact
        if self.status == "miss_above":
            return self.q_fact - self.upper
        return 0.0

    @property
    def signed_miss_distance(self) -> float:
        """Вернуть направленное расстояние: выше положительно, ниже отрицательно."""

        if self.status == "miss_below":
            return self.q_fact - self.lower
        if self.status == "miss_above":
            return self.q_fact - self.upper
        return 0.0


@dataclass(frozen=True, slots=True)
class IntervalCoverageSummary:
    """Сводные показатели покрытия диагностическими интервалами."""

    total: int
    covered_count: int
    miss_above_count: int
    miss_below_count: int
    coverage_rate: float
    mean_interval_width: float
    median_interval_width: float
    mean_distance_all: float
    mean_distance_misses: float
    max_distance: float
    max_distance_scenario: str



def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    """Прочитать непустой CSV-файл."""

    if not path.is_file():
        raise FileNotFoundError(f"Не найден входной файл: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Входной файл не содержит строк: {path}")
    return rows



def _validate_columns(
    rows: Sequence[Mapping[str, str]],
    required: Sequence[str],
    *,
    source_name: str,
) -> None:
    """Проверить наличие обязательных колонок."""

    missing = [column for column in required if column not in rows[0]]
    if missing:
        raise ValueError(
            f"В {source_name} отсутствуют обязательные колонки: {', '.join(missing)}"
        )



def _index_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    source_name: str,
) -> dict[tuple[str, str], Mapping[str, str]]:
    """Индексировать строки по ключам сценария и протокола."""

    indexed: dict[tuple[str, str], Mapping[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        key = tuple(str(row[column]).strip() for column in JOIN_KEYS)
        if any(not value for value in key):
            raise ValueError(
                f"В {source_name}, строка {row_number}, обнаружен пустой ключ объединения."
            )
        if key in indexed:
            raise ValueError(
                f"В {source_name} обнаружен дублирующий ключ: {key[0]} / {key[1]}."
            )
        indexed[key] = row
    return indexed



def _parse_unit_value(raw: str, *, column: str, key: tuple[str, str]) -> float:
    """Преобразовать значение и проверить принадлежность диапазону [0; 1]."""

    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Колонка {column}, ключ {key[0]} / {key[1]}: требуется число."
        ) from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            f"Колонка {column}, ключ {key[0]} / {key[1]}: "
            "значение должно находиться в диапазоне [0; 1]."
        )
    return value



def load_prediction_interval_points(
    intervals_path: Path,
    q_fact_path: Path,
) -> tuple[PredictionIntervalPoint, ...]:
    """Загрузить интервалы и согласовать их с фактическим качеством."""

    interval_rows = _read_rows(intervals_path)
    fact_rows = _read_rows(q_fact_path)
    _validate_columns(
        interval_rows,
        [
            *JOIN_KEYS,
            "q_pred",
            "q_pred_lower",
            "q_pred_upper",
            "uncertainty_score",
        ],
        source_name=intervals_path.name,
    )
    _validate_columns(
        fact_rows,
        [*JOIN_KEYS, "integral_quality"],
        source_name=q_fact_path.name,
    )
    interval_index = _index_rows(interval_rows, source_name=intervals_path.name)
    fact_index = _index_rows(fact_rows, source_name=q_fact_path.name)
    if set(interval_index) != set(fact_index):
        raise ValueError("Наборы ключей интервального и фактического качества не совпадают.")
    if len(interval_index) < 3:
        raise ValueError("Для проверки интервалов требуется не менее трёх сценариев.")

    points: list[PredictionIntervalPoint] = []
    for key in sorted(interval_index):
        interval_row = interval_index[key]
        q_pred = _parse_unit_value(interval_row["q_pred"], column="q_pred", key=key)
        lower = _parse_unit_value(
            interval_row["q_pred_lower"], column="q_pred_lower", key=key
        )
        upper = _parse_unit_value(
            interval_row["q_pred_upper"], column="q_pred_upper", key=key
        )
        uncertainty_score = _parse_unit_value(
            interval_row["uncertainty_score"],
            column="uncertainty_score",
            key=key,
        )
        q_fact = _parse_unit_value(
            fact_index[key]["integral_quality"],
            column="integral_quality",
            key=key,
        )
        if lower > upper:
            raise ValueError(
                f"Ключ {key[0]} / {key[1]}: нижняя граница интервала выше верхней."
            )
        if not lower <= q_pred <= upper:
            raise ValueError(
                f"Ключ {key[0]} / {key[1]}: q_pred должен находиться внутри интервала."
            )
        points.append(
            PredictionIntervalPoint(
                scenario_id=key[0],
                protocol_id=key[1],
                q_pred=q_pred,
                q_fact=q_fact,
                lower=lower,
                upper=upper,
                uncertainty_score=uncertainty_score,
            )
        )
    return tuple(points)



def calculate_interval_coverage_summary(
    points: Sequence[PredictionIntervalPoint],
) -> IntervalCoverageSummary:
    """Рассчитать покрытие, ширины интервалов и расстояния промахов."""

    if len(points) < 3:
        raise ValueError("Для расчёта покрытия требуется не менее трёх сценариев.")
    covered_count = sum(point.status == "covered" for point in points)
    miss_above_count = sum(point.status == "miss_above" for point in points)
    miss_below_count = sum(point.status == "miss_below" for point in points)
    widths = np.asarray([point.interval_width for point in points], dtype=float)
    distances = np.asarray([point.distance_to_interval for point in points], dtype=float)
    miss_distances = distances[distances > 0.0]
    maximum_index = int(np.argmax(distances))
    return IntervalCoverageSummary(
        total=len(points),
        covered_count=covered_count,
        miss_above_count=miss_above_count,
        miss_below_count=miss_below_count,
        coverage_rate=covered_count / len(points),
        mean_interval_width=float(np.mean(widths)),
        median_interval_width=float(np.median(widths)),
        mean_distance_all=float(np.mean(distances)),
        mean_distance_misses=(
            float(np.mean(miss_distances)) if miss_distances.size else 0.0
        ),
        max_distance=float(distances[maximum_index]),
        max_distance_scenario=points[maximum_index].scenario_id,
    )



def _draw_status_legend(axis: plt.Axes) -> None:
    """Добавить компактную легенду статусов интервала."""

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=STATUS_COLORS[status],
            marker=("o" if status == "covered" else "^" if status == "miss_above" else "v"),
            linewidth=2.1,
            markersize=7,
            label=STATUS_LABELS[status],
        )
        for status in STATUS_ORDER
    ]
    handles.extend(
        [
            plt.Line2D(
                [0],
                [0],
                color="#1f4e79",
                marker="s",
                linewidth=0,
                markersize=5,
                label="Q_pred",
            ),
            plt.Line2D(
                [0],
                [0],
                color="#1d252c",
                marker=".",
                linewidth=1.0,
                markersize=8,
                label="Q_fact",
            ),
        ]
    )
    axis.legend(
        handles=handles,
        loc="upper left",
        ncol=5,
        frameon=True,
        fontsize=9.4,
        borderpad=0.55,
        handlelength=1.6,
        columnspacing=1.2,
    )



def build_figure(
    points: Sequence[PredictionIntervalPoint],
    summary: IntervalCoverageSummary,
) -> plt.Figure:
    """Построить интервальный график и диагностические панели."""

    configure_dissertation_style()
    sorted_points = tuple(sorted(points, key=lambda point: (point.q_fact, point.scenario_id)))
    ranks = np.arange(1, len(sorted_points) + 1)
    q_fact = np.asarray([point.q_fact for point in sorted_points], dtype=float)
    q_pred = np.asarray([point.q_pred for point in sorted_points], dtype=float)
    lower = np.asarray([point.lower for point in sorted_points], dtype=float)
    upper = np.asarray([point.upper for point in sorted_points], dtype=float)
    statuses = np.asarray([point.status for point in sorted_points], dtype=object)
    signed_distances = np.asarray(
        [point.signed_miss_distance for point in sorted_points], dtype=float
    )

    figure = plt.figure(figsize=(20.6, 8.8))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(4.7, 1.35),
        height_ratios=(3.45, 1.2),
        hspace=0.35,
        wspace=0.24,
    )
    interval_axis = figure.add_subplot(grid[0, 0])
    miss_axis = figure.add_subplot(grid[1, 0], sharex=interval_axis)
    outcome_axis = figure.add_subplot(grid[0, 1])
    info_axis = figure.add_subplot(grid[1, 1])

    figure.suptitle(
        "Рисунок 6.6 — Покрытие фактического качества диагностическими интервалами",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )

    # Фоновые области классов качества помогают сопоставить интервалы с порогами.
    interval_axis.axhspan(0.0, LOW_THRESHOLD, color="#f7d9d9", alpha=0.30)
    interval_axis.axhspan(LOW_THRESHOLD, HIGH_THRESHOLD, color="#f8edc9", alpha=0.28)
    interval_axis.axhspan(HIGH_THRESHOLD, 1.0, color="#dcefdc", alpha=0.28)
    interval_axis.axhline(
        LOW_THRESHOLD,
        color="#9a6b00",
        linestyle="--",
        linewidth=1.1,
    )
    interval_axis.axhline(
        HIGH_THRESHOLD,
        color="#3d7a4a",
        linestyle="--",
        linewidth=1.1,
    )

    for status in STATUS_ORDER:
        mask = statuses == status
        if not np.any(mask):
            continue
        interval_axis.vlines(
            ranks[mask],
            lower[mask],
            upper[mask],
            color=STATUS_COLORS[status],
            alpha=0.82,
            linewidth=1.5,
            zorder=2,
        )
        marker = "o" if status == "covered" else "^" if status == "miss_above" else "v"
        interval_axis.scatter(
            ranks[mask],
            q_fact[mask],
            color=STATUS_COLORS[status],
            marker=marker,
            s=23,
            edgecolor="white",
            linewidth=0.35,
            zorder=4,
        )

    interval_axis.scatter(
        ranks,
        q_pred,
        color="#1f4e79",
        marker="s",
        s=11,
        alpha=0.90,
        zorder=3,
    )
    interval_axis.plot(
        ranks,
        q_fact,
        color="#1d252c",
        linewidth=0.75,
        alpha=0.65,
        zorder=1,
    )
    interval_axis.set_ylim(0.0, 1.0)
    interval_axis.set_xlim(0, len(sorted_points) + 1)
    interval_axis.set_ylabel("Интегральное качество")
    interval_axis.set_title(
        "Интервалы прогнозов, упорядоченные по фактическому качеству",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    interval_axis.grid(axis="y", alpha=0.20)
    interval_axis.tick_params(axis="x", labelbottom=False)
    interval_axis.text(
        0.985,
        0.055,
        (
            f"Покрыто: {summary.covered_count}/{summary.total} "
            f"({summary.coverage_rate * 100:.1f}%)\n"
            f"Промах выше: {summary.miss_above_count}; "
            f"промах ниже: {summary.miss_below_count}"
        ),
        transform=interval_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#8796a5",
            "alpha": 0.92,
        },
    )
    interval_axis.text(
        len(sorted_points) + 0.2,
        LOW_THRESHOLD,
        "0,45",
        ha="left",
        va="center",
        fontsize=9,
        color="#7d5b00",
    )
    interval_axis.text(
        len(sorted_points) + 0.2,
        HIGH_THRESHOLD,
        "0,70",
        ha="left",
        va="center",
        fontsize=9,
        color="#2f663c",
    )
    _draw_status_legend(interval_axis)

    miss_axis.axhline(0.0, color="#17202a", linewidth=1.0)
    for status in STATUS_ORDER:
        mask = statuses == status
        if not np.any(mask):
            continue
        miss_axis.scatter(
            ranks[mask],
            signed_distances[mask],
            color=STATUS_COLORS[status],
            marker=("o" if status == "covered" else "^" if status == "miss_above" else "v"),
            s=(12 if status == "covered" else 22),
            alpha=0.85,
        )
    miss_axis.fill_between(
        [0, len(sorted_points) + 1],
        [0, 0],
        [max(0.01, float(np.max(signed_distances)) * 1.08)] * 2,
        color="#f7d9d9",
        alpha=0.18,
    )
    miss_axis.fill_between(
        [0, len(sorted_points) + 1],
        [min(-0.01, float(np.min(signed_distances)) * 1.08)] * 2,
        [0, 0],
        color="#e8dff0",
        alpha=0.25,
    )
    max_abs_distance = max(0.05, float(np.max(np.abs(signed_distances))))
    miss_axis.set_ylim(-max_abs_distance * 1.12, max_abs_distance * 1.12)
    miss_axis.set_xlabel("Порядковый номер сценария после сортировки по Q_fact")
    miss_axis.set_ylabel("Расстояние до\nграницы интервала")
    miss_axis.set_title(
        "Промахи выше и ниже интервала",
        fontsize=12.8,
        fontweight="bold",
        pad=8,
    )
    miss_axis.grid(axis="y", alpha=0.22)
    miss_axis.text(
        0.995,
        0.92,
        "выше интервала",
        transform=miss_axis.transAxes,
        ha="right",
        va="top",
        fontsize=9.2,
        color=STATUS_COLORS["miss_above"],
    )
    miss_axis.text(
        0.995,
        0.08,
        "ниже интервала",
        transform=miss_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.2,
        color=STATUS_COLORS["miss_below"],
    )

    counts = np.asarray(
        [summary.covered_count, summary.miss_above_count, summary.miss_below_count],
        dtype=int,
    )
    labels = [STATUS_LABELS[status] for status in STATUS_ORDER]
    colors = [STATUS_COLORS[status] for status in STATUS_ORDER]
    bars = outcome_axis.barh(labels, counts, color=colors, alpha=0.88)
    outcome_axis.invert_yaxis()
    outcome_axis.set_xlim(0, max(counts) * 1.28)
    outcome_axis.set_xlabel("Число сценариев")
    outcome_axis.set_title(
        "Исходы покрытия",
        fontsize=13.5,
        fontweight="bold",
        pad=12,
    )
    outcome_axis.grid(axis="x", alpha=0.20)
    for bar, count in zip(bars, counts, strict=True):
        outcome_axis.text(
            bar.get_width() + max(counts) * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{count}\n({count / summary.total * 100:.1f}%)",
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    info_axis.axis("off")
    info_axis.set_title(
        "Диагностика интервалов",
        fontsize=13.2,
        fontweight="bold",
        pad=9,
    )
    info_axis.text(
        0.03,
        0.94,
        (
            f"Coverage rate: {summary.coverage_rate:.4f}\n"
            f"Средняя ширина: {summary.mean_interval_width:.4f}\n"
            f"Медианная ширина: {summary.median_interval_width:.4f}\n"
            f"Среднее расстояние до интервала: {summary.mean_distance_all:.4f}\n"
            f"Среди промахов: {summary.mean_distance_misses:.4f}\n"
            f"Максимальный промах: {summary.max_distance:.4f}\n"
            f"Сценарий: {summary.max_distance_scenario}"
        ),
        transform=info_axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.8,
        linespacing=1.38,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#f3f6f8",
            "edgecolor": "#8796a5",
        },
    )

    figure.text(
        0.5,
        0.022,
        (
            "Методическое ограничение: интервалы являются диагностическими. "
            "Низкое эмпирическое покрытие показывает отсутствие абсолютной калибровки; "
            "ширина интервала не должна трактоваться как подтверждённый доверительный уровень."
        ),
        ha="center",
        va="bottom",
        fontsize=10.6,
        color="#3f4a52",
    )
    figure.subplots_adjust(left=0.055, right=0.975, top=0.91, bottom=0.13)
    return figure



def generate(
    *,
    project_root: Path,
    intervals_path: Path | None = None,
    q_fact_path: Path | None = None,
    dpi: int = 300,
) -> FigureExportResult:
    """Сформировать рисунок 6.6 в PNG и SVG."""

    root = project_root.resolve()
    resolved_intervals_path = root / (intervals_path or DEFAULT_INTERVALS_PATH)
    resolved_q_fact_path = root / (q_fact_path or DEFAULT_Q_FACT_PATH)
    points = load_prediction_interval_points(
        resolved_intervals_path,
        resolved_q_fact_path,
    )
    summary = calculate_interval_coverage_summary(points)
    figure = build_figure(points, summary)
    return export_figure(
        figure,
        project_root=root,
        relative_output_dir=OUTPUT_DIR,
        file_stem=FILE_STEM,
        dpi=dpi,
    )



def build_parser() -> argparse.ArgumentParser:
    """Создать парсер аргументов командной строки."""

    parser = argparse.ArgumentParser(
        description="Сформировать рисунок 6.6 с диагностическими интервалами прогноза."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Корень проекта manual_coding_sim.",
    )
    parser.add_argument(
        "--intervals",
        type=Path,
        default=None,
        help="Путь к prediction_uncertainty.csv относительно корня проекта.",
    )
    parser.add_argument(
        "--q-fact",
        type=Path,
        default=None,
        help="Путь к quality_targets.csv относительно корня проекта.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Разрешение PNG, dpi.",
    )
    return parser



def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить CLI генератора рисунка 6.6."""

    args = build_parser().parse_args(argv)
    result = generate(
        project_root=args.project_root,
        intervals_path=args.intervals,
        q_fact_path=args.q_fact,
        dpi=args.dpi,
    )
    print("Рисунок 6.6 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
