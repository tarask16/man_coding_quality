"""Генерация рисунка 5.4 с распределением интегрального показателя Q_pred."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from manual_coding_sim.dissertation_figures.common import (
    FigureExportResult,
    configure_dissertation_style,
    export_figure,
)

OUTPUT_DIR = Path("reports/chapter5/figures")
FILE_STEM = "figure_5_4_q_pred_distribution"
DEFAULT_INPUT_PATH = Path("reports/chapter5/q_pred.csv")

LOW_THRESHOLD = 0.45
HIGH_THRESHOLD = 0.70


@dataclass(frozen=True, slots=True)
class QualityClassSpec:
    """Описание одного класса априорного интегрального показателя."""

    code: str
    label: str
    lower: float
    upper: float
    color: str


QUALITY_CLASSES = (
    QualityClassSpec("low", "Низкий", 0.0, LOW_THRESHOLD, "#D95F5F"),
    QualityClassSpec("medium", "Средний", LOW_THRESHOLD, HIGH_THRESHOLD, "#E6A23C"),
    QualityClassSpec("high", "Высокий", HIGH_THRESHOLD, 1.0, "#4F9D69"),
)


@dataclass(frozen=True, slots=True)
class QPredRow:
    """Интегральный прогнозный показатель одного сценария."""

    scenario_id: str
    protocol_id: str
    q_pred: float


@dataclass(frozen=True, slots=True)
class QPredSummary:
    """Описательная статистика распределения Q_pred."""

    count: int
    minimum: float
    first_quartile: float
    median: float
    mean: float
    third_quartile: float
    maximum: float
    standard_deviation: float


@dataclass(frozen=True, slots=True)
class QualityClassCount:
    """Численность и доля сценариев одного класса качества."""

    code: str
    label: str
    count: int
    share: float
    color: str


def classify_q_pred(value: float) -> str:
    """Классифицировать Q_pred по порогам главы 5."""

    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("Q_pred должен быть конечным и лежать в диапазоне [0; 1].")
    if value < LOW_THRESHOLD:
        return "low"
    if value < HIGH_THRESHOLD:
        return "medium"
    return "high"


def load_q_pred(path: str | Path) -> tuple[QPredRow, ...]:
    """Загрузить интегральный прогнозный показатель из q_pred.csv."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Не найден файл интегрального прогноза: {source}")

    rows: list[QPredRow] = []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("CSV-файл не содержит строки заголовка.")

        q_pred_column = next(
            (
                candidate
                for candidate in (
                    "q_pred",
                    "Q_pred",
                    "integral_quality_pred",
                    "predicted_integral_quality",
                )
                if candidate in reader.fieldnames
            ),
            None,
        )
        required = {"scenario_id", "protocol_id"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "В q_pred.csv отсутствуют обязательные колонки: "
                + ", ".join(sorted(missing))
            )
        if q_pred_column is None:
            raise ValueError(
                "В q_pred.csv отсутствует колонка q_pred с интегральным прогнозом."
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                rows.append(
                    QPredRow(
                        scenario_id=str(row["scenario_id"]).strip(),
                        protocol_id=str(row["protocol_id"]).strip(),
                        q_pred=float(row[q_pred_column]),
                    )
                )
            except (TypeError, ValueError, KeyError) as error:
                raise ValueError(
                    f"Некорректная строка {row_number} q_pred.csv: {error}."
                ) from error

    result = tuple(rows)
    validate_q_pred_rows(result)
    return result


def validate_q_pred_rows(rows: Sequence[QPredRow]) -> None:
    """Проверить полноту, уникальность и диапазон интегрального прогноза."""

    if not rows:
        raise ValueError("Файл интегрального прогноза не должен быть пустым.")

    scenario_ids: set[str] = set()
    protocol_ids: set[str] = set()
    for row in rows:
        if not row.scenario_id or not row.protocol_id:
            raise ValueError("Идентификаторы сценария и протокола не должны быть пустыми.")
        if row.scenario_id in scenario_ids:
            raise ValueError("Идентификаторы сценариев должны быть уникальными.")
        if row.protocol_id in protocol_ids:
            raise ValueError("Идентификаторы протоколов должны быть уникальными.")
        scenario_ids.add(row.scenario_id)
        protocol_ids.add(row.protocol_id)
        classify_q_pred(row.q_pred)


def q_pred_values(rows: Sequence[QPredRow]) -> np.ndarray:
    """Вернуть одномерный массив значений Q_pred."""

    validate_q_pred_rows(rows)
    values = np.asarray([row.q_pred for row in rows], dtype=float)
    if values.shape != (len(rows),):
        raise ValueError("Некорректная размерность массива Q_pred.")
    return values


def calculate_summary(rows: Sequence[QPredRow]) -> QPredSummary:
    """Рассчитать описательную статистику Q_pred."""

    values = q_pred_values(rows)
    return QPredSummary(
        count=len(rows),
        minimum=float(np.min(values)),
        first_quartile=float(np.quantile(values, 0.25)),
        median=float(median(values.tolist())),
        mean=float(mean(values.tolist())),
        third_quartile=float(np.quantile(values, 0.75)),
        maximum=float(np.max(values)),
        standard_deviation=float(np.std(values, ddof=0)),
    )


def calculate_class_counts(rows: Sequence[QPredRow]) -> tuple[QualityClassCount, ...]:
    """Рассчитать численность и долю low, medium и high."""

    validate_q_pred_rows(rows)
    counts = {spec.code: 0 for spec in QUALITY_CLASSES}
    for row in rows:
        counts[classify_q_pred(row.q_pred)] += 1

    total = len(rows)
    return tuple(
        QualityClassCount(
            code=spec.code,
            label=spec.label,
            count=counts[spec.code],
            share=counts[spec.code] / total,
            color=spec.color,
        )
        for spec in QUALITY_CLASSES
    )


def build_figure(rows: Sequence[QPredRow]) -> plt.Figure:
    """Построить гистограмму Q_pred и диаграмму численности классов."""

    values = q_pred_values(rows)
    summary = calculate_summary(rows)
    class_counts = calculate_class_counts(rows)

    configure_dissertation_style()
    figure = plt.figure(figsize=(17.6, 9.4))
    outer_grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(3.25, 1.30),
        left=0.065,
        right=0.975,
        top=0.82,
        bottom=0.18,
        wspace=0.22,
    )
    histogram_axis = figure.add_subplot(outer_grid[0, 0])
    right_grid = outer_grid[0, 1].subgridspec(
        2,
        1,
        height_ratios=(1.08, 0.92),
        hspace=0.28,
    )
    class_axis = figure.add_subplot(right_grid[0, 0])
    summary_axis = figure.add_subplot(right_grid[1, 0])

    figure.suptitle(
        "Рисунок 5.4 — Распределение интегрального прогнозного показателя Q_pred",
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )
    figure.text(
        0.5,
        0.918,
        "Q_pred = Σ_j w_j · q_j,pred;  Σ_j w_j = 1",
        ha="center",
        fontsize=12.0,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.882,
        f"N = {len(rows)} сценариев; пороги классов: low < {LOW_THRESHOLD:.2f}, "
        f"medium < {HIGH_THRESHOLD:.2f}, далее high.",
        ha="center",
        fontsize=10.7,
    )

    for spec in QUALITY_CLASSES:
        histogram_axis.axvspan(
            spec.lower,
            spec.upper,
            color=spec.color,
            alpha=0.105,
            linewidth=0,
            zorder=0,
        )

    bins = np.linspace(0.0, 1.0, 21)
    histogram_axis.hist(
        values,
        bins=bins,
        color="#547B9A",
        edgecolor="white",
        linewidth=0.9,
        alpha=0.90,
        zorder=2,
    )

    histogram_axis.axvline(
        LOW_THRESHOLD,
        color="#A64B4B",
        linewidth=2.0,
        linestyle="--",
        zorder=4,
    )
    histogram_axis.axvline(
        HIGH_THRESHOLD,
        color="#8B6F18",
        linewidth=2.0,
        linestyle="--",
        zorder=4,
    )
    histogram_axis.axvline(
        summary.mean,
        color="#1D6F8A",
        linewidth=2.2,
        linestyle="-",
        zorder=4,
    )
    histogram_axis.axvline(
        summary.median,
        color="#6F4A8E",
        linewidth=2.0,
        linestyle=":",
        zorder=4,
    )

    maximum_height = max(histogram_axis.get_ylim()[1], 1.0)
    histogram_axis.text(
        LOW_THRESHOLD,
        maximum_height * 0.965,
        "граница 0,45",
        ha="right",
        va="top",
        rotation=90,
        fontsize=9.2,
        color="#8E3F3F",
        fontweight="bold",
    )
    histogram_axis.text(
        HIGH_THRESHOLD,
        maximum_height * 0.965,
        "граница 0,70",
        ha="right",
        va="top",
        rotation=90,
        fontsize=9.2,
        color="#765D12",
        fontweight="bold",
    )

    class_count_by_code = {item.code: item for item in class_counts}
    zone_centers = (
        LOW_THRESHOLD / 2,
        (LOW_THRESHOLD + HIGH_THRESHOLD) / 2,
        (HIGH_THRESHOLD + 1.0) / 2,
    )
    for spec, center in zip(QUALITY_CLASSES, zone_centers, strict=True):
        item = class_count_by_code[spec.code]
        histogram_axis.text(
            center,
            maximum_height * 0.865,
            f"{spec.code}\n{item.count} ({item.share * 100:.1f}%)",
            ha="center",
            va="top",
            fontsize=10.0,
            color="#34495E",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.34",
                "facecolor": "white",
                "edgecolor": spec.color,
                "linewidth": 0.9,
                "alpha": 0.92,
            },
        )

    histogram_axis.set_xlim(0.0, 1.0)
    histogram_axis.set_xlabel("Интегральный априорный показатель Q_pred")
    histogram_axis.set_ylabel("Число сценариев")
    histogram_axis.set_title(
        "А. Гистограмма Q_pred и пороговые границы классов",
        fontweight="bold",
        pad=13,
    )
    histogram_axis.grid(axis="y", alpha=0.28, linewidth=0.8)
    histogram_axis.set_axisbelow(True)
    histogram_axis.legend(
        handles=(
            Line2D([0], [0], color="#1D6F8A", linewidth=2.2, label=f"Среднее = {summary.mean:.3f}"),
            Line2D([0], [0], color="#6F4A8E", linewidth=2.0, linestyle=":", label=f"Медиана = {summary.median:.3f}"),
            Patch(facecolor="#D8E4EC", edgecolor="#547B9A", label="Гистограмма"),
        ),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        frameon=False,
        fontsize=9.3,
    )

    class_axis.set_title(
        "Б. Численность классов качества",
        fontweight="bold",
        pad=12,
    )
    y_positions = np.arange(len(class_counts), dtype=float)[::-1]
    counts = np.asarray([item.count for item in class_counts], dtype=float)
    bars = class_axis.barh(
        y_positions,
        counts,
        color=[item.color for item in class_counts],
        alpha=0.82,
        height=0.56,
    )
    class_axis.set_yticks(y_positions)
    class_axis.set_yticklabels(
        [f"{item.label}\n{item.code}" for item in class_counts],
        fontsize=10.0,
    )
    class_axis.set_xlabel("Число сценариев")
    class_axis.set_xlim(0.0, max(counts) * 1.30)
    class_axis.grid(axis="x", alpha=0.24, linewidth=0.8)
    class_axis.set_axisbelow(True)
    class_axis.spines[["top", "right"]].set_visible(False)
    for bar, item in zip(bars, class_counts, strict=True):
        class_axis.text(
            bar.get_width() + max(counts) * 0.035,
            bar.get_y() + bar.get_height() / 2,
            f"{item.count}\n{item.share * 100:.1f}%",
            va="center",
            ha="left",
            fontsize=10.0,
            fontweight="bold",
            color="#2C3E50",
        )

    summary_axis.set_axis_off()
    summary_axis.set_title(
        "В. Описательная статистика",
        fontweight="bold",
        pad=12,
    )
    summary_lines = (
        ("Минимум", summary.minimum),
        ("Первый квартиль", summary.first_quartile),
        ("Медиана", summary.median),
        ("Среднее", summary.mean),
        ("Третий квартиль", summary.third_quartile),
        ("Максимум", summary.maximum),
        ("Стандартное отклонение", summary.standard_deviation),
    )
    y_start = 0.88
    y_step = 0.105
    for index, (label, value) in enumerate(summary_lines):
        y = y_start - index * y_step
        summary_axis.text(
            0.04,
            y,
            label,
            transform=summary_axis.transAxes,
            fontsize=9.7,
            va="center",
        )
        summary_axis.text(
            0.96,
            y,
            f"{value:.6f}",
            transform=summary_axis.transAxes,
            fontsize=9.7,
            va="center",
            ha="right",
            fontweight="bold",
        )
        if index < len(summary_lines) - 1:
            summary_axis.plot(
                [0.04, 0.96],
                [y - 0.052, y - 0.052],
                transform=summary_axis.transAxes,
                color="#D7DBDD",
                linewidth=0.7,
            )

    figure.text(
        0.5,
        0.055,
        "Методическое ограничение: классы low, medium и high сформированы по заранее заданным порогам Q_pred. "
        "Их распределение характеризует структуру априорного индекса, но само по себе не подтверждает внешнюю точность или абсолютную калибровку прогноза.",
        ha="center",
        va="center",
        fontsize=9.7,
        bbox={
            "boxstyle": "round,pad=0.50",
            "facecolor": "#FFF8E7",
            "edgecolor": "#C9A227",
            "linewidth": 0.9,
        },
    )

    return figure


def generate(
    *,
    project_root: str | Path,
    input_path: str | Path | None = None,
    dpi: int = 300,
) -> FigureExportResult:
    """Сформировать рисунок 5.4 в PNG и SVG."""

    root = Path(project_root).resolve()
    source = Path(input_path) if input_path is not None else root / DEFAULT_INPUT_PATH
    rows = load_q_pred(source)
    figure = build_figure(rows)
    return export_figure(
        figure,
        project_root=root,
        relative_output_dir=OUTPUT_DIR,
        file_stem=FILE_STEM,
        dpi=dpi,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Создать парсер аргументов командной строки."""

    parser = argparse.ArgumentParser(
        description="Сформировать рисунок 5.4 с распределением Q_pred."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Корень проекта manual_coding_quality.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Путь к q_pred.csv; по умолчанию используется reports/chapter5/q_pred.csv.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Разрешение PNG; по умолчанию 300 dpi.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить CLI генератора рисунка 5.4."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    result = generate(
        project_root=args.project_root,
        input_path=args.input,
        dpi=args.dpi,
    )
    print("Рисунок 5.4 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
