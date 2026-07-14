"""Генерация рисунка 5.5 с неопределённостью и радиусом интервала."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

from manual_coding_sim.dissertation_figures.common import (
    FigureExportResult,
    configure_dissertation_style,
    export_figure,
)

OUTPUT_DIR = Path("reports/chapter5/figures")
FILE_STEM = "figure_5_5_uncertainty_and_interval_radius"
DEFAULT_INPUT_PATH = Path("reports/chapter5/prediction_uncertainty.csv")
DEFAULT_REPORT_PATH = Path("reports/chapter5/prediction_uncertainty_report.json")
DEFAULT_DELTA = 0.15
DEFAULT_WEIGHTS = {
    "theta_entropy": 0.50,
    "lda_stability": 0.30,
    "input_quality": 0.20,
}


@dataclass(frozen=True, slots=True)
class UncertaintyRow:
    """Показатели неопределённости одного сценария."""

    scenario_id: str
    protocol_id: str
    q_pred: float
    uncertainty_score: float
    interval_radius: float
    q_pred_lower: float
    q_pred_upper: float


@dataclass(frozen=True, slots=True)
class UncertaintyMetadata:
    """Параметры алгоритма интервальной диагностики."""

    delta: float
    weights: Mapping[str, float]
    mean_stability: float | None
    input_missing_share: float | None


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    """Описательная статистика одного числового показателя."""

    minimum: float
    first_quartile: float
    median: float
    mean: float
    third_quartile: float
    maximum: float
    standard_deviation: float


@dataclass(frozen=True, slots=True)
class UncertaintySummary:
    """Сводка неопределённости и радиуса интервала."""

    count: int
    uncertainty: DistributionSummary
    radius: DistributionSummary
    delta: float
    pearson: float


def load_uncertainty(path: str | Path) -> tuple[UncertaintyRow, ...]:
    """Загрузить ``prediction_uncertainty.csv`` и проверить его структуру."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Не найден файл неопределённости прогноза: {source}")

    required = {
        "scenario_id",
        "protocol_id",
        "q_pred",
        "uncertainty_score",
        "interval_radius",
        "q_pred_lower",
        "q_pred_upper",
    }
    rows: list[UncertaintyRow] = []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("CSV-файл не содержит строки заголовка.")
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "В prediction_uncertainty.csv отсутствуют обязательные колонки: "
                + ", ".join(sorted(missing))
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                rows.append(
                    UncertaintyRow(
                        scenario_id=str(row["scenario_id"]).strip(),
                        protocol_id=str(row["protocol_id"]).strip(),
                        q_pred=float(row["q_pred"]),
                        uncertainty_score=float(row["uncertainty_score"]),
                        interval_radius=float(row["interval_radius"]),
                        q_pred_lower=float(row["q_pred_lower"]),
                        q_pred_upper=float(row["q_pred_upper"]),
                    )
                )
            except (TypeError, ValueError, KeyError) as error:
                raise ValueError(
                    f"Некорректная строка {row_number} prediction_uncertainty.csv: {error}."
                ) from error

    result = tuple(rows)
    validate_uncertainty_rows(result)
    return result


def load_metadata(path: str | Path | None) -> UncertaintyMetadata:
    """Загрузить параметры расчёта или вернуть параметры по умолчанию."""

    if path is None:
        return UncertaintyMetadata(
            delta=DEFAULT_DELTA,
            weights=DEFAULT_WEIGHTS,
            mean_stability=None,
            input_missing_share=None,
        )

    source = Path(path)
    if not source.is_file():
        return UncertaintyMetadata(
            delta=DEFAULT_DELTA,
            weights=DEFAULT_WEIGHTS,
            mean_stability=None,
            input_missing_share=None,
        )

    payload = json.loads(source.read_text(encoding="utf-8"))
    delta = float(payload.get("delta", DEFAULT_DELTA))
    raw_weights = payload.get("weights", DEFAULT_WEIGHTS)
    if not isinstance(raw_weights, Mapping):
        raise ValueError("Поле weights отчёта неопределённости должно быть словарём.")
    weights = {
        key: float(raw_weights.get(key, DEFAULT_WEIGHTS[key]))
        for key in DEFAULT_WEIGHTS
    }
    metadata = UncertaintyMetadata(
        delta=delta,
        weights=weights,
        mean_stability=(
            float(payload["mean_stability"])
            if payload.get("mean_stability") is not None
            else None
        ),
        input_missing_share=(
            float(payload["input_missing_share"])
            if payload.get("input_missing_share") is not None
            else None
        ),
    )
    validate_metadata(metadata)
    return metadata


def validate_metadata(metadata: UncertaintyMetadata) -> None:
    """Проверить параметры формулы неопределённости."""

    if not math.isfinite(metadata.delta) or metadata.delta <= 0.0:
        raise ValueError("Параметр delta должен быть положительным конечным числом.")
    weight_sum = sum(metadata.weights.values())
    if any(not math.isfinite(value) or value < 0.0 for value in metadata.weights.values()):
        raise ValueError("Веса источников неопределённости должны быть неотрицательными.")
    if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Сумма весов источников неопределённости должна быть равна единице.")


def validate_uncertainty_rows(rows: Sequence[UncertaintyRow]) -> None:
    """Проверить диапазоны, уникальность и согласованность интервальных данных."""

    if not rows:
        raise ValueError("Таблица неопределённости не должна быть пустой.")

    keys: set[tuple[str, str]] = set()
    for row in rows:
        if not row.scenario_id or not row.protocol_id:
            raise ValueError("Идентификаторы сценария и протокола не должны быть пустыми.")
        key = (row.scenario_id, row.protocol_id)
        if key in keys:
            raise ValueError("Пары scenario_id и protocol_id должны быть уникальными.")
        keys.add(key)

        numeric_values = (
            row.q_pred,
            row.uncertainty_score,
            row.interval_radius,
            row.q_pred_lower,
            row.q_pred_upper,
        )
        if any(not math.isfinite(value) for value in numeric_values):
            raise ValueError("Все числовые значения должны быть конечными.")
        if not 0.0 <= row.q_pred <= 1.0:
            raise ValueError("Q_pred должен лежать в диапазоне [0; 1].")
        if not 0.0 <= row.uncertainty_score <= 1.0:
            raise ValueError("uncertainty_score должен лежать в диапазоне [0; 1].")
        if not 0.0 <= row.interval_radius <= 1.0:
            raise ValueError("interval_radius должен лежать в диапазоне [0; 1].")
        if not 0.0 <= row.q_pred_lower <= row.q_pred_upper <= 1.0:
            raise ValueError("Границы интервала должны быть упорядочены в диапазоне [0; 1].")
        if row.q_pred < row.q_pred_lower - 1e-9 or row.q_pred > row.q_pred_upper + 1e-9:
            raise ValueError("Q_pred должен находиться внутри собственного интервала.")


def infer_delta(rows: Sequence[UncertaintyRow]) -> float:
    """Определить коэффициент пропорциональности r = delta · U по данным."""

    validate_uncertainty_rows(rows)
    ratios = [
        row.interval_radius / row.uncertainty_score
        for row in rows
        if row.uncertainty_score > 1e-12
    ]
    if not ratios:
        raise ValueError("Невозможно определить delta при нулевой неопределённости всех строк.")
    delta = float(median(ratios))
    if any(not math.isclose(value, delta, rel_tol=0.0, abs_tol=1e-9) for value in ratios):
        raise ValueError("interval_radius не является постоянной долей uncertainty_score.")
    return delta


def validate_proportionality(
    rows: Sequence[UncertaintyRow],
    *,
    expected_delta: float,
) -> None:
    """Проверить формулу ``interval_radius = delta · uncertainty_score``."""

    inferred = infer_delta(rows)
    if not math.isclose(inferred, expected_delta, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "Коэффициент пропорциональности интервального радиуса не совпадает с delta: "
            f"получено {inferred:.12f}, ожидалось {expected_delta:.12f}."
        )


def _distribution_summary(values: np.ndarray) -> DistributionSummary:
    """Рассчитать описательную статистику массива."""

    return DistributionSummary(
        minimum=float(np.min(values)),
        first_quartile=float(np.quantile(values, 0.25)),
        median=float(np.median(values)),
        mean=float(np.mean(values)),
        third_quartile=float(np.quantile(values, 0.75)),
        maximum=float(np.max(values)),
        standard_deviation=float(np.std(values, ddof=0)),
    )


def calculate_summary(
    rows: Sequence[UncertaintyRow],
    *,
    delta: float,
) -> UncertaintySummary:
    """Рассчитать сводные характеристики двух распределений."""

    validate_proportionality(rows, expected_delta=delta)
    uncertainty = np.asarray([row.uncertainty_score for row in rows], dtype=float)
    radius = np.asarray([row.interval_radius for row in rows], dtype=float)
    if np.std(uncertainty) <= 1e-15 or np.std(radius) <= 1e-15:
        pearson = 1.0
    else:
        pearson = float(np.corrcoef(uncertainty, radius)[0, 1])
    return UncertaintySummary(
        count=len(rows),
        uncertainty=_distribution_summary(uncertainty),
        radius=_distribution_summary(radius),
        delta=delta,
        pearson=pearson,
    )


def _format_decimal(value: float, digits: int = 4) -> str:
    """Отформатировать число с десятичной запятой."""

    return f"{value:.{digits}f}".replace(".", ",")


def _draw_summary_box(
    axis: plt.Axes,
    *,
    summary: UncertaintySummary,
    metadata: UncertaintyMetadata,
) -> None:
    """Нарисовать компактную сводку параметров алгоритма."""

    axis.set_axis_off()
    box = FancyBboxPatch(
        (0.02, 0.05),
        0.96,
        0.90,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        transform=axis.transAxes,
        facecolor="#F5F7FA",
        edgecolor="#7B8794",
        linewidth=1.2,
    )
    axis.add_patch(box)

    axis.text(
        0.08,
        0.85,
        "Параметры расчёта",
        transform=axis.transAxes,
        fontsize=11.2,
        fontweight="bold",
        va="top",
    )
    rows = [
        f"N = {summary.count}",
        f"δ = {_format_decimal(summary.delta, 2)}",
        f"mean(U) = {_format_decimal(summary.uncertainty.mean, 4)}",
        f"mean(r) = {_format_decimal(summary.radius.mean, 4)}",
        f"Pearson(U, r) = {_format_decimal(summary.pearson, 4)}",
    ]
    axis.text(
        0.08,
        0.68,
        "\n".join(rows),
        transform=axis.transAxes,
        fontsize=9.25,
        va="top",
        linespacing=1.20,
    )



def build_figure(
    rows: Sequence[UncertaintyRow],
    *,
    metadata: UncertaintyMetadata,
) -> plt.Figure:
    """Построить scatter-график с маргинальными распределениями."""

    validate_metadata(metadata)
    summary = calculate_summary(rows, delta=metadata.delta)
    uncertainty = np.asarray([row.uncertainty_score for row in rows], dtype=float)
    radius = np.asarray([row.interval_radius for row in rows], dtype=float)

    configure_dissertation_style()
    figure = plt.figure(figsize=(17.4, 9.8))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(4.25, 1.35),
        height_ratios=(1.45, 4.20),
        left=0.075,
        right=0.965,
        top=0.82,
        bottom=0.17,
        hspace=0.20,
        wspace=0.13,
    )
    top_axis = figure.add_subplot(grid[0, 0])
    summary_axis = figure.add_subplot(grid[0, 1])
    scatter_axis = figure.add_subplot(grid[1, 0], sharex=top_axis)
    right_axis = figure.add_subplot(grid[1, 1], sharey=scatter_axis)

    figure.suptitle(
        "Рисунок 5.5 — Неопределённость прогноза и радиус диагностического интервала",
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )
    figure.text(
        0.5,
        0.918,
        "interval_radius = δ · uncertainty_score;  δ = 0,15",
        ha="center",
        fontsize=12.1,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.882,
        "U = 0,50 · Hθ + 0,30 · (1 − mean_stability) + 0,20 · input_missing_share",
        ha="center",
        fontsize=10.7,
    )

    x_limit = max(0.60, float(uncertainty.max()) * 1.08)
    y_limit = max(0.09, float(radius.max()) * 1.10)
    x_bins = np.linspace(0.0, x_limit, 17)
    y_bins = np.linspace(0.0, y_limit, 17)

    top_axis.hist(
        uncertainty,
        bins=x_bins,
        color="#5A84A2",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.92,
    )
    top_axis.axvline(summary.uncertainty.mean, color="#8A3D52", linewidth=1.8)
    top_axis.axvline(
        summary.uncertainty.median,
        color="#6F4A8E",
        linewidth=1.8,
        linestyle="--",
    )
    top_axis.set_ylabel("Частота")
    top_axis.set_title("Маргинальное распределение uncertainty_score", loc="left", pad=8)
    top_axis.grid(axis="y", linestyle=":", alpha=0.35)
    top_axis.tick_params(axis="x", labelbottom=False)
    top_axis.spines[["top", "right"]].set_visible(False)

    scatter_axis.scatter(
        uncertainty,
        radius,
        s=36,
        color="#355F7A",
        edgecolor="white",
        linewidth=0.45,
        alpha=0.78,
        zorder=3,
    )
    line_x = np.linspace(0.0, x_limit, 200)
    scatter_axis.plot(
        line_x,
        metadata.delta * line_x,
        color="#A33E3E",
        linewidth=2.2,
        linestyle="--",
        zorder=2,
        label="r = 0,15 · U",
    )
    scatter_axis.scatter(
        [summary.uncertainty.mean],
        [summary.radius.mean],
        s=105,
        marker="D",
        color="#D4A72C",
        edgecolor="#513E00",
        linewidth=0.9,
        zorder=5,
    )
    scatter_axis.annotate(
        "среднее",
        xy=(summary.uncertainty.mean, summary.radius.mean),
        xytext=(13, -18),
        textcoords="offset points",
        fontsize=9.4,
        fontweight="bold",
        color="#6B5200",
    )
    scatter_axis.set_xlim(0.0, x_limit)
    scatter_axis.set_ylim(0.0, y_limit)
    scatter_axis.set_xlabel("uncertainty_score, U")
    scatter_axis.set_ylabel("interval_radius, r")
    scatter_axis.text(
        0.52,
        0.965,
        "Линейная зависимость радиуса от априорной неопределённости",
        transform=scatter_axis.transAxes,
        ha="center",
        va="top",
        fontsize=11.3,
        fontweight="bold",
        color="#2F3740",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
        },
    )
    scatter_axis.grid(linestyle=":", alpha=0.38)
    scatter_axis.spines[["top", "right"]].set_visible(False)
    scatter_axis.legend(
        handles=[
            Line2D([0], [0], color="#A33E3E", linestyle="--", linewidth=2.2, label="r = 0,15 · U"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="#D4A72C", markeredgecolor="#513E00", markersize=7, label="Средняя точка"),
        ],
        loc="upper left",
        frameon=True,
        fontsize=9.4,
    )

    right_axis.hist(
        radius,
        bins=y_bins,
        orientation="horizontal",
        color="#B88746",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.90,
    )
    right_axis.axhline(summary.radius.mean, color="#8A3D52", linewidth=1.8)
    right_axis.axhline(
        summary.radius.median,
        color="#6F4A8E",
        linewidth=1.8,
        linestyle="--",
    )
    right_axis.set_xlabel("Частота")
    right_axis.set_title("Распределение\ninterval_radius", pad=6, fontsize=11.0)
    right_axis.grid(axis="x", linestyle=":", alpha=0.35)
    right_axis.tick_params(axis="y", labelleft=False)
    right_axis.spines[["top", "right"]].set_visible(False)

    _draw_summary_box(summary_axis, summary=summary, metadata=metadata)

    top_axis.legend(
        handles=[
            Line2D([0], [0], color="#8A3D52", linewidth=1.8, label="Среднее"),
            Line2D([0], [0], color="#6F4A8E", linewidth=1.8, linestyle="--", label="Медиана"),
        ],
        loc="upper right",
        ncol=2,
        frameon=False,
        fontsize=9.0,
    )

    figure.text(
        0.075,
        0.105,
        "Методическое ограничение: линейность r = δ · U задана алгоритмом. "
        "Она не доказывает абсолютную калибровку, вероятностное покрытие или "
        "соответствие величины интервала фактической ошибке прогноза.",
        ha="left",
        va="center",
        fontsize=10.0,
        color="#3F4650",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#FFF8E8",
            "edgecolor": "#B98A2E",
            "linewidth": 1.0,
        },
    )
    figure.text(
        0.075,
        0.045,
        "Источник: reports/chapter5/prediction_uncertainty.csv; "
        "маргинальные распределения построены по тем же 150 сценариям.",
        ha="left",
        fontsize=9.2,
        color="#5C6570",
    )
    return figure


def generate(
    *,
    project_root: str | Path,
    input_path: str | Path | None = None,
    report_path: str | Path | None = None,
    dpi: int = 300,
) -> FigureExportResult:
    """Сформировать рисунок 5.5 в PNG и SVG."""

    root = Path(project_root).resolve()
    source = Path(input_path) if input_path is not None else root / DEFAULT_INPUT_PATH
    if report_path is None:
        metadata_source: Path | None = root / DEFAULT_REPORT_PATH
    else:
        metadata_source = Path(report_path)
    rows = load_uncertainty(source)
    metadata = load_metadata(metadata_source)
    figure = build_figure(rows, metadata=metadata)
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
        description=(
            "Сформировать рисунок 5.5 с uncertainty_score, interval_radius "
            "и маргинальными распределениями."
        )
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
        help=(
            "Путь к prediction_uncertainty.csv; по умолчанию используется "
            "reports/chapter5/prediction_uncertainty.csv."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Путь к prediction_uncertainty_report.json; при отсутствии "
            "используются параметры главы 5 по умолчанию."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Разрешение PNG; по умолчанию 300 dpi.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить CLI генератора рисунка 5.5."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    result = generate(
        project_root=args.project_root,
        input_path=args.input,
        report_path=args.report,
        dpi=args.dpi,
    )
    print("Рисунок 5.5 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
