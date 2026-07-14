"""Генерация рисунка 6.3 с распределением абсолютной ошибки прогноза."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from manual_coding_sim.dissertation_figures.common import (
    FigureExportResult,
    configure_dissertation_style,
    export_figure,
)
from manual_coding_sim.dissertation_figures.figure_6_2_residuals_vs_q_fact import (
    DEFAULT_Q_FACT_PATH,
    DEFAULT_Q_PRED_PATH,
    ResidualPoint,
    load_q_fact,
    load_q_pred,
    merge_residual_points,
)

OUTPUT_DIR = Path("reports/chapter6/figures")
FILE_STEM = "absolute_error_distribution"
DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30)


@dataclass(frozen=True, slots=True)
class AbsoluteErrorSummary:
    """Сводные характеристики абсолютной ошибки интегрального прогноза."""

    count: int
    minimum: float
    q1: float
    median: float
    mean: float
    q3: float
    p90: float
    p95: float
    maximum: float
    standard_deviation: float
    rmse: float
    maximum_scenario: str
    maximum_protocol: str
    threshold_counts: tuple[tuple[float, int, float], ...]


def calculate_absolute_errors(points: Sequence[ResidualPoint]) -> np.ndarray:
    """Вернуть массив абсолютных ошибок |Q_pred - Q_fact|."""

    if len(points) < 3:
        raise ValueError("Для анализа абсолютной ошибки требуется не менее трёх сценариев.")
    errors = np.asarray([abs(point.residual) for point in points], dtype=float)
    if not np.all(np.isfinite(errors)):
        raise ValueError("Абсолютные ошибки не должны содержать NaN или бесконечность.")
    if np.any(errors < 0.0) or np.any(errors > 1.0):
        raise ValueError("Абсолютные ошибки должны находиться в диапазоне [0; 1].")
    return errors


def calculate_summary(
    points: Sequence[ResidualPoint],
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> AbsoluteErrorSummary:
    """Рассчитать показатели распределения абсолютной ошибки."""

    errors = calculate_absolute_errors(points)
    normalized_thresholds: list[float] = []
    for threshold in thresholds:
        value = float(threshold)
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError("Порог абсолютной ошибки должен принадлежать диапазону (0; 1].")
        normalized_thresholds.append(value)
    if len(set(normalized_thresholds)) != len(normalized_thresholds):
        raise ValueError("Пороги абсолютной ошибки не должны повторяться.")

    maximum_index = int(np.argmax(errors))
    threshold_counts = tuple(
        (
            threshold,
            int(np.sum(errors <= threshold)),
            float(np.mean(errors <= threshold)),
        )
        for threshold in sorted(normalized_thresholds)
    )
    residuals = np.asarray([point.residual for point in points], dtype=float)
    return AbsoluteErrorSummary(
        count=len(points),
        minimum=float(np.min(errors)),
        q1=float(np.quantile(errors, 0.25)),
        median=float(np.median(errors)),
        mean=float(np.mean(errors)),
        q3=float(np.quantile(errors, 0.75)),
        p90=float(np.quantile(errors, 0.90)),
        p95=float(np.quantile(errors, 0.95)),
        maximum=float(np.max(errors)),
        standard_deviation=float(np.std(errors, ddof=1)),
        rmse=float(np.sqrt(np.mean(np.square(residuals)))),
        maximum_scenario=points[maximum_index].scenario_id,
        maximum_protocol=points[maximum_index].protocol_id,
        threshold_counts=threshold_counts,
    )


def empirical_cdf(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Построить упорядоченные координаты эмпирической функции распределения."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("Для ЭФР требуется непустой одномерный ряд.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Ряд ЭФР не должен содержать NaN или бесконечность.")
    x_values = np.sort(array)
    probabilities = np.arange(1, len(array) + 1, dtype=float) / len(array)
    return x_values, probabilities


def _add_summary_card(
    axis: plt.Axes,
    *,
    y: float,
    title: str,
    lines: Sequence[str],
    facecolor: str,
    height: float,
) -> None:
    """Добавить информационную карточку на сводную панель."""

    patch = FancyBboxPatch(
        (0.035, y - height),
        0.93,
        height,
        transform=axis.transAxes,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="#7a8793",
        facecolor=facecolor,
    )
    axis.add_patch(patch)
    axis.text(
        0.075,
        y - 0.032,
        title,
        transform=axis.transAxes,
        fontsize=12.1,
        fontweight="bold",
        va="top",
    )
    axis.text(
        0.075,
        y - 0.080,
        "\n".join(lines),
        transform=axis.transAxes,
        fontsize=10.2,
        va="top",
        linespacing=1.28,
    )


def build_figure(
    points: Sequence[ResidualPoint],
    summary: AbsoluteErrorSummary,
) -> plt.Figure:
    """Построить гистограмму и ЭФР абсолютной ошибки."""

    configure_dissertation_style()
    errors = calculate_absolute_errors(points)
    cdf_x, cdf_y = empirical_cdf(errors)

    figure = plt.figure(figsize=(18.0, 9.0))
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=(1.72, 1.42, 0.92),
        height_ratios=(1.0, 1.0),
        left=0.055,
        right=0.965,
        top=0.835,
        bottom=0.155,
        wspace=0.20,
        hspace=0.31,
    )
    histogram_axis = figure.add_subplot(grid[:, 0])
    cdf_axis = figure.add_subplot(grid[0, 1])
    threshold_axis = figure.add_subplot(grid[1, 1])
    info_axis = figure.add_subplot(grid[:, 2])
    info_axis.axis("off")

    upper = max(0.40, summary.maximum + 0.025)
    bins = np.linspace(0.0, upper, 17)
    histogram_axis.hist(
        errors,
        bins=bins,
        edgecolor="white",
        linewidth=0.9,
        alpha=0.86,
    )
    histogram_axis.axvline(
        summary.mean,
        linestyle="-",
        linewidth=2.4,
        color="#c0392b",
        label=f"MAE = {summary.mean:.3f}",
    )
    histogram_axis.axvline(
        summary.median,
        linestyle="--",
        linewidth=2.2,
        color="#2c3e50",
        label=f"Median |e| = {summary.median:.3f}",
    )
    histogram_axis.axvline(
        summary.maximum,
        linestyle=":",
        linewidth=2.4,
        color="#8e44ad",
        label=f"Max |e| = {summary.maximum:.3f}",
    )
    histogram_axis.annotate(
        f"максимальная ошибка\n{summary.maximum_scenario}\n|e| = {summary.maximum:.3f}",
        xy=(summary.maximum, 0.5),
        xycoords=("data", "axes fraction"),
        xytext=(-145, 66),
        textcoords="offset points",
        fontsize=10.0,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.94},
        arrowprops={"arrowstyle": "->", "linewidth": 1.1},
    )
    histogram_axis.set_xlim(0.0, upper)
    histogram_axis.set_xlabel(r"Абсолютная ошибка $|e_i|=|Q_{pred,i}-Q_{fact,i}|$")
    histogram_axis.set_ylabel("Число сценариев")
    histogram_axis.set_title(
        "Гистограмма абсолютной ошибки",
        fontweight="bold",
        pad=12,
    )
    histogram_axis.grid(axis="y", alpha=0.24)
    histogram_axis.legend(loc="upper right", fontsize=10.0, framealpha=0.96)

    cdf_axis.step(
        cdf_x,
        cdf_y,
        where="post",
        linewidth=2.7,
        color="#2468a2",
    )
    cdf_axis.fill_between(cdf_x, cdf_y, step="post", alpha=0.10, color="#2468a2")
    for threshold, count, share in summary.threshold_counts:
        cdf_axis.plot(threshold, share, "o", markersize=6.5)
        if threshold in (0.10, 0.20, 0.30):
            cdf_axis.annotate(
                f"≤ {threshold:.2f}: {share * 100:.1f}%",
                xy=(threshold, share),
                xytext=(6, 8),
                textcoords="offset points",
                fontsize=9.3,
            )
    cdf_axis.axhline(0.5, linestyle=":", linewidth=1.3, color="#6c757d")
    cdf_axis.axvline(summary.median, linestyle="--", linewidth=1.7, color="#2c3e50")
    cdf_axis.set_xlim(0.0, upper)
    cdf_axis.set_ylim(0.0, 1.02)
    cdf_axis.set_xlabel(r"Порог абсолютной ошибки $\varepsilon$")
    cdf_axis.set_ylabel(r"$P(|e_i|\leq\varepsilon)$")
    cdf_axis.set_title(
        "Эмпирическая функция распределения",
        fontweight="bold",
        pad=10,
    )
    cdf_axis.grid(alpha=0.25)

    thresholds = [item[0] for item in summary.threshold_counts]
    shares = [item[2] * 100.0 for item in summary.threshold_counts]
    bars = threshold_axis.barh(
        [f"|e| ≤ {threshold:.2f}" for threshold in thresholds],
        shares,
        alpha=0.85,
    )
    threshold_axis.set_xlim(0.0, 100.0)
    threshold_axis.set_xlabel("Доля сценариев, %")
    threshold_axis.set_title(
        "Покрытие фиксированных порогов ошибки",
        fontweight="bold",
        pad=10,
    )
    threshold_axis.grid(axis="x", alpha=0.24)
    for bar, (_, count, share) in zip(bars, summary.threshold_counts, strict=True):
        threshold_axis.text(
            min(share * 100.0 + 1.3, 94.0),
            bar.get_y() + bar.get_height() / 2,
            f"{count} ({share * 100:.1f}%)",
            va="center",
            fontsize=9.6,
        )

    _add_summary_card(
        info_axis,
        y=0.98,
        title="Центр распределения",
        lines=(
            f"MAE = {summary.mean:.6f}",
            f"Median |e| = {summary.median:.6f}",
            f"RMSE = {summary.rmse:.6f}",
            f"σ(|e|) = {summary.standard_deviation:.6f}",
        ),
        facecolor="#eef5fb",
        height=0.225,
    )
    _add_summary_card(
        info_axis,
        y=0.70,
        title="Квартильная структура",
        lines=(
            f"min = {summary.minimum:.6f}",
            f"Q1 = {summary.q1:.6f}",
            f"Q3 = {summary.q3:.6f}",
            f"P90 = {summary.p90:.6f}",
            f"P95 = {summary.p95:.6f}",
        ),
        facecolor="#f7f4ec",
        height=0.255,
    )
    _add_summary_card(
        info_axis,
        y=0.40,
        title="Максимальная абсолютная ошибка",
        lines=(
            f"Max |e| = {summary.maximum:.6f}",
            f"сценарий: {summary.maximum_scenario}",
            f"протокол: {summary.maximum_protocol}",
            f"N = {summary.count}",
        ),
        facecolor="#fbefef",
        height=0.205,
    )
    _add_summary_card(
        info_axis,
        y=0.17,
        title="Интерпретация",
        lines=(
            "|e| измеряет величину расхождения",
            "без направления; меньшая ошибка лучше.",
        ),
        facecolor="#f1f6f0",
        height=0.135,
    )

    figure.suptitle(
        "Рисунок 6.3 — Распределение абсолютной ошибки интегрального прогноза",
        fontsize=17,
        fontweight="bold",
        y=0.965,
    )
    figure.text(
        0.5,
        0.905,
        "MAE характеризует среднюю величину расхождения, Median |e| — типичную ошибку, Max |e| — крайнее наблюдаемое отклонение.",
        ha="center",
        fontsize=11.3,
    )
    figure.text(
        0.5,
        0.045,
        "Методическое примечание: распределение |e| используется для внешней диагностики величины ошибки. "
        "Оно не сохраняет направление смещения и само по себе не подтверждает абсолютную калибровку, "
        "вероятностное покрытие или переносимость модели за пределы вычислительного корпуса.",
        ha="center",
        va="bottom",
        fontsize=10.6,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#f3f4f6",
            "edgecolor": "#9aa3ad",
        },
    )
    return figure


def generate(
    *,
    project_root: Path,
    q_pred_path: Path | None = None,
    q_fact_path: Path | None = None,
    dpi: int = 300,
) -> FigureExportResult:
    """Загрузить данные, построить рисунок и экспортировать PNG/SVG."""

    root = project_root.resolve()
    predictions = load_q_pred(root / (q_pred_path or DEFAULT_Q_PRED_PATH))
    facts = load_q_fact(root / (q_fact_path or DEFAULT_Q_FACT_PATH))
    points = merge_residual_points(predictions, facts)
    summary = calculate_summary(points)
    figure = build_figure(points, summary)
    return export_figure(
        figure,
        project_root=root,
        relative_output_dir=OUTPUT_DIR,
        file_stem=FILE_STEM,
        dpi=dpi,
    )


def build_parser() -> argparse.ArgumentParser:
    """Создать CLI-парсер генератора рисунка 6.3."""

    parser = argparse.ArgumentParser(
        description="Сформировать распределение абсолютной ошибки Q_pred относительно Q_fact."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--q-pred", type=Path, default=None)
    parser.add_argument("--q-fact", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить генерацию рисунка из командной строки."""

    args = build_parser().parse_args(argv)
    result = generate(
        project_root=args.project_root,
        q_pred_path=args.q_pred,
        q_fact_path=args.q_fact,
        dpi=args.dpi,
    )
    print("Рисунок 6.3 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
