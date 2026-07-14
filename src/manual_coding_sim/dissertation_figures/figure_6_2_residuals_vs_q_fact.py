"""Генерация рисунка 6.2 с остатками прогноза относительно Q_fact."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from manual_coding_sim.dissertation_figures.common import (
    FigureExportResult,
    configure_dissertation_style,
    export_figure,
)

OUTPUT_DIR = Path("reports/chapter6/figures")
FILE_STEM = "residuals_vs_q_fact"
DEFAULT_Q_PRED_PATH = Path("reports/chapter5/q_pred.csv")
DEFAULT_Q_FACT_PATH = Path("data/processed/quality_targets.csv")
KEY_COLUMNS = ("scenario_id", "protocol_id")
DEFAULT_SMOOTHING_BANDWIDTH = 0.12


@dataclass(frozen=True, slots=True)
class PredictionValue:
    """Интегральный априорный прогноз одного сценария."""

    scenario_id: str
    protocol_id: str
    q_pred: float


@dataclass(frozen=True, slots=True)
class FactValue:
    """Фактическое интегральное качество одного сценария."""

    scenario_id: str
    protocol_id: str
    q_fact: float


@dataclass(frozen=True, slots=True)
class ResidualPoint:
    """Согласованная пара качества и соответствующий остаток прогноза."""

    scenario_id: str
    protocol_id: str
    q_pred: float
    q_fact: float

    @property
    def residual(self) -> float:
        """Вернуть остаток e = Q_pred - Q_fact."""

        return self.q_pred - self.q_fact


@dataclass(frozen=True, slots=True)
class ResidualSummary:
    """Сводные характеристики распределения остатков."""

    count: int
    bias: float
    residual_std: float
    residual_min: float
    residual_q1: float
    residual_median: float
    residual_q3: float
    residual_max: float
    underestimation_count: int
    overestimation_count: int
    zero_count: int
    correlation_with_q_fact: float
    minimum_scenario: str
    maximum_scenario: str


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Прочитать CSV-файл с проверкой наличия заголовка."""

    if not path.is_file():
        raise FileNotFoundError(f"Не найден входной файл: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV-файл не содержит заголовка: {path}")
        return list(reader)


def _parse_probability(raw: str, *, column: str, row_number: int) -> float:
    """Преобразовать показатель качества и проверить диапазон [0; 1]."""

    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Колонка {column}: строка {row_number} содержит нечисловое значение."
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Колонка {column}: строка {row_number} содержит NaN или бесконечность."
        )
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"Колонка {column}: значение {value} вне допустимого диапазона [0; 1]."
        )
    return value


def _validate_unique_keys(
    rows: Iterable[PredictionValue | FactValue], *, source_name: str
) -> None:
    """Проверить непустые и уникальные составные ключи."""

    keys: set[tuple[str, str]] = set()
    count = 0
    for row in rows:
        count += 1
        key = (row.scenario_id, row.protocol_id)
        if not all(key):
            raise ValueError(f"В {source_name} обнаружен пустой ключ сценария.")
        if key in keys:
            raise ValueError(
                f"Ключи scenario_id и protocol_id в {source_name} должны быть уникальными."
            )
        keys.add(key)
    if count == 0:
        raise ValueError(f"В {source_name} отсутствуют строки данных.")


def load_q_pred(path: Path) -> tuple[PredictionValue, ...]:
    """Загрузить прогнозные значения Q_pred."""

    rows = _read_rows(path)
    required = {*KEY_COLUMNS, "q_pred"}
    missing = required.difference(rows[0].keys()) if rows else required
    if missing:
        raise ValueError(
            "В q_pred.csv отсутствуют обязательные колонки: "
            + ", ".join(sorted(missing))
        )
    result = tuple(
        PredictionValue(
            scenario_id=row["scenario_id"].strip(),
            protocol_id=row["protocol_id"].strip(),
            q_pred=_parse_probability(
                row["q_pred"], column="q_pred", row_number=index
            ),
        )
        for index, row in enumerate(rows, start=2)
    )
    _validate_unique_keys(result, source_name="q_pred.csv")
    return result


def load_q_fact(path: Path) -> tuple[FactValue, ...]:
    """Загрузить фактические значения Q_fact."""

    rows = _read_rows(path)
    required = {*KEY_COLUMNS, "integral_quality"}
    missing = required.difference(rows[0].keys()) if rows else required
    if missing:
        raise ValueError(
            "В quality_targets.csv отсутствуют обязательные колонки: "
            + ", ".join(sorted(missing))
        )
    result = tuple(
        FactValue(
            scenario_id=row["scenario_id"].strip(),
            protocol_id=row["protocol_id"].strip(),
            q_fact=_parse_probability(
                row["integral_quality"],
                column="integral_quality",
                row_number=index,
            ),
        )
        for index, row in enumerate(rows, start=2)
    )
    _validate_unique_keys(result, source_name="quality_targets.csv")
    return result


def merge_residual_points(
    predictions: Sequence[PredictionValue], facts: Sequence[FactValue]
) -> tuple[ResidualPoint, ...]:
    """Объединить прогнозные и фактические данные в режиме one-to-one."""

    prediction_map = {
        (row.scenario_id, row.protocol_id): row.q_pred for row in predictions
    }
    fact_map = {(row.scenario_id, row.protocol_id): row.q_fact for row in facts}
    if prediction_map.keys() != fact_map.keys():
        missing_fact = prediction_map.keys() - fact_map.keys()
        missing_prediction = fact_map.keys() - prediction_map.keys()
        raise ValueError(
            "Наборы ключей прогнозных и фактических данных не совпадают: "
            f"без факта={len(missing_fact)}, без прогноза={len(missing_prediction)}."
        )
    points = tuple(
        ResidualPoint(
            scenario_id=scenario_id,
            protocol_id=protocol_id,
            q_pred=prediction_map[(scenario_id, protocol_id)],
            q_fact=fact_map[(scenario_id, protocol_id)],
        )
        for scenario_id, protocol_id in sorted(prediction_map)
    )
    validate_points(points)
    return points


def validate_points(points: Sequence[ResidualPoint]) -> None:
    """Проверить достаточность и корректность набора остатков."""

    if len(points) < 3:
        raise ValueError("Для анализа остатков требуется не менее трёх сценариев.")
    keys: set[tuple[str, str]] = set()
    for point in points:
        key = (point.scenario_id, point.protocol_id)
        if key in keys:
            raise ValueError("Ключи объединённого набора должны быть уникальными.")
        keys.add(key)
        for name, value in (("q_pred", point.q_pred), ("q_fact", point.q_fact)):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Значение {name} должно находиться в диапазоне [0; 1].")
    q_fact = np.asarray([point.q_fact for point in points], dtype=float)
    if np.ptp(q_fact) <= 1e-15:
        raise ValueError("Анализ относительно Q_fact невозможен для постоянного ряда.")


def calculate_summary(points: Sequence[ResidualPoint]) -> ResidualSummary:
    """Рассчитать описательные показатели остатков."""

    validate_points(points)
    q_fact = np.asarray([point.q_fact for point in points], dtype=float)
    residuals = np.asarray([point.residual for point in points], dtype=float)
    minimum_index = int(np.argmin(residuals))
    maximum_index = int(np.argmax(residuals))
    correlation = float(np.corrcoef(q_fact, residuals)[0, 1])
    return ResidualSummary(
        count=len(points),
        bias=float(np.mean(residuals)),
        residual_std=float(np.std(residuals, ddof=1)),
        residual_min=float(np.min(residuals)),
        residual_q1=float(np.quantile(residuals, 0.25)),
        residual_median=float(np.median(residuals)),
        residual_q3=float(np.quantile(residuals, 0.75)),
        residual_max=float(np.max(residuals)),
        underestimation_count=int(np.sum(residuals < 0.0)),
        overestimation_count=int(np.sum(residuals > 0.0)),
        zero_count=int(np.sum(np.isclose(residuals, 0.0, atol=1e-12))),
        correlation_with_q_fact=correlation,
        minimum_scenario=points[minimum_index].scenario_id,
        maximum_scenario=points[maximum_index].scenario_id,
    )


def gaussian_smooth(
    q_fact: np.ndarray,
    residuals: np.ndarray,
    *,
    bandwidth: float = DEFAULT_SMOOTHING_BANDWIDTH,
    grid_size: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    """Вычислить гауссово-взвешенную сглаженную тенденцию остатков."""

    if bandwidth <= 0.0 or not math.isfinite(bandwidth):
        raise ValueError("Ширина сглаживания должна быть положительной.")
    if len(q_fact) != len(residuals) or len(q_fact) < 3:
        raise ValueError("Для сглаживания требуются согласованные ряды длиной не менее 3.")
    if not np.all(np.isfinite(q_fact)) or not np.all(np.isfinite(residuals)):
        raise ValueError("Сглаживаемые ряды не должны содержать NaN или бесконечность.")
    grid = np.linspace(float(np.min(q_fact)), float(np.max(q_fact)), grid_size)
    distances = (grid[:, None] - q_fact[None, :]) / bandwidth
    weights = np.exp(-0.5 * np.square(distances))
    totals = np.sum(weights, axis=1)
    smooth = np.sum(weights * residuals[None, :], axis=1) / totals
    return grid, smooth


def _add_metric_card(
    axis: plt.Axes,
    *,
    y: float,
    title: str,
    lines: Sequence[str],
    facecolor: str,
    height: float = 0.205,
) -> None:
    """Добавить информационную карточку в правую панель."""

    patch = FancyBboxPatch(
        (0.04, y - height),
        0.92,
        height,
        transform=axis.transAxes,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="#7a8793",
        facecolor=facecolor,
    )
    axis.add_patch(patch)
    axis.text(
        0.08,
        y - 0.032,
        title,
        transform=axis.transAxes,
        fontsize=12.2,
        fontweight="bold",
        va="top",
    )
    axis.text(
        0.08,
        y - 0.080,
        "\n".join(lines),
        transform=axis.transAxes,
        fontsize=10.4,
        va="top",
        linespacing=1.25,
    )


def build_figure(
    points: Sequence[ResidualPoint],
    summary: ResidualSummary,
    *,
    smoothing_bandwidth: float = DEFAULT_SMOOTHING_BANDWIDTH,
) -> plt.Figure:
    """Построить рисунок остатков относительно фактического качества."""

    configure_dissertation_style()
    figure = plt.figure(figsize=(16.6, 8.9))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(2.28, 1.0),
        left=0.065,
        right=0.965,
        top=0.84,
        bottom=0.15,
        wspace=0.16,
    )
    residual_axis = figure.add_subplot(grid[0, 0])
    info_axis = figure.add_subplot(grid[0, 1])
    info_axis.axis("off")

    q_fact = np.asarray([point.q_fact for point in points], dtype=float)
    residuals = np.asarray([point.residual for point in points], dtype=float)
    negative_mask = residuals < 0.0
    nonnegative_mask = ~negative_mask

    residual_axis.axhspan(
        min(-0.5, summary.residual_min - 0.04),
        0.0,
        alpha=0.07,
        color="#c94d4d",
        label="область занижения: e < 0",
        zorder=0,
    )
    residual_axis.axhspan(
        0.0,
        max(0.5, summary.residual_max + 0.04),
        alpha=0.06,
        color="#4b8f5f",
        label="область завышения: e > 0",
        zorder=0,
    )
    residual_axis.scatter(
        q_fact[negative_mask],
        residuals[negative_mask],
        s=48,
        alpha=0.78,
        edgecolors="white",
        linewidths=0.7,
        label=f"занижение, n = {summary.underestimation_count}",
        zorder=3,
    )
    if np.any(nonnegative_mask):
        residual_axis.scatter(
            q_fact[nonnegative_mask],
            residuals[nonnegative_mask],
            s=48,
            alpha=0.82,
            marker="s",
            edgecolors="white",
            linewidths=0.7,
            label=f"завышение/совпадение, n = {summary.overestimation_count + summary.zero_count}",
            zorder=3,
        )

    residual_axis.axhline(
        0.0,
        linestyle="--",
        linewidth=2.0,
        color="#263238",
        label="нулевая линия e = 0",
        zorder=2,
    )
    residual_axis.axhline(
        summary.bias,
        linestyle=":",
        linewidth=2.2,
        color="#8e44ad",
        label=f"средний остаток Bias = {summary.bias:+.3f}",
        zorder=2,
    )
    smooth_x, smooth_y = gaussian_smooth(
        q_fact,
        residuals,
        bandwidth=smoothing_bandwidth,
    )
    residual_axis.plot(
        smooth_x,
        smooth_y,
        linewidth=2.8,
        color="#d17a00",
        label=f"сглаженная тенденция, h = {smoothing_bandwidth:.2f}",
        zorder=4,
    )

    minimum_index = int(np.argmin(residuals))
    maximum_index = int(np.argmax(residuals))
    for index, label, offset in (
        (minimum_index, "минимальный остаток", (18, -34)),
        (maximum_index, "максимальный остаток", (18, 18)),
    ):
        point = points[index]
        residual_axis.annotate(
            f"{label}: {point.residual:+.3f}\n{point.scenario_id}",
            xy=(point.q_fact, point.residual),
            xytext=offset,
            textcoords="offset points",
            fontsize=9.9,
            bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "alpha": 0.9},
            arrowprops={"arrowstyle": "->", "linewidth": 1.0},
            zorder=6,
        )

    y_min = min(-0.45, summary.residual_min - 0.06)
    y_max = max(0.22, summary.residual_max + 0.06)
    residual_axis.set_xlim(max(0.0, float(np.min(q_fact)) - 0.04), min(1.0, float(np.max(q_fact)) + 0.04))
    residual_axis.set_ylim(y_min, y_max)
    residual_axis.set_xlabel(r"Фактическое интегральное качество $Q_{fact}$")
    residual_axis.set_ylabel(r"Остаток $e_i = Q_{pred,i} - Q_{fact,i}$")
    residual_axis.set_title(
        "Остатки интегрального прогноза относительно Q_fact",
        pad=12,
        fontweight="bold",
    )
    residual_axis.grid(True, alpha=0.25, linewidth=0.8)
    residual_axis.legend(loc="upper left", fontsize=9.0, framealpha=0.96, ncol=2)

    _add_metric_card(
        info_axis,
        y=0.98,
        title="Центр распределения",
        lines=(
            f"Bias = {summary.bias:+.3f}",
            f"медиана e = {summary.residual_median:+.3f}",
            f"стандартное отклонение = {summary.residual_std:.3f}",
            "Отрицательный центр: систематическое занижение",
        ),
        facecolor="#fbefef",
    )
    _add_metric_card(
        info_axis,
        y=0.73,
        title="Диапазон и квартильная структура",
        lines=(
            f"min = {summary.residual_min:+.3f}",
            f"Q1 = {summary.residual_q1:+.3f}",
            f"Q3 = {summary.residual_q3:+.3f}",
            f"max = {summary.residual_max:+.3f}",
        ),
        facecolor="#f7f4ec",
    )
    _add_metric_card(
        info_axis,
        y=0.48,
        title="Направление ошибок",
        lines=(
            f"Q_pred < Q_fact: {summary.underestimation_count}",
            f"Q_pred > Q_fact: {summary.overestimation_count}",
            f"Q_pred = Q_fact: {summary.zero_count}",
            f"N = {summary.count}",
        ),
        facecolor="#eef5fb",
    )
    _add_metric_card(
        info_axis,
        y=0.23,
        title="Связь с уровнем Q_fact",
        lines=(
            f"Pearson(Q_fact, e) = {summary.correlation_with_q_fact:+.3f}",
            "Сглаженная тенденция показывает",
            "изменение систематической ошибки",
            "по диапазону фактического качества",
        ),
        facecolor="#f1f6f0",
    )

    figure.suptitle(
        r"Рисунок 6.2 — остатки $e_i=Q_{pred,i}-Q_{fact,i}$",
        fontsize=17,
        fontweight="bold",
        y=0.965,
    )
    figure.text(
        0.5,
        0.905,
        "Нулевая линия соответствует отсутствию абсолютной ошибки; отрицательные значения означают занижение прогноза.",
        ha="center",
        fontsize=11.3,
    )
    figure.text(
        0.5,
        0.045,
        "Методическое примечание: остаток e = Q_pred − Q_fact используется для внешней диагностики. "
        "Сглаженная тенденция описывает структуру смещения, но не является абсолютной калибровкой модели "
        "и не должна интерпретироваться как причинная зависимость.",
        ha="center",
        va="bottom",
        fontsize=10.7,
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
    smoothing_bandwidth: float = DEFAULT_SMOOTHING_BANDWIDTH,
) -> FigureExportResult:
    """Загрузить данные, построить рисунок и экспортировать PNG/SVG."""

    root = project_root.resolve()
    predictions = load_q_pred(root / (q_pred_path or DEFAULT_Q_PRED_PATH))
    facts = load_q_fact(root / (q_fact_path or DEFAULT_Q_FACT_PATH))
    points = merge_residual_points(predictions, facts)
    summary = calculate_summary(points)
    figure = build_figure(
        points,
        summary,
        smoothing_bandwidth=smoothing_bandwidth,
    )
    return export_figure(
        figure,
        project_root=root,
        relative_output_dir=OUTPUT_DIR,
        file_stem=FILE_STEM,
        dpi=dpi,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Создать CLI-парсер генератора рисунка 6.2."""

    parser = argparse.ArgumentParser(
        description="Сформировать рисунок 6.2 с остатками относительно Q_fact."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Корень проекта manual_coding_sim.",
    )
    parser.add_argument(
        "--q-pred",
        type=Path,
        default=None,
        help="Путь к q_pred.csv относительно корня проекта.",
    )
    parser.add_argument(
        "--q-fact",
        type=Path,
        default=None,
        help="Путь к quality_targets.csv относительно корня проекта.",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=DEFAULT_SMOOTHING_BANDWIDTH,
        help="Ширина гауссова сглаживания по шкале Q_fact.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Разрешение PNG.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить CLI генератора и вывести пути к артефактам."""

    args = build_argument_parser().parse_args(argv)
    result = generate(
        project_root=args.project_root,
        q_pred_path=args.q_pred,
        q_fact_path=args.q_fact,
        dpi=args.dpi,
        smoothing_bandwidth=args.bandwidth,
    )
    print("Рисунок 6.2 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
