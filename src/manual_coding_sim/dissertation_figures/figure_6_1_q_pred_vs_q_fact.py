"""Генерация рисунка 6.1 для сопоставления Q_pred и Q_fact."""

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
FILE_STEM = "q_pred_vs_q_fact"
DEFAULT_Q_PRED_PATH = Path("reports/chapter5/q_pred.csv")
DEFAULT_Q_FACT_PATH = Path("data/processed/quality_targets.csv")
KEY_COLUMNS = ("scenario_id", "protocol_id")


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
class ValidationPoint:
    """Согласованная пара прогнозного и фактического качества."""

    scenario_id: str
    protocol_id: str
    q_pred: float
    q_fact: float

    @property
    def error(self) -> float:
        """Вернуть знаковую ошибку Q_pred - Q_fact."""

        return self.q_pred - self.q_fact


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """Сводные метрики интегральной внешней проверки."""

    count: int
    q_pred_mean: float
    q_fact_mean: float
    pearson: float
    spearman: float
    bias: float
    mae: float
    rmse: float
    median_absolute_error: float
    maximum_absolute_error: float
    regression_slope: float
    regression_intercept: float
    underestimation_count: int
    overestimation_count: int
    maximum_error_scenario: str


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Прочитать CSV и вернуть строки с проверкой наличия файла."""

    if not path.is_file():
        raise FileNotFoundError(f"Не найден входной файл: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV-файл не содержит заголовка: {path}")
        return list(reader)


def _parse_probability(raw: str, *, column: str, row_number: int) -> float:
    """Преобразовать значение показателя качества и проверить диапазон."""

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


def load_q_pred(path: Path) -> tuple[PredictionValue, ...]:
    """Загрузить интегральные прогнозы главы 5."""

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
    """Загрузить фактическое интегральное качество корпуса."""

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


def merge_validation_points(
    predictions: Sequence[PredictionValue], facts: Sequence[FactValue]
) -> tuple[ValidationPoint, ...]:
    """Объединить прогнозы и фактические значения в режиме one-to-one."""

    prediction_map = {
        (row.scenario_id, row.protocol_id): row.q_pred for row in predictions
    }
    fact_map = {(row.scenario_id, row.protocol_id): row.q_fact for row in facts}
    if prediction_map.keys() != fact_map.keys():
        missing_fact = sorted(prediction_map.keys() - fact_map.keys())
        missing_prediction = sorted(fact_map.keys() - prediction_map.keys())
        raise ValueError(
            "Наборы ключей прогнозных и фактических данных не совпадают: "
            f"без факта={len(missing_fact)}, без прогноза={len(missing_prediction)}."
        )
    points = tuple(
        ValidationPoint(
            scenario_id=scenario_id,
            protocol_id=protocol_id,
            q_pred=prediction_map[(scenario_id, protocol_id)],
            q_fact=fact_map[(scenario_id, protocol_id)],
        )
        for scenario_id, protocol_id in sorted(prediction_map)
    )
    validate_points(points)
    return points


def validate_points(points: Sequence[ValidationPoint]) -> None:
    """Проверить достаточность и числовую корректность пар значений."""

    if len(points) < 3:
        raise ValueError("Для корреляционного анализа требуется не менее трёх сценариев.")
    keys: set[tuple[str, str]] = set()
    for point in points:
        key = (point.scenario_id, point.protocol_id)
        if key in keys:
            raise ValueError("Ключи объединённого набора должны быть уникальными.")
        keys.add(key)
        for name, value in (("q_pred", point.q_pred), ("q_fact", point.q_fact)):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Значение {name} должно находиться в диапазоне [0; 1].")
    q_pred = np.asarray([point.q_pred for point in points], dtype=float)
    q_fact = np.asarray([point.q_fact for point in points], dtype=float)
    if np.ptp(q_pred) <= 1e-15 or np.ptp(q_fact) <= 1e-15:
        raise ValueError("Корреляция не определена для постоянного ряда значений.")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Вычислить средние ранги с корректной обработкой совпадающих значений."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        ranks[order[position:end]] = average_rank
        position = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Вычислить коэффициент корреляции Пирсона двух рядов."""

    return float(np.corrcoef(left, right)[0, 1])


def calculate_summary(points: Sequence[ValidationPoint]) -> ValidationSummary:
    """Рассчитать метрики интегральной внешней проверки."""

    validate_points(points)
    q_pred = np.asarray([point.q_pred for point in points], dtype=float)
    q_fact = np.asarray([point.q_fact for point in points], dtype=float)
    errors = q_pred - q_fact
    absolute_errors = np.abs(errors)
    slope, intercept = np.polyfit(q_pred, q_fact, deg=1)
    maximum_index = int(np.argmax(absolute_errors))
    return ValidationSummary(
        count=len(points),
        q_pred_mean=float(np.mean(q_pred)),
        q_fact_mean=float(np.mean(q_fact)),
        pearson=_correlation(q_pred, q_fact),
        spearman=_correlation(_average_ranks(q_pred), _average_ranks(q_fact)),
        bias=float(np.mean(errors)),
        mae=float(np.mean(absolute_errors)),
        rmse=float(np.sqrt(np.mean(np.square(errors)))),
        median_absolute_error=float(np.median(absolute_errors)),
        maximum_absolute_error=float(np.max(absolute_errors)),
        regression_slope=float(slope),
        regression_intercept=float(intercept),
        underestimation_count=int(np.sum(errors < 0.0)),
        overestimation_count=int(np.sum(errors > 0.0)),
        maximum_error_scenario=points[maximum_index].scenario_id,
    )


def _add_metric_card(
    axis: plt.Axes,
    *,
    y: float,
    title: str,
    lines: Sequence[str],
    facecolor: str,
) -> None:
    """Добавить информационную карточку в правую панель."""

    height = 0.205
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
        fontsize=12.4,
        fontweight="bold",
        va="top",
    )
    axis.text(
        0.08,
        y - 0.080,
        "\n".join(lines),
        transform=axis.transAxes,
        fontsize=10.6,
        va="top",
        linespacing=1.27,
    )


def build_figure(
    points: Sequence[ValidationPoint], summary: ValidationSummary
) -> plt.Figure:
    """Построить диссертационный рисунок сопоставления Q_pred и Q_fact."""

    configure_dissertation_style()
    figure = plt.figure(figsize=(16.4, 8.8))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(2.18, 1.0),
        left=0.065,
        right=0.965,
        top=0.84,
        bottom=0.15,
        wspace=0.16,
    )
    scatter_axis = figure.add_subplot(grid[0, 0])
    info_axis = figure.add_subplot(grid[0, 1])
    info_axis.axis("off")

    q_pred = np.asarray([point.q_pred for point in points], dtype=float)
    q_fact = np.asarray([point.q_fact for point in points], dtype=float)
    errors = q_pred - q_fact

    scatter_axis.scatter(
        q_pred,
        q_fact,
        s=48,
        alpha=0.78,
        edgecolors="white",
        linewidths=0.7,
        label=f"сценарии, N = {summary.count}",
        zorder=3,
    )
    line = np.linspace(0.0, 1.0, 200)
    scatter_axis.plot(
        line,
        line,
        linestyle="--",
        linewidth=2.0,
        label="идеальное соответствие y = x",
        zorder=2,
    )
    regression = summary.regression_slope * line + summary.regression_intercept
    scatter_axis.plot(
        line,
        regression,
        linestyle="-.",
        linewidth=2.0,
        label=(
            "линейная тенденция: "
            f"y = {summary.regression_slope:.3f}x + {summary.regression_intercept:.3f}"
        ),
        zorder=2,
    )
    scatter_axis.scatter(
        [summary.q_pred_mean],
        [summary.q_fact_mean],
        marker="D",
        s=95,
        edgecolors="black",
        linewidths=0.8,
        label="средняя точка корпуса",
        zorder=5,
    )

    maximum_index = int(np.argmax(np.abs(errors)))
    point = points[maximum_index]
    scatter_axis.annotate(
        f"макс. |e| = {summary.maximum_absolute_error:.3f}\n{point.scenario_id}",
        xy=(point.q_pred, point.q_fact),
        xytext=(18, 18),
        textcoords="offset points",
        fontsize=10.2,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9},
        arrowprops={"arrowstyle": "->", "linewidth": 1.0},
        zorder=6,
    )

    scatter_axis.set_xlim(0.0, 1.0)
    scatter_axis.set_ylim(0.0, 1.0)
    scatter_axis.set_aspect("equal", adjustable="box")
    scatter_axis.set_xlabel(r"Априорный прогноз $Q_{pred}$")
    scatter_axis.set_ylabel(r"Фактическое качество $Q_{fact}$")
    scatter_axis.set_title(
        "Сопоставление прогнозного и фактического интегрального качества",
        pad=12,
        fontweight="bold",
    )
    scatter_axis.grid(True, alpha=0.25, linewidth=0.8)
    scatter_axis.legend(loc="lower right", fontsize=9.5, framealpha=0.96)

    _add_metric_card(
        info_axis,
        y=0.98,
        title="Корреляционная согласованность",
        lines=(
            f"Pearson = {summary.pearson:.3f}",
            f"Spearman = {summary.spearman:.3f}",
            "Высокая согласованность ранжирования",
        ),
        facecolor="#eef5fb",
    )
    _add_metric_card(
        info_axis,
        y=0.73,
        title="Ошибка интегрального прогноза",
        lines=(
            f"MAE = {summary.mae:.3f}",
            f"RMSE = {summary.rmse:.3f}",
            f"Median |e| = {summary.median_absolute_error:.3f}",
            f"Max |e| = {summary.maximum_absolute_error:.3f}",
        ),
        facecolor="#f7f4ec",
    )
    _add_metric_card(
        info_axis,
        y=0.48,
        title="Систематическое смещение",
        lines=(
            f"Bias = {summary.bias:+.3f}",
            f"среднее Q_pred = {summary.q_pred_mean:.3f}",
            f"среднее Q_fact = {summary.q_fact_mean:.3f}",
            "Отрицательный Bias: систематическое занижение",
        ),
        facecolor="#fbefef",
    )
    _add_metric_card(
        info_axis,
        y=0.23,
        title="Направление ошибок",
        lines=(
            f"Q_pred < Q_fact: {summary.underestimation_count}",
            f"Q_pred > Q_fact: {summary.overestimation_count}",
            "Фактические данные используются",
            "только для внешней проверки",
        ),
        facecolor="#f1f6f0",
    )

    figure.suptitle(
        r"Рисунок 6.1 — $Q_{pred}$ против $Q_{fact}$",
        fontsize=17,
        fontweight="bold",
        y=0.965,
    )
    figure.text(
        0.5,
        0.905,
        "Линия y = x соответствует идеальному абсолютному совпадению; "
        "корреляция оценивает согласованность изменения и ранжирования.",
        ha="center",
        fontsize=11.3,
    )
    figure.text(
        0.5,
        0.045,
        "Методическое примечание: высокая Pearson/Spearman-корреляция подтверждает "
        "сравнительную информативность априорного индекса, но отрицательный Bias и "
        "отклонение от y = x показывают отсутствие абсолютной калибровки.",
        ha="center",
        va="bottom",
        fontsize=10.8,
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
    points = merge_validation_points(predictions, facts)
    summary = calculate_summary(points)
    figure = build_figure(points, summary)
    return export_figure(
        figure,
        project_root=root,
        relative_output_dir=OUTPUT_DIR,
        file_stem=FILE_STEM,
        dpi=dpi,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Создать CLI-парсер генератора рисунка 6.1."""

    parser = argparse.ArgumentParser(
        description="Сформировать рисунок 6.1 Q_pred против Q_fact."
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
    )
    print("Рисунок 6.1 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
