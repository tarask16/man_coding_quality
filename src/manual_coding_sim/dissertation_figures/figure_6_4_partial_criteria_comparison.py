"""Генерация рисунка 6.4 со сравнением метрик частных критериев."""

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
FILE_STEM = "partial_criteria_comparison"
DEFAULT_Q_COMPONENTS_PATH = Path("reports/chapter5/q_pred_components.csv")
DEFAULT_Q_FACT_PATH = Path("data/processed/quality_targets.csv")
JOIN_KEYS = ("scenario_id", "protocol_id")

CRITERIA: tuple[tuple[str, str], ...] = (
    ("acc", "q_acc — точность"),
    ("time", "q_time — время"),
    ("effort", "q_effort — трудоёмкость"),
    ("res", "q_res — контроль"),
    ("rep", "q_rep — повторяемость"),
    ("fit", "q_fit — соответствие"),
)


@dataclass(frozen=True, slots=True)
class CriterionPair:
    """Согласованные прогнозные и фактические значения одного критерия."""

    code: str
    label: str
    q_pred: tuple[float, ...]
    q_fact: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CriterionMetrics:
    """Метрики качества прогноза для одного частного критерия."""

    code: str
    label: str
    count: int
    mae: float
    rmse: float
    bias: float
    spearman: float


@dataclass(frozen=True, slots=True)
class MetricsSummary:
    """Сводка по шести частным критериям."""

    count: int
    mean_mae: float
    mean_rmse: float
    mean_bias: float
    mean_spearman: float
    best_mae_code: str
    worst_mae_code: str
    best_spearman_code: str
    worst_spearman_code: str


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    """Прочитать CSV и вернуть непустой набор строк."""

    if not path.is_file():
        raise FileNotFoundError(f"Не найден входной файл: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Входной файл не содержит строк: {path}")
    return rows


def _validate_required_columns(
    rows: Sequence[Mapping[str, str]],
    required: Sequence[str],
    *,
    source_name: str,
) -> None:
    """Проверить наличие обязательных колонок в CSV."""

    columns = set(rows[0])
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(
            f"В {source_name} отсутствуют обязательные колонки: {', '.join(missing)}"
        )


def _parse_unit_value(raw: str, *, column: str, row_number: int) -> float:
    """Преобразовать значение критерия и проверить диапазон [0; 1]."""

    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Колонка {column}, строка {row_number}: требуется числовое значение."
        ) from error
    if not math.isfinite(value):
        raise ValueError(
            f"Колонка {column}, строка {row_number}: NaN и бесконечность недопустимы."
        )
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"Колонка {column}, строка {row_number}: значение должно быть в диапазоне [0; 1]."
        )
    return value


def _index_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    source_name: str,
) -> dict[tuple[str, str], Mapping[str, str]]:
    """Индексировать строки по ключам сценария и протокола без дубликатов."""

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


def load_criterion_pairs(
    q_components_path: Path,
    q_fact_path: Path,
) -> tuple[CriterionPair, ...]:
    """Загрузить и согласовать шесть пар прогнозных и фактических критериев."""

    predicted_rows = _read_rows(q_components_path)
    fact_rows = _read_rows(q_fact_path)
    predicted_columns = [*JOIN_KEYS, *(f"q_{code}_pred" for code, _ in CRITERIA)]
    fact_columns = [*JOIN_KEYS, *(f"q_{code}" for code, _ in CRITERIA)]
    _validate_required_columns(
        predicted_rows,
        predicted_columns,
        source_name=q_components_path.name,
    )
    _validate_required_columns(fact_rows, fact_columns, source_name=q_fact_path.name)

    predicted_index = _index_rows(predicted_rows, source_name=q_components_path.name)
    fact_index = _index_rows(fact_rows, source_name=q_fact_path.name)
    predicted_keys = set(predicted_index)
    fact_keys = set(fact_index)
    if predicted_keys != fact_keys:
        missing_fact = len(predicted_keys - fact_keys)
        missing_predicted = len(fact_keys - predicted_keys)
        raise ValueError(
            "Наборы ключей не совпадают: "
            f"без фактической строки — {missing_fact}, "
            f"без прогнозной строки — {missing_predicted}."
        )
    if len(predicted_keys) < 3:
        raise ValueError("Для сравнения частных критериев требуется не менее трёх сценариев.")

    ordered_keys = sorted(predicted_keys)
    pairs: list[CriterionPair] = []
    for code, label in CRITERIA:
        pred_column = f"q_{code}_pred"
        fact_column = f"q_{code}"
        q_pred = tuple(
            _parse_unit_value(
                predicted_index[key][pred_column],
                column=pred_column,
                row_number=index + 2,
            )
            for index, key in enumerate(ordered_keys)
        )
        q_fact = tuple(
            _parse_unit_value(
                fact_index[key][fact_column],
                column=fact_column,
                row_number=index + 2,
            )
            for index, key in enumerate(ordered_keys)
        )
        pairs.append(CriterionPair(code=code, label=label, q_pred=q_pred, q_fact=q_fact))
    return tuple(pairs)


def _rank_average(values: np.ndarray) -> np.ndarray:
    """Рассчитать средние ранги с корректной обработкой совпадающих значений."""

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman_correlation(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    """Рассчитать коэффициент Спирмена через корреляцию средних рангов."""

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 3:
        raise ValueError("Для Spearman требуются два одномерных ряда одинаковой длины.")
    x_rank = _rank_average(x)
    y_rank = _rank_average(y)
    if np.std(x_rank) == 0.0 or np.std(y_rank) == 0.0:
        raise ValueError("Spearman не определён для постоянного ряда.")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def calculate_metrics(pair: CriterionPair) -> CriterionMetrics:
    """Рассчитать MAE, RMSE, Bias и Spearman для одного критерия."""

    q_pred = np.asarray(pair.q_pred, dtype=float)
    q_fact = np.asarray(pair.q_fact, dtype=float)
    if q_pred.shape != q_fact.shape or len(q_pred) < 3:
        raise ValueError("Прогнозный и фактический ряды должны иметь одинаковую длину.")
    residuals = q_pred - q_fact
    return CriterionMetrics(
        code=pair.code,
        label=pair.label,
        count=len(q_pred),
        mae=float(np.mean(np.abs(residuals))),
        rmse=float(np.sqrt(np.mean(np.square(residuals)))),
        bias=float(np.mean(residuals)),
        spearman=spearman_correlation(q_pred, q_fact),
    )


def calculate_all_metrics(
    pairs: Sequence[CriterionPair],
) -> tuple[CriterionMetrics, ...]:
    """Рассчитать метрики в установленном порядке шести критериев."""

    expected_codes = [code for code, _ in CRITERIA]
    actual_codes = [pair.code for pair in pairs]
    if actual_codes != expected_codes:
        raise ValueError(
            "Критерии должны следовать порядку: " + ", ".join(expected_codes)
        )
    return tuple(calculate_metrics(pair) for pair in pairs)


def summarize_metrics(metrics: Sequence[CriterionMetrics]) -> MetricsSummary:
    """Сформировать сводные показатели и определить крайние критерии."""

    if len(metrics) != len(CRITERIA):
        raise ValueError("Сводка должна включать ровно шесть частных критериев.")
    maes = np.asarray([item.mae for item in metrics], dtype=float)
    rmses = np.asarray([item.rmse for item in metrics], dtype=float)
    biases = np.asarray([item.bias for item in metrics], dtype=float)
    correlations = np.asarray([item.spearman for item in metrics], dtype=float)
    return MetricsSummary(
        count=metrics[0].count,
        mean_mae=float(np.mean(maes)),
        mean_rmse=float(np.mean(rmses)),
        mean_bias=float(np.mean(biases)),
        mean_spearman=float(np.mean(correlations)),
        best_mae_code=metrics[int(np.argmin(maes))].code,
        worst_mae_code=metrics[int(np.argmax(maes))].code,
        best_spearman_code=metrics[int(np.argmax(correlations))].code,
        worst_spearman_code=metrics[int(np.argmin(correlations))].code,
    )


def _annotate_horizontal_bars(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    signed: bool = False,
) -> None:
    """Добавить числовые подписи к горизонтальным столбцам."""

    for index, value in enumerate(values):
        if signed and value < 0:
            x = value - 0.008
            alignment = "right"
        else:
            x = value + 0.008
            alignment = "left"
        axis.text(
            x,
            index,
            f"{value:+.3f}" if signed else f"{value:.3f}",
            va="center",
            ha=alignment,
            fontsize=10.2,
            fontweight="bold",
        )


def build_figure(
    metrics: Sequence[CriterionMetrics],
    summary: MetricsSummary,
) -> plt.Figure:
    """Построить четыре панели сопоставления частных критериев."""

    configure_dissertation_style()
    labels = [item.label for item in metrics]
    positions = np.arange(len(metrics))
    mae = np.asarray([item.mae for item in metrics])
    rmse = np.asarray([item.rmse for item in metrics])
    bias = np.asarray([item.bias for item in metrics])
    spearman = np.asarray([item.spearman for item in metrics])

    figure, axes = plt.subplots(2, 2, figsize=(18.4, 9.2))
    figure.subplots_adjust(
        left=0.16,
        right=0.965,
        top=0.84,
        bottom=0.175,
        wspace=0.32,
        hspace=0.42,
    )
    mae_axis, rmse_axis, bias_axis, correlation_axis = axes.ravel()

    mae_axis.barh(positions, mae, alpha=0.88, edgecolor="white", linewidth=0.8)
    mae_axis.axvline(summary.mean_mae, linestyle="--", linewidth=1.8, color="#4b5563")
    mae_axis.set_yticks(positions, labels)
    mae_axis.invert_yaxis()
    mae_axis.set_xlim(0.0, max(0.40, float(mae.max()) + 0.055))
    mae_axis.set_xlabel("MAE — меньше лучше")
    mae_axis.set_title("Средняя абсолютная ошибка (MAE)", fontweight="bold")
    mae_axis.grid(axis="x", alpha=0.22)
    _annotate_horizontal_bars(mae_axis, mae)
    mae_axis.text(
        0.98,
        0.04,
        f"среднее = {summary.mean_mae:.3f}",
        transform=mae_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.1,
    )

    rmse_axis.barh(
        positions,
        rmse,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.8,
        color="#dd8452",
    )
    rmse_axis.axvline(summary.mean_rmse, linestyle="--", linewidth=1.8, color="#4b5563")
    rmse_axis.set_yticks(positions, labels)
    rmse_axis.invert_yaxis()
    rmse_axis.set_xlim(0.0, max(0.44, float(rmse.max()) + 0.055))
    rmse_axis.set_xlabel("RMSE — меньше лучше")
    rmse_axis.set_title("Среднеквадратическая ошибка (RMSE)", fontweight="bold")
    rmse_axis.grid(axis="x", alpha=0.22)
    _annotate_horizontal_bars(rmse_axis, rmse)
    rmse_axis.text(
        0.98,
        0.04,
        f"среднее = {summary.mean_rmse:.3f}",
        transform=rmse_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.1,
    )

    bias_colors = ["#c44e52" if value < 0 else "#55a868" for value in bias]
    bias_axis.barh(
        positions,
        bias,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.8,
        color=bias_colors,
    )
    bias_axis.axvline(0.0, color="#222222", linewidth=1.4)
    bias_axis.axvline(summary.mean_bias, linestyle="--", linewidth=1.8, color="#4b5563")
    bias_axis.set_yticks(positions, labels)
    bias_axis.invert_yaxis()
    bias_axis.set_xlim(min(-0.40, float(bias.min()) - 0.06), 0.08)
    bias_axis.set_xlabel(r"Bias = mean($q_{pred}-q_{fact}$)")
    bias_axis.set_title("Систематическое смещение (Bias)", fontweight="bold")
    bias_axis.grid(axis="x", alpha=0.22)
    _annotate_horizontal_bars(bias_axis, bias, signed=True)
    bias_axis.text(
        0.98,
        0.04,
        f"среднее = {summary.mean_bias:+.3f}",
        transform=bias_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.1,
    )

    correlation_axis.barh(
        positions,
        spearman,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.8,
        color="#55a868",
    )
    correlation_axis.axvline(
        summary.mean_spearman,
        linestyle="--",
        linewidth=1.8,
        color="#4b5563",
    )
    correlation_axis.axvline(0.60, linestyle=":", linewidth=1.6, color="#8172b3")
    correlation_axis.set_yticks(positions, labels)
    correlation_axis.invert_yaxis()
    correlation_axis.set_xlim(0.0, 1.0)
    correlation_axis.set_xlabel("Spearman — больше лучше")
    correlation_axis.set_title("Ранговая корреляция Spearman", fontweight="bold")
    correlation_axis.grid(axis="x", alpha=0.22)
    _annotate_horizontal_bars(correlation_axis, spearman)
    correlation_axis.text(
        0.98,
        0.04,
        f"среднее = {summary.mean_spearman:.3f}",
        transform=correlation_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.1,
    )

    figure.suptitle(
        "Сравнение прогнозных и фактических частных критериев",
        fontsize=18,
        fontweight="bold",
        y=0.965,
    )
    figure.text(
        0.5,
        0.915,
        (
            f"N = {summary.count}; лучший MAE: q_{summary.best_mae_code}; "
            f"наибольший MAE: q_{summary.worst_mae_code}; "
            f"mean Spearman = {summary.mean_spearman:.3f}"
        ),
        ha="center",
        va="center",
        fontsize=11.5,
    )
    figure.text(
        0.5,
        0.065,
        (
            "MAE и RMSE характеризуют величину расхождения, Bias — его направление, "
            "а Spearman — сохранение рангового порядка. Высокая ранговая корреляция "
            "не устраняет систематическое занижение и не подтверждает абсолютную калибровку."
        ),
        ha="center",
        va="center",
        fontsize=10.5,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#f3f5f7",
            "edgecolor": "#9aa4ad",
        },
    )
    return figure


def generate(
    *,
    project_root: Path,
    q_components_path: Path | None = None,
    q_fact_path: Path | None = None,
    dpi: int = 300,
) -> FigureExportResult:
    """Сформировать рисунок 6.4 в PNG и SVG."""

    root = project_root.resolve()
    components = (
        q_components_path.resolve()
        if q_components_path is not None
        else root / DEFAULT_Q_COMPONENTS_PATH
    )
    facts = q_fact_path.resolve() if q_fact_path is not None else root / DEFAULT_Q_FACT_PATH
    pairs = load_criterion_pairs(components, facts)
    metrics = calculate_all_metrics(pairs)
    summary = summarize_metrics(metrics)
    figure = build_figure(metrics, summary)
    return export_figure(
        figure,
        project_root=root,
        relative_output_dir=OUTPUT_DIR,
        file_stem=FILE_STEM,
        dpi=dpi,
    )


def build_parser() -> argparse.ArgumentParser:
    """Создать CLI-парсер генератора рисунка 6.4."""

    parser = argparse.ArgumentParser(
        description="Сформировать рисунок 6.4 со сравнением частных критериев."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--components", type=Path, default=None)
    parser.add_argument("--q-fact", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Запустить генерацию рисунка 6.4 из командной строки."""

    arguments = build_parser().parse_args(argv)
    result = generate(
        project_root=arguments.project_root,
        q_components_path=arguments.components,
        q_fact_path=arguments.q_fact,
        dpi=arguments.dpi,
    )
    print("Рисунок 6.4 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
