"""Генерация рисунка 6.7 со сравнением полной модели и базовых прогнозов."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from manual_coding_sim.dissertation_figures.common import (
    FigureExportResult,
    configure_dissertation_style,
    export_figure,
)

OUTPUT_DIR = Path("reports/chapter6/figures")
FILE_STEM = "baseline_comparison"
DEFAULT_INPUT_PATH = Path("reports/chapter6/baseline_comparison.csv")

MODEL_ORDER = ("mean", "prior_only", "full", "theta_only")
MODEL_LABELS = {
    "mean": "Mean baseline",
    "prior_only": "Prior-only",
    "full": "Полная модель",
    "theta_only": "Theta-only",
}
MODEL_DESCRIPTIONS = {
    "mean": "OOF-среднее фактического качества",
    "prior_only": "только априорные признаковые компоненты",
    "full": "априорные признаки + латентный профиль",
    "theta_only": "только латентная компонента q_latent",
}

_MODEL_COLUMN_CANDIDATES = (
    "model",
    "model_name",
    "baseline",
    "baseline_name",
    "prediction_model",
)
_METRIC_COLUMN_CANDIDATES = {
    "mae": ("mae", "MAE", "mean_absolute_error"),
    "rmse": ("rmse", "RMSE", "root_mean_squared_error"),
}

_MODEL_ALIASES = {
    "mean": "mean",
    "mean_baseline": "mean",
    "mean_oof": "mean",
    "oof_mean": "mean",
    "prior": "prior_only",
    "prior_only": "prior_only",
    "prior_baseline": "prior_only",
    "prior_only_baseline": "prior_only",
    "full": "full",
    "full_model": "full",
    "full_pipeline": "full",
    "chapter5_full": "full",
    "chapter5_model": "full",
    "q_pred": "full",
    "theta": "theta_only",
    "theta_only": "theta_only",
    "theta_baseline": "theta_only",
    "theta_only_baseline": "theta_only",
}


@dataclass(frozen=True, slots=True)
class BaselineMetrics:
    """Абсолютные ошибки одной сравниваемой модели."""

    model: str
    mae: float
    rmse: float


@dataclass(frozen=True, slots=True)
class BaselineSummary:
    """Сводка ранжирования моделей по двум метрикам ошибки."""

    best_mae_model: str
    best_rmse_model: str
    full_mae_rank: int
    full_rmse_rank: int
    full_better_than_theta: bool
    full_better_than_mean: bool
    full_better_than_prior: bool
    full_mae_relative_to_mean: float
    full_rmse_relative_to_mean: float


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    """Прочитать CSV-отчёт и вернуть непустой набор строк."""

    if not path.is_file():
        raise FileNotFoundError(f"Не найден входной файл: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Входной файл не содержит строк: {path}")
    return rows


def _resolve_column(columns: Sequence[str], candidates: Sequence[str], *, purpose: str) -> str:
    """Найти колонку по набору допустимых имён."""

    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"Не найдена колонка для {purpose}. Допустимые имена: {', '.join(candidates)}."
    )


def _normalise_model_name(raw_name: str) -> str:
    """Преобразовать имя модели из отчёта в канонический идентификатор."""

    normalised = (
        str(raw_name)
        .strip()
        .lower()
        .replace("—", "-")
        .replace("–", "-")
        .replace(" ", "_")
        .replace("-", "_")
    )
    while "__" in normalised:
        normalised = normalised.replace("__", "_")
    if normalised not in _MODEL_ALIASES:
        raise ValueError(f"Неизвестная модель в baseline-отчёте: {raw_name!r}.")
    return _MODEL_ALIASES[normalised]


def _parse_nonnegative_metric(raw: str, *, metric: str, row_number: int) -> float:
    """Преобразовать метрику ошибки и проверить её корректность."""

    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Строка {row_number}, метрика {metric}: требуется числовое значение."
        ) from error
    if not math.isfinite(value):
        raise ValueError(
            f"Строка {row_number}, метрика {metric}: NaN и бесконечность недопустимы."
        )
    if value < 0.0:
        raise ValueError(
            f"Строка {row_number}, метрика {metric}: значение не может быть отрицательным."
        )
    return value


def load_baseline_metrics(path: Path) -> tuple[BaselineMetrics, ...]:
    """Загрузить MAE и RMSE четырёх обязательных моделей из CSV-отчёта."""

    rows = _read_rows(path)
    columns = tuple(rows[0])
    model_column = _resolve_column(
        columns,
        _MODEL_COLUMN_CANDIDATES,
        purpose="идентификатора модели",
    )
    mae_column = _resolve_column(
        columns,
        _METRIC_COLUMN_CANDIDATES["mae"],
        purpose="MAE",
    )
    rmse_column = _resolve_column(
        columns,
        _METRIC_COLUMN_CANDIDATES["rmse"],
        purpose="RMSE",
    )

    by_model: dict[str, BaselineMetrics] = {}
    for row_number, row in enumerate(rows, start=2):
        model = _normalise_model_name(row[model_column])
        if model in by_model:
            raise ValueError(f"Модель {model!r} указана в отчёте более одного раза.")
        by_model[model] = BaselineMetrics(
            model=model,
            mae=_parse_nonnegative_metric(row[mae_column], metric="MAE", row_number=row_number),
            rmse=_parse_nonnegative_metric(
                row[rmse_column], metric="RMSE", row_number=row_number
            ),
        )

    missing = [model for model in MODEL_ORDER if model not in by_model]
    if missing:
        raise ValueError(
            "В baseline-отчёте отсутствуют обязательные модели: " + ", ".join(missing)
        )
    return tuple(by_model[model] for model in MODEL_ORDER)


def summarise_baselines(metrics: Sequence[BaselineMetrics]) -> BaselineSummary:
    """Рассчитать ранги и сравнительные выводы для полной модели."""

    by_model = {item.model: item for item in metrics}
    missing = [model for model in MODEL_ORDER if model not in by_model]
    if missing:
        raise ValueError("Для сводки отсутствуют модели: " + ", ".join(missing))

    mae_order = sorted(MODEL_ORDER, key=lambda model: by_model[model].mae)
    rmse_order = sorted(MODEL_ORDER, key=lambda model: by_model[model].rmse)
    full = by_model["full"]
    mean = by_model["mean"]
    prior = by_model["prior_only"]
    theta = by_model["theta_only"]

    if mean.mae == 0.0 or mean.rmse == 0.0:
        raise ValueError("Метрики mean baseline должны быть положительными для сравнения.")

    return BaselineSummary(
        best_mae_model=mae_order[0],
        best_rmse_model=rmse_order[0],
        full_mae_rank=mae_order.index("full") + 1,
        full_rmse_rank=rmse_order.index("full") + 1,
        full_better_than_theta=full.mae < theta.mae and full.rmse < theta.rmse,
        full_better_than_mean=full.mae < mean.mae and full.rmse < mean.rmse,
        full_better_than_prior=full.mae < prior.mae and full.rmse < prior.rmse,
        full_mae_relative_to_mean=full.mae / mean.mae - 1.0,
        full_rmse_relative_to_mean=full.rmse / mean.rmse - 1.0,
    )


def _add_value_labels(axis: plt.Axes, bars: Sequence[plt.Rectangle]) -> None:
    """Добавить над столбцами подписи с четырьмя знаками после запятой."""

    for bar in bars:
        height = float(bar.get_height())
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.009,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="semibold",
        )


def _add_summary_card(
    axis: plt.Axes,
    *,
    y: float,
    title: str,
    body: str,
    edge_color: str,
    face_color: str,
) -> None:
    """Добавить текстовую карточку диагностического вывода."""

    card = FancyBboxPatch(
        (0.04, y),
        0.92,
        0.19,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.5,
        edgecolor=edge_color,
        facecolor=face_color,
        transform=axis.transAxes,
        clip_on=False,
    )
    axis.add_patch(card)
    axis.text(
        0.08,
        y + 0.145,
        title,
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    axis.text(
        0.08,
        y + 0.075,
        body,
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=10.3,
        linespacing=1.35,
    )


def build_figure(metrics: Sequence[BaselineMetrics]) -> plt.Figure:
    """Построить группированную диаграмму MAE/RMSE и диагностическую сводку."""

    configure_dissertation_style()
    summary = summarise_baselines(metrics)
    by_model = {item.model: item for item in metrics}

    figure = plt.figure(figsize=(17.2, 8.2))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(3.45, 1.55),
        left=0.065,
        right=0.975,
        top=0.80,
        bottom=0.19,
        wspace=0.16,
    )
    chart_axis = figure.add_subplot(grid[0, 0])
    summary_axis = figure.add_subplot(grid[0, 1])

    x_positions = np.arange(len(MODEL_ORDER), dtype=float)
    bar_width = 0.34
    mae_values = [by_model[model].mae for model in MODEL_ORDER]
    rmse_values = [by_model[model].rmse for model in MODEL_ORDER]

    mae_bars = chart_axis.bar(
        x_positions - bar_width / 2.0,
        mae_values,
        width=bar_width,
        label="MAE",
        color="#4C78A8",
        edgecolor="#244A6A",
        linewidth=0.9,
        zorder=3,
    )
    rmse_bars = chart_axis.bar(
        x_positions + bar_width / 2.0,
        rmse_values,
        width=bar_width,
        label="RMSE",
        color="#F28E2B",
        edgecolor="#8C4F12",
        linewidth=0.9,
        zorder=3,
    )

    # Полную модель выделяем штриховкой, не меняя порядка моделей.
    for bars in (mae_bars, rmse_bars):
        bars[MODEL_ORDER.index("full")].set_hatch("///")
        bars[MODEL_ORDER.index("full")].set_linewidth(1.8)
        bars[MODEL_ORDER.index("full")].set_edgecolor("#1F1F1F")

    _add_value_labels(chart_axis, mae_bars)
    _add_value_labels(chart_axis, rmse_bars)

    upper_limit = max(max(mae_values), max(rmse_values)) * 1.19
    chart_axis.set_ylim(0.0, upper_limit)
    chart_axis.set_ylabel("Ошибка относительно $Q_{fact}$")
    chart_axis.set_xticks(x_positions)
    chart_axis.set_xticklabels(
        [
            "Mean baseline\nOOF-среднее",
            "Prior-only\nбез $q_{latent}$",
            "Полная модель\nглавы 5",
            "Theta-only\nтолько $q_{latent}$",
        ],
        fontsize=11,
    )
    chart_axis.set_title("MAE и RMSE четырёх вариантов прогноза", pad=14, fontweight="bold")
    chart_axis.grid(axis="y", alpha=0.28, zorder=0)
    chart_axis.spines[["top", "right"]].set_visible(False)
    chart_axis.legend(loc="upper left", frameon=False, ncols=2)
    chart_axis.text(
        0.99,
        0.96,
        "меньше — лучше ↓",
        transform=chart_axis.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color="#3D3D3D",
    )

    # Визуальная отметка лучшей пары метрик.
    best_index = MODEL_ORDER.index(summary.best_mae_model)
    chart_axis.axvspan(best_index - 0.46, best_index + 0.46, color="#59A14F", alpha=0.08)

    summary_axis.set_axis_off()
    summary_axis.set_title("Интерпретация результата", pad=14, fontweight="bold")

    _add_summary_card(
        summary_axis,
        y=0.72,
        title="Лучшее абсолютное приближение",
        body=(
            f"MAE: {MODEL_LABELS[summary.best_mae_model]}\n"
            f"RMSE: {MODEL_LABELS[summary.best_rmse_model]}"
        ),
        edge_color="#4E8D45",
        face_color="#F1F8EF",
    )
    _add_summary_card(
        summary_axis,
        y=0.47,
        title="Положение полной модели",
        body=(
            f"Ранг по MAE: {summary.full_mae_rank} из 4\n"
            f"Ранг по RMSE: {summary.full_rmse_rank} из 4"
        ),
        edge_color="#8A6D1D",
        face_color="#FFF8E2",
    )
    _add_summary_card(
        summary_axis,
        y=0.22,
        title="Относительно Mean baseline",
        body=(
            f"MAE выше на {summary.full_mae_relative_to_mean * 100:.1f}%\n"
            f"RMSE выше на {summary.full_rmse_relative_to_mean * 100:.1f}%"
        ),
        edge_color="#A84A44",
        face_color="#FFF1F0",
    )

    verdict = (
        "Полная модель лучше theta-only, но уступает mean baseline и prior-only "
        "по обеим метрикам абсолютной ошибки."
        if summary.full_better_than_theta
        and not summary.full_better_than_mean
        and not summary.full_better_than_prior
        else "Сравнительный вывод определяется фактическими значениями baseline-отчёта."
    )
    summary_axis.text(
        0.04,
        0.115,
        verdict,
        transform=summary_axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        linespacing=1.35,
        wrap=True,
    )

    figure.suptitle(
        "Рисунок 6.7 — Сравнение полной модели с базовыми прогнозами",
        y=0.965,
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.895,
        "Группированная диаграмма абсолютных ошибок на 150 сценариях",
        ha="center",
        va="center",
        fontsize=11.5,
    )
    figure.text(
        0.5,
        0.085,
        (
            "Mean baseline строится out-of-fold; prior-only использует только "
            "q_*_feature_component; theta-only использует q_latent; полная модель — "
            "неизменный Q_pred главы 5. Преимущество по MAE/RMSE характеризует "
            "абсолютную ошибку и не заменяет анализ ранговой согласованности."
        ),
        ha="center",
        va="center",
        fontsize=10.4,
        linespacing=1.35,
    )
    return figure


def generate(
    *,
    project_root: Path,
    input_path: Path | None = None,
    dpi: int = 300,
) -> FigureExportResult:
    """Загрузить baseline-отчёт и сформировать рисунок 6.7."""

    root = project_root.resolve()
    source = input_path if input_path is not None else DEFAULT_INPUT_PATH
    if not source.is_absolute():
        source = root / source
    metrics = load_baseline_metrics(source)
    figure = build_figure(metrics)
    return export_figure(
        figure,
        project_root=root,
        relative_output_dir=OUTPUT_DIR,
        file_stem=FILE_STEM,
        dpi=dpi,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Создать CLI-парсер генератора рисунка 6.7."""

    parser = argparse.ArgumentParser(
        description="Сформировать рисунок 6.7 со сравнением baseline-моделей."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Корень проекта man_coding_quality.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "CSV с метриками baseline-моделей. По умолчанию используется "
            "reports/chapter6/baseline_comparison.csv."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300, help="Разрешение PNG.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить CLI-генерацию рисунка 6.7."""

    arguments = build_argument_parser().parse_args(argv)
    result = generate(
        project_root=arguments.project_root,
        input_path=arguments.input,
        dpi=arguments.dpi,
    )
    print("Рисунок 6.7 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
