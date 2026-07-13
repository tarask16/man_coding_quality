"""Генерация рисунка 3.3 с распределениями ключевых априорных признаков."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from manual_coding_sim.dissertation_figures.common import (
    FigureExportResult,
    configure_dissertation_style,
    export_figure,
)

OUTPUT_DIR = Path("reports/chapter3/figures")
FILE_STEM = "figure_3_3_prior_feature_distributions"
DEFAULT_INPUT_PATH = Path("data/processed/prior_features.csv")
EXPECTED_SCENARIO_COUNT = 150


@dataclass(frozen=True, slots=True)
class PriorFeatureSpec:
    """Описание априорного признака, отображаемого на рисунке."""

    column: str
    title: str
    x_label: str
    panel_label: str
    discrete: bool
    expected_min: float
    expected_max: float


FEATURE_SPECS: tuple[PriorFeatureSpec, ...] = (
    PriorFeatureSpec(
        column="prior_mean_complexity",
        title="Средняя сложность сообщения",
        x_label="Уровень сложности",
        panel_label="а",
        discrete=True,
        expected_min=1.0,
        expected_max=5.0,
    ),
    PriorFeatureSpec(
        column="prior_mean_message_criticality",
        title="Средняя критичность сообщения",
        x_label="Уровень критичности",
        panel_label="б",
        discrete=True,
        expected_min=1.0,
        expected_max=5.0,
    ),
    PriorFeatureSpec(
        column="prior_operator_total_estimated_time",
        title="Расчётное время оператора",
        x_label="Время, усл. ед.",
        panel_label="в",
        discrete=False,
        expected_min=0.0,
        expected_max=np.inf,
    ),
    PriorFeatureSpec(
        column="prior_condition_time_pressure",
        title="Давление времени",
        x_label="Уровень давления времени",
        panel_label="г",
        discrete=True,
        expected_min=0.0,
        expected_max=3.0,
    ),
    PriorFeatureSpec(
        column="prior_operator_attention",
        title="Априорный уровень внимания",
        x_label="Уровень внимания",
        panel_label="д",
        discrete=True,
        expected_min=1.0,
        expected_max=5.0,
    ),
    PriorFeatureSpec(
        column="prior_expected_error_probability",
        title="Ожидаемая ошибкоопасность",
        x_label="Вероятность ошибки",
        panel_label="е",
        discrete=False,
        expected_min=0.0,
        expected_max=1.0,
    ),
)

REQUIRED_COLUMNS: tuple[str, ...] = tuple(spec.column for spec in FEATURE_SPECS)


def load_prior_features(path: Path) -> pd.DataFrame:
    """Загрузить и проверить таблицу априорных признаков."""

    if not path.is_file():
        raise FileNotFoundError(f"Файл априорных признаков не найден: {path}")

    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "В таблице отсутствуют обязательные колонки: " + ", ".join(missing)
        )

    if frame.empty:
        raise ValueError("Таблица априорных признаков не должна быть пустой.")

    selected = frame.loc[:, REQUIRED_COLUMNS].copy()
    for spec in FEATURE_SPECS:
        series = pd.to_numeric(selected[spec.column], errors="coerce")
        if series.isna().any():
            raise ValueError(f"Признак {spec.column} содержит пропуски или нечисловые значения.")
        values = series.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Признак {spec.column} содержит inf или -inf.")
        if float(values.min()) < spec.expected_min:
            raise ValueError(
                f"Признак {spec.column} выходит ниже допустимой границы "
                f"{spec.expected_min}."
            )
        if float(values.max()) > spec.expected_max:
            raise ValueError(
                f"Признак {spec.column} выходит выше допустимой границы "
                f"{spec.expected_max}."
            )
        if spec.discrete and not np.allclose(values, np.round(values)):
            raise ValueError(f"Дискретный признак {spec.column} содержит дробные значения.")
        selected[spec.column] = series

    return selected


def summarize_prior_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Рассчитать описательные характеристики шести признаков."""

    rows: list[dict[str, float | str]] = []
    for spec in FEATURE_SPECS:
        series = frame[spec.column].astype(float)
        rows.append(
            {
                "column": spec.column,
                "count": float(series.count()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std(ddof=1)),
                "min": float(series.min()),
                "max": float(series.max()),
                "unique": float(series.nunique()),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def _draw_discrete_distribution(
    axis: plt.Axes,
    *,
    values: np.ndarray,
    spec: PriorFeatureSpec,
) -> None:
    """Построить столбчатое распределение дискретного признака."""

    levels = np.arange(int(spec.expected_min), int(spec.expected_max) + 1)
    counts = np.array([(values == level).sum() for level in levels], dtype=int)
    bars = axis.bar(
        levels,
        counts,
        width=0.72,
        color="#6F8FAF",
        edgecolor="#314452",
        linewidth=0.8,
    )
    axis.set_xticks(levels)
    axis.set_ylim(0, max(int(counts.max() * 1.22), 1))
    for bar, count in zip(bars, counts, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.max() * 0.025, 0.4),
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="#263746",
        )


def _draw_continuous_distribution(
    axis: plt.Axes,
    *,
    values: np.ndarray,
) -> None:
    """Построить гистограмму непрерывного признака и отметить медиану."""

    bin_count = max(8, min(14, int(round(np.sqrt(values.size)))))
    axis.hist(
        values,
        bins=bin_count,
        color="#89A7C2",
        edgecolor="#314452",
        linewidth=0.8,
        alpha=0.95,
    )
    median = float(np.median(values))
    axis.axvline(
        median,
        color="#A34A4A",
        linewidth=1.7,
        linestyle="--",
        label=f"Медиана = {median:.3f}",
    )
    axis.legend(loc="upper right", frameon=True, fontsize=7.7)


def build_figure(frame: pd.DataFrame) -> plt.Figure:
    """Построить шестипанельный рисунок распределений априорных признаков."""

    configure_dissertation_style()
    summary = summarize_prior_features(frame)

    figure, axes = plt.subplots(2, 3, figsize=(13.8, 8.4), constrained_layout=False)
    figure.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.12, wspace=0.25, hspace=0.40)
    figure.suptitle(
        "Распределения ключевых априорных признаков расширенного корпуса",
        fontsize=15,
        fontweight="bold",
        y=0.965,
        color="#17212B",
    )
    figure.text(
        0.5,
        0.925,
        f"Число сценариев: N = {len(frame)}; данные используются до фактического выполнения процедуры",
        ha="center",
        va="center",
        fontsize=9.4,
        color="#405464",
    )

    for axis, spec in zip(axes.flat, FEATURE_SPECS, strict=True):
        values = frame[spec.column].to_numpy(dtype=float)
        if spec.discrete:
            _draw_discrete_distribution(axis, values=values, spec=spec)
        else:
            _draw_continuous_distribution(axis, values=values)

        stats = summary.loc[spec.column]
        axis.set_title(
            f"{spec.panel_label}) {spec.title}",
            fontsize=10.5,
            fontweight="bold",
            pad=8,
            color="#263746",
        )
        axis.set_xlabel(spec.x_label, fontsize=9.2)
        axis.set_ylabel("Число сценариев", fontsize=9.2)
        axis.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.55)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=8.5)
        axis.text(
            0.02,
            0.96,
            (
                f"min={stats['min']:.3f}; max={stats['max']:.3f}\n"
                f"mean={stats['mean']:.3f}; std={stats['std']:.3f}"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.4,
            color="#314452",
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": "#B8C4CC",
                "alpha": 0.90,
            },
        )

    figure.text(
        0.5,
        0.045,
        (
            "Примечание — рисунок характеризует вариативность входных априорных признаков; "
            "фактические ошибки и целевые показатели качества в расчёт не включены."
        ),
        ha="center",
        va="center",
        fontsize=8.6,
        color="#405464",
    )
    return figure


def generate(
    *,
    project_root: Path,
    input_path: Path | None = None,
    dpi: int = 300,
) -> FigureExportResult:
    """Сформировать рисунок 3.3 в PNG и SVG."""

    root = project_root.resolve()
    source_path = input_path or (root / DEFAULT_INPUT_PATH)
    if not source_path.is_absolute():
        source_path = root / source_path

    frame = load_prior_features(source_path)
    figure = build_figure(frame)
    return export_figure(
        figure,
        project_root=root,
        relative_output_dir=OUTPUT_DIR,
        file_stem=FILE_STEM,
        dpi=dpi,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Создать анализатор аргументов CLI."""

    parser = argparse.ArgumentParser(
        description="Сформировать рисунок 3.3 с распределениями априорных признаков."
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
        help="Путь к prior_features.csv; по умолчанию data/processed/prior_features.csv.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Разрешение PNG; рекомендуемое значение — 300 dpi.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Запустить генерацию из командной строки."""

    args = build_argument_parser().parse_args(argv)
    result = generate(
        project_root=args.project_root,
        input_path=args.input,
        dpi=args.dpi,
    )
    print("Рисунок 3.3 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
