"""Генерация рисунка 3.5 с результатами проверки достаточности корпуса."""

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
FILE_STEM = "figure_3_5_corpus_sufficiency"
DEFAULT_INPUT_PATH = Path("data/processed/prior_features.csv")
DEFAULT_TOKEN_COUNT = 96
EXPECTED_SCENARIO_COUNT = 150

REQUIRED_COLUMNS: tuple[str, ...] = (
    "scenario_id",
    "protocol_id",
    "prior_mean_complexity",
    "prior_mean_message_criticality",
    "prior_operator_total_estimated_time",
    "prior_condition_total_adjusted_time",
)


@dataclass(frozen=True, slots=True)
class SufficiencyMetric:
    """Фактическое и минимально допустимое значение показателя достаточности."""

    key: str
    label: str
    actual: int
    minimum: int
    group: str

    @property
    def passed(self) -> bool:
        """Вернуть результат прохождения порога достаточности."""

        return self.actual >= self.minimum

    @property
    def ratio(self) -> float:
        """Вернуть отношение фактического значения к минимальному порогу."""

        return self.actual / self.minimum


DEFAULT_MINIMUMS: dict[str, int] = {
    "documents": 100,
    "scenarios": 100,
    "protocols": 100,
    "tokens": 30,
    "complexity_levels": 3,
    "criticality_levels": 3,
    "operator_time_levels": 3,
    "condition_time_levels": 3,
}


def load_prior_features(path: Path) -> pd.DataFrame:
    """Загрузить таблицу априорных признаков и проверить обязательные поля."""

    if not path.is_file():
        raise FileNotFoundError(f"Файл априорных признаков не найден: {path}")

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("Таблица априорных признаков не должна быть пустой.")

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "В таблице отсутствуют обязательные колонки: " + ", ".join(missing)
        )

    selected = frame.loc[:, REQUIRED_COLUMNS].copy()
    if selected[["scenario_id", "protocol_id"]].isna().any().any():
        raise ValueError("Идентификаторы сценариев и протоколов не должны содержать NaN.")

    numeric_columns = REQUIRED_COLUMNS[2:]
    for column in numeric_columns:
        series = pd.to_numeric(selected[column], errors="coerce")
        if series.isna().any():
            raise ValueError(
                f"Признак {column} содержит пропуски или нечисловые значения."
            )
        values = series.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Признак {column} содержит inf или -inf.")
        selected[column] = series

    return selected


def build_sufficiency_metrics(
    frame: pd.DataFrame,
    *,
    token_count: int = DEFAULT_TOKEN_COUNT,
    minimums: dict[str, int] | None = None,
) -> tuple[SufficiencyMetric, ...]:
    """Сформировать показатели достаточности корпуса и покрытия признаков."""

    if token_count <= 0:
        raise ValueError("Число токенов словаря должно быть положительным.")

    thresholds = dict(DEFAULT_MINIMUMS if minimums is None else minimums)
    missing_thresholds = [key for key in DEFAULT_MINIMUMS if key not in thresholds]
    if missing_thresholds:
        raise ValueError(
            "Не заданы пороги для показателей: " + ", ".join(missing_thresholds)
        )
    if any(int(value) <= 0 for value in thresholds.values()):
        raise ValueError("Все минимальные пороги должны быть положительными.")

    return (
        SufficiencyMetric(
            "documents",
            "Документы корпуса",
            int(len(frame)),
            int(thresholds["documents"]),
            "corpus",
        ),
        SufficiencyMetric(
            "scenarios",
            "Уникальные сценарии",
            int(frame["scenario_id"].nunique(dropna=False)),
            int(thresholds["scenarios"]),
            "corpus",
        ),
        SufficiencyMetric(
            "protocols",
            "Уникальные протоколы",
            int(frame["protocol_id"].nunique(dropna=False)),
            int(thresholds["protocols"]),
            "corpus",
        ),
        SufficiencyMetric(
            "tokens",
            "Токены словаря LDA",
            int(token_count),
            int(thresholds["tokens"]),
            "corpus",
        ),
        SufficiencyMetric(
            "complexity_levels",
            "Уровни сложности сообщения",
            int(frame["prior_mean_complexity"].nunique()),
            int(thresholds["complexity_levels"]),
            "categorical",
        ),
        SufficiencyMetric(
            "criticality_levels",
            "Уровни критичности сообщения",
            int(frame["prior_mean_message_criticality"].nunique()),
            int(thresholds["criticality_levels"]),
            "categorical",
        ),
        SufficiencyMetric(
            "operator_time_levels",
            "Уникальные значения расчётного времени оператора",
            int(frame["prior_operator_total_estimated_time"].nunique()),
            int(thresholds["operator_time_levels"]),
            "continuous",
        ),
        SufficiencyMetric(
            "condition_time_levels",
            "Уникальные значения скорректированного времени условий",
            int(frame["prior_condition_total_adjusted_time"].nunique()),
            int(thresholds["condition_time_levels"]),
            "continuous",
        ),
    )


def _draw_metric_panel(
    axis: plt.Axes,
    metrics: Sequence[SufficiencyMetric],
    *,
    title: str,
    x_label: str,
) -> None:
    """Нарисовать панель с парными фактическими и пороговыми значениями."""

    positions = np.arange(len(metrics), dtype=float)
    bar_height = 0.32
    actual_values = np.array([metric.actual for metric in metrics], dtype=float)
    minimum_values = np.array([metric.minimum for metric in metrics], dtype=float)

    axis.barh(
        positions - bar_height / 2,
        actual_values,
        height=bar_height,
        color="#6E91AE",
        edgecolor="#314452",
        linewidth=0.8,
        label="Фактическое значение",
        zorder=3,
    )
    axis.barh(
        positions + bar_height / 2,
        minimum_values,
        height=bar_height,
        color="#D8DEE3",
        edgecolor="#6B7780",
        linewidth=0.8,
        hatch="//",
        label="Минимальный порог",
        zorder=3,
    )

    max_value = max(float(actual_values.max()), float(minimum_values.max()))
    axis.set_xlim(0, max_value * 1.28)
    axis.set_yticks(positions)
    axis.set_yticklabels([metric.label for metric in metrics], fontsize=9.2)
    axis.invert_yaxis()
    axis.set_title(title, loc="left", fontsize=12.2, fontweight="bold", pad=9)
    axis.set_xlabel(x_label, fontsize=9.4)
    axis.grid(axis="x", linestyle=":", linewidth=0.75, alpha=0.60)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#7D8D98")
    axis.spines["bottom"].set_color("#7D8D98")

    for position, metric in zip(positions, metrics, strict=True):
        axis.text(
            metric.actual + max_value * 0.018,
            position - bar_height / 2,
            f"{metric.actual}",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#24343F",
            fontweight="bold",
        )
        axis.text(
            metric.minimum + max_value * 0.018,
            position + bar_height / 2,
            f"{metric.minimum}",
            ha="left",
            va="center",
            fontsize=8.1,
            color="#56636D",
        )
        status = "passed" if metric.passed else "failed"
        status_color = "#2E6E50" if metric.passed else "#A64040"
        axis.text(
            max_value * 1.245,
            position,
            f"{status}\n×{metric.ratio:.1f}",
            ha="right",
            va="center",
            fontsize=8.0,
            color=status_color,
            fontweight="bold",
        )


def build_figure(metrics: Sequence[SufficiencyMetric]) -> plt.Figure:
    """Построить комбинированную диаграмму достаточности корпуса."""

    configure_dissertation_style()
    groups = {
        "corpus": [metric for metric in metrics if metric.group == "corpus"],
        "categorical": [
            metric for metric in metrics if metric.group == "categorical"
        ],
        "continuous": [metric for metric in metrics if metric.group == "continuous"],
    }
    if any(not group for group in groups.values()):
        raise ValueError("Для рисунка должны быть заданы все три группы показателей.")

    figure = plt.figure(figsize=(14.2, 11.4), constrained_layout=False)
    grid = figure.add_gridspec(
        3,
        1,
        height_ratios=(1.45, 0.86, 0.86),
        left=0.27,
        right=0.965,
        top=0.84,
        bottom=0.11,
        hspace=0.50,
    )
    axis_corpus = figure.add_subplot(grid[0, 0])
    axis_categorical = figure.add_subplot(grid[1, 0])
    axis_continuous = figure.add_subplot(grid[2, 0])

    figure.suptitle(
        "Проверка достаточности расширенного вычислительного корпуса",
        fontsize=15,
        fontweight="bold",
        y=0.955,
        color="#17212B",
    )
    overall_passed = all(metric.passed for metric in metrics)
    figure.text(
        0.5,
        0.908,
        (
            "Фактические значения сопоставлены с минимальными порогами; "
            f"итоговый статус CorpusSufficiencyAnalyzer: "
            f"{'passed' if overall_passed else 'failed'}"
        ),
        ha="center",
        va="center",
        fontsize=9.6,
        color="#2E6E50" if overall_passed else "#A64040",
        fontweight="bold",
    )

    _draw_metric_panel(
        axis_corpus,
        groups["corpus"],
        title="а) Размер корпуса и словаря",
        x_label="Количество элементов",
    )
    _draw_metric_panel(
        axis_categorical,
        groups["categorical"],
        title="б) Покрытие дискретных уровней",
        x_label="Количество уникальных уровней",
    )
    _draw_metric_panel(
        axis_continuous,
        groups["continuous"],
        title="в) Разнообразие непрерывных признаков",
        x_label="Количество уникальных значений",
    )

    handles, labels = axis_corpus.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=2,
        frameon=False,
        fontsize=8.8,
    )
    figure.text(
        0.5,
        0.055,
        (
            "Примечание — значение ×k показывает кратность превышения минимального порога. "
            "Порог достаточности используется как вычислительный gate и не заменяет "
            "проверку внешней валидности модели."
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
    token_count: int = DEFAULT_TOKEN_COUNT,
    dpi: int = 300,
) -> FigureExportResult:
    """Сформировать рисунок 3.5 в PNG и SVG."""

    root = project_root.resolve()
    source_path = input_path or (root / DEFAULT_INPUT_PATH)
    if not source_path.is_absolute():
        source_path = root / source_path

    frame = load_prior_features(source_path)
    metrics = build_sufficiency_metrics(frame, token_count=token_count)
    figure = build_figure(metrics)
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
        description=(
            "Сформировать рисунок 3.5 с фактическими и пороговыми "
            "показателями достаточности корпуса."
        )
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
            "Путь к prior_features.csv; по умолчанию "
            "data/processed/prior_features.csv."
        ),
    )
    parser.add_argument(
        "--token-count",
        type=int,
        default=DEFAULT_TOKEN_COUNT,
        help="Число токенов итогового словаря LDA; по умолчанию 96.",
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
        token_count=args.token_count,
        dpi=args.dpi,
    )
    print("Рисунок 3.5 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
