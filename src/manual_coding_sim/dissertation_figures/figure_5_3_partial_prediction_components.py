"""Генерация рисунка 5.3 с распределениями частных прогнозных критериев."""

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
FILE_STEM = "figure_5_3_partial_prediction_components"
DEFAULT_INPUT_PATH = Path("reports/chapter5/q_pred_components.csv")


@dataclass(frozen=True, slots=True)
class CriterionSpec:
    """Описание одного частного прогнозного критерия."""

    code: str
    prediction_column: str
    feature_weight_column: str
    latent_weight_column: str
    short_label: str
    full_name: str


CRITERIA = (
    CriterionSpec(
        "q_acc",
        "q_acc_pred",
        "q_acc_observed_weight",
        "q_acc_latent_weight",
        "Точность",
        "точность восстановления",
    ),
    CriterionSpec(
        "q_time",
        "q_time_pred",
        "q_time_observed_weight",
        "q_time_latent_weight",
        "Временная\nэффективность",
        "временная эффективность",
    ),
    CriterionSpec(
        "q_effort",
        "q_effort_pred",
        "q_effort_observed_weight",
        "q_effort_latent_weight",
        "Трудоёмкость",
        "трудоёмкость выполнения",
    ),
    CriterionSpec(
        "q_res",
        "q_res_pred",
        "q_res_observed_weight",
        "q_res_latent_weight",
        "Результативность\nконтроля",
        "результативность контроля",
    ),
    CriterionSpec(
        "q_rep",
        "q_rep_pred",
        "q_rep_observed_weight",
        "q_rep_latent_weight",
        "Повторяемость",
        "повторяемость результата",
    ),
    CriterionSpec(
        "q_fit",
        "q_fit_pred",
        "q_fit_observed_weight",
        "q_fit_latent_weight",
        "Соответствие\nусловиям",
        "соответствие условиям",
    ),
)

COLORS = (
    "#4C78A8",
    "#59A14F",
    "#F28E2B",
    "#E15759",
    "#B279A2",
    "#76B7B2",
)


@dataclass(frozen=True, slots=True)
class PartialPredictionRow:
    """Частные прогнозные критерии одного сценария."""

    scenario_id: str
    protocol_id: str
    values: tuple[float, ...]
    feature_weights: tuple[float, ...]
    latent_weights: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CriterionSummary:
    """Описательная статистика одного частного критерия."""

    code: str
    minimum: float
    first_quartile: float
    median: float
    mean: float
    third_quartile: float
    maximum: float
    standard_deviation: float
    feature_weight: float
    latent_weight: float


def _required_columns() -> set[str]:
    """Вернуть множество обязательных колонок входного CSV."""

    columns = {"scenario_id", "protocol_id"}
    for criterion in CRITERIA:
        columns.update(
            {
                criterion.prediction_column,
                criterion.feature_weight_column,
                criterion.latent_weight_column,
            }
        )
    return columns


def load_partial_predictions(path: str | Path) -> tuple[PartialPredictionRow, ...]:
    """Загрузить частные прогнозные критерии из CSV-файла."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Не найден файл частных прогнозных критериев: {source}")

    rows: list[PartialPredictionRow] = []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("CSV-файл не содержит строки заголовка.")
        missing = _required_columns().difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "В q_pred_components.csv отсутствуют обязательные колонки: "
                + ", ".join(sorted(missing))
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                rows.append(
                    PartialPredictionRow(
                        scenario_id=str(row["scenario_id"]).strip(),
                        protocol_id=str(row["protocol_id"]).strip(),
                        values=tuple(
                            float(row[criterion.prediction_column])
                            for criterion in CRITERIA
                        ),
                        feature_weights=tuple(
                            float(row[criterion.feature_weight_column])
                            for criterion in CRITERIA
                        ),
                        latent_weights=tuple(
                            float(row[criterion.latent_weight_column])
                            for criterion in CRITERIA
                        ),
                    )
                )
            except (TypeError, ValueError, KeyError) as error:
                raise ValueError(
                    f"Некорректная строка {row_number} q_pred_components.csv: {error}."
                ) from error

    result = tuple(rows)
    validate_partial_predictions(result)
    return result


def validate_partial_predictions(rows: Sequence[PartialPredictionRow]) -> None:
    """Проверить полноту, диапазоны и постоянство весов критериев."""

    if not rows:
        raise ValueError("Файл частных прогнозных критериев не должен быть пустым.")

    scenario_ids: set[str] = set()
    protocol_ids: set[str] = set()
    reference_feature_weights = rows[0].feature_weights
    reference_latent_weights = rows[0].latent_weights

    if not (
        len(reference_feature_weights)
        == len(reference_latent_weights)
        == len(CRITERIA)
    ):
        raise ValueError("Число весов должно совпадать с числом критериев.")

    for row in rows:
        if not row.scenario_id or not row.protocol_id:
            raise ValueError("Идентификаторы сценария и протокола не должны быть пустыми.")
        if row.scenario_id in scenario_ids:
            raise ValueError("Идентификаторы сценариев должны быть уникальными.")
        if row.protocol_id in protocol_ids:
            raise ValueError("Идентификаторы протоколов должны быть уникальными.")
        scenario_ids.add(row.scenario_id)
        protocol_ids.add(row.protocol_id)

        if not (
            len(row.values)
            == len(row.feature_weights)
            == len(row.latent_weights)
            == len(CRITERIA)
        ):
            raise ValueError("Каждая строка должна содержать шесть критериев и их веса.")

        for value in row.values:
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    "Частные прогнозные критерии должны быть конечными и лежать в [0; 1]."
                )

        for feature_weight, latent_weight in zip(
            row.feature_weights, row.latent_weights, strict=True
        ):
            if not (
                math.isfinite(feature_weight)
                and math.isfinite(latent_weight)
                and 0.0 <= feature_weight <= 1.0
                and 0.0 <= latent_weight <= 1.0
            ):
                raise ValueError("Веса компонентов должны лежать в диапазоне [0; 1].")
            if not math.isclose(
                feature_weight + latent_weight,
                1.0,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError("Сумма весов признаковой и латентной компонент должна быть равна 1.")

        if any(
            not math.isclose(current, reference, rel_tol=1e-9, abs_tol=1e-9)
            for current, reference in zip(
                row.feature_weights, reference_feature_weights, strict=True
            )
        ):
            raise ValueError("Веса априорных признаков должны быть постоянными по сценариям.")
        if any(
            not math.isclose(current, reference, rel_tol=1e-9, abs_tol=1e-9)
            for current, reference in zip(
                row.latent_weights, reference_latent_weights, strict=True
            )
        ):
            raise ValueError("Веса латентной компоненты должны быть постоянными по сценариям.")


def prediction_matrix(rows: Sequence[PartialPredictionRow]) -> np.ndarray:
    """Сформировать матрицу значений размерности N × 6."""

    validate_partial_predictions(rows)
    matrix = np.asarray([row.values for row in rows], dtype=float)
    if matrix.shape != (len(rows), len(CRITERIA)):
        raise ValueError("Некорректная размерность матрицы частных критериев.")
    return matrix


def calculate_summaries(
    rows: Sequence[PartialPredictionRow],
) -> tuple[CriterionSummary, ...]:
    """Рассчитать статистику распределения каждого частного критерия."""

    matrix = prediction_matrix(rows)
    feature_weights = rows[0].feature_weights
    latent_weights = rows[0].latent_weights
    summaries: list[CriterionSummary] = []

    for index, criterion in enumerate(CRITERIA):
        values = matrix[:, index]
        summaries.append(
            CriterionSummary(
                code=criterion.code,
                minimum=float(np.min(values)),
                first_quartile=float(np.quantile(values, 0.25)),
                median=float(median(values.tolist())),
                mean=float(mean(values.tolist())),
                third_quartile=float(np.quantile(values, 0.75)),
                maximum=float(np.max(values)),
                standard_deviation=float(np.std(values, ddof=1)),
                feature_weight=float(feature_weights[index]),
                latent_weight=float(latent_weights[index]),
            )
        )
    return tuple(summaries)


def build_figure(rows: Sequence[PartialPredictionRow]) -> plt.Figure:
    """Построить violin- и boxplot частных прогнозных критериев."""

    matrix = prediction_matrix(rows)
    summaries = calculate_summaries(rows)

    configure_dissertation_style()
    figure = plt.figure(figsize=(17.8, 9.7))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(3.35, 1.35),
        left=0.065,
        right=0.975,
        top=0.82,
        bottom=0.19,
        wspace=0.20,
    )
    distribution_axis = figure.add_subplot(grid[0, 0])
    summary_axis = figure.add_subplot(grid[0, 1])

    figure.suptitle(
        "Рисунок 5.3 — Распределения частных прогнозных критериев",
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )
    figure.text(
        0.5,
        0.918,
        "q_j,pred = α_j · B_j(X_prior,norm) + (1 − α_j) · q_latent",
        ha="center",
        fontsize=12.0,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.883,
        f"N = {len(rows)} сценариев; единая шкала [0; 1] обеспечивает сопоставимость шести критериев.",
        ha="center",
        fontsize=10.7,
    )

    positions = np.arange(1, len(CRITERIA) + 1, dtype=float)
    series = [matrix[:, index] for index in range(matrix.shape[1])]
    violins = distribution_axis.violinplot(
        series,
        positions=positions,
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.28,
    )
    for body, color in zip(violins["bodies"], COLORS, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.38)
        body.set_linewidth(1.0)

    boxplot = distribution_axis.boxplot(
        series,
        positions=positions,
        widths=0.18,
        patch_artist=True,
        showfliers=True,
        medianprops={"linewidth": 2.2, "color": "#B23A48"},
        whiskerprops={"linewidth": 1.2, "color": "#37474F"},
        capprops={"linewidth": 1.2, "color": "#37474F"},
        boxprops={"linewidth": 1.1, "facecolor": "white", "alpha": 0.92},
        flierprops={
            "marker": "o",
            "markersize": 3.0,
            "markerfacecolor": "#5C6770",
            "markeredgecolor": "none",
            "alpha": 0.40,
        },
    )
    if len(boxplot["boxes"]) != len(CRITERIA):
        raise RuntimeError("Число boxplot не соответствует числу критериев.")

    means = np.asarray([summary.mean for summary in summaries], dtype=float)
    distribution_axis.scatter(
        positions,
        means,
        marker="D",
        s=54,
        color="#1D6F8A",
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
    )

    for position, summary in zip(positions, summaries, strict=True):
        distribution_axis.text(
            position,
            min(summary.maximum + 0.045, 1.035),
            f"μ={summary.mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.1,
            color="#1D6F8A",
            fontweight="bold",
        )

    distribution_axis.set_xlim(0.40, len(CRITERIA) + 0.60)
    distribution_axis.set_ylim(0.0, 1.08)
    distribution_axis.set_xticks(positions)
    distribution_axis.set_xticklabels(
        [f"{criterion.short_label}\n{criterion.prediction_column}" for criterion in CRITERIA],
        fontsize=9.2,
    )
    distribution_axis.set_ylabel("Априорная прогнозная оценка")
    distribution_axis.set_title(
        "А. Форма распределений и квартильная структура",
        fontweight="bold",
        pad=13,
    )
    distribution_axis.grid(axis="y", alpha=0.28, linewidth=0.8)
    distribution_axis.set_axisbelow(True)

    legend_handles = (
        Patch(facecolor="#AFC9D8", edgecolor="#4C78A8", alpha=0.55, label="Плотность"),
        Patch(facecolor="white", edgecolor="#37474F", label="Межквартильный диапазон"),
        Line2D([0], [0], color="#B23A48", linewidth=2.2, label="Медиана"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="#1D6F8A",
            markeredgecolor="white",
            markersize=7,
            label="Среднее",
        ),
    )
    distribution_axis.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=False,
        fontsize=9.2,
    )

    summary_axis.set_axis_off()
    summary_axis.set_title(
        "Б. Сводные характеристики и веса компонентов",
        fontweight="bold",
        pad=13,
    )
    summary_axis.text(
        0.02,
        0.955,
        "Критерий",
        transform=summary_axis.transAxes,
        fontsize=9.3,
        fontweight="bold",
        va="top",
    )
    summary_axis.text(
        0.56,
        0.955,
        "μ / Me",
        transform=summary_axis.transAxes,
        fontsize=9.3,
        fontweight="bold",
        va="top",
        ha="center",
    )
    summary_axis.text(
        0.86,
        0.955,
        "α / (1−α)",
        transform=summary_axis.transAxes,
        fontsize=9.3,
        fontweight="bold",
        va="top",
        ha="center",
    )
    summary_axis.plot(
        [0.01, 0.99],
        [0.91, 0.91],
        transform=summary_axis.transAxes,
        color="#7F8C8D",
        linewidth=0.9,
    )

    y_positions = np.linspace(0.84, 0.24, len(CRITERIA))
    for index, (criterion, summary, y_position, color) in enumerate(
        zip(CRITERIA, summaries, y_positions, COLORS, strict=True)
    ):
        summary_axis.add_patch(
            plt.Rectangle(
                (0.015, y_position - 0.028),
                0.020,
                0.056,
                transform=summary_axis.transAxes,
                facecolor=color,
                edgecolor="none",
                alpha=0.85,
            )
        )
        summary_axis.text(
            0.055,
            y_position,
            f"{criterion.code}\n{criterion.full_name}",
            transform=summary_axis.transAxes,
            fontsize=8.9,
            va="center",
        )
        summary_axis.text(
            0.56,
            y_position,
            f"{summary.mean:.3f} / {summary.median:.3f}",
            transform=summary_axis.transAxes,
            fontsize=9.0,
            ha="center",
            va="center",
        )
        summary_axis.text(
            0.86,
            y_position,
            f"{summary.feature_weight:.2f} / {summary.latent_weight:.2f}",
            transform=summary_axis.transAxes,
            fontsize=9.0,
            ha="center",
            va="center",
        )
        if index < len(CRITERIA) - 1:
            summary_axis.plot(
                [0.02, 0.98],
                [y_position - 0.066, y_position - 0.066],
                transform=summary_axis.transAxes,
                color="#D6DBDF",
                linewidth=0.7,
            )

    summary_axis.text(
        0.04,
        0.105,
        "Обозначения весов:\n"
        "α — вклад направленно нормированных\n"
        "априорных признаков; 1−α — вклад q_latent.",
        transform=summary_axis.transAxes,
        fontsize=9.2,
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#F4F7F9",
            "edgecolor": "#AAB7B8",
            "linewidth": 0.8,
        },
    )

    figure.text(
        0.5,
        0.055,
        "Методическое ограничение: показаны априорные q_j,pred, а не фактические q_j,fact. "
        "Общая латентная компонента q_latent входит во все шесть критериев, поэтому их распределения не являются независимыми.",
        ha="center",
        va="center",
        fontsize=9.8,
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
    """Сформировать рисунок 5.3 в PNG и SVG."""

    root = Path(project_root).resolve()
    source = Path(input_path) if input_path is not None else root / DEFAULT_INPUT_PATH
    rows = load_partial_predictions(source)
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
        description="Сформировать рисунок 5.3 с частными прогнозными критериями."
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
        help="Путь к q_pred_components.csv; по умолчанию используется reports/chapter5.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Разрешение PNG; по умолчанию 300 dpi.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Запустить генератор рисунка 5.3 из командной строки."""

    arguments = build_argument_parser().parse_args(argv)
    result = generate(
        project_root=arguments.project_root,
        input_path=arguments.input,
        dpi=arguments.dpi,
    )
    print("Рисунок 5.3 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
