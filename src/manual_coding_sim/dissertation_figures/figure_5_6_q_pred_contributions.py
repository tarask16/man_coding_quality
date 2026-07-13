"""Генерация рисунка 5.6 с декомпозицией интегрального прогноза."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch

from manual_coding_sim.dissertation_figures.common import (
    FigureExportResult,
    configure_dissertation_style,
    export_figure,
)

OUTPUT_DIR = Path("reports/chapter5/figures")
FILE_STEM = "figure_5_6_q_pred_contributions"
DEFAULT_COMPONENTS_PATH = Path("reports/chapter5/q_pred_components.csv")
DEFAULT_Q_PRED_PATH = Path("reports/chapter5/q_pred.csv")


@dataclass(frozen=True, slots=True)
class CriterionSpec:
    """Описание частного критерия прогнозной оценки."""

    key: str
    symbol: str
    label: str


CRITERIA: tuple[CriterionSpec, ...] = (
    CriterionSpec("q_acc", "q_acc", "точность восстановления"),
    CriterionSpec("q_time", "q_time", "временная эффективность"),
    CriterionSpec("q_effort", "q_effort", "трудоёмкость выполнения"),
    CriterionSpec("q_res", "q_res", "результативность контроля"),
    CriterionSpec("q_rep", "q_rep", "повторяемость результата"),
    CriterionSpec("q_fit", "q_fit", "соответствие условиям"),
)


@dataclass(frozen=True, slots=True)
class ComponentRow:
    """Компоненты частных прогнозных критериев одного сценария."""

    scenario_id: str
    protocol_id: str
    q_latent: float
    feature_components: Mapping[str, float]
    latent_components: Mapping[str, float]
    observed_weights: Mapping[str, float]
    latent_weights: Mapping[str, float]
    predictions: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class QPredRow:
    """Интегральный прогноз и вклады критериев одного сценария."""

    scenario_id: str
    protocol_id: str
    criterion_weights: Mapping[str, float]
    criterion_contributions: Mapping[str, float]
    q_pred: float


@dataclass(frozen=True, slots=True)
class CriterionContribution:
    """Средние вклады одного критерия в интегральный прогноз."""

    criterion: CriterionSpec
    criterion_weight: float
    feature_contribution: float
    latent_contribution: float
    total_contribution: float
    prediction_mean: float


@dataclass(frozen=True, slots=True)
class ContributionSummary:
    """Сводная декомпозиция среднего интегрального прогноза."""

    count: int
    criteria: tuple[CriterionContribution, ...]
    q_pred_mean: float
    feature_total: float
    latent_total: float
    feature_share: float
    latent_share: float
    maximum_reconstruction_error: float


def _read_float(row: Mapping[str, str], column: str, row_number: int) -> float:
    """Прочитать конечное числовое значение из CSV-строки."""

    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Некорректное значение колонки {column!r} в строке {row_number}."
        ) from error
    if not math.isfinite(value):
        raise ValueError(
            f"Колонка {column!r} в строке {row_number} содержит нечисловое значение."
        )
    return value


def load_components(path: str | Path) -> tuple[ComponentRow, ...]:
    """Загрузить компоненты частных прогнозных критериев."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Не найден файл компонентов прогноза: {source}")

    required = {"scenario_id", "protocol_id", "q_latent"}
    for spec in CRITERIA:
        required.update(
            {
                f"{spec.key}_feature_component",
                f"{spec.key}_latent_component",
                f"{spec.key}_observed_weight",
                f"{spec.key}_latent_weight",
                f"{spec.key}_pred",
            }
        )

    rows: list[ComponentRow] = []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("CSV-файл компонентов не содержит строки заголовка.")
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "В q_pred_components.csv отсутствуют обязательные колонки: "
                + ", ".join(sorted(missing))
            )

        for row_number, row in enumerate(reader, start=2):
            rows.append(
                ComponentRow(
                    scenario_id=str(row["scenario_id"]).strip(),
                    protocol_id=str(row["protocol_id"]).strip(),
                    q_latent=_read_float(row, "q_latent", row_number),
                    feature_components={
                        spec.key: _read_float(
                            row, f"{spec.key}_feature_component", row_number
                        )
                        for spec in CRITERIA
                    },
                    latent_components={
                        spec.key: _read_float(
                            row, f"{spec.key}_latent_component", row_number
                        )
                        for spec in CRITERIA
                    },
                    observed_weights={
                        spec.key: _read_float(
                            row, f"{spec.key}_observed_weight", row_number
                        )
                        for spec in CRITERIA
                    },
                    latent_weights={
                        spec.key: _read_float(
                            row, f"{spec.key}_latent_weight", row_number
                        )
                        for spec in CRITERIA
                    },
                    predictions={
                        spec.key: _read_float(row, f"{spec.key}_pred", row_number)
                        for spec in CRITERIA
                    },
                )
            )

    result = tuple(rows)
    validate_component_rows(result)
    return result


def load_q_pred(path: str | Path) -> tuple[QPredRow, ...]:
    """Загрузить интегральный прогноз и критерийные вклады."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Не найден файл интегрального прогноза: {source}")

    required = {"scenario_id", "protocol_id", "q_pred"}
    for spec in CRITERIA:
        required.update({f"{spec.key}_weight", f"{spec.key}_contribution"})

    rows: list[QPredRow] = []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("CSV-файл Q_pred не содержит строки заголовка.")
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "В q_pred.csv отсутствуют обязательные колонки: "
                + ", ".join(sorted(missing))
            )

        for row_number, row in enumerate(reader, start=2):
            rows.append(
                QPredRow(
                    scenario_id=str(row["scenario_id"]).strip(),
                    protocol_id=str(row["protocol_id"]).strip(),
                    criterion_weights={
                        spec.key: _read_float(row, f"{spec.key}_weight", row_number)
                        for spec in CRITERIA
                    },
                    criterion_contributions={
                        spec.key: _read_float(
                            row, f"{spec.key}_contribution", row_number
                        )
                        for spec in CRITERIA
                    },
                    q_pred=_read_float(row, "q_pred", row_number),
                )
            )

    result = tuple(rows)
    validate_q_pred_rows(result)
    return result


def validate_component_rows(rows: Sequence[ComponentRow]) -> None:
    """Проверить компоненты частных критериев и их весовые формулы."""

    if not rows:
        raise ValueError("Таблица компонентов прогноза не должна быть пустой.")

    keys: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.scenario_id, row.protocol_id)
        if not all(key):
            raise ValueError("Идентификаторы сценария и протокола не должны быть пустыми.")
        if key in keys:
            raise ValueError("Пары scenario_id и protocol_id должны быть уникальными.")
        keys.add(key)

        if not 0.0 <= row.q_latent <= 1.0:
            raise ValueError("q_latent должен лежать в диапазоне [0; 1].")

        for spec in CRITERIA:
            criterion = spec.key
            values = (
                row.feature_components[criterion],
                row.latent_components[criterion],
                row.observed_weights[criterion],
                row.latent_weights[criterion],
                row.predictions[criterion],
            )
            if any(not math.isfinite(value) for value in values):
                raise ValueError("Компоненты и веса критериев должны быть конечными.")
            if any(not 0.0 <= value <= 1.0 for value in values):
                raise ValueError("Компоненты и веса критериев должны лежать в [0; 1].")
            if not math.isclose(
                row.observed_weights[criterion] + row.latent_weights[criterion],
                1.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"Веса признаковой и латентной частей {criterion} должны давать единицу."
                )
            if not math.isclose(
                row.latent_components[criterion],
                row.q_latent,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"Латентная компонента {criterion} должна совпадать с q_latent."
                )
            reconstructed = (
                row.observed_weights[criterion] * row.feature_components[criterion]
                + row.latent_weights[criterion] * row.latent_components[criterion]
            )
            if not math.isclose(
                reconstructed,
                row.predictions[criterion],
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"Частный прогноз {criterion} не совпадает с суммой компонентов."
                )


def validate_q_pred_rows(rows: Sequence[QPredRow]) -> None:
    """Проверить веса критериев и точность сборки Q_pred."""

    if not rows:
        raise ValueError("Таблица интегрального прогноза не должна быть пустой.")

    keys: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.scenario_id, row.protocol_id)
        if not all(key):
            raise ValueError("Идентификаторы сценария и протокола не должны быть пустыми.")
        if key in keys:
            raise ValueError("Пары scenario_id и protocol_id должны быть уникальными.")
        keys.add(key)

        weights = tuple(row.criterion_weights[spec.key] for spec in CRITERIA)
        contributions = tuple(
            row.criterion_contributions[spec.key] for spec in CRITERIA
        )
        if any(not math.isfinite(value) for value in (*weights, *contributions, row.q_pred)):
            raise ValueError("Веса и вклады интегрального прогноза должны быть конечными.")
        if any(value < 0.0 for value in (*weights, *contributions)):
            raise ValueError("Веса и вклады интегрального прогноза неотрицательны.")
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("Сумма весов частных критериев должна быть равна единице.")
        if not 0.0 <= row.q_pred <= 1.0:
            raise ValueError("Q_pred должен лежать в диапазоне [0; 1].")
        if not math.isclose(
            sum(contributions), row.q_pred, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("Q_pred должен совпадать с суммой критерийных вкладов.")


def calculate_summary(
    component_rows: Sequence[ComponentRow],
    q_pred_rows: Sequence[QPredRow],
) -> ContributionSummary:
    """Рассчитать среднюю декомпозицию Q_pred по критериям и источникам."""

    validate_component_rows(component_rows)
    validate_q_pred_rows(q_pred_rows)

    components_by_key = {
        (row.scenario_id, row.protocol_id): row for row in component_rows
    }
    q_pred_by_key = {(row.scenario_id, row.protocol_id): row for row in q_pred_rows}
    if set(components_by_key) != set(q_pred_by_key):
        raise ValueError(
            "Наборы ключей q_pred_components.csv и q_pred.csv должны совпадать."
        )

    feature_sums = {spec.key: 0.0 for spec in CRITERIA}
    latent_sums = {spec.key: 0.0 for spec in CRITERIA}
    prediction_sums = {spec.key: 0.0 for spec in CRITERIA}
    criterion_weight_sums = {spec.key: 0.0 for spec in CRITERIA}
    maximum_error = 0.0
    q_pred_sum = 0.0

    ordered_keys = sorted(components_by_key)
    for key in ordered_keys:
        component_row = components_by_key[key]
        q_pred_row = q_pred_by_key[key]
        scenario_reconstruction = 0.0
        for spec in CRITERIA:
            criterion = spec.key
            criterion_weight = q_pred_row.criterion_weights[criterion]
            feature_contribution = (
                criterion_weight
                * component_row.observed_weights[criterion]
                * component_row.feature_components[criterion]
            )
            latent_contribution = (
                criterion_weight
                * component_row.latent_weights[criterion]
                * component_row.latent_components[criterion]
            )
            total = feature_contribution + latent_contribution
            expected = q_pred_row.criterion_contributions[criterion]
            maximum_error = max(maximum_error, abs(total - expected))
            scenario_reconstruction += total
            feature_sums[criterion] += feature_contribution
            latent_sums[criterion] += latent_contribution
            prediction_sums[criterion] += component_row.predictions[criterion]
            criterion_weight_sums[criterion] += criterion_weight

        maximum_error = max(
            maximum_error, abs(scenario_reconstruction - q_pred_row.q_pred)
        )
        q_pred_sum += q_pred_row.q_pred

    count = len(ordered_keys)
    criterion_summaries = tuple(
        CriterionContribution(
            criterion=spec,
            criterion_weight=criterion_weight_sums[spec.key] / count,
            feature_contribution=feature_sums[spec.key] / count,
            latent_contribution=latent_sums[spec.key] / count,
            total_contribution=(feature_sums[spec.key] + latent_sums[spec.key]) / count,
            prediction_mean=prediction_sums[spec.key] / count,
        )
        for spec in CRITERIA
    )
    q_pred_mean = q_pred_sum / count
    feature_total = sum(item.feature_contribution for item in criterion_summaries)
    latent_total = sum(item.latent_contribution for item in criterion_summaries)
    if not math.isclose(
        feature_total + latent_total,
        q_pred_mean,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Средние вклады не восстанавливают средний Q_pred.")

    return ContributionSummary(
        count=count,
        criteria=criterion_summaries,
        q_pred_mean=q_pred_mean,
        feature_total=feature_total,
        latent_total=latent_total,
        feature_share=feature_total / q_pred_mean,
        latent_share=latent_total / q_pred_mean,
        maximum_reconstruction_error=maximum_error,
    )


def _format_decimal(value: float, digits: int = 4) -> str:
    """Форматировать число с десятичной запятой."""

    return f"{value:.{digits}f}".replace(".", ",")


def _draw_header(figure: plt.Figure, summary: ContributionSummary) -> None:
    """Добавить заголовок и формульную строку рисунка."""

    figure.suptitle(
        "Вклад частных критериев и латентной компоненты в Q_pred",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.940,
        "Q_pred = Σ_j w_j · [α_j · B_j(X_prior,norm) + (1 − α_j) · q_latent]",
        ha="center",
        va="center",
        fontsize=12.5,
        color="#263238",
    )
    figure.text(
        0.5,
        0.908,
        (
            f"Усреднение по N = {summary.count} сценариям; "
            f"средний Q_pred = {_format_decimal(summary.q_pred_mean, 6)}"
        ),
        ha="center",
        va="center",
        fontsize=10.8,
        color="#455A64",
    )


def _draw_criterion_panel(axis: plt.Axes, summary: ContributionSummary) -> None:
    """Показать средние критерийные вклады с разделением источников."""

    items = list(summary.criteria)
    y = np.arange(len(items))
    feature = np.array([item.feature_contribution for item in items])
    latent = np.array([item.latent_contribution for item in items])
    totals = feature + latent

    axis.barh(
        y,
        feature,
        height=0.62,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.8,
        label="априорные признаки",
    )
    axis.barh(
        y,
        latent,
        left=feature,
        height=0.62,
        color="#F2A65A",
        edgecolor="white",
        linewidth=0.8,
        label="латентная компонента",
    )

    labels = [f"{item.criterion.symbol} — {item.criterion.label}" for item in items]
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Средний вклад в Q_pred")
    axis.set_title(
        "а) Средний вклад частных критериев",
        loc="left",
        fontweight="bold",
        pad=13,
    )
    axis.grid(axis="x", alpha=0.25, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.set_xlim(0.0, max(totals) * 1.28)

    for index, (feature_value, latent_value, total_value) in enumerate(
        zip(feature, latent, totals, strict=True)
    ):
        if feature_value > 0.018:
            axis.text(
                feature_value / 2,
                index,
                _format_decimal(feature_value, 3),
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
        if latent_value > 0.016:
            axis.text(
                feature_value + latent_value / 2,
                index,
                _format_decimal(latent_value, 3),
                ha="center",
                va="center",
                fontsize=8.7,
                color="#3E2723",
                fontweight="bold",
            )
        axis.text(
            total_value + 0.0022,
            index,
            _format_decimal(total_value, 4),
            ha="left",
            va="center",
            fontsize=9.4,
            color="#263238",
        )

    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    axis.spines[["top", "right"]].set_visible(False)


def _draw_criterion_total_panel(axis: plt.Axes, summary: ContributionSummary) -> None:
    """Показать состав среднего Q_pred по шести критериям."""

    palette = ("#4C78A8", "#59A14F", "#E15759", "#B279A2", "#76B7B2", "#EDC948")
    left = 0.0
    handles: list[Patch] = []
    for item, color in zip(summary.criteria, palette, strict=True):
        value = item.total_contribution
        axis.barh(
            [0],
            [value],
            left=[left],
            height=0.48,
            color=color,
            edgecolor="white",
            linewidth=1.0,
        )
        axis.text(
            left + value / 2,
            0,
            f"{item.criterion.symbol}\n{_format_decimal(value, 3)}",
            ha="center",
            va="center",
            fontsize=8.3,
            color="white" if color not in {"#EDC948", "#76B7B2"} else "#263238",
            fontweight="bold",
        )
        handles.append(Patch(facecolor=color, label=item.criterion.symbol))
        left += value

    axis.axvline(
        summary.q_pred_mean,
        color="#263238",
        linewidth=1.2,
        linestyle="--",
        alpha=0.9,
    )
    axis.text(
        summary.q_pred_mean,
        0.37,
        f"средний Q_pred = {_format_decimal(summary.q_pred_mean, 4)}",
        ha="right",
        va="bottom",
        fontsize=9.6,
        color="#263238",
    )
    axis.set_xlim(0.0, max(0.56, summary.q_pred_mean * 1.1))
    axis.set_ylim(-0.65, 0.75)
    axis.set_yticks([])
    axis.set_xlabel("Суммарный вклад")
    axis.set_title(
        "б) Состав среднего Q_pred по критериям",
        loc="left",
        fontweight="bold",
        pad=13,
    )
    axis.grid(axis="x", alpha=0.22, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=6,
        frameon=False,
        fontsize=8.2,
        columnspacing=0.8,
        handlelength=1.0,
    )
    axis.spines[["top", "right", "left"]].set_visible(False)


def _draw_source_panel(axis: plt.Axes, summary: ContributionSummary) -> None:
    """Показать доли признакового и латентного источников среднего Q_pred."""

    values = (summary.feature_total, summary.latent_total)
    colors = ("#4C78A8", "#F2A65A")
    labels = ("априорные признаки", "латентная компонента")
    shares = (summary.feature_share, summary.latent_share)

    left = 0.0
    for value, color, label, share in zip(values, colors, labels, shares, strict=True):
        axis.barh(
            [0],
            [value],
            left=[left],
            height=0.52,
            color=color,
            edgecolor="white",
            linewidth=1.0,
        )
        axis.text(
            left + value / 2,
            0,
            f"{label}\n{_format_decimal(value, 4)} · {share * 100:.1f}%".replace(".", ","),
            ha="center",
            va="center",
            fontsize=9.4,
            color="white" if color == "#4C78A8" else "#3E2723",
            fontweight="bold",
        )
        left += value

    axis.set_xlim(0.0, max(0.56, summary.q_pred_mean * 1.1))
    axis.set_ylim(-0.62, 0.78)
    axis.set_yticks([])
    axis.set_xlabel("Средний вклад источника")
    axis.set_title(
        "в) Источники среднего Q_pred",
        loc="left",
        fontweight="bold",
        pad=13,
    )
    axis.grid(axis="x", alpha=0.22, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.text(
        0.0,
        0.82,
        (
            "Точная декомпозиция: "
            f"{_format_decimal(summary.feature_total, 4)} + "
            f"{_format_decimal(summary.latent_total, 4)} = "
            f"{_format_decimal(summary.q_pred_mean, 4)}"
        ),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#37474F",
    )
    axis.spines[["top", "right", "left"]].set_visible(False)


def _draw_footer(figure: plt.Figure, summary: ContributionSummary) -> None:
    """Добавить методическое примечание и сведения о точности сборки."""

    box = FancyBboxPatch(
        (0.035, 0.025),
        0.93,
        0.080,
        transform=figure.transFigure,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor="#F5F7FA",
        edgecolor="#B0BEC5",
        linewidth=1.0,
    )
    figure.add_artist(box)
    figure.text(
        0.055,
        0.074,
        (
            "Интерпретация: столбцы показывают средние аддитивные вклады в расчётный индекс. "
            "Доли источников не являются причинными эффектами и не подтверждают внешнюю точность модели."
        ),
        ha="left",
        va="center",
        fontsize=10.1,
        color="#263238",
    )
    figure.text(
        0.055,
        0.043,
        (
            "Максимальная ошибка численного восстановления Q_pred: "
            f"{summary.maximum_reconstruction_error:.2e}; веса критериев нормированы к единице."
        ),
        ha="left",
        va="center",
        fontsize=9.4,
        color="#546E7A",
    )


def build_figure(summary: ContributionSummary) -> plt.Figure:
    """Построить составной рисунок с декомпозицией Q_pred."""

    configure_dissertation_style()
    figure = plt.figure(figsize=(17.0, 8.8))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.25, 1.0),
        height_ratios=(1.0, 1.0),
        left=0.16,
        right=0.975,
        bottom=0.19,
        top=0.85,
        wspace=0.28,
        hspace=0.58,
    )
    criterion_axis = figure.add_subplot(grid[:, 0])
    total_axis = figure.add_subplot(grid[0, 1])
    source_axis = figure.add_subplot(grid[1, 1])

    _draw_header(figure, summary)
    _draw_criterion_panel(criterion_axis, summary)
    _draw_criterion_total_panel(total_axis, summary)
    _draw_source_panel(source_axis, summary)
    _draw_footer(figure, summary)
    return figure


def generate(
    *,
    project_root: str | Path,
    components_path: str | Path | None = None,
    q_pred_path: str | Path | None = None,
    dpi: int = 300,
) -> FigureExportResult:
    """Сформировать рисунок 5.6 в PNG и SVG."""

    root = Path(project_root).resolve()
    components_source = (
        Path(components_path)
        if components_path is not None
        else root / DEFAULT_COMPONENTS_PATH
    )
    q_pred_source = (
        Path(q_pred_path) if q_pred_path is not None else root / DEFAULT_Q_PRED_PATH
    )
    component_rows = load_components(components_source)
    q_pred_rows = load_q_pred(q_pred_source)
    summary = calculate_summary(component_rows, q_pred_rows)
    figure = build_figure(summary)
    return export_figure(
        figure,
        project_root=root,
        relative_output_dir=OUTPUT_DIR,
        file_stem=FILE_STEM,
        dpi=dpi,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Создать парсер аргументов командной строки."""

    parser = argparse.ArgumentParser(
        description=(
            "Сформировать рисунок 5.6 с вкладами частных критериев "
            "и латентной компоненты в Q_pred."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Корень проекта manual_coding_sim.",
    )
    parser.add_argument(
        "--components",
        type=Path,
        default=None,
        help="Путь к q_pred_components.csv.",
    )
    parser.add_argument(
        "--q-pred",
        type=Path,
        default=None,
        help="Путь к q_pred.csv.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Разрешение PNG, точек на дюйм.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить CLI генератора рисунка 5.6."""

    args = _build_parser().parse_args(argv)
    result = generate(
        project_root=args.project_root,
        components_path=args.components,
        q_pred_path=args.q_pred,
        dpi=args.dpi,
    )
    print("Рисунок 5.6 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
