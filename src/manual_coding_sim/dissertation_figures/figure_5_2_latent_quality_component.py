"""Генерация рисунка 5.2 с распределением латентной компоненты качества."""

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

from manual_coding_sim.dissertation_figures.common import (
    FigureExportResult,
    configure_dissertation_style,
    export_figure,
)

OUTPUT_DIR = Path("reports/chapter5/figures")
FILE_STEM = "figure_5_2_latent_quality_component"
DEFAULT_INPUT_PATH = Path("reports/chapter4/theta_prior.csv")
THETA_COLUMNS = ("theta_0", "theta_1", "theta_2")
DIRECTION_WEIGHTS = (-1.0, -1.0, 1.0)
EXPECTED_TOPIC_COUNT = 3

TOPIC_NAMES = (
    "θ₀ — процедурная трудоёмкость",
    "θ₁ — операционный риск",
    "θ₂ — благоприятные условия",
)


@dataclass(frozen=True, slots=True)
class ThetaProfile:
    """Латентный профиль одного сценария."""

    scenario_id: str
    protocol_id: str
    theta_0: float
    theta_1: float
    theta_2: float
    selected_k: int

    @property
    def values(self) -> tuple[float, float, float]:
        """Вернуть компоненты профиля в фиксированном порядке."""

        return (self.theta_0, self.theta_1, self.theta_2)


@dataclass(frozen=True, slots=True)
class LatentQualitySummary:
    """Описательная статистика латентной компоненты качества."""

    sample_size: int
    minimum: float
    first_quartile: float
    median: float
    mean: float
    third_quartile: float
    maximum: float
    standard_deviation: float
    mean_signed_terms: tuple[float, float, float]
    max_identity_error: float



def load_theta_profiles(path: str | Path) -> tuple[ThetaProfile, ...]:
    """Загрузить компоненты ``theta_prior`` из CSV-файла."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Не найден файл латентных профилей: {source}")

    required = {
        "scenario_id",
        "protocol_id",
        "theta_0",
        "theta_1",
        "theta_2",
        "selected_k",
    }
    profiles: list[ThetaProfile] = []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("CSV-файл не содержит строки заголовка.")
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "В theta_prior.csv отсутствуют обязательные колонки: "
                + ", ".join(sorted(missing))
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                profiles.append(
                    ThetaProfile(
                        scenario_id=str(row["scenario_id"]).strip(),
                        protocol_id=str(row["protocol_id"]).strip(),
                        theta_0=float(row["theta_0"]),
                        theta_1=float(row["theta_1"]),
                        theta_2=float(row["theta_2"]),
                        selected_k=int(row["selected_k"]),
                    )
                )
            except (TypeError, ValueError, KeyError) as error:
                raise ValueError(
                    f"Некорректная строка {row_number} theta_prior.csv: {error}."
                ) from error

    result = tuple(profiles)
    validate_theta_profiles(result)
    return result



def validate_theta_profiles(profiles: Sequence[ThetaProfile]) -> None:
    """Проверить полноту, диапазоны и нормировку латентных профилей."""

    if not profiles:
        raise ValueError("Файл латентных профилей не должен быть пустым.")

    scenario_ids: set[str] = set()
    protocol_ids: set[str] = set()
    for profile in profiles:
        if not profile.scenario_id or not profile.protocol_id:
            raise ValueError("Идентификаторы сценария и протокола не должны быть пустыми.")
        if profile.scenario_id in scenario_ids:
            raise ValueError("Идентификаторы сценариев должны быть уникальными.")
        if profile.protocol_id in protocol_ids:
            raise ValueError("Идентификаторы протоколов должны быть уникальными.")
        scenario_ids.add(profile.scenario_id)
        protocol_ids.add(profile.protocol_id)

        if profile.selected_k != EXPECTED_TOPIC_COUNT:
            raise ValueError("Для рисунка 5.2 ожидается selected_k = 3.")
        for value in profile.values:
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("Компоненты theta должны быть конечными и лежать в [0; 1].")
        if not math.isclose(sum(profile.values), 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("Сумма компонентов theta каждого сценария должна быть равна единице.")



def calculate_latent_quality(profile: ThetaProfile) -> float:
    """Рассчитать латентную компоненту качества для одного сценария."""

    directed_sum = sum(
        direction * value
        for direction, value in zip(DIRECTION_WEIGHTS, profile.values, strict=True)
    )
    result = (directed_sum + 1.0) / 2.0
    if not -1e-10 <= result <= 1.0 + 1e-10:
        raise ValueError("Латентная компонента качества вышла за диапазон [0; 1].")
    return float(np.clip(result, 0.0, 1.0))



def calculate_latent_values(profiles: Sequence[ThetaProfile]) -> np.ndarray:
    """Рассчитать латентную компоненту качества для набора сценариев."""

    validate_theta_profiles(profiles)
    return np.asarray([calculate_latent_quality(profile) for profile in profiles], dtype=float)



def calculate_summary(profiles: Sequence[ThetaProfile]) -> LatentQualitySummary:
    """Рассчитать описательную статистику и направленные члены формулы."""

    values = calculate_latent_values(profiles)
    theta = np.asarray([profile.values for profile in profiles], dtype=float)
    signed_terms = theta * np.asarray(DIRECTION_WEIGHTS, dtype=float)
    identity_error = np.max(np.abs(values - theta[:, 2]))

    return LatentQualitySummary(
        sample_size=len(profiles),
        minimum=float(np.min(values)),
        first_quartile=float(np.quantile(values, 0.25)),
        median=float(median(values.tolist())),
        mean=float(mean(values.tolist())),
        third_quartile=float(np.quantile(values, 0.75)),
        maximum=float(np.max(values)),
        standard_deviation=float(np.std(values, ddof=1)),
        mean_signed_terms=tuple(float(item) for item in np.mean(signed_terms, axis=0)),
        max_identity_error=float(identity_error),
    )



def build_figure(profiles: Sequence[ThetaProfile]) -> plt.Figure:
    """Построить гистограмму, boxplot и панель направлений компонентов theta."""

    validate_theta_profiles(profiles)
    latent_values = calculate_latent_values(profiles)
    summary = calculate_summary(profiles)

    configure_dissertation_style()
    figure = plt.figure(figsize=(17.4, 9.6))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.55, 1.0),
        height_ratios=(1.0, 1.0),
        left=0.065,
        right=0.975,
        top=0.82,
        bottom=0.18,
        hspace=0.54,
        wspace=0.30,
    )
    histogram_axis = figure.add_subplot(grid[:, 0])
    box_axis = figure.add_subplot(grid[0, 1])
    direction_axis = figure.add_subplot(grid[1, 1])

    figure.suptitle(
        "Рисунок 5.2 — Распределение латентной компоненты качества",
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )
    figure.text(
        0.5,
        0.918,
        "q_latent = (−θ₀ − θ₁ + θ₂ + 1) / 2,  где d = (−1, −1, +1)",
        ha="center",
        fontsize=12.0,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.885,
        f"N = {summary.sample_size};  вследствие θ₀ + θ₁ + θ₂ = 1 выполняется "
        "точное тождество q_latent = θ₂.",
        ha="center",
        fontsize=10.7,
    )

    bin_count = 18
    counts, bins, patches = histogram_axis.hist(
        latent_values,
        bins=bin_count,
        range=(0.0, 1.0),
        edgecolor="white",
        linewidth=0.8,
        alpha=0.78,
    )
    if len(patches) != bin_count:
        raise RuntimeError("Число столбцов гистограммы не соответствует настройке.")

    markers = (
        (summary.minimum, "min", "#6A7885", "--"),
        (summary.median, "медиана", "#B23A48", "-"),
        (summary.mean, "среднее", "#1D6F8A", "-"),
        (summary.maximum, "max", "#6A7885", "--"),
    )
    ymax = max(float(np.max(counts)) if counts.size else 1.0, 1.0)
    label_levels = {"min": 0.94, "медиана": 0.82, "среднее": 0.70, "max": 0.94}
    for x_value, label, color, line_style in markers:
        histogram_axis.axvline(
            x_value,
            color=color,
            linestyle=line_style,
            linewidth=1.8,
            zorder=4,
        )
        horizontal = "left" if x_value < 0.82 else "right"
        offset = 0.010 if horizontal == "left" else -0.010
        histogram_axis.text(
            x_value + offset,
            ymax * label_levels[label],
            f"{label}: {x_value:.3f}",
            ha=horizontal,
            va="top",
            fontsize=9.4,
            color=color,
            fontweight="bold" if label in {"медиана", "среднее"} else "normal",
            bbox={
                "boxstyle": "round,pad=0.20",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.6,
                "alpha": 0.88,
            },
        )

    histogram_axis.set_xlim(0.0, 1.0)
    histogram_axis.set_ylim(0.0, ymax * 1.08)
    histogram_axis.set_xlabel("Значение latent_quality_component")
    histogram_axis.set_ylabel("Число сценариев")
    histogram_axis.set_title(
        "А. Частотное распределение латентной компоненты",
        fontweight="bold",
        pad=13,
    )
    histogram_axis.grid(axis="y", alpha=0.28, linewidth=0.8)
    histogram_axis.set_axisbelow(True)

    box_axis.boxplot(
        latent_values,
        vert=False,
        widths=0.52,
        patch_artist=True,
        showfliers=True,
        medianprops={"linewidth": 2.2, "color": "#B23A48"},
        whiskerprops={"linewidth": 1.3},
        capprops={"linewidth": 1.3},
        boxprops={"linewidth": 1.3, "facecolor": "#D7E7F0", "alpha": 0.85},
        flierprops={"markersize": 3.8, "alpha": 0.45},
    )
    box_axis.scatter(
        [summary.mean],
        [1.0],
        marker="D",
        s=62,
        zorder=5,
        color="#1D6F8A",
    )
    box_axis.set_xlim(0.0, 1.0)
    box_axis.set_yticks([])
    box_axis.set_xlabel("Значение latent_quality_component")
    box_axis.set_title("Б. Диапазон и квартильная структура", fontweight="bold", pad=13)
    box_axis.grid(axis="x", alpha=0.28, linewidth=0.8)
    box_axis.set_axisbelow(True)
    box_axis.text(
        0.5,
        1.18,
        f"min={summary.minimum:.3f}   Q₁={summary.first_quartile:.3f}   "
        f"Me={summary.median:.3f}   Q₃={summary.third_quartile:.3f}   "
        f"max={summary.maximum:.3f}",
        ha="center",
        va="bottom",
        transform=box_axis.transAxes,
        fontsize=9.4,
    )
    box_axis.text(
        0.98,
        0.10,
        f"◇ среднее = {summary.mean:.3f};  σ = {summary.standard_deviation:.3f}",
        ha="right",
        va="bottom",
        transform=box_axis.transAxes,
        fontsize=9.3,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "0.78",
            "linewidth": 0.6,
            "alpha": 0.90,
        },
    )

    signed = np.asarray(summary.mean_signed_terms, dtype=float)
    y_positions = np.arange(EXPECTED_TOPIC_COUNT)
    direction_axis.barh(y_positions, signed, height=0.54, alpha=0.80)
    direction_axis.axvline(0.0, color="#334A5A", linewidth=1.0)
    limit = max(0.48, float(np.max(np.abs(signed))) + 0.08)
    direction_axis.set_xlim(-limit, limit)
    direction_axis.set_yticks(
        y_positions,
        (
            "θ₀ — трудоёмкость",
            "θ₁ — операционный риск",
            "θ₂ — благоприятные условия",
        ),
    )
    direction_axis.invert_yaxis()
    direction_axis.set_xlabel("Средний направленный член  dₖ · θₖ")
    direction_axis.set_title(
        "В. Направления латентных компонентов",
        fontweight="bold",
        pad=13,
    )
    direction_axis.grid(axis="x", alpha=0.25, linewidth=0.8)
    direction_axis.set_axisbelow(True)
    for y_value, value in zip(y_positions, signed, strict=True):
        direction_axis.text(
            value + (0.012 if value >= 0.0 else -0.012),
            y_value,
            f"{value:+.3f}",
            ha="left" if value >= 0.0 else "right",
            va="center",
            fontsize=9.4,
            fontweight="bold",
        )
    figure.text(
        0.5,
        0.055,
        "Латентная компонента отражает направленную свёртку априорного профиля LDA. "
        "Она не является наблюдаемой фактической оценкой качества и не доказывает "
        "причинное влияние отдельных факторов. Тождество q_latent = θ₂ обусловлено "
        "выбранными направлениями d и нормировкой суммы компонентов θ.",
        ha="center",
        va="bottom",
        fontsize=10.0,
        wrap=True,
    )
    return figure



def generate(
    *,
    project_root: str | Path,
    input_path: str | Path | None = None,
    dpi: int = 300,
) -> FigureExportResult:
    """Сформировать рисунок 5.2 в PNG и SVG."""

    root = Path(project_root).resolve()
    source = Path(input_path) if input_path is not None else root / DEFAULT_INPUT_PATH
    profiles = load_theta_profiles(source)
    figure = build_figure(profiles)
    return export_figure(
        figure,
        project_root=root,
        relative_output_dir=OUTPUT_DIR,
        file_stem=FILE_STEM,
        dpi=dpi,
    )



def build_argument_parser() -> argparse.ArgumentParser:
    """Создать разборщик аргументов командной строки."""

    parser = argparse.ArgumentParser(
        description="Сформировать рисунок 5.2 с латентной компонентой качества."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Корень проекта manual_coding_sim.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Путь к theta_prior.csv; по умолчанию используется reports/chapter4.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Разрешение PNG; по умолчанию 300 dpi.",
    )
    return parser



def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить генерацию рисунка 5.2 из командной строки."""

    arguments = build_argument_parser().parse_args(argv)
    result = generate(
        project_root=arguments.project_root,
        input_path=arguments.input,
        dpi=arguments.dpi,
    )
    print("Рисунок 5.2 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
