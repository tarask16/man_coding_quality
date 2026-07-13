"""Генерация рисунка 4.6 с распределением латентных профилей theta_prior."""

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

OUTPUT_DIR = Path("reports/chapter4/figures")
FILE_STEM = "figure_4_6_theta_profile_distribution"
DEFAULT_INPUT_PATH = Path("reports/chapter4/theta_prior.csv")
THETA_COLUMNS = ("theta_0", "theta_1", "theta_2")
EXPECTED_TOPIC_COUNT = 3

TOPIC_NAMES = {
    0: "Процедурная трудоёмкость\nи ресурсная нагрузка",
    1: "Операционный риск\nи дефицит внимания",
    2: "Благоприятные условия\nи низкий риск",
}


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
        """Вернуть компоненты профиля в порядке theta_0, theta_1, theta_2."""

        return (self.theta_0, self.theta_1, self.theta_2)

    @property
    def dominant_topic(self) -> int:
        """Вернуть индекс компоненты с максимальным весом."""

        return int(np.argmax(np.asarray(self.values, dtype=float)))


@dataclass(frozen=True, slots=True)
class TopicProfileSummary:
    """Описательная статистика одной компоненты латентного профиля."""

    topic_id: int
    topic_name: str
    minimum: float
    first_quartile: float
    median: float
    mean: float
    third_quartile: float
    maximum: float
    dominant_count: int
    dominant_share: float


def load_theta_profiles(path: str | Path) -> tuple[ThetaProfile, ...]:
    """Загрузить латентные профили из ``theta_prior.csv``."""

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
    """Проверить полноту, нормировку и идентификаторы латентных профилей."""

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
            raise ValueError("Для рисунка 4.6 ожидается selected_k = 3.")
        for value in profile.values:
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("Компоненты theta должны быть конечными и лежать в [0; 1].")
        if not math.isclose(sum(profile.values), 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("Сумма компонентов theta каждого сценария должна быть равна единице.")


def calculate_topic_summaries(
    profiles: Sequence[ThetaProfile],
) -> tuple[TopicProfileSummary, ...]:
    """Рассчитать распределения компонент и доли доминирующих факторов."""

    validate_theta_profiles(profiles)
    values = np.asarray([profile.values for profile in profiles], dtype=float)
    dominant = np.asarray([profile.dominant_topic for profile in profiles], dtype=int)
    sample_size = len(profiles)

    summaries: list[TopicProfileSummary] = []
    for topic_id in range(EXPECTED_TOPIC_COUNT):
        component = values[:, topic_id]
        summaries.append(
            TopicProfileSummary(
                topic_id=topic_id,
                topic_name=TOPIC_NAMES[topic_id],
                minimum=float(np.min(component)),
                first_quartile=float(np.quantile(component, 0.25)),
                median=float(median(component.tolist())),
                mean=float(mean(component.tolist())),
                third_quartile=float(np.quantile(component, 0.75)),
                maximum=float(np.max(component)),
                dominant_count=int(np.sum(dominant == topic_id)),
                dominant_share=float(np.mean(dominant == topic_id)),
            )
        )

    if sum(item.dominant_count for item in summaries) != sample_size:
        raise RuntimeError("Число доминирующих факторов не совпадает с числом сценариев.")
    return tuple(summaries)


def build_figure(profiles: Sequence[ThetaProfile]) -> plt.Figure:
    """Построить boxplot компонент theta и доли доминирующих факторов."""

    validate_theta_profiles(profiles)
    summaries = calculate_topic_summaries(profiles)
    values = np.asarray([profile.values for profile in profiles], dtype=float)
    sample_size = len(profiles)

    configure_dissertation_style()
    figure = plt.figure(figsize=(17.4, 9.4))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.65, 1.0),
        left=0.065,
        right=0.975,
        top=0.81,
        bottom=0.20,
        wspace=0.22,
    )
    box_axis = figure.add_subplot(grid[0, 0])
    share_axis = figure.add_subplot(grid[0, 1])

    figure.suptitle(
        "Рисунок 4.6 — Распределение априорных латентных профилей θ_prior",
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )
    figure.text(
        0.5,
        0.922,
        f"Компоненты θ₀–θ₂ и доминирующий фактор для {sample_size} сценариев; "
        "в каждой строке сумма компонентов θ равна 1.",
        ha="center",
        fontsize=11.5,
    )
    figure.text(
        0.5,
        0.892,
        "Ромб — среднее значение; горизонтальная линия внутри boxplot — медиана.",
        ha="center",
        fontsize=10.2,
    )

    boxplot = box_axis.boxplot(
        [values[:, index] for index in range(EXPECTED_TOPIC_COUNT)],
        positions=np.arange(EXPECTED_TOPIC_COUNT),
        widths=0.58,
        patch_artist=True,
        showfliers=True,
        medianprops={"linewidth": 2.0},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        boxprops={"linewidth": 1.2},
        flierprops={"markersize": 3.8, "alpha": 0.45},
    )
    for patch in boxplot["boxes"]:
        patch.set_alpha(0.62)

    x_positions = np.arange(EXPECTED_TOPIC_COUNT)
    means = np.asarray([item.mean for item in summaries], dtype=float)
    box_axis.scatter(
        x_positions,
        means,
        marker="D",
        s=54,
        zorder=4,
        label="Среднее значение",
    )
    box_axis.set_xticks(
        x_positions,
        [
            "θ₀\nПроцедурная\nтрудоёмкость",
            "θ₁\nОперационный\nриск",
            "θ₂\nБлагоприятные\nусловия",
        ],
    )
    box_axis.set_ylim(0.0, 1.05)
    box_axis.set_ylabel("Вес компоненты латентного профиля")
    box_axis.set_title("А. Распределение компонент θ₀–θ₂", fontweight="bold", pad=13)
    box_axis.grid(axis="y", alpha=0.28, linewidth=0.8)
    box_axis.set_axisbelow(True)
    for x_value, item in zip(x_positions, summaries, strict=True):
        box_axis.text(
            x_value,
            1.015,
            f"μ={item.mean:.3f}\nMe={item.median:.3f}",
            ha="center",
            va="top",
            fontsize=9.2,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "0.78",
                "linewidth": 0.7,
            },
        )

    counts = np.asarray([item.dominant_count for item in summaries], dtype=int)
    shares = np.asarray([item.dominant_share for item in summaries], dtype=float)
    bars = share_axis.bar(x_positions, shares, width=0.62)
    share_axis.set_ylim(0.0, max(0.55, float(np.max(shares) + 0.1)))
    share_axis.set_ylabel("Доля сценариев")
    share_axis.set_title("Б. Доминирующий фактор", fontweight="bold", pad=13)
    share_axis.set_xticks(
        x_positions,
        [
            "θ₀\nПроцедурная\nтрудоёмкость",
            "θ₁\nОперационный\nриск",
            "θ₂\nБлагоприятные\nусловия",
        ],
    )
    share_axis.grid(axis="y", alpha=0.28, linewidth=0.8)
    share_axis.set_axisbelow(True)

    for bar, count, share in zip(bars, counts, shares, strict=True):
        share_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.018,
            f"{count} сценариев\n{share:.1%}",
            ha="center",
            va="bottom",
            fontsize=10.2,
            fontweight="bold",
        )


    figure.text(
        0.5,
        0.055,
        "Доминирующий фактор определяется как argmax(θ₀, θ₁, θ₂). "
        "Профиль характеризует относительную выраженность априорных латентных "
        "состояний сценария и не является вероятностью фактического успеха или "
        "коэффициентом причинного влияния на качество.",
        ha="center",
        va="bottom",
        fontsize=10.2,
        wrap=True,
    )
    return figure


def generate(
    *,
    project_root: str | Path,
    input_path: str | Path | None = None,
    dpi: int = 300,
) -> FigureExportResult:
    """Сформировать рисунок 4.6 в PNG и SVG."""

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
        description="Сформировать рисунок 4.6 с распределением профилей theta_prior."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Корень проекта manual_coding_sim.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Путь к theta_prior.csv; по умолчанию reports/chapter4/theta_prior.csv.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Разрешение PNG; по умолчанию 300 dpi.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить CLI-генерацию рисунка 4.6."""

    arguments = build_argument_parser().parse_args(argv)
    result = generate(
        project_root=arguments.project_root,
        input_path=arguments.input,
        dpi=arguments.dpi,
    )
    print("Рисунок 4.6 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
