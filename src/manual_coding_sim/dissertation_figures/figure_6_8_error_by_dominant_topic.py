"""Генерация рисунка 6.8 с абсолютной ошибкой по доминирующему фактору."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from manual_coding_sim.dissertation_figures.common import (
    FigureExportResult,
    configure_dissertation_style,
    export_figure,
)

OUTPUT_DIR = Path("reports/chapter6/figures")
FILE_STEM = "error_by_dominant_topic"
DEFAULT_Q_PRED_PATH = Path("reports/chapter5/q_pred.csv")
DEFAULT_Q_FACT_PATH = Path("data/processed/quality_targets.csv")
DEFAULT_THETA_PATH = Path("reports/chapter4/theta_prior.csv")

KEY_COLUMNS = ("scenario_id", "protocol_id")
THETA_COLUMNS = ("theta_0", "theta_1", "theta_2")
TOPIC_ORDER = (0, 1, 2)
TOPIC_LABELS = {
    0: "Процедурная\nтрудоёмкость",
    1: "Операционный\nриск",
    2: "Благоприятные\nусловия",
}
TOPIC_LONG_LABELS = {
    0: "Процедурная трудоёмкость",
    1: "Операционный риск",
    2: "Благоприятные условия",
}


@dataclass(frozen=True, slots=True)
class TopicErrorPoint:
    """Одна согласованная запись прогноза, факта и латентного профиля."""

    scenario_id: str
    protocol_id: str
    q_pred: float
    q_fact: float
    theta_0: float
    theta_1: float
    theta_2: float
    dominant_topic: int

    @property
    def absolute_error(self) -> float:
        """Вернуть абсолютную ошибку интегрального прогноза."""

        return abs(self.q_pred - self.q_fact)


@dataclass(frozen=True, slots=True)
class TopicErrorSummary:
    """Описательная статистика абсолютной ошибки внутри одной темы."""

    topic_id: int
    count: int
    mean: float
    median: float
    q1: float
    q3: float
    minimum: float
    maximum: float
    mean_q_pred: float
    mean_q_fact: float


@dataclass(frozen=True, slots=True)
class ErrorByTopicSummary:
    """Полная сводка группового анализа абсолютной ошибки."""

    topics: tuple[TopicErrorSummary, ...]
    overall_mae: float
    best_observed_topic: int
    worst_observed_topic: int


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    """Прочитать непустой CSV-файл."""

    if not path.is_file():
        raise FileNotFoundError(f"Не найден входной файл: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Входной файл не содержит строк: {path}")
    return rows


def _parse_probability(raw: str, *, column: str, row_number: int) -> float:
    """Преобразовать числовое значение и проверить диапазон [0; 1]."""

    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Строка {row_number}, колонка {column}: требуется числовое значение."
        ) from error
    if not math.isfinite(value):
        raise ValueError(
            f"Строка {row_number}, колонка {column}: NaN и бесконечность недопустимы."
        )
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"Строка {row_number}, колонка {column}: значение должно быть в диапазоне [0; 1]."
        )
    return value


def _build_key(row: dict[str, str], *, row_number: int, source: str) -> tuple[str, str]:
    """Сформировать составной ключ и проверить его непустоту."""

    try:
        scenario_id = str(row["scenario_id"]).strip()
        protocol_id = str(row["protocol_id"]).strip()
    except KeyError as error:
        raise ValueError(
            f"В источнике {source} отсутствует обязательная колонка {error.args[0]!r}."
        ) from error
    if not scenario_id or not protocol_id:
        raise ValueError(
            f"Строка {row_number} источника {source}: ключевые идентификаторы не должны быть пустыми."
        )
    return scenario_id, protocol_id


def _index_rows(
    rows: Sequence[dict[str, str]],
    *,
    source: str,
) -> dict[tuple[str, str], tuple[int, dict[str, str]]]:
    """Проиндексировать строки по составному ключу с контролем дубликатов."""

    indexed: dict[tuple[str, str], tuple[int, dict[str, str]]] = {}
    for row_number, row in enumerate(rows, start=2):
        key = _build_key(row, row_number=row_number, source=source)
        if key in indexed:
            raise ValueError(
                f"Источник {source} содержит повтор ключа {key[0]!r}, {key[1]!r}."
            )
        indexed[key] = (row_number, row)
    return indexed


def load_topic_error_points(
    q_pred_path: Path,
    q_fact_path: Path,
    theta_path: Path,
) -> tuple[TopicErrorPoint, ...]:
    """Загрузить и согласовать прогноз, факт и априорный латентный профиль."""

    prediction_rows = _read_rows(q_pred_path)
    fact_rows = _read_rows(q_fact_path)
    theta_rows = _read_rows(theta_path)

    predictions = _index_rows(prediction_rows, source="q_pred")
    facts = _index_rows(fact_rows, source="quality_targets")
    theta = _index_rows(theta_rows, source="theta_prior")

    key_sets = (set(predictions), set(facts), set(theta))
    if not (key_sets[0] == key_sets[1] == key_sets[2]):
        only_prediction = sorted(key_sets[0] - key_sets[1] - key_sets[2])
        only_fact = sorted(key_sets[1] - key_sets[0])
        only_theta = sorted(key_sets[2] - key_sets[0])
        raise ValueError(
            "Ключи q_pred, quality_targets и theta_prior не совпадают. "
            f"Только q_pred: {only_prediction[:3]}; "
            f"только quality_targets: {only_fact[:3]}; "
            f"только theta_prior: {only_theta[:3]}."
        )

    points: list[TopicErrorPoint] = []
    for key in sorted(predictions):
        prediction_row_number, prediction_row = predictions[key]
        fact_row_number, fact_row = facts[key]
        theta_row_number, theta_row = theta[key]

        if "q_pred" not in prediction_row:
            raise ValueError("В q_pred отсутствует обязательная колонка 'q_pred'.")
        if "integral_quality" not in fact_row:
            raise ValueError(
                "В quality_targets отсутствует обязательная колонка 'integral_quality'."
            )
        missing_theta = [column for column in THETA_COLUMNS if column not in theta_row]
        if missing_theta:
            raise ValueError(
                "В theta_prior отсутствуют обязательные колонки: " + ", ".join(missing_theta)
            )

        q_pred = _parse_probability(
            prediction_row["q_pred"],
            column="q_pred",
            row_number=prediction_row_number,
        )
        q_fact = _parse_probability(
            fact_row["integral_quality"],
            column="integral_quality",
            row_number=fact_row_number,
        )
        theta_values = tuple(
            _parse_probability(
                theta_row[column],
                column=column,
                row_number=theta_row_number,
            )
            for column in THETA_COLUMNS
        )
        if not math.isclose(sum(theta_values), 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError(
                f"Строка {theta_row_number} theta_prior: сумма theta_0, theta_1 и theta_2 "
                "должна быть равна 1."
            )

        dominant_topic = int(np.argmax(np.asarray(theta_values, dtype=float)))
        points.append(
            TopicErrorPoint(
                scenario_id=key[0],
                protocol_id=key[1],
                q_pred=q_pred,
                q_fact=q_fact,
                theta_0=theta_values[0],
                theta_1=theta_values[1],
                theta_2=theta_values[2],
                dominant_topic=dominant_topic,
            )
        )

    if len(points) < 3:
        raise ValueError("Для группового анализа требуется не менее трёх сценариев.")
    present_topics = {point.dominant_topic for point in points}
    missing_topics = [topic for topic in TOPIC_ORDER if topic not in present_topics]
    if missing_topics:
        raise ValueError(
            "В данных отсутствуют сценарии с доминирующими темами: "
            + ", ".join(str(topic) for topic in missing_topics)
        )
    return tuple(points)


def summarise_errors_by_topic(points: Sequence[TopicErrorPoint]) -> ErrorByTopicSummary:
    """Рассчитать описательные показатели абсолютной ошибки по трём темам."""

    if len(points) < 3:
        raise ValueError("Для сводки требуется не менее трёх сценариев.")

    summaries: list[TopicErrorSummary] = []
    for topic_id in TOPIC_ORDER:
        group = [point for point in points if point.dominant_topic == topic_id]
        if not group:
            raise ValueError(f"Отсутствует группа dominant_topic={topic_id}.")
        errors = np.asarray([point.absolute_error for point in group], dtype=float)
        q_pred = np.asarray([point.q_pred for point in group], dtype=float)
        q_fact = np.asarray([point.q_fact for point in group], dtype=float)
        summaries.append(
            TopicErrorSummary(
                topic_id=topic_id,
                count=len(group),
                mean=float(np.mean(errors)),
                median=float(np.median(errors)),
                q1=float(np.quantile(errors, 0.25)),
                q3=float(np.quantile(errors, 0.75)),
                minimum=float(np.min(errors)),
                maximum=float(np.max(errors)),
                mean_q_pred=float(np.mean(q_pred)),
                mean_q_fact=float(np.mean(q_fact)),
            )
        )

    all_errors = np.asarray([point.absolute_error for point in points], dtype=float)
    best = min(summaries, key=lambda item: item.mean).topic_id
    worst = max(summaries, key=lambda item: item.mean).topic_id
    return ErrorByTopicSummary(
        topics=tuple(summaries),
        overall_mae=float(np.mean(all_errors)),
        best_observed_topic=best,
        worst_observed_topic=worst,
    )


def _add_summary_card(
    axis: plt.Axes,
    *,
    y: float,
    summary: TopicErrorSummary,
    edge_color: str,
    face_color: str,
) -> None:
    """Добавить карточку описательной статистики одной группы."""

    card = FancyBboxPatch(
        (0.035, y),
        0.93,
        0.235,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.35,
        edgecolor=edge_color,
        facecolor=face_color,
        transform=axis.transAxes,
        clip_on=False,
    )
    axis.add_patch(card)
    axis.text(
        0.075,
        y + 0.185,
        f"Тема {summary.topic_id}: {TOPIC_LONG_LABELS[summary.topic_id]}",
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=11.6,
        fontweight="bold",
    )
    axis.text(
        0.075,
        y + 0.105,
        (
            f"n = {summary.count};  mean |e| = {summary.mean:.4f};  "
            f"median = {summary.median:.4f}\n"
            f"IQR = [{summary.q1:.4f}; {summary.q3:.4f}];  max = {summary.maximum:.4f}\n"
            f"mean Q_pred = {summary.mean_q_pred:.4f};  mean Q_fact = {summary.mean_q_fact:.4f}"
        ),
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=9.7,
        linespacing=1.32,
    )


def build_figure(points: Sequence[TopicErrorPoint]) -> plt.Figure:
    """Построить boxplot абсолютной ошибки и описательную сводку групп."""

    configure_dissertation_style()
    summary = summarise_errors_by_topic(points)

    figure = plt.figure(figsize=(17.5, 8.5))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(3.35, 1.65),
        left=0.065,
        right=0.975,
        top=0.79,
        bottom=0.20,
        wspace=0.13,
    )
    chart_axis = figure.add_subplot(grid[0, 0])
    summary_axis = figure.add_subplot(grid[0, 1])

    grouped_errors = [
        np.asarray(
            [point.absolute_error for point in points if point.dominant_topic == topic],
            dtype=float,
        )
        for topic in TOPIC_ORDER
    ]
    topic_summaries = {item.topic_id: item for item in summary.topics}
    positions = np.arange(1, 4, dtype=float)
    colors = ("#D9A66B", "#CF6D6D", "#6AAE8B")
    edge_colors = ("#8C5A25", "#8C3232", "#2E6B4E")

    boxplot = chart_axis.boxplot(
        grouped_errors,
        positions=positions,
        widths=0.52,
        orientation="vertical",
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#8B1E1E", "linewidth": 2.0},
        whiskerprops={"color": "#555555", "linewidth": 1.25},
        capprops={"color": "#555555", "linewidth": 1.25},
        boxprops={"linewidth": 1.3},
    )
    for patch, face_color, edge_color in zip(
        boxplot["boxes"], colors, edge_colors, strict=True
    ):
        patch.set_facecolor(face_color)
        patch.set_edgecolor(edge_color)
        patch.set_alpha(0.82)

    for index, (topic_id, values) in enumerate(zip(TOPIC_ORDER, grouped_errors, strict=True)):
        random = np.random.default_rng(20260714 + topic_id)
        jitter = random.uniform(-0.17, 0.17, size=len(values))
        chart_axis.scatter(
            np.full(len(values), positions[index]) + jitter,
            values,
            s=23,
            alpha=0.50,
            color=edge_colors[index],
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
        chart_axis.scatter(
            positions[index],
            topic_summaries[topic_id].mean,
            marker="D",
            s=78,
            color="#F4C542",
            edgecolor="#6F5500",
            linewidth=1.0,
            zorder=5,
        )
        chart_axis.text(
            positions[index],
            topic_summaries[topic_id].maximum + 0.014,
            f"mean={topic_summaries[topic_id].mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=10.2,
            fontweight="semibold",
        )

    chart_axis.axhline(
        summary.overall_mae,
        color="#4C78A8",
        linewidth=1.8,
        linestyle="--",
        label=f"Общая MAE = {summary.overall_mae:.4f}",
        zorder=2,
    )
    chart_axis.set_xlim(0.45, 3.55)
    chart_axis.set_ylim(0.0, max(max(values) for values in grouped_errors) + 0.065)
    chart_axis.set_xticks(positions)
    chart_axis.set_xticklabels(
        [
            f"Тема {topic}\n{TOPIC_LABELS[topic]}\n(n={topic_summaries[topic].count})"
            for topic in TOPIC_ORDER
        ],
        fontsize=10.6,
    )
    chart_axis.set_ylabel(r"Абсолютная ошибка $|Q_{pred}-Q_{fact}|$")
    chart_axis.set_title(
        "Распределение |Q_pred − Q_fact| внутри групп dominant_topic",
        pad=14,
        fontweight="bold",
    )
    chart_axis.grid(axis="y", alpha=0.24, linewidth=0.8, zorder=0)
    chart_axis.legend(loc="upper right", frameon=True, fontsize=10.2)

    chart_axis.text(
        0.02,
        0.96,
        "ромб — среднее; красная линия — медиана; точки — отдельные сценарии",
        transform=chart_axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.8,
        color="#444444",
    )

    summary_axis.axis("off")
    _add_summary_card(
        summary_axis,
        y=0.705,
        summary=topic_summaries[0],
        edge_color=edge_colors[0],
        face_color="#FFF7ED",
    )
    _add_summary_card(
        summary_axis,
        y=0.405,
        summary=topic_summaries[1],
        edge_color=edge_colors[1],
        face_color="#FFF0F0",
    )
    _add_summary_card(
        summary_axis,
        y=0.105,
        summary=topic_summaries[2],
        edge_color=edge_colors[2],
        face_color="#EEF8F2",
    )
    summary_axis.set_title(
        "Описательная сводка групп",
        pad=14,
        fontweight="bold",
    )

    best_label = TOPIC_LONG_LABELS[summary.best_observed_topic]
    worst_label = TOPIC_LONG_LABELS[summary.worst_observed_topic]
    figure.suptitle(
        "Абсолютная ошибка по доминирующему латентному фактору",
        fontsize=18,
        fontweight="bold",
        y=0.955,
    )
    figure.text(
        0.5,
        0.895,
        (
            f"Наблюдаемая средняя ошибка минимальна в группе «{best_label}» и максимальна "
            f"в группе «{worst_label}»"
        ),
        ha="center",
        va="center",
        fontsize=11.5,
    )
    figure.text(
        0.5,
        0.085,
        (
            "Методическое ограничение: различия между группами dominant_topic имеют только "
            "ассоциативную интерпретацию. Доминирующий фактор не рассматривается как причина "
            "ошибки, а результат не подтверждает переносимость или абсолютную калибровку модели."
        ),
        ha="center",
        va="center",
        fontsize=10.5,
        wrap=True,
    )
    figure.text(
        0.5,
        0.035,
        "Источник группировки — априорный профиль θ_prior; фактическое качество используется только для внешней проверки.",
        ha="center",
        va="center",
        fontsize=9.8,
        color="#444444",
    )
    return figure


def generate(
    *,
    project_root: Path,
    q_pred_path: Path | None = None,
    q_fact_path: Path | None = None,
    theta_path: Path | None = None,
    dpi: int = 300,
) -> FigureExportResult:
    """Сформировать рисунок 6.8 в PNG и SVG."""

    root = project_root.resolve()
    resolved_q_pred = q_pred_path or root / DEFAULT_Q_PRED_PATH
    resolved_q_fact = q_fact_path or root / DEFAULT_Q_FACT_PATH
    resolved_theta = theta_path or root / DEFAULT_THETA_PATH
    points = load_topic_error_points(
        resolved_q_pred,
        resolved_q_fact,
        resolved_theta,
    )
    figure = build_figure(points)
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
        description="Сформировать рисунок 6.8 с ошибкой по dominant_topic."
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
        help="Путь к q_pred.csv. По умолчанию используется reports/chapter5/q_pred.csv.",
    )
    parser.add_argument(
        "--q-fact",
        type=Path,
        default=None,
        help="Путь к quality_targets.csv.",
    )
    parser.add_argument(
        "--theta",
        type=Path,
        default=None,
        help="Путь к theta_prior.csv.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Разрешение PNG в dpi.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить CLI-генерацию рисунка 6.8."""

    args = build_argument_parser().parse_args(argv)
    result = generate(
        project_root=args.project_root,
        q_pred_path=args.q_pred,
        q_fact_path=args.q_fact,
        theta_path=args.theta,
        dpi=args.dpi,
    )
    print("Рисунок 6.8 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
