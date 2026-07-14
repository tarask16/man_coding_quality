"""Генерация рисунка 6.5 с матрицей ошибок классов качества."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from manual_coding_sim.dissertation_figures.common import (
    FigureExportResult,
    configure_dissertation_style,
    export_figure,
)

OUTPUT_DIR = Path("reports/chapter6/figures")
FILE_STEM = "confusion_matrix"
DEFAULT_Q_PRED_PATH = Path("reports/chapter5/q_pred.csv")
DEFAULT_Q_FACT_PATH = Path("data/processed/quality_targets.csv")
JOIN_KEYS = ("scenario_id", "protocol_id")
CLASS_ORDER = ("low", "medium", "high")
CLASS_LABELS = {
    "low": "Низкое\n(low)",
    "medium": "Среднее\n(medium)",
    "high": "Высокое\n(high)",
}
LOW_THRESHOLD = 0.45
HIGH_THRESHOLD = 0.70


@dataclass(frozen=True, slots=True)
class QualityClassPoint:
    """Согласованные прогнозный и фактический классы одного сценария."""

    scenario_id: str
    protocol_id: str
    q_pred: float
    q_fact: float
    predicted_class: str
    actual_class: str


@dataclass(frozen=True, slots=True)
class ConfusionSummary:
    """Матрица ошибок и диагностические показатели классификации."""

    matrix: tuple[tuple[int, ...], ...]
    total: int
    correct: int
    adjacent_errors: int
    critical_low_to_high: int
    critical_high_to_low: int
    accuracy: float
    actual_support: tuple[int, ...]
    predicted_support: tuple[int, ...]

    @property
    def critical_errors(self) -> int:
        """Вернуть суммарное число критических ошибок через класс."""

        return self.critical_low_to_high + self.critical_high_to_low


def classify_quality(value: float) -> str:
    """Отнести показатель качества к low, medium или high."""

    if not math.isfinite(value):
        raise ValueError("Показатель качества должен быть конечным числом.")
    if not 0.0 <= value <= 1.0:
        raise ValueError("Показатель качества должен находиться в диапазоне [0; 1].")
    if value < LOW_THRESHOLD:
        return "low"
    if value < HIGH_THRESHOLD:
        return "medium"
    return "high"


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    """Прочитать непустой CSV-файл."""

    if not path.is_file():
        raise FileNotFoundError(f"Не найден входной файл: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Входной файл не содержит строк: {path}")
    return rows


def _validate_columns(
    rows: Sequence[Mapping[str, str]],
    required: Sequence[str],
    *,
    source_name: str,
) -> None:
    """Проверить наличие обязательных колонок."""

    missing = [column for column in required if column not in rows[0]]
    if missing:
        raise ValueError(
            f"В {source_name} отсутствуют обязательные колонки: {', '.join(missing)}"
        )


def _index_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    source_name: str,
) -> dict[tuple[str, str], Mapping[str, str]]:
    """Индексировать строки по ключам сценария и протокола."""

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


def _parse_quality(raw: str, *, column: str, key: tuple[str, str]) -> float:
    """Преобразовать значение качества и проверить диапазон."""

    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Колонка {column}, ключ {key[0]} / {key[1]}: требуется число."
        ) from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            f"Колонка {column}, ключ {key[0]} / {key[1]}: "
            "значение должно быть конечным числом в диапазоне [0; 1]."
        )
    return value


def load_quality_class_points(
    q_pred_path: Path,
    q_fact_path: Path,
) -> tuple[QualityClassPoint, ...]:
    """Загрузить согласованные классы прогнозного и фактического качества."""

    prediction_rows = _read_rows(q_pred_path)
    fact_rows = _read_rows(q_fact_path)
    _validate_columns(
        prediction_rows,
        [*JOIN_KEYS, "q_pred"],
        source_name=q_pred_path.name,
    )
    _validate_columns(
        fact_rows,
        [*JOIN_KEYS, "integral_quality"],
        source_name=q_fact_path.name,
    )
    prediction_index = _index_rows(prediction_rows, source_name=q_pred_path.name)
    fact_index = _index_rows(fact_rows, source_name=q_fact_path.name)
    if set(prediction_index) != set(fact_index):
        raise ValueError("Наборы ключей прогнозного и фактического качества не совпадают.")
    if len(prediction_index) < 3:
        raise ValueError("Для построения матрицы ошибок требуется не менее трёх сценариев.")

    points: list[QualityClassPoint] = []
    for key in sorted(prediction_index):
        q_pred = _parse_quality(
            prediction_index[key]["q_pred"],
            column="q_pred",
            key=key,
        )
        q_fact = _parse_quality(
            fact_index[key]["integral_quality"],
            column="integral_quality",
            key=key,
        )
        points.append(
            QualityClassPoint(
                scenario_id=key[0],
                protocol_id=key[1],
                q_pred=q_pred,
                q_fact=q_fact,
                predicted_class=classify_quality(q_pred),
                actual_class=classify_quality(q_fact),
            )
        )
    return tuple(points)


def calculate_confusion_summary(
    points: Sequence[QualityClassPoint],
) -> ConfusionSummary:
    """Рассчитать абсолютную матрицу ошибок и диагностические показатели."""

    if len(points) < 3:
        raise ValueError("Для матрицы ошибок требуется не менее трёх сценариев.")
    class_index = {name: index for index, name in enumerate(CLASS_ORDER)}
    matrix = np.zeros((3, 3), dtype=int)
    for point in points:
        if point.actual_class not in class_index or point.predicted_class not in class_index:
            raise ValueError("Обнаружен неизвестный класс качества.")
        matrix[class_index[point.actual_class], class_index[point.predicted_class]] += 1

    correct = int(np.trace(matrix))
    critical_low_to_high = int(matrix[0, 2])
    critical_high_to_low = int(matrix[2, 0])
    adjacent_errors = int(matrix[0, 1] + matrix[1, 0] + matrix[1, 2] + matrix[2, 1])
    total = int(matrix.sum())
    return ConfusionSummary(
        matrix=tuple(tuple(int(value) for value in row) for row in matrix),
        total=total,
        correct=correct,
        adjacent_errors=adjacent_errors,
        critical_low_to_high=critical_low_to_high,
        critical_high_to_low=critical_high_to_low,
        accuracy=correct / total,
        actual_support=tuple(int(value) for value in matrix.sum(axis=1)),
        predicted_support=tuple(int(value) for value in matrix.sum(axis=0)),
    )


def _cell_text_color(value: float, maximum: float) -> str:
    """Выбрать контрастный цвет текста для ячейки тепловой карты."""

    return "white" if maximum > 0 and value / maximum >= 0.48 else "#17202a"


def build_figure(summary: ConfusionSummary) -> plt.Figure:
    """Построить тепловую карту и диагностическую панель."""

    configure_dissertation_style()
    matrix = np.asarray(summary.matrix, dtype=int)
    row_shares = np.divide(
        matrix,
        np.asarray(summary.actual_support, dtype=float)[:, None],
        out=np.zeros_like(matrix, dtype=float),
        where=np.asarray(summary.actual_support, dtype=float)[:, None] != 0,
    )

    figure = plt.figure(figsize=(18.8, 8.9))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.62, 0.88),
        left=0.075,
        right=0.965,
        top=0.82,
        bottom=0.17,
        wspace=0.20,
    )
    heat_axis = figure.add_subplot(grid[0, 0])
    info_axis = figure.add_subplot(grid[0, 1])

    colormap = LinearSegmentedColormap.from_list(
        "dissertation_confusion",
        ["#f8fafc", "#bdd7e7", "#5b9bd5", "#244a73"],
    )
    image = heat_axis.imshow(matrix, cmap=colormap, vmin=0, vmax=max(1, int(matrix.max())))
    heat_axis.set_xticks(np.arange(3), [CLASS_LABELS[name] for name in CLASS_ORDER])
    heat_axis.set_yticks(np.arange(3), [CLASS_LABELS[name] for name in CLASS_ORDER])
    heat_axis.set_xlabel("Прогнозный класс качества", labelpad=12, fontweight="bold")
    heat_axis.set_ylabel("Фактический класс качества", labelpad=12, fontweight="bold")
    heat_axis.set_title(
        "Матрица ошибок в абсолютных значениях",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    heat_axis.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    heat_axis.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    heat_axis.grid(which="minor", color="white", linewidth=3.0)
    heat_axis.tick_params(which="minor", bottom=False, left=False)

    maximum = float(matrix.max())
    for row in range(3):
        for column in range(3):
            value = int(matrix[row, column])
            color = _cell_text_color(value, maximum)
            heat_axis.text(
                column,
                row - 0.08,
                str(value),
                ha="center",
                va="center",
                fontsize=25,
                fontweight="bold",
                color=color,
            )
            heat_axis.text(
                column,
                row + 0.23,
                f"{row_shares[row, column] * 100:.1f}% строки",
                ha="center",
                va="center",
                fontsize=10.3,
                color=color,
            )

    # Критические угловые ошибки выделяются независимо от их значения.
    for row, column in ((0, 2), (2, 0)):
        heat_axis.add_patch(
            plt.Rectangle(
                (column - 0.48, row - 0.48),
                0.96,
                0.96,
                fill=False,
                edgecolor="#c44e52",
                linewidth=3.0,
                linestyle="--",
            )
        )
    colorbar = figure.colorbar(image, ax=heat_axis, fraction=0.042, pad=0.035)
    colorbar.set_label("Число сценариев", rotation=90, labelpad=12)

    info_axis.axis("off")
    info_axis.set_title(
        "Диагностика классификации",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    info_axis.text(
        0.04,
        0.92,
        (
            f"N = {summary.total}\n"
            f"Точное совпадение: {summary.correct} ({summary.accuracy * 100:.1f}%)\n"
            f"Соседние ошибки: {summary.adjacent_errors}\n"
            f"Критические ошибки: {summary.critical_errors}"
        ),
        transform=info_axis.transAxes,
        ha="left",
        va="top",
        fontsize=12.5,
        linespacing=1.55,
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "#f3f6f8",
            "edgecolor": "#8796a5",
        },
    )

    info_axis.text(
        0.04,
        0.61,
        "Критические переходы",
        transform=info_axis.transAxes,
        ha="left",
        va="center",
        fontsize=12.3,
        fontweight="bold",
    )
    info_axis.text(
        0.04,
        0.53,
        (
            f"low → high: {summary.critical_low_to_high}\n"
            f"high → low: {summary.critical_high_to_low}\n"
            "Критические ошибки отсутствуют"
            if summary.critical_errors == 0
            else (
                f"low → high: {summary.critical_low_to_high}\n"
                f"high → low: {summary.critical_high_to_low}\n"
                "Критические ошибки обнаружены"
            )
        ),
        transform=info_axis.transAxes,
        ha="left",
        va="top",
        fontsize=12.2,
        linespacing=1.6,
        color="#236b3a" if summary.critical_errors == 0 else "#9e2f2f",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#edf7ef" if summary.critical_errors == 0 else "#fbecec",
            "edgecolor": "#55a868" if summary.critical_errors == 0 else "#c44e52",
        },
    )

    support_y = 0.18
    info_axis.text(
        0.04,
        support_y + 0.08,
        "Поддержка классов: факт / прогноз",
        transform=info_axis.transAxes,
        fontsize=11.8,
        fontweight="bold",
        ha="left",
    )
    for index, class_name in enumerate(CLASS_ORDER):
        y = support_y - index * 0.075
        info_axis.text(
            0.04,
            y,
            f"{class_name}: {summary.actual_support[index]} / {summary.predicted_support[index]}",
            transform=info_axis.transAxes,
            fontsize=11.4,
            ha="left",
        )

    figure.suptitle(
        "Матрица ошибок классов интегрального качества",
        fontsize=18,
        fontweight="bold",
        y=0.965,
    )
    figure.text(
        0.5,
        0.905,
        (
            "Пороги классов: low < 0,45; medium < 0,70; high ≥ 0,70. "
            "Строки — фактический класс, столбцы — прогнозный класс."
        ),
        ha="center",
        va="center",
        fontsize=11.5,
    )
    figure.text(
        0.5,
        0.065,
        (
            "Матрица показывает абсолютные переходы между классами. Отсутствие ошибок "
            "low → high и high → low исключает наиболее критические межклассовые "
            "подмены, но точность классификации не заменяет анализ непрерывных ошибок "
            "и абсолютной калибровки Q_pred."
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
    q_pred_path: Path | None = None,
    q_fact_path: Path | None = None,
    dpi: int = 300,
) -> FigureExportResult:
    """Сформировать рисунок 6.5 в PNG и SVG."""

    root = project_root.resolve()
    predictions = q_pred_path.resolve() if q_pred_path else root / DEFAULT_Q_PRED_PATH
    facts = q_fact_path.resolve() if q_fact_path else root / DEFAULT_Q_FACT_PATH
    points = load_quality_class_points(predictions, facts)
    summary = calculate_confusion_summary(points)
    figure = build_figure(summary)
    return export_figure(
        figure,
        project_root=root,
        relative_output_dir=OUTPUT_DIR,
        file_stem=FILE_STEM,
        dpi=dpi,
    )


def build_parser() -> argparse.ArgumentParser:
    """Создать CLI-парсер генератора рисунка 6.5."""

    parser = argparse.ArgumentParser(
        description="Сформировать рисунок 6.5 с матрицей ошибок классов качества."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--q-pred", type=Path, default=None)
    parser.add_argument("--q-fact", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Запустить генерацию рисунка 6.5 из командной строки."""

    arguments = build_parser().parse_args(argv)
    result = generate(
        project_root=arguments.project_root,
        q_pred_path=arguments.q_pred,
        q_fact_path=arguments.q_fact,
        dpi=arguments.dpi,
    )
    print("Рисунок 6.5 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
