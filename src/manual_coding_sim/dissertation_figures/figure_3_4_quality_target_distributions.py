"""Генерация рисунка 3.4 с распределениями фактических показателей качества."""

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
FILE_STEM = "figure_3_4_quality_target_distributions"
DEFAULT_INPUT_PATH = Path("data/processed/quality_targets.csv")
EXPECTED_SCENARIO_COUNT = 150


@dataclass(frozen=True, slots=True)
class QualityTargetSpec:
    """Описание фактического показателя качества для общего графика."""

    column: str
    short_label: str
    title: str


TARGET_SPECS: tuple[QualityTargetSpec, ...] = (
    QualityTargetSpec("q_acc", r"$q_{acc}$", "Точность восстановления"),
    QualityTargetSpec("q_time", r"$q_{time}$", "Временная эффективность"),
    QualityTargetSpec("q_effort", r"$q_{effort}$", "Трудоёмкость"),
    QualityTargetSpec("q_res", r"$q_{res}$", "Результативность контроля"),
    QualityTargetSpec("q_rep", r"$q_{rep}$", "Повторяемость результата"),
    QualityTargetSpec("q_fit", r"$q_{fit}$", "Соответствие условиям"),
    QualityTargetSpec("integral_quality", r"$Q_{fact}$", "Интегральное качество"),
)

REQUIRED_COLUMNS: tuple[str, ...] = tuple(spec.column for spec in TARGET_SPECS)


def load_quality_targets(path: Path) -> pd.DataFrame:
    """Загрузить и проверить фактические показатели качества."""

    if not path.is_file():
        raise FileNotFoundError(f"Файл фактических показателей не найден: {path}")

    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "В таблице отсутствуют обязательные колонки: " + ", ".join(missing)
        )
    if frame.empty:
        raise ValueError("Таблица фактических показателей не должна быть пустой.")

    selected = frame.loc[:, REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        series = pd.to_numeric(selected[column], errors="coerce")
        if series.isna().any():
            raise ValueError(
                f"Показатель {column} содержит пропуски или нечисловые значения."
            )
        values = series.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Показатель {column} содержит inf или -inf.")
        if float(values.min()) < 0.0 or float(values.max()) > 1.0:
            raise ValueError(
                f"Показатель {column} должен находиться в диапазоне [0; 1]."
            )
        selected[column] = series

    return selected


def summarize_quality_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Рассчитать описательные характеристики показателей качества."""

    rows: list[dict[str, float | str]] = []
    for spec in TARGET_SPECS:
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
                "q1": float(series.quantile(0.25)),
                "q3": float(series.quantile(0.75)),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def _style_violin_parts(parts: dict[str, object]) -> None:
    """Применить единое оформление к элементам violin-графика."""

    bodies = parts["bodies"]
    for body in bodies:  # type: ignore[union-attr]
        body.set_facecolor("#789AB7")
        body.set_edgecolor("#314452")
        body.set_linewidth(0.9)
        body.set_alpha(0.72)

    for key in ("cmins", "cmaxes", "cbars"):
        element = parts.get(key)
        if element is not None:
            element.set_color("#526C7E")  # type: ignore[union-attr]
            element.set_linewidth(0.9)  # type: ignore[union-attr]


def build_figure(frame: pd.DataFrame) -> plt.Figure:
    """Построить violin- и boxplot-графики в единой шкале [0; 1]."""

    configure_dissertation_style()
    summary = summarize_quality_targets(frame)
    data = [frame[spec.column].to_numpy(dtype=float) for spec in TARGET_SPECS]
    positions = np.arange(1, len(TARGET_SPECS) + 1)

    figure, axis = plt.subplots(figsize=(13.5, 7.8), constrained_layout=False)
    figure.subplots_adjust(left=0.19, right=0.965, top=0.86, bottom=0.20)
    figure.suptitle(
        "Распределения фактических частных и интегрального показателей качества",
        fontsize=15,
        fontweight="bold",
        y=0.955,
        color="#17212B",
    )
    figure.text(
        0.5,
        0.905,
        (
            f"Расширенный вычислительный корпус: N = {len(frame)} сценариев; "
            "все показатели приведены к общей шкале [0; 1]"
        ),
        ha="center",
        va="center",
        fontsize=9.4,
        color="#405464",
    )

    violin = axis.violinplot(
        data,
        positions=positions,
        orientation="horizontal",
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=True,
        points=120,
        bw_method=0.24,
    )
    _style_violin_parts(violin)

    box = axis.boxplot(
        data,
        positions=positions,
        orientation="horizontal",
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#9A3F3F", "linewidth": 1.8},
        boxprops={
            "facecolor": "white",
            "edgecolor": "#314452",
            "linewidth": 1.0,
            "alpha": 0.92,
        },
        whiskerprops={"color": "#314452", "linewidth": 1.0},
        capprops={"color": "#314452", "linewidth": 1.0},
    )
    _ = box

    means = np.array([summary.loc[spec.column, "mean"] for spec in TARGET_SPECS])
    axis.scatter(
        means,
        positions,
        marker="D",
        s=38,
        facecolor="#E7A83D",
        edgecolor="#6B4E1F",
        linewidth=0.8,
        zorder=5,
    )

    for position, spec, mean in zip(positions, TARGET_SPECS, means, strict=True):
        stats = summary.loc[spec.column]
        label_x = min(float(stats["max"]) + 0.018, 1.025)
        axis.text(
            label_x,
            position,
            f"μ={mean:.3f}",
            ha="left",
            va="center",
            fontsize=8.0,
            color="#314452",
            clip_on=False,
        )

    axis.set_yticks(positions)
    axis.set_yticklabels(
        [f"{spec.short_label} — {spec.title}" for spec in TARGET_SPECS],
        fontsize=9.4,
    )
    axis.invert_yaxis()
    axis.set_xlim(-0.01, 1.08)
    axis.set_xticks(np.arange(0.0, 1.01, 0.1))
    axis.set_xlabel("Нормированное значение показателя качества", fontsize=10.4)
    axis.grid(axis="x", linestyle=":", linewidth=0.75, alpha=0.62)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#7D8D98")
    axis.spines["bottom"].set_color("#7D8D98")
    axis.tick_params(axis="x", labelsize=8.8)

    figure.text(
        0.5,
        0.115,
        (
            "Обозначения: ширина violin-графика — локальная плотность; белый блок — "
            "межквартильный диапазон; красная линия — медиана; ромб — среднее значение."
        ),
        ha="center",
        va="center",
        fontsize=8.2,
        color="#405464",
    )

    figure.text(
        0.5,
        0.050,
        (
            "Примечание — показатели сформированы после выполнения сценариев и используются "
            "как фактические целевые данные; они не входят в априорный контур LDA_prior."
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
    """Сформировать рисунок 3.4 в PNG и SVG."""

    root = project_root.resolve()
    source_path = input_path or (root / DEFAULT_INPUT_PATH)
    if not source_path.is_absolute():
        source_path = root / source_path

    frame = load_quality_targets(source_path)
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
        description=(
            "Сформировать рисунок 3.4 с распределениями фактических "
            "показателей качества."
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
            "Путь к quality_targets.csv; по умолчанию "
            "data/processed/quality_targets.csv."
        ),
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
    print("Рисунок 3.4 успешно сформирован.")
    print(f"PNG: {result.png_path}")
    print(f"SVG: {result.svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
