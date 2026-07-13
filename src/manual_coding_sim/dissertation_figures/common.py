"""Общие средства оформления и экспорта диссертационных рисунков."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@dataclass(frozen=True, slots=True)
class FigureExportResult:
    """Пути к двум обязательным форматам сформированного рисунка."""

    png_path: Path
    svg_path: Path


def configure_dissertation_style() -> None:
    """Настроить единый нейтральный стиль графических материалов."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "manual-coding-quality-dissertation",
        }
    )


def export_figure(
    figure: plt.Figure,
    *,
    project_root: Path,
    relative_output_dir: Path,
    file_stem: str,
    dpi: int = 300,
) -> FigureExportResult:
    """Сохранить рисунок в PNG и SVG и вернуть пути к артефактам."""

    if dpi < 150:
        raise ValueError("Разрешение PNG должно быть не ниже 150 dpi.")

    output_dir = project_root.resolve() / relative_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{file_stem}.png"
    svg_path = output_dir / f"{file_stem}.svg"

    figure.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.08,
        metadata={"Software": "manual_coding_sim"},
    )
    figure.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.08,
        metadata={"Date": None, "Creator": "manual_coding_sim"},
    )
    plt.close(figure)

    return FigureExportResult(png_path=png_path, svg_path=svg_path)
