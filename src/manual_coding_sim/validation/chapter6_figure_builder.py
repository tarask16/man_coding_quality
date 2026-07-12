"""Воспроизводимое построение графических материалов главы 6.

Модуль реализует этап 12 программного контура главы 6. Все рисунки
строятся только по сохраненным расчетным артефактам этапов 3--11.
Исходные таблицы не изменяются, ручная подмена значений не выполняется.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from manual_coding_sim.validation.chapter6_config import Chapter6ValidationConfig


FIGURE_FILENAMES: tuple[str, ...] = (
    "q_pred_vs_q_fact.png",
    "residuals_vs_q_fact.png",
    "absolute_error_distribution.png",
    "confusion_matrix.png",
    "baseline_comparison.png",
    "prediction_intervals.png",
    "error_by_dominant_topic.png",
    "partial_criteria_comparison.png",
)

REQUIRED_BASELINE_MODELS: tuple[str, ...] = (
    "mean_baseline",
    "prior_only_baseline",
    "theta_only_baseline",
    "chapter5_model",
)

REQUIRED_PARTIAL_CRITERIA: tuple[str, ...] = (
    "q_acc",
    "q_time",
    "q_effort",
    "q_res",
    "q_rep",
    "q_fit",
)

CLASS_LABELS: tuple[str, ...] = ("low", "medium", "high")
THETA_LABELS: tuple[str, ...] = ("theta_0", "theta_1", "theta_2")
UNIT_INTERVAL_TOLERANCE = 1e-12
DEFAULT_DPI = 300
DEFAULT_FIGURE_SIZE = (11.8, 7.0)


class Chapter6FigureBuildError(ValueError):
    """Ошибка проверки источников или построения рисунков главы 6."""


@dataclass(frozen=True)
class Chapter6FigureBuildResult:
    """Результат построения комплекта рисунков этапа 12."""

    figure_paths: Mapping[str, Path]
    manifest: Mapping[str, Any]
    manifest_json_path: Path
    manifest_markdown_path: Path

    @property
    def passed(self) -> bool:
        """Вернуть итоговый статус этапа."""

        return bool(self.manifest["passed"])


class Chapter6FigureBuilder:
    """Построить восемь воспроизводимых рисунков для главы 6."""

    def __init__(
        self,
        config: Chapter6ValidationConfig,
        project_root: str | Path = ".",
        *,
        dpi: int = DEFAULT_DPI,
        figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
    ) -> None:
        """Сохранить конфигурацию, корень проекта и параметры рендеринга."""

        self.config = config
        self.project_root = Path(project_root)
        self.dpi = int(dpi)
        self.figure_size = (float(figure_size[0]), float(figure_size[1]))
        self.config.validate()
        if self.dpi < 100:
            raise Chapter6FigureBuildError(
                "Разрешение рисунков должно быть не меньше 100 DPI."
            )
        if min(self.figure_size) <= 0:
            raise Chapter6FigureBuildError(
                "Размер рисунка должен содержать положительные значения."
            )

    def build_and_save(self) -> Chapter6FigureBuildResult:
        """Проверить источники, построить рисунки и сохранить манифест."""

        sources = self._load_sources()
        row_count = self._validate_sources(sources)
        figures_dir = self._figures_dir()
        figures_dir.mkdir(parents=True, exist_ok=True)

        source_paths = self._source_paths()
        source_hashes_before = {
            name: self._sha256(path) for name, path in source_paths.items()
        }

        builders: tuple[tuple[str, Callable[[Mapping[str, pd.DataFrame], Path], None]], ...] = (
            (FIGURE_FILENAMES[0], self._build_q_pred_vs_q_fact),
            (FIGURE_FILENAMES[1], self._build_residuals_vs_q_fact),
            (FIGURE_FILENAMES[2], self._build_absolute_error_distribution),
            (FIGURE_FILENAMES[3], self._build_confusion_matrix),
            (FIGURE_FILENAMES[4], self._build_baseline_comparison),
            (FIGURE_FILENAMES[5], self._build_prediction_intervals),
            (FIGURE_FILENAMES[6], self._build_error_by_dominant_topic),
            (FIGURE_FILENAMES[7], self._build_partial_criteria_comparison),
        )

        figure_paths: dict[str, Path] = {}
        for filename, builder in builders:
            path = figures_dir / filename
            builder(sources, path)
            self._validate_png(path)
            figure_paths[filename] = path

        source_hashes_after = {
            name: self._sha256(path) for name, path in source_paths.items()
        }
        if source_hashes_before != source_hashes_after:
            raise Chapter6FigureBuildError(
                "Один или несколько расчетных источников изменились при построении рисунков."
            )

        figure_records = []
        for filename in FIGURE_FILENAMES:
            path = figure_paths[filename]
            width, height = self._png_dimensions(path)
            figure_records.append(
                {
                    "filename": filename,
                    "path": self._relative_path(path),
                    "sha256": self._sha256(path),
                    "width_px": width,
                    "height_px": height,
                    "size_bytes": path.stat().st_size,
                }
            )

        manifest: dict[str, Any] = {
            "stage": 12,
            "report_type": "chapter6_figure_manifest",
            "passed": True,
            "row_count": row_count,
            "expected_row_count": self._expected_row_count(),
            "figure_count": len(figure_records),
            "dpi": self.dpi,
            "figure_size_inches": list(self.figure_size),
            "matplotlib_backend": matplotlib.get_backend(),
            "language": "ru",
            "source_data_modified": False,
            "manual_data_substitution": False,
            "source_files": [
                {
                    "name": name,
                    "path": self._relative_path(path),
                    "sha256": source_hashes_before[name],
                }
                for name, path in source_paths.items()
            ],
            "figures": figure_records,
            "methodological_note": (
                "Рисунки построены программно по зафиксированным расчетным "
                "артефактам этапов 3--11. Исходные значения и результаты "
                "главы 5 не изменялись."
            ),
        }

        manifest_json_path = figures_dir / "figure_manifest.json"
        manifest_markdown_path = figures_dir / "figure_manifest.md"
        manifest_json_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest_markdown_path.write_text(
            self._render_manifest_markdown(manifest),
            encoding="utf-8",
        )

        return Chapter6FigureBuildResult(
            figure_paths=figure_paths,
            manifest=manifest,
            manifest_json_path=manifest_json_path,
            manifest_markdown_path=manifest_markdown_path,
        )

    def _load_sources(self) -> dict[str, pd.DataFrame]:
        """Загрузить расчетные CSV-источники этапов 3--11."""

        sources: dict[str, pd.DataFrame] = {}
        for name, path in self._source_paths().items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Не найден расчетный источник рисунков: {path}"
                )
            try:
                sources[name] = pd.read_csv(path)
            except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
                raise Chapter6FigureBuildError(
                    f"Не удалось прочитать расчетный источник {path}: {error}"
                ) from error
            if sources[name].empty:
                raise Chapter6FigureBuildError(
                    f"Расчетный источник не содержит строк: {path}"
                )
        return sources

    def _validate_sources(self, sources: Mapping[str, pd.DataFrame]) -> int:
        """Проверить схемы, диапазоны и взаимную согласованность источников."""

        validation = sources["validation_dataset"]
        self._require_columns(
            validation,
            "validation_dataset",
            ("scenario_id", "protocol_id", "q_pred", "q_fact"),
        )
        row_count = len(validation)
        if row_count != self._expected_row_count():
            raise Chapter6FigureBuildError(
                "Неверное число строк в validation_dataset.csv: "
                f"ожидалось {self._expected_row_count()}, получено {row_count}."
            )
        keys = list(self._join_keys())
        if validation.duplicated(subset=keys).any():
            raise Chapter6FigureBuildError(
                "Составной ключ validation_dataset.csv не является уникальным."
            )
        self._validate_numeric_columns(
            validation,
            "validation_dataset",
            ("q_pred", "q_fact"),
            unit_interval=True,
        )

        errors = sources["integral_prediction_errors"]
        self._require_columns(
            errors,
            "integral_prediction_errors",
            (
                "scenario_id",
                "protocol_id",
                "q_pred",
                "q_fact",
                "prediction_error",
                "absolute_error",
            ),
        )
        self._validate_row_aligned_source(errors, validation, "integral_prediction_errors")
        self._validate_numeric_columns(
            errors,
            "integral_prediction_errors",
            ("q_pred", "q_fact", "prediction_error", "absolute_error"),
        )
        expected_errors = (
            pd.to_numeric(errors["q_pred"], errors="raise")
            - pd.to_numeric(errors["q_fact"], errors="raise")
        )
        if not np.allclose(
            expected_errors.to_numpy(dtype=float),
            pd.to_numeric(errors["prediction_error"], errors="raise").to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise Chapter6FigureBuildError(
                "prediction_error не совпадает с q_pred - q_fact."
            )

        confusion = sources["confusion_matrix"]
        self._require_columns(
            confusion,
            "confusion_matrix",
            ("factual_class", *CLASS_LABELS),
        )
        if set(confusion["factual_class"].astype(str)) != set(CLASS_LABELS):
            raise Chapter6FigureBuildError(
                "Матрица ошибок не содержит полный набор классов low/medium/high."
            )
        confusion_values = confusion[list(CLASS_LABELS)].apply(
            pd.to_numeric, errors="raise"
        )
        if not np.isfinite(confusion_values.to_numpy(dtype=float)).all():
            raise Chapter6FigureBuildError(
                "Матрица ошибок содержит NaN или бесконечные значения."
            )
        if (confusion_values.to_numpy(dtype=float) < 0).any():
            raise Chapter6FigureBuildError(
                "Матрица ошибок содержит отрицательные значения."
            )
        if int(confusion_values.to_numpy(dtype=float).sum()) != row_count:
            raise Chapter6FigureBuildError(
                "Сумма элементов confusion_matrix.csv не равна числу сценариев."
            )

        baseline = sources["baseline_comparison"]
        self._require_columns(
            baseline,
            "baseline_comparison",
            ("model", "mae", "rmse", "spearman", "kendall"),
        )
        if set(baseline["model"].astype(str)) != set(REQUIRED_BASELINE_MODELS):
            raise Chapter6FigureBuildError(
                "Таблица baseline не содержит четыре обязательные модели."
            )
        self._validate_numeric_columns(
            baseline,
            "baseline_comparison",
            ("mae", "rmse", "spearman", "kendall"),
        )

        intervals = sources["interval_coverage_details"]
        self._require_columns(
            intervals,
            "interval_coverage_details",
            (
                "scenario_id",
                "protocol_id",
                "q_pred",
                "q_fact",
                "q_pred_lower",
                "q_pred_upper",
                "is_covered",
            ),
        )
        self._validate_row_aligned_source(intervals, validation, "interval_coverage_details")
        self._validate_numeric_columns(
            intervals,
            "interval_coverage_details",
            ("q_pred", "q_fact", "q_pred_lower", "q_pred_upper"),
            unit_interval=True,
        )
        lower = pd.to_numeric(intervals["q_pred_lower"], errors="raise")
        upper = pd.to_numeric(intervals["q_pred_upper"], errors="raise")
        if (lower > upper).any():
            raise Chapter6FigureBuildError(
                "Нижняя граница прогнозного интервала превышает верхнюю."
            )

        groups = sources["error_group_analysis"]
        self._require_columns(
            groups,
            "error_group_analysis",
            ("analysis_dimension", "group", "count", "mae", "bias"),
        )
        dominant = groups.loc[
            groups["analysis_dimension"].astype(str) == "dominant_factor"
        ]
        if set(dominant["group"].astype(str)) != set(THETA_LABELS):
            raise Chapter6FigureBuildError(
                "Групповой анализ не содержит theta_0, theta_1 и theta_2."
            )
        self._validate_numeric_columns(
            dominant,
            "error_group_analysis/dominant_factor",
            ("count", "mae", "bias"),
        )
        if int(pd.to_numeric(dominant["count"], errors="raise").sum()) != row_count:
            raise Chapter6FigureBuildError(
                "Суммарное число сценариев по доминирующим факторам некорректно."
            )

        partial = sources["partial_criteria_validation"]
        self._require_columns(
            partial,
            "partial_criteria_validation",
            ("criterion", "mae", "rmse", "spearman", "kendall"),
        )
        if set(partial["criterion"].astype(str)) != set(REQUIRED_PARTIAL_CRITERIA):
            raise Chapter6FigureBuildError(
                "Таблица частных критериев не содержит все шесть критериев."
            )
        self._validate_numeric_columns(
            partial,
            "partial_criteria_validation",
            ("mae", "rmse", "spearman", "kendall"),
        )
        return row_count

    def _build_q_pred_vs_q_fact(
        self,
        sources: Mapping[str, pd.DataFrame],
        path: Path,
    ) -> None:
        """Построить диаграмму соответствия прогнозного и фактического качества."""

        data = sources["validation_dataset"]
        q_fact = pd.to_numeric(data["q_fact"], errors="raise").to_numpy(dtype=float)
        q_pred = pd.to_numeric(data["q_pred"], errors="raise").to_numpy(dtype=float)
        figure, axis = self._new_figure()
        axis.scatter(q_fact, q_pred, alpha=0.75, label="Сценарии")
        axis.plot([0, 1], [0, 1], linestyle="--", label="Идеальное совпадение")
        slope, intercept = np.polyfit(q_fact, q_pred, 1)
        x_line = np.linspace(0.0, 1.0, 200)
        axis.plot(
            x_line,
            slope * x_line + intercept,
            label="Линейная тенденция",
        )
        axis.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Фактическое интегральное качество $Q_{fact}$",
            ylabel="Априорный прогноз $Q_{pred}$",
            title="Сопоставление априорного прогноза и фактического качества",
        )
        axis.grid(True, alpha=0.25)
        axis.legend()
        self._save_figure(figure, path)

    def _build_residuals_vs_q_fact(
        self,
        sources: Mapping[str, pd.DataFrame],
        path: Path,
    ) -> None:
        """Построить зависимость ошибки прогноза от фактического качества."""

        data = sources["integral_prediction_errors"]
        q_fact = pd.to_numeric(data["q_fact"], errors="raise").to_numpy(dtype=float)
        residuals = pd.to_numeric(
            data["prediction_error"], errors="raise"
        ).to_numpy(dtype=float)
        figure, axis = self._new_figure()
        axis.scatter(q_fact, residuals, alpha=0.75)
        axis.axhline(0.0, linestyle="--", label="Нулевая ошибка")
        slope, intercept = np.polyfit(q_fact, residuals, 1)
        x_line = np.linspace(float(q_fact.min()), float(q_fact.max()), 200)
        axis.plot(
            x_line,
            slope * x_line + intercept,
            label="Линейная тенденция ошибки",
        )
        axis.set(
            xlabel="Фактическое интегральное качество $Q_{fact}$",
            ylabel="Ошибка $Q_{pred} - Q_{fact}$",
            title="Остатки априорного прогноза",
        )
        axis.grid(True, alpha=0.25)
        axis.legend()
        self._save_figure(figure, path)

    def _build_absolute_error_distribution(
        self,
        sources: Mapping[str, pd.DataFrame],
        path: Path,
    ) -> None:
        """Построить распределение абсолютной ошибки прогноза."""

        errors = pd.to_numeric(
            sources["integral_prediction_errors"]["absolute_error"],
            errors="raise",
        ).to_numpy(dtype=float)
        figure, axis = self._new_figure()
        axis.hist(errors, bins=18, edgecolor="black", alpha=0.8)
        axis.axvline(
            float(np.mean(errors)),
            linestyle="--",
            label=f"Среднее: {np.mean(errors):.3f}",
        )
        axis.axvline(
            float(np.median(errors)),
            linestyle=":",
            label=f"Медиана: {np.median(errors):.3f}",
        )
        axis.set(
            xlabel="Абсолютная ошибка $|Q_{pred} - Q_{fact}|$",
            ylabel="Число сценариев",
            title="Распределение абсолютной ошибки априорного прогноза",
        )
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend()
        self._save_figure(figure, path)

    def _build_confusion_matrix(
        self,
        sources: Mapping[str, pd.DataFrame],
        path: Path,
    ) -> None:
        """Построить матрицу ошибок классификации уровней качества."""

        source = sources["confusion_matrix"].set_index("factual_class")
        matrix = source.loc[list(CLASS_LABELS), list(CLASS_LABELS)].to_numpy(dtype=float)
        figure, axis = self._new_figure()
        image = axis.imshow(matrix)
        figure.colorbar(image, ax=axis, label="Число сценариев")
        axis.set_xticks(range(len(CLASS_LABELS)), ["Низкий", "Средний", "Высокий"])
        axis.set_yticks(range(len(CLASS_LABELS)), ["Низкий", "Средний", "Высокий"])
        axis.set(
            xlabel="Прогнозный класс",
            ylabel="Фактический класс",
            title="Матрица ошибок классификации уровней качества",
        )
        threshold = float(matrix.max()) / 2.0
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = int(round(matrix[row, column]))
                axis.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                    fontsize=13,
                )
        self._save_figure(figure, path)

    def _build_baseline_comparison(
        self,
        sources: Mapping[str, pd.DataFrame],
        path: Path,
    ) -> None:
        """Построить сравнение абсолютных и ранговых метрик моделей."""

        data = sources["baseline_comparison"].set_index("model").loc[
            list(REQUIRED_BASELINE_MODELS)
        ]
        labels = [
            "Среднее OOF",
            "Только признаки",
            "Только LDA",
            "Модель главы 5",
        ]
        x = np.arange(len(labels), dtype=float)
        width = 0.36
        figure, axes = plt.subplots(
            2,
            1,
            figsize=self.figure_size,
            constrained_layout=True,
        )
        axes[0].bar(x - width / 2, data["mae"], width, label="MAE")
        axes[0].bar(x + width / 2, data["rmse"], width, label="RMSE")
        axes[0].set_ylabel("Ошибка")
        axes[0].set_title("Абсолютная точность моделей")
        axes[0].grid(True, axis="y", alpha=0.25)
        axes[0].legend()

        axes[1].bar(x - width / 2, data["spearman"], width, label="Spearman")
        axes[1].bar(x + width / 2, data["kendall"], width, label="Kendall")
        axes[1].axhline(0.0, linewidth=0.8)
        axes[1].set_ylabel("Коэффициент корреляции")
        axes[1].set_title("Ранговая согласованность моделей")
        axes[1].grid(True, axis="y", alpha=0.25)
        axes[1].legend()
        axes[1].set_xticks(x, labels, rotation=12, ha="right")
        figure.suptitle("Сравнение модели главы 5 с базовыми схемами")
        self._save_figure(figure, path)

    def _build_prediction_intervals(
        self,
        sources: Mapping[str, pd.DataFrame],
        path: Path,
    ) -> None:
        """Построить интервалы прогноза и фактические значения по сценариям."""

        data = sources["interval_coverage_details"].sort_values(
            ["q_fact", "scenario_id"], kind="mergesort"
        )
        x = np.arange(len(data), dtype=float)
        lower = pd.to_numeric(data["q_pred_lower"], errors="raise").to_numpy(dtype=float)
        upper = pd.to_numeric(data["q_pred_upper"], errors="raise").to_numpy(dtype=float)
        q_pred = pd.to_numeric(data["q_pred"], errors="raise").to_numpy(dtype=float)
        q_fact = pd.to_numeric(data["q_fact"], errors="raise").to_numpy(dtype=float)
        figure, axis = self._new_figure()
        axis.vlines(x, lower, upper, alpha=0.5, label="Прогнозный интервал")
        axis.scatter(x, q_pred, s=16, marker="o", label="$Q_{pred}$")
        axis.scatter(x, q_fact, s=18, marker="x", label="$Q_{fact}$")
        axis.set(
            xlim=(-2, len(data) + 1),
            ylim=(0, 1),
            xlabel="Сценарии, упорядоченные по $Q_{fact}$",
            ylabel="Качество",
            title="Покрытие фактического качества прогнозными интервалами",
        )
        axis.grid(True, alpha=0.2)
        axis.legend(ncol=3)
        self._save_figure(figure, path)

    def _build_error_by_dominant_topic(
        self,
        sources: Mapping[str, pd.DataFrame],
        path: Path,
    ) -> None:
        """Построить ошибки по доминирующему латентному фактору."""

        data = sources["error_group_analysis"]
        data = data.loc[data["analysis_dimension"] == "dominant_factor"].copy()
        data = data.set_index("group").loc[list(THETA_LABELS)]
        labels = ["θ₀", "θ₁", "θ₂"]
        x = np.arange(len(labels), dtype=float)
        width = 0.36
        figure, axis = self._new_figure()
        axis.bar(x - width / 2, data["mae"], width, label="MAE")
        axis.bar(x + width / 2, np.abs(data["bias"]), width, label="|Bias|")
        axis.set_xticks(x, labels)
        axis.set(
            xlabel="Доминирующий латентный фактор",
            ylabel="Ошибка",
            title="Ошибка прогноза по доминирующим латентным факторам",
        )
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend()
        self._save_figure(figure, path)

    def _build_partial_criteria_comparison(
        self,
        sources: Mapping[str, pd.DataFrame],
        path: Path,
    ) -> None:
        """Построить сравнение ошибок и рангов частных критериев."""

        data = sources["partial_criteria_validation"].set_index("criterion").loc[
            list(REQUIRED_PARTIAL_CRITERIA)
        ]
        labels = [
            "Точность",
            "Время",
            "Трудоемкость",
            "Результативность",
            "Повторяемость",
            "Соответствие",
        ]
        x = np.arange(len(labels), dtype=float)
        width = 0.36
        figure, axes = plt.subplots(
            2,
            1,
            figsize=self.figure_size,
            constrained_layout=True,
        )
        axes[0].bar(x - width / 2, data["mae"], width, label="MAE")
        axes[0].bar(x + width / 2, data["rmse"], width, label="RMSE")
        axes[0].set_ylabel("Ошибка")
        axes[0].set_title("Абсолютная точность частных критериев")
        axes[0].grid(True, axis="y", alpha=0.25)
        axes[0].legend()

        axes[1].bar(x - width / 2, data["spearman"], width, label="Spearman")
        axes[1].bar(x + width / 2, data["kendall"], width, label="Kendall")
        axes[1].set_ylabel("Коэффициент корреляции")
        axes[1].set_title("Ранговая согласованность частных критериев")
        axes[1].grid(True, axis="y", alpha=0.25)
        axes[1].legend()
        axes[1].set_xticks(x, labels, rotation=14, ha="right")
        figure.suptitle("Сравнение качества прогноза частных критериев")
        self._save_figure(figure, path)

    def _new_figure(self) -> tuple[plt.Figure, plt.Axes]:
        """Создать одиночную фигуру с едиными параметрами."""

        return plt.subplots(figsize=self.figure_size, constrained_layout=True)

    def _save_figure(self, figure: plt.Figure, path: Path) -> None:
        """Сохранить PNG с разрешением для вставки в диссертацию."""

        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            path,
            dpi=self.dpi,
            format="png",
            bbox_inches="tight",
            metadata={
                "Title": "Графический материал главы 6",
                "Software": "manual_coding_sim.validation.chapter6_figure_builder",
            },
        )
        plt.close(figure)

    def _source_paths(self) -> dict[str, Path]:
        """Вернуть зафиксированный перечень расчетных источников."""

        return {
            "validation_dataset": self._resolve_output_path(
                ("validation_dataset_path", "validation_dataset"),
                "reports/chapter6/validation_dataset.csv",
            ),
            "integral_prediction_errors": self._resolve_output_path(
                ("integral_prediction_errors_path", "integral_prediction_errors"),
                "reports/chapter6/integral_prediction_errors.csv",
            ),
            "confusion_matrix": self._resolve_output_path(
                ("confusion_matrix_path", "confusion_matrix"),
                "reports/chapter6/confusion_matrix.csv",
            ),
            "baseline_comparison": self._resolve_output_path(
                ("baseline_comparison_path", "baseline_comparison"),
                "reports/chapter6/baseline_comparison.csv",
            ),
            "interval_coverage_details": self._resolve_output_path(
                ("interval_coverage_details_path", "interval_coverage_details"),
                "reports/chapter6/interval_coverage_details.csv",
            ),
            "error_group_analysis": self._resolve_output_path(
                ("error_group_analysis_path", "error_group_analysis"),
                "reports/chapter6/error_group_analysis.csv",
            ),
            "partial_criteria_validation": self._resolve_output_path(
                ("partial_criteria_validation_path", "partial_criteria_validation"),
                "reports/chapter6/partial_criteria_validation.csv",
            ),
        }

    def _figures_dir(self) -> Path:
        """Разрешить каталог рисунков через API конфигурации этапа 1."""

        outputs = self.config.outputs
        value = getattr(outputs, "figures_dir", None)
        path = Path(value) if value is not None else Path("reports/chapter6/figures")
        return path if path.is_absolute() else self.project_root / path

    def _resolve_output_path(
        self,
        candidates: Sequence[str],
        fallback: str,
    ) -> Path:
        """Разрешить путь входного расчетного артефакта."""

        outputs = self.config.outputs
        value: Any | None = None
        for name in candidates:
            if hasattr(outputs, name):
                value = getattr(outputs, name)
                break
        path = Path(value) if value is not None else Path(fallback)
        return path if path.is_absolute() else self.project_root / path

    def _validate_row_aligned_source(
        self,
        source: pd.DataFrame,
        reference: pd.DataFrame,
        source_name: str,
    ) -> None:
        """Проверить число строк и множество составных ключей."""

        keys = list(self._join_keys())
        self._require_columns(source, source_name, keys)
        if len(source) != len(reference):
            raise Chapter6FigureBuildError(
                f"Источник {source_name} содержит неверное число строк."
            )
        source_keys = set(map(tuple, source[keys].astype(str).to_numpy()))
        reference_keys = set(map(tuple, reference[keys].astype(str).to_numpy()))
        if source_keys != reference_keys:
            raise Chapter6FigureBuildError(
                f"Составные ключи источника {source_name} не согласованы с датасетом."
            )

    @staticmethod
    def _require_columns(
        table: pd.DataFrame,
        table_name: str,
        required: Sequence[str],
    ) -> None:
        """Проверить наличие обязательных колонок."""

        missing = [column for column in required if column not in table.columns]
        if missing:
            raise Chapter6FigureBuildError(
                f"В источнике {table_name} отсутствуют колонки: {', '.join(missing)}."
            )

    @staticmethod
    def _validate_numeric_columns(
        table: pd.DataFrame,
        table_name: str,
        columns: Sequence[str],
        *,
        unit_interval: bool = False,
    ) -> None:
        """Проверить конечность и при необходимости диапазон чисел."""

        try:
            numeric = table[list(columns)].apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as error:
            raise Chapter6FigureBuildError(
                f"Источник {table_name} содержит нечисловые значения."
            ) from error
        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise Chapter6FigureBuildError(
                f"Источник {table_name} содержит NaN или бесконечные значения."
            )
        if unit_interval and (
            (values < -UNIT_INTERVAL_TOLERANCE).any()
            or (values > 1.0 + UNIT_INTERVAL_TOLERANCE).any()
        ):
            raise Chapter6FigureBuildError(
                f"Источник {table_name} содержит значения вне диапазона [0; 1]."
            )

    def _expected_row_count(self) -> int:
        """Вернуть ожидаемое число сценариев."""

        return int(self.config.merge.expected_row_count)

    def _join_keys(self) -> tuple[str, ...]:
        """Вернуть составной ключ проверки."""

        keys = getattr(self.config.merge, "key_columns", None)
        if keys is None:
            keys = getattr(self.config.merge, "keys", None)
        if keys is None:
            raise Chapter6FigureBuildError(
                "В конфигурации отсутствуют ключи объединения."
            )
        return tuple(str(key) for key in keys)

    def _relative_path(self, path: Path) -> str:
        """Вернуть путь относительно корня проекта, если это возможно."""

        try:
            return str(path.resolve().relative_to(self.project_root.resolve()))
        except ValueError:
            return str(path)

    @staticmethod
    def _sha256(path: Path) -> str:
        """Рассчитать SHA-256 файла."""

        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _png_dimensions(path: Path) -> tuple[int, int]:
        """Прочитать размеры PNG без внешней зависимости Pillow."""

        with path.open("rb") as stream:
            signature = stream.read(8)
            if signature != b"\x89PNG\r\n\x1a\n":
                raise Chapter6FigureBuildError(f"Файл не является PNG: {path}")
            length = struct.unpack(">I", stream.read(4))[0]
            chunk = stream.read(4)
            if chunk != b"IHDR" or length < 8:
                raise Chapter6FigureBuildError(
                    f"PNG не содержит корректный заголовок IHDR: {path}"
                )
            width, height = struct.unpack(">II", stream.read(8))
        return int(width), int(height)

    def _validate_png(self, path: Path) -> None:
        """Проверить существование, размер и ненулевое содержимое PNG."""

        if not path.exists() or path.stat().st_size < 10_000:
            raise Chapter6FigureBuildError(
                f"Рисунок не создан или имеет подозрительно малый размер: {path}"
            )
        width, height = self._png_dimensions(path)
        expected_width = int(self.figure_size[0] * self.dpi * 0.70)
        expected_height = int(self.figure_size[1] * self.dpi * 0.70)
        if width < expected_width or height < expected_height:
            raise Chapter6FigureBuildError(
                "Разрешение созданного рисунка ниже ожидаемого: "
                f"{path}, {width}x{height}."
            )

    @staticmethod
    def _render_manifest_markdown(manifest: Mapping[str, Any]) -> str:
        """Сформировать русскоязычный Markdown-манифест рисунков."""

        lines = [
            "# Манифест графических материалов главы 6",
            "",
            f"- Этап: `{manifest['stage']}`",
            f"- Статус: `{manifest['passed']}`",
            f"- Сценариев: `{manifest['row_count']}`",
            f"- Рисунков: `{manifest['figure_count']}`",
            f"- Разрешение: `{manifest['dpi']} DPI`",
            "- Исходные данные изменены: `False`",
            "- Ручная подмена данных: `False`",
            "",
            "## Сформированные рисунки",
            "",
            "| Файл | Размер, px | SHA-256 |",
            "|---|---:|---|",
        ]
        for item in manifest["figures"]:
            lines.append(
                f"| `{item['filename']}` | "
                f"{item['width_px']}×{item['height_px']} | "
                f"`{item['sha256']}` |"
            )
        lines.extend(
            [
                "",
                "## Методическое ограничение",
                "",
                str(manifest["methodological_note"]),
                "",
            ]
        )
        return "\n".join(lines)


__all__ = [
    "Chapter6FigureBuildError",
    "Chapter6FigureBuildResult",
    "Chapter6FigureBuilder",
    "DEFAULT_DPI",
    "FIGURE_FILENAMES",
]
