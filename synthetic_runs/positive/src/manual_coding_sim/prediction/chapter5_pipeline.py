"""Сводный отчет полного CLI-контура главы 5.

Модуль фиксирует состояние единого запуска этапов 4--9 через runner главы 5:
проверку входов, нормировку, расчет латентной компоненты, частных критериев,
интегрального ``Q_pred`` и интервальной неопределенности. Отчет нужен для
технической трассируемости этапа 10 и не использует фактические значения
качества.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from manual_coding_sim.prediction.chapter5_config import Chapter5PredictionConfig
from manual_coding_sim.prediction.integral_quality_predictor import (
    IntegralQualityPredictionResult,
)
from manual_coding_sim.prediction.latent_quality_component import (
    LatentQualityComponentResult,
)
from manual_coding_sim.prediction.partial_quality_predictor import (
    PartialQualityPredictionResult,
)
from manual_coding_sim.prediction.paths import resolve_project_path
from manual_coding_sim.prediction.prediction_uncertainty import (
    PredictionUncertaintyResult,
)
from manual_coding_sim.prediction.prior_feature_normalizer import (
    PriorFeatureNormalizationResult,
)


@dataclass(frozen=True)
class Chapter5PipelineRunReport:
    """JSON-совместимый отчет полного запуска главы 5."""

    stage: int
    run_mode: str
    completed_steps: dict[str, bool]
    row_counts: dict[str, int | None]
    artifact_paths: dict[str, str]
    quality_ranges: dict[str, dict[str, float] | None]

    def to_dict(self) -> dict[str, object]:
        """Преобразовать отчет в словарь для сохранения JSON."""

        return asdict(self)


class Chapter5PipelineRunReporter:
    """Формирует и сохраняет отчет полного CLI-запуска главы 5."""

    def build_report(
        self,
        *,
        config: Chapter5PredictionConfig,
        project_root: Path,
        normalization_result: PriorFeatureNormalizationResult | None,
        latent_result: LatentQualityComponentResult | None,
        partial_result: PartialQualityPredictionResult | None,
        q_pred_result: IntegralQualityPredictionResult | None,
        uncertainty_result: PredictionUncertaintyResult | None,
        run_mode: str = "full_pipeline",
    ) -> Chapter5PipelineRunReport:
        """Собрать отчет по выполненным шагам полного контура."""

        return Chapter5PipelineRunReport(
            stage=10,
            run_mode=run_mode,
            completed_steps={
                "input_validation": True,
                "leakage_guard": True,
                "normalization": normalization_result is not None,
                "latent_quality": latent_result is not None,
                "partial_criteria": partial_result is not None,
                "integral_quality": q_pred_result is not None,
                "uncertainty": uncertainty_result is not None,
            },
            row_counts={
                "normalized_prior_features": self._row_count(normalization_result),
                "latent_quality_component": self._row_count(latent_result),
                "q_pred_components": self._row_count(partial_result),
                "q_pred": self._row_count(q_pred_result),
                "prediction_uncertainty": self._row_count(uncertainty_result),
            },
            artifact_paths=self._artifact_paths(config, project_root),
            quality_ranges={
                "q_latent": self._range(
                    latent_result,
                    minimum_name="q_latent_min",
                    maximum_name="q_latent_max",
                ),
                "q_pred": self._range(
                    q_pred_result,
                    minimum_name="q_pred_min",
                    maximum_name="q_pred_max",
                ),
                "uncertainty_score": self._range(
                    uncertainty_result,
                    minimum_name="uncertainty_score_min",
                    maximum_name="uncertainty_score_max",
                ),
                "interval_radius": self._range(
                    uncertainty_result,
                    minimum_name="interval_radius_min",
                    maximum_name="interval_radius_max",
                ),
            },
        )

    def save_report(self, report: Chapter5PipelineRunReport, report_path: Path) -> None:
        """Сохранить отчет полного запуска в JSON-файл."""

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _artifact_paths(
        self,
        config: Chapter5PredictionConfig,
        project_root: Path,
    ) -> dict[str, str]:
        """Вернуть ключевые выходные пути полного контура главы 5."""

        output_paths: Mapping[str, Path] = {
            "leakage_report": config.outputs.reports_dir / "chapter5_leakage_report.json",
            "normalized_prior_features": config.outputs.normalized_prior_features_path,
            "normalization_report": config.outputs.normalization_report_path,
            "latent_quality_component": config.outputs.latent_quality_component_path,
            "latent_quality_component_report": config.outputs.latent_quality_component_report_path,
            "q_pred_components": config.outputs.q_pred_components_path,
            "q_pred_components_report": config.outputs.q_pred_components_report_path,
            "q_pred": config.outputs.q_pred_path,
            "q_pred_report": config.outputs.q_pred_report_path,
            "prediction_uncertainty": config.outputs.prediction_uncertainty_path,
            "prediction_uncertainty_report": config.outputs.prediction_uncertainty_report_path,
            "pipeline_run_report": config.outputs.pipeline_run_report_path,
        }
        return {
            name: str(resolve_project_path(project_root, path))
            for name, path in output_paths.items()
        }

    def _row_count(self, result: Any | None) -> int | None:
        """Извлечь число строк из отчета результата, если результат создан."""

        if result is None:
            return None
        return int(result.report.row_count)

    def _range(
        self,
        result: Any | None,
        *,
        minimum_name: str,
        maximum_name: str,
    ) -> dict[str, float] | None:
        """Извлечь минимальное и максимальное значение из отчета результата."""

        if result is None:
            return None
        return {
            "min": float(getattr(result.report, minimum_name)),
            "max": float(getattr(result.report, maximum_name)),
        }
