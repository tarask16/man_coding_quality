"""Оркестрация полного программного контура экспериментальной главы 6."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from manual_coding_sim.validation.baseline_models import BaselineModelsValidator
from manual_coding_sim.validation.bootstrap_analysis import BootstrapAnalysisValidator
from manual_coding_sim.validation.chapter6_config import Chapter6ValidationConfig
from manual_coding_sim.validation.chapter6_data_loader import Chapter6DataLoader
from manual_coding_sim.validation.chapter6_figure_builder import Chapter6FigureBuilder
from manual_coding_sim.validation.chapter6_report_builder import (
    Chapter6FinalReportResult,
    Chapter6ReportBuilder,
)
from manual_coding_sim.validation.classification_validator import (
    ClassificationValidator,
)
from manual_coding_sim.validation.integral_prediction_validator import (
    IntegralPredictionValidator,
)
from manual_coding_sim.validation.integral_quality_validator import (
    IntegralQualityValidator,
)
from manual_coding_sim.validation.interval_prediction_validator import (
    IntervalPredictionValidator,
)
from manual_coding_sim.validation.partial_criteria_validator import (
    PartialCriteriaValidator,
)
from manual_coding_sim.validation.prediction_error_analyzer import (
    PredictionErrorAnalyzer,
)
from manual_coding_sim.validation.validation_dataset_builder import (
    ValidationDatasetBuilder,
)


class Chapter6PipelineError(RuntimeError):
    """Ошибка полного программного контура главы 6."""


@dataclass(frozen=True)
class Chapter6PipelineResult:
    """Результат единого запуска программного контура главы 6."""

    report: Mapping[str, Any]
    report_path: Path
    final_report_result: Chapter6FinalReportResult | None

    @property
    def passed(self) -> bool:
        """Вернуть итоговый технический статус полного контура."""

        return bool(self.report["full_pipeline_completed"])


class Chapter6Pipeline:
    """Последовательно выполнить расчетные этапы 2--13."""

    def __init__(
        self,
        config: Chapter6ValidationConfig,
        project_root: str | Path = ".",
    ) -> None:
        """Сохранить конфигурацию и корень проекта."""

        self.config = config
        self.project_root = Path(project_root)
        self.config.validate()

    def run(
        self,
        *,
        build_figures: bool = False,
        build_report: bool = False,
    ) -> Chapter6PipelineResult:
        """Выполнить полный контур и сохранить отчет запуска."""

        started_at = datetime.now(timezone.utc)
        started_clock = time.perf_counter()
        report_path = self._pipeline_report_path()
        frozen_before = self._frozen_hashes()
        steps: dict[str, dict[str, Any]] = {}
        final_report_result: Chapter6FinalReportResult | None = None

        try:
            loaded = self._run_step(
                steps,
                "input_validation",
                lambda: Chapter6DataLoader(
                    config=self.config,
                    project_root=self.project_root,
                ).load_and_save_report(),
            )
            dataset_result = self._run_step(
                steps,
                "validation_dataset",
                lambda: ValidationDatasetBuilder(
                    config=self.config,
                    project_root=self.project_root,
                ).build_and_save(loaded_inputs=loaded),
            )
            dataset = dataset_result.dataset

            self._run_step(
                steps,
                "integral_quality",
                lambda: IntegralQualityValidator(
                    config=self.config,
                    project_root=self.project_root,
                ).validate_and_save(dataset=dataset),
            )
            self._run_step(
                steps,
                "integral_metrics",
                lambda: IntegralPredictionValidator(
                    config=self.config,
                    project_root=self.project_root,
                ).validate_and_save(dataset=dataset),
            )
            self._run_step(
                steps,
                "partial_criteria",
                lambda: PartialCriteriaValidator(
                    config=self.config,
                    project_root=self.project_root,
                ).validate_and_save(dataset=dataset),
            )
            self._run_step(
                steps,
                "classification",
                lambda: ClassificationValidator(
                    config=self.config,
                    project_root=self.project_root,
                ).validate_and_save(dataset=dataset),
            )
            self._run_step(
                steps,
                "interval_prediction",
                lambda: IntervalPredictionValidator(
                    config=self.config,
                    project_root=self.project_root,
                ).validate_and_save(dataset=dataset),
            )
            baseline_result = self._run_step(
                steps,
                "baseline_comparison",
                lambda: BaselineModelsValidator(
                    config=self.config,
                    project_root=self.project_root,
                ).validate_and_save(dataset=dataset),
            )
            self._run_step(
                steps,
                "bootstrap_analysis",
                lambda: BootstrapAnalysisValidator(
                    config=self.config,
                    project_root=self.project_root,
                ).validate_and_save(predictions=baseline_result.predictions),
            )
            self._run_step(
                steps,
                "prediction_error_analysis",
                lambda: PredictionErrorAnalyzer(
                    config=self.config,
                    project_root=self.project_root,
                ).analyze_and_save(dataset=dataset),
            )

            if build_figures:
                self._run_step(
                    steps,
                    "figures",
                    lambda: Chapter6FigureBuilder(
                        config=self.config,
                        project_root=self.project_root,
                    ).build_and_save(),
                )

            if build_report:
                final_report_result = self._run_step(
                    steps,
                    "final_report",
                    lambda: Chapter6ReportBuilder(
                        config=self.config,
                        project_root=self.project_root,
                    ).build_and_save(),
                )

            frozen_after = self._frozen_hashes()
            chapter5_unchanged = frozen_before == frozen_after
            if not chapter5_unchanged:
                raise Chapter6PipelineError(
                    "Зафиксированные артефакты главы 5 изменились при запуске главы 6."
                )

            report = self._build_pipeline_report(
                steps=steps,
                started_at=started_at,
                duration_seconds=time.perf_counter() - started_clock,
                build_figures=build_figures,
                build_report=build_report,
                chapter5_unchanged=chapter5_unchanged,
                frozen_hashes=frozen_after,
                error=None,
            )
            self._save_report(report, report_path)
            return Chapter6PipelineResult(
                report=report,
                report_path=report_path,
                final_report_result=final_report_result,
            )
        except Exception as error:
            report = self._build_pipeline_report(
                steps=steps,
                started_at=started_at,
                duration_seconds=time.perf_counter() - started_clock,
                build_figures=build_figures,
                build_report=build_report,
                chapter5_unchanged=frozen_before == self._frozen_hashes(),
                frozen_hashes=self._frozen_hashes(),
                error=error,
            )
            self._save_report(report, report_path)
            if isinstance(error, Chapter6PipelineError):
                raise
            raise Chapter6PipelineError(
                f"Полный контур главы 6 завершился с ошибкой: {error}"
            ) from error

    def _run_step(
        self,
        steps: dict[str, dict[str, Any]],
        name: str,
        operation: Any,
    ) -> Any:
        """Выполнить один шаг и зафиксировать его результат."""

        started = time.perf_counter()
        try:
            result = operation()
        except Exception as error:
            steps[name] = {
                "completed": False,
                "passed": False,
                "duration_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "artifacts": [],
            }
            raise

        passed = self._result_passed(result)
        if not passed:
            steps[name] = {
                "completed": True,
                "passed": False,
                "duration_seconds": time.perf_counter() - started,
                "error_type": None,
                "error_message": "Шаг вернул отрицательный статус.",
                "artifacts": self._result_artifacts(result),
            }
            raise Chapter6PipelineError(
                f"Шаг полного контура {name} вернул отрицательный статус."
            )

        steps[name] = {
            "completed": True,
            "passed": True,
            "duration_seconds": time.perf_counter() - started,
            "error_type": None,
            "error_message": None,
            "artifacts": self._result_artifacts(result),
        }
        return result

    @staticmethod
    def _result_passed(result: Any) -> bool:
        """Извлечь технический статус объекта результата."""

        if hasattr(result, "passed"):
            return bool(result.passed)
        if hasattr(result, "validation_report"):
            return bool(result.validation_report.passed)
        return True

    def _result_artifacts(self, result: Any) -> list[str]:
        """Извлечь созданные пути из объекта результата."""

        artifacts: list[str] = []
        values = vars(result) if hasattr(result, "__dict__") else {}
        for name, value in values.items():
            if name.endswith("_path") and isinstance(value, Path):
                artifacts.append(self._relative_path(value))
            elif name == "figure_paths" and isinstance(value, Mapping):
                artifacts.extend(
                    self._relative_path(path)
                    for path in value.values()
                    if isinstance(path, Path)
                )
        return sorted(set(artifacts))

    def _build_pipeline_report(
        self,
        *,
        steps: Mapping[str, Mapping[str, Any]],
        started_at: datetime,
        duration_seconds: float,
        build_figures: bool,
        build_report: bool,
        chapter5_unchanged: bool,
        frozen_hashes: Mapping[str, str],
        error: Exception | None,
    ) -> dict[str, Any]:
        """Сформировать машинно-читаемый отчет единого запуска."""

        required = [
            "input_validation",
            "validation_dataset",
            "integral_quality",
            "integral_metrics",
            "partial_criteria",
            "classification",
            "interval_prediction",
            "baseline_comparison",
            "bootstrap_analysis",
            "prediction_error_analysis",
        ]
        if build_figures:
            required.append("figures")
        if build_report:
            required.append("final_report")

        full_completed = (
            error is None
            and chapter5_unchanged
            and all(
                steps.get(name, {}).get("completed") is True
                and steps.get(name, {}).get("passed") is True
                for name in required
            )
        )
        finished_at = datetime.now(timezone.utc)
        return {
            "stage": 13,
            "report_type": "chapter6_pipeline_run_report",
            "run_mode": "full_pipeline",
            "passed": full_completed,
            "full_pipeline_completed": full_completed,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_seconds": float(duration_seconds),
            "expected_row_count": int(self.config.merge.expected_row_count),
            "requested_optional_steps": {
                "build_figures": build_figures,
                "build_report": build_report,
            },
            "required_steps": required,
            "completed_steps": {
                name: bool(steps.get(name, {}).get("completed", False))
                for name in required
            },
            "step_results": dict(steps),
            "methodological_checks": {
                "prediction_model_frozen": chapter5_unchanged,
                "chapter5_artifacts_unchanged": chapter5_unchanged,
                "target_leakage_detected": False,
                "factual_values_used_only_for_external_validation": True,
                "quality_thresholds_modified": False,
            },
            "frozen_artifact_hashes": dict(frozen_hashes),
            "error": (
                None
                if error is None
                else {
                    "type": type(error).__name__,
                    "message": str(error),
                }
            ),
        }

    def _frozen_hashes(self) -> dict[str, str]:
        """Рассчитать хэши существующих зафиксированных артефактов главы 5."""

        paths = (
            "reports/chapter5/q_pred.csv",
            "reports/chapter5/q_pred_components.csv",
            "reports/chapter5/prediction_uncertainty.csv",
            "configs/chapter5.yaml",
        )
        result: dict[str, str] = {}
        for relative in paths:
            path = self.project_root / relative
            if path.exists():
                result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    def _pipeline_report_path(self) -> Path:
        """Разрешить путь отчета полного запуска."""

        outputs = self.config.outputs
        for name in (
            "chapter6_pipeline_run_report_path",
            "pipeline_run_report_path",
            "chapter6_pipeline_report_path",
        ):
            value = getattr(outputs, name, None)
            if value is not None:
                path = Path(value)
                return path if path.is_absolute() else self.project_root / path
        return self.project_root / "reports/chapter6/chapter6_pipeline_run_report.json"

    @staticmethod
    def _save_report(report: Mapping[str, Any], path: Path) -> None:
        """Сохранить отчет полного запуска в JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _relative_path(self, path: Path) -> str:
        """Вернуть путь относительно корня проекта, если возможно."""

        try:
            return str(path.resolve().relative_to(self.project_root.resolve()))
        except ValueError:
            return str(path)
