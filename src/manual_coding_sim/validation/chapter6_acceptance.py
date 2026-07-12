"""Финальная приемка программного контура экспериментальной главы 6.

Модуль выполняет независимую повторную проверку результатов этапов 2--13,
сопоставляет машинно-читаемые отчеты с фактически существующими артефактами
и формирует итоговый акт технической приемки. Научный статус гипотезы не
подменяется техническим статусом: корректно выполненный эксперимент может
быть принят при полном, частичном или отрицательном подтверждении гипотезы.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from manual_coding_sim.validation.chapter6_config import Chapter6ValidationConfig


class Chapter6AcceptanceError(RuntimeError):
    """Ошибка финальной приемки программного контура главы 6."""


@dataclass(frozen=True)
class Chapter6AcceptanceResult:
    """Результат финальной приемки главы 6."""

    report: Mapping[str, Any]
    json_path: Path | None
    markdown_path: Path | None

    @property
    def accepted(self) -> bool:
        """Вернуть итоговый статус технической приемки."""

        return bool(self.report["accepted"])

    @property
    def passed(self) -> bool:
        """Вернуть совместимый положительный статус результата."""

        return self.accepted


@dataclass(frozen=True)
class _SourceSpec:
    """Описание обязательного JSON-источника финальной приемки."""

    name: str
    stage: int
    candidates: tuple[str, ...]
    fallback: str


SOURCE_SPECS: tuple[_SourceSpec, ...] = (
    _SourceSpec(
        "input_validation",
        2,
        ("input_validation_report_json_path",),
        "reports/chapter6/chapter6_input_validation_report.json",
    ),
    _SourceSpec(
        "integral_quality",
        4,
        (
            "integral_quality_report_json_path",
            "integral_quality_consistency_report_json_path",
        ),
        "reports/chapter6/integral_quality_consistency_report.json",
    ),
    _SourceSpec(
        "integral_prediction",
        5,
        ("validation_metrics_json_path", "integral_prediction_report_json_path"),
        "reports/chapter6/validation_metrics.json",
    ),
    _SourceSpec(
        "partial_criteria",
        6,
        ("partial_criteria_validation_report_json_path",),
        "reports/chapter6/partial_criteria_validation_report.json",
    ),
    _SourceSpec(
        "classification",
        7,
        ("classification_report_json_path",),
        "reports/chapter6/classification_report.json",
    ),
    _SourceSpec(
        "interval_prediction",
        8,
        ("interval_coverage_report_json_path",),
        "reports/chapter6/interval_coverage_report.json",
    ),
    _SourceSpec(
        "baselines",
        9,
        ("baseline_comparison_report_json_path", "baseline_report_json"),
        "reports/chapter6/baseline_comparison_report.json",
    ),
    _SourceSpec(
        "bootstrap",
        10,
        ("bootstrap_report_json_path", "bootstrap_report_json"),
        "reports/chapter6/bootstrap_report.json",
    ),
    _SourceSpec(
        "error_analysis",
        11,
        ("prediction_error_analysis_json_path",),
        "reports/chapter6/prediction_error_analysis.json",
    ),
    _SourceSpec(
        "figures",
        12,
        ("figure_manifest_json_path",),
        "reports/chapter6/figures/figure_manifest.json",
    ),
    _SourceSpec(
        "final_report",
        13,
        (
            "chapter6_validation_report_json_path",
            "validation_report_json_path",
            "chapter6_report_json_path",
        ),
        "reports/chapter6/chapter6_validation_report.json",
    ),
    _SourceSpec(
        "pipeline",
        13,
        (
            "chapter6_pipeline_run_report_path",
            "pipeline_run_report_path",
            "chapter6_pipeline_report_path",
        ),
        "reports/chapter6/chapter6_pipeline_run_report.json",
    ),
)


class Chapter6AcceptanceBuilder:
    """Выполнить финальную техническую приемку результатов главы 6."""

    def __init__(
        self,
        config: Chapter6ValidationConfig,
        project_root: str | Path = ".",
    ) -> None:
        """Сохранить конфигурацию и корень проекта."""

        self.config = config
        self.project_root = Path(project_root)
        self.config.validate()

    def build(self) -> Chapter6AcceptanceResult:
        """Проверить все обязательные условия приемки в памяти."""

        expected_row_count = int(self.config.merge.expected_row_count)
        dataset_path = self._resolve_output_path(
            ("validation_dataset_path", "validation_dataset"),
            "reports/chapter6/validation_dataset.csv",
        )
        dataset = self._load_dataset(dataset_path, expected_row_count)

        report_paths = {
            spec.name: self._resolve_output_path(spec.candidates, spec.fallback)
            for spec in SOURCE_SPECS
        }
        reports = {
            spec.name: self._load_json_report(
                report_paths[spec.name],
                expected_stage=spec.stage,
                source_name=spec.name,
            )
            for spec in SOURCE_SPECS
        }

        checks = self._run_checks(
            dataset=dataset,
            dataset_path=dataset_path,
            reports=reports,
            report_paths=report_paths,
            expected_row_count=expected_row_count,
        )
        accepted = all(bool(item["passed"]) for item in checks)
        failed_checks = [item["id"] for item in checks if not item["passed"]]

        final_report = reports["final_report"]
        pipeline = reports["pipeline"]
        report: dict[str, Any] = {
            "stage": 14,
            "report_type": "chapter6_final_acceptance_report",
            "passed": accepted,
            "accepted": accepted,
            "full_pipeline_completed": bool(
                pipeline.get("full_pipeline_completed") is True
            ),
            "prediction_model_frozen": bool(
                pipeline.get("methodological_checks", {}).get(
                    "prediction_model_frozen"
                )
                is True
            ),
            "target_leakage_detected": bool(
                pipeline.get("methodological_checks", {}).get(
                    "target_leakage_detected"
                )
                is True
            ),
            "row_count": int(len(dataset)),
            "expected_row_count": expected_row_count,
            "hypothesis_status": final_report.get("hypothesis_status"),
            "acceptance_scope": "technical_and_methodological_reproducibility",
            "scientific_status_is_not_overridden": True,
            "check_count": len(checks),
            "passed_check_count": sum(bool(item["passed"]) for item in checks),
            "failed_check_count": len(failed_checks),
            "failed_checks": failed_checks,
            "checks": checks,
            "methodological_checks": {
                "main_dataset_used": True,
                "synthetic_holdout_used_as_main_evidence": False,
                "chapter5_artifacts_unchanged": self._chapter5_hashes_match(
                    reports["pipeline"]
                ),
                "prediction_model_frozen": (
                    pipeline.get("methodological_checks", {}).get(
                        "prediction_model_frozen"
                    )
                    is True
                ),
                "target_leakage_detected": (
                    pipeline.get("methodological_checks", {}).get(
                        "target_leakage_detected"
                    )
                    is True
                ),
                "factual_values_used_only_for_external_validation": (
                    pipeline.get("methodological_checks", {}).get(
                        "factual_values_used_only_for_external_validation"
                    )
                    is True
                ),
                "quality_thresholds_modified": (
                    pipeline.get("methodological_checks", {}).get(
                        "quality_thresholds_modified"
                    )
                    is True
                ),
                "manual_data_substitution_detected": (
                    reports["figures"].get("manual_data_substitution") is True
                ),
            },
            "accepted_artifacts": self._accepted_artifacts(
                dataset_path=dataset_path,
                report_paths=report_paths,
            ),
            "scientific_conclusion": {
                "hypothesis_status": final_report.get("hypothesis_status"),
                "recommended_claim": final_report.get(
                    "methodological_conclusion", {}
                ).get("recommended_claim"),
                "limitations": final_report.get("limitations", []),
                "acceptance_note": (
                    "Техническая приемка подтверждает воспроизводимость и "
                    "методическую корректность вычислительного контура, но не "
                    "изменяет фактический статус основной гипотезы."
                ),
            },
        }

        if not accepted:
            raise Chapter6AcceptanceError(
                "Финальная приемка главы 6 не пройдена. Непройденные проверки: "
                + ", ".join(failed_checks)
            )

        return Chapter6AcceptanceResult(
            report=report,
            json_path=None,
            markdown_path=None,
        )

    def build_and_save(self) -> Chapter6AcceptanceResult:
        """Сформировать и сохранить JSON- и Markdown-акт приемки."""

        result = self.build()
        json_path = self._resolve_output_path(
            (
                "chapter6_acceptance_report_json_path",
                "acceptance_report_json_path",
            ),
            "reports/chapter6/chapter6_acceptance_report.json",
        )
        markdown_path = self._resolve_output_path(
            (
                "chapter6_acceptance_report_md_path",
                "acceptance_report_md_path",
            ),
            "reports/chapter6/chapter6_acceptance_report.md",
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(result.report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        markdown_path.write_text(
            self._render_markdown(result.report),
            encoding="utf-8",
        )
        return Chapter6AcceptanceResult(
            report=result.report,
            json_path=json_path,
            markdown_path=markdown_path,
        )

    def _run_checks(
        self,
        *,
        dataset: pd.DataFrame,
        dataset_path: Path,
        reports: Mapping[str, Mapping[str, Any]],
        report_paths: Mapping[str, Path],
        expected_row_count: int,
    ) -> list[dict[str, Any]]:
        """Выполнить полный перечень обязательных приемочных проверок."""

        input_report = reports["input_validation"]
        integral_quality = reports["integral_quality"]
        partial = reports["partial_criteria"]
        classification = reports["classification"]
        baselines = reports["baselines"]
        bootstrap = reports["bootstrap"]
        error_analysis = reports["error_analysis"]
        figures = reports["figures"]
        final_report = reports["final_report"]
        pipeline = reports["pipeline"]

        artifact_checks = input_report.get("artifact_checks", {})
        checked_csv_count = int(input_report.get("checked_csv_count", -1))
        checked_json_count = int(input_report.get("checked_json_count", -1))
        all_input_artifacts = (
            input_report.get("passed") is True
            and isinstance(artifact_checks, Mapping)
            and bool(artifact_checks)
            and checked_csv_count == len(artifact_checks)
            and checked_json_count == 2
            and all(
                isinstance(item, Mapping)
                and int(item.get("row_count", -1)) == expected_row_count
                and int(item.get("column_count", 1)) > 0
                and item.get("required_columns_present", True) is True
                and item.get("unique_keys") is True
                and item.get("finite_values") is True
                and item.get("unit_interval_values") is True
                and item.get("key_set_aligned") is True
                and self._project_path(str(item.get("path", ""))).is_file()
                for item in artifact_checks.values()
            )
        )

        key_alignment_reference = input_report.get("key_alignment_reference")
        all_input_key_sets_aligned = (
            isinstance(artifact_checks, Mapping)
            and "q_pred" in artifact_checks
            and bool(artifact_checks)
            and all(
                isinstance(item, Mapping)
                and item.get("key_set_aligned") is True
                for item in artifact_checks.values()
            )
        )
        input_key_alignment_confirmed = (
            key_alignment_reference == "q_pred"
            or (
                input_report.get("passed") is True
                and all_input_key_sets_aligned
                and input_report.get("q_pred_consistency", {}).get("passed")
                is True
            )
        )

        report_row_counts = all(
            int(
                report.get(
                    "row_count",
                    report.get("expected_row_count", expected_row_count),
                )
            )
            == expected_row_count
            for name, report in reports.items()
            if name not in {"pipeline"}
        )

        key_columns = self._join_keys()
        keys_unique = not dataset.duplicated(list(key_columns), keep=False).any()
        bounded_columns = self._bounded_numeric_columns(dataset)
        values_in_unit_interval = bool(bounded_columns) and all(
            self._column_in_unit_interval(dataset[column])
            for column in bounded_columns
        )

        consistency_metrics = integral_quality.get("metrics", {})
        integral_quality_consistent = (
            integral_quality.get("passed") is True
            and int(consistency_metrics.get("outside_tolerance_count", 1)) == 0
            and math.isclose(
                float(consistency_metrics.get("max_q_fact_alias_difference", 1.0)),
                0.0,
                abs_tol=1e-12,
            )
        )

        confusion_path = self._resolve_output_path(
            ("confusion_matrix_path",),
            "reports/chapter6/confusion_matrix.csv",
        )
        confusion_valid = self._validate_confusion_matrix(
            confusion_path,
            expected_row_count,
        )

        baseline_path = self._resolve_output_path(
            ("baseline_comparison_path", "baseline_comparison_csv_path"),
            "reports/chapter6/baseline_comparison.csv",
        )
        baseline_models_valid = self._validate_baseline_table(baseline_path)

        bootstrap_differences_path = self._resolve_output_path(
            ("bootstrap_model_differences_path",),
            "reports/chapter6/bootstrap_model_differences.csv",
        )
        bootstrap_differences_valid = self._validate_bootstrap_differences(
            bootstrap_differences_path,
            expected_count=18,
        )

        top_errors_path = self._resolve_output_path(
            ("top_prediction_errors_path",),
            "reports/chapter6/top_prediction_errors.csv",
        )
        top_errors_valid = self._validate_top_errors(top_errors_path)

        figure_files_valid = self._validate_figure_files(
            figures,
            report_paths["figures"],
        )
        final_markdown_path = self._resolve_output_path(
            (
                "chapter6_validation_report_md_path",
                "validation_report_md_path",
                "chapter6_report_md_path",
            ),
            "reports/chapter6/chapter6_validation_report.md",
        )

        source_hashes_valid = self._validate_final_report_source_hashes(
            final_report
        )
        pipeline_steps = pipeline.get("completed_steps", {})
        pipeline_optional = pipeline.get("requested_optional_steps", {})
        full_pipeline_valid = (
            pipeline.get("passed") is True
            and pipeline.get("full_pipeline_completed") is True
            and pipeline.get("error") is None
            and bool(pipeline_steps)
            and all(value is True for value in pipeline_steps.values())
            and pipeline_optional.get("build_figures") is True
            and pipeline_optional.get("build_report") is True
        )

        stage13_checks = final_report.get("technical_checks", {})
        all_stage13_checks = bool(stage13_checks) and all(
            value is True for value in stage13_checks.values()
        )
        pipeline_methodology = pipeline.get("methodological_checks", {})
        main_sources_only = self._main_sources_only(
            dataset_path=dataset_path,
            final_report=final_report,
        )

        checks: list[dict[str, Any]] = []

        def add(check_id: str, description: str, passed: bool, evidence: Any) -> None:
            checks.append(
                {
                    "id": check_id,
                    "description": description,
                    "passed": bool(passed),
                    "evidence": evidence,
                }
            )

        add(
            "input_files_found",
            "Все обязательные входные CSV- и JSON-файлы найдены.",
            all_input_artifacts,
            {
                "checked_csv_count": input_report.get("checked_csv_count"),
                "checked_json_count": input_report.get("checked_json_count"),
                "artifact_count": len(artifact_checks),
            },
        )
        add(
            "all_tables_have_expected_rows",
            "Все основные таблицы и отчеты содержат ожидаемое число сценариев.",
            len(dataset) == expected_row_count and report_row_counts,
            {"expected_row_count": expected_row_count, "dataset_rows": len(dataset)},
        )
        add(
            "merge_without_row_loss",
            "Объединение выполнено без потери строк.",
            len(dataset) == expected_row_count
            and input_key_alignment_confirmed,
            {
                "dataset_rows": len(dataset),
                "key_alignment_reference": key_alignment_reference,
                "all_input_key_sets_aligned": all_input_key_sets_aligned,
                "alignment_confirmation_mode": (
                    "explicit_reference"
                    if key_alignment_reference == "q_pred"
                    else "derived_from_stage2_artifact_checks"
                ),
            },
        )
        add(
            "composite_keys_unique",
            "Составной ключ проверочного датасета уникален.",
            keys_unique,
            {"key_columns": list(key_columns)},
        )
        add(
            "quality_values_in_unit_interval",
            "Контролируемые показатели находятся в диапазоне [0; 1].",
            values_in_unit_interval,
            {"checked_columns": bounded_columns},
        )
        add(
            "integral_quality_consistent",
            "integral_quality согласован с шестью частными критериями.",
            integral_quality_consistent,
            consistency_metrics,
        )
        add(
            "integral_metrics_calculated",
            "Интегральные метрики рассчитаны и прошли техническую проверку.",
            reports["integral_prediction"].get("passed") is True,
            reports["integral_prediction"].get("metrics", {}),
        )
        add(
            "six_partial_criteria_validated",
            "Проверены шесть частных прогнозных критериев.",
            partial.get("passed") is True
            and int(partial.get("criterion_count", -1)) == 6,
            {"criterion_count": partial.get("criterion_count")},
        )
        add(
            "confusion_matrix_created",
            "Матрица ошибок сформирована и содержит все сценарии.",
            confusion_valid,
            {"path": self._relative_path(confusion_path)},
        )
        add(
            "classification_metrics_calculated",
            "Классификационные метрики рассчитаны.",
            classification.get("passed") is True
            and isinstance(classification.get("metrics"), Mapping),
            classification.get("metrics", {}),
        )
        add(
            "prediction_intervals_checked",
            "Интервальный прогноз проверен.",
            reports["interval_prediction"].get("passed") is True,
            reports["interval_prediction"].get("metrics", {}),
        )
        add(
            "baseline_models_built",
            "Построены четыре сравниваемые модели.",
            baselines.get("passed") is True and baseline_models_valid,
            {"path": self._relative_path(baseline_path)},
        )
        leakage = baselines.get("leakage_checks", {})
        add(
            "mean_baseline_has_no_target_leakage",
            "Mean baseline построен out-of-fold без целевой утечки.",
            leakage.get("mean_baseline_is_out_of_fold") is True
            and leakage.get("test_fold_targets_excluded_from_training_mean")
            is True,
            leakage,
        )
        add(
            "bootstrap_completed",
            "Парный cluster-bootstrap выполнен.",
            bootstrap.get("passed") is True
            and int(bootstrap.get("sampling", {}).get("resamples", 0)) == 1000,
            bootstrap.get("sampling", {}),
        )
        add(
            "paired_difference_intervals_calculated",
            "Рассчитаны доверительные интервалы парных разностей моделей.",
            bootstrap_differences_valid
            and int(bootstrap.get("summary", {}).get("comparison_count", -1))
            == 18,
            {
                "comparison_count": bootstrap.get("summary", {}).get(
                    "comparison_count"
                ),
                "path": self._relative_path(bootstrap_differences_path),
            },
        )
        add(
            "top10_errors_created",
            "Сформирован top-10 сценариев с наибольшей ошибкой.",
            error_analysis.get("passed") is True
            and int(error_analysis.get("top_error_count", -1)) == 10
            and top_errors_valid,
            {"path": self._relative_path(top_errors_path)},
        )
        add(
            "figures_created",
            "Сформированы восемь графических материалов.",
            figures.get("passed") is True
            and int(figures.get("figure_count", -1)) == 8
            and figure_files_valid,
            {
                "figure_count": figures.get("figure_count"),
                "dpi": figures.get("dpi"),
            },
        )
        add(
            "final_json_report_created",
            "Создан итоговый JSON-отчет главы 6.",
            report_paths["final_report"].exists()
            and report_paths["final_report"].stat().st_size > 0,
            {"path": self._relative_path(report_paths["final_report"])},
        )
        add(
            "final_markdown_report_created",
            "Создан итоговый Markdown-отчет главы 6.",
            final_markdown_path.exists() and final_markdown_path.stat().st_size > 0,
            {"path": self._relative_path(final_markdown_path)},
        )
        add(
            "stage13_technical_checks_passed",
            "Все повторные технические проверки итогового отчета пройдены.",
            all_stage13_checks,
            stage13_checks,
        )
        add(
            "final_report_source_hashes_match",
            "Хэши источников итогового отчета соответствуют текущим файлам.",
            source_hashes_valid,
            {"source_count": len(final_report.get("source_artifacts", []))},
        )
        add(
            "chapter5_artifacts_unchanged",
            "Зафиксированные артефакты главы 5 не изменены.",
            self._chapter5_hashes_match(pipeline),
            pipeline.get("frozen_artifact_hashes", {}),
        )
        add(
            "chapter5_parameters_not_tuned",
            "Параметры и пороги главы 5 не подгонялись по фактической выборке.",
            pipeline_methodology.get("prediction_model_frozen") is True
            and pipeline_methodology.get("quality_thresholds_modified") is False
            and final_report.get("technical_checks", {}).get(
                "prediction_model_not_modified"
            )
            is True
            and final_report.get("technical_checks", {}).get(
                "quality_thresholds_not_modified"
            )
            is True,
            pipeline_methodology,
        )
        add(
            "target_leakage_absent",
            "Целевая утечка в основном экспериментальном контуре отсутствует.",
            pipeline_methodology.get("target_leakage_detected") is False
            and pipeline_methodology.get(
                "factual_values_used_only_for_external_validation"
            )
            is True,
            pipeline_methodology,
        )
        add(
            "manual_data_substitution_absent",
            "Ручная подмена расчетных данных при построении рисунков отсутствует.",
            figures.get("manual_data_substitution") is False
            and figures.get("source_data_modified") is False,
            {
                "manual_data_substitution": figures.get(
                    "manual_data_substitution"
                ),
                "source_data_modified": figures.get("source_data_modified"),
            },
        )
        add(
            "synthetic_holdout_excluded_from_main_acceptance",
            "Изолированный синтетический holdout не подменяет основной корпус.",
            main_sources_only,
            {"dataset_path": self._relative_path(dataset_path)},
        )
        add(
            "full_cli_pipeline_completed",
            "Полный CLI-контур этапов 2--13 завершен без ошибок.",
            full_pipeline_valid,
            {
                "completed_steps": pipeline_steps,
                "requested_optional_steps": pipeline_optional,
                "error": pipeline.get("error"),
            },
        )
        return checks

    def _load_dataset(self, path: Path, expected: int) -> pd.DataFrame:
        """Загрузить и минимально проверить итоговый датасет."""

        if not path.exists():
            raise FileNotFoundError(f"Не найден проверочный датасет: {path}")
        try:
            dataset = pd.read_csv(path)
        except Exception as error:
            raise Chapter6AcceptanceError(
                f"Не удалось прочитать проверочный датасет: {path}"
            ) from error
        if len(dataset) != expected:
            raise Chapter6AcceptanceError(
                "Проверочный датасет содержит неверное число строк: "
                f"ожидалось {expected}, получено {len(dataset)}."
            )
        missing = [name for name in self._join_keys() if name not in dataset.columns]
        if missing:
            raise Chapter6AcceptanceError(
                "В проверочном датасете отсутствуют ключевые колонки: "
                + ", ".join(missing)
            )
        return dataset

    def _load_json_report(
        self,
        path: Path,
        *,
        expected_stage: int,
        source_name: str,
    ) -> dict[str, Any]:
        """Загрузить обязательный JSON-отчет и проверить его этап."""

        if not path.exists():
            raise FileNotFoundError(
                f"Не найден обязательный отчет {source_name}: {path}"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise Chapter6AcceptanceError(
                f"Не удалось прочитать отчет {source_name}: {path}"
            ) from error
        if not isinstance(payload, dict):
            raise Chapter6AcceptanceError(
                f"Отчет {source_name} должен содержать JSON-объект."
            )
        if int(payload.get("stage", -1)) != expected_stage:
            raise Chapter6AcceptanceError(
                f"Отчет {source_name} имеет неверный номер этапа."
            )
        if payload.get("passed") is not True:
            raise Chapter6AcceptanceError(
                f"Отчет {source_name} не имеет положительного статуса passed."
            )
        return payload

    def _validate_confusion_matrix(self, path: Path, expected: int) -> bool:
        """Проверить размер и сумму элементов матрицы ошибок."""

        if not path.exists():
            return False
        try:
            frame = pd.read_csv(path)
        except Exception:
            return False
        value_columns = [name for name in ("low", "medium", "high") if name in frame]
        if len(value_columns) != 3 or len(frame) != 3:
            return False
        values = frame[value_columns].apply(pd.to_numeric, errors="coerce")
        return not values.isna().any().any() and int(values.to_numpy().sum()) == expected

    def _validate_baseline_table(self, path: Path) -> bool:
        """Проверить наличие утвержденных baseline-моделей."""

        if not path.exists():
            return False
        try:
            frame = pd.read_csv(path)
        except Exception:
            return False
        if "model" not in frame.columns:
            return False
        required = {
            "mean_baseline",
            "prior_only_baseline",
            "theta_only_baseline",
            "chapter5_model",
        }
        return set(frame["model"].astype(str)) == required

    def _validate_bootstrap_differences(
        self,
        path: Path,
        expected_count: int,
    ) -> bool:
        """Проверить таблицу доверительных интервалов парных разностей."""

        if not path.exists():
            return False
        try:
            frame = pd.read_csv(path)
        except Exception:
            return False
        required = {"baseline", "metric", "ci_lower", "ci_upper", "conclusion"}
        if not required.issubset(frame.columns) or len(frame) != expected_count:
            return False
        values = frame[["ci_lower", "ci_upper"]].apply(
            pd.to_numeric,
            errors="coerce",
        )
        return not values.isna().any().any() and bool(
            (values["ci_lower"] <= values["ci_upper"]).all()
        )

    def _validate_top_errors(self, path: Path) -> bool:
        """Проверить таблицу десяти крупнейших ошибок."""

        if not path.exists():
            return False
        try:
            frame = pd.read_csv(path)
        except Exception:
            return False
        required = {"scenario_id", "protocol_id", "absolute_error"}
        return len(frame) == 10 and required.issubset(frame.columns)

    def _validate_figure_files(
        self,
        manifest: Mapping[str, Any],
        manifest_path: Path,
    ) -> bool:
        """Проверить существование и ненулевой размер восьми рисунков."""

        figures = manifest.get("figures")
        if not isinstance(figures, list) or len(figures) != 8:
            return False
        for item in figures:
            if not isinstance(item, Mapping) or not item.get("filename"):
                return False
            configured = item.get("path")
            if configured:
                path = self._project_path(str(configured))
            else:
                path = manifest_path.parent / str(item["filename"])
            if not path.exists() or path.stat().st_size <= 0:
                return False
        return True

    def _validate_final_report_source_hashes(
        self,
        final_report: Mapping[str, Any],
    ) -> bool:
        """Сопоставить хэши источников этапа 13 с текущими файлами."""

        items = final_report.get("source_artifacts")
        if not isinstance(items, list) or not items:
            return False
        for item in items:
            if not isinstance(item, Mapping):
                return False
            path_value = item.get("path")
            expected_hash = str(item.get("sha256", ""))
            if not path_value or len(expected_hash) != 64:
                return False
            path = self._project_path(str(path_value))
            if not path.exists() or self._sha256(path) != expected_hash:
                return False
        return True

    def _chapter5_hashes_match(self, pipeline: Mapping[str, Any]) -> bool:
        """Проверить текущие хэши зафиксированных артефактов главы 5."""

        hashes = pipeline.get("frozen_artifact_hashes")
        if not isinstance(hashes, Mapping) or not hashes:
            return False
        for path_value, expected_hash in hashes.items():
            path = self._project_path(str(path_value))
            if not path.exists() or self._sha256(path) != str(expected_hash):
                return False
        return True

    def _main_sources_only(
        self,
        *,
        dataset_path: Path,
        final_report: Mapping[str, Any],
    ) -> bool:
        """Убедиться, что основной отчет не ссылается на synthetic_runs."""

        values = [str(dataset_path)]
        for item in final_report.get("source_artifacts", []):
            if isinstance(item, Mapping):
                values.append(str(item.get("path", "")))
        normalized = [value.replace("\\", "/").lower() for value in values]
        return all("synthetic_runs/" not in value for value in normalized)

    def _accepted_artifacts(
        self,
        *,
        dataset_path: Path,
        report_paths: Mapping[str, Path],
    ) -> list[dict[str, Any]]:
        """Зафиксировать хэши основных принятых артефактов."""

        paths = {"validation_dataset": dataset_path, **dict(report_paths)}
        return [
            {
                "name": name,
                "path": self._relative_path(path),
                "sha256": self._sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        ]

    @staticmethod
    def _bounded_numeric_columns(dataset: pd.DataFrame) -> list[str]:
        """Вернуть числовые колонки, которые должны лежать в [0; 1]."""

        result: list[str] = []
        for name in dataset.columns:
            if name.endswith("_class"):
                continue
            if not (
                name.startswith("q_")
                or name.startswith("theta_")
                or name.endswith("_norm")
                or name == "integral_quality"
                or name == "uncertainty_score"
            ):
                continue
            values = pd.to_numeric(dataset[name], errors="coerce")
            if not values.isna().all():
                result.append(name)
        return result

    @staticmethod
    def _column_in_unit_interval(series: pd.Series) -> bool:
        """Проверить конечность и диапазон числовой колонки."""

        values = pd.to_numeric(series, errors="coerce")
        if values.isna().any():
            return False
        return bool(
            values.map(math.isfinite).all()
            and (values >= -1e-12).all()
            and (values <= 1.0 + 1e-12).all()
        )

    def _resolve_output_path(
        self,
        candidates: Sequence[str],
        fallback: str,
    ) -> Path:
        """Разрешить путь через совместимый API конфигурации этапа 1."""

        outputs = self.config.outputs
        for name in candidates:
            value = getattr(outputs, name, None)
            if value is not None:
                path = Path(value)
                return path if path.is_absolute() else self.project_root / path
        return self.project_root / fallback

    def _join_keys(self) -> tuple[str, ...]:
        """Вернуть ключевые колонки объединения."""

        keys = getattr(self.config.merge, "key_columns", None)
        if keys is None:
            keys = getattr(self.config.merge, "keys", None)
        if keys is None:
            raise Chapter6AcceptanceError(
                "Конфигурация не содержит ключи объединения."
            )
        return tuple(str(value) for value in keys)

    def _project_path(self, value: str) -> Path:
        """Преобразовать строковый путь в путь относительно проекта."""

        path = Path(value)
        return path if path.is_absolute() else self.project_root / path

    def _relative_path(self, path: Path) -> str:
        """Вернуть путь относительно корня проекта, если возможно."""

        try:
            return str(path.resolve().relative_to(self.project_root.resolve()))
        except ValueError:
            return str(path)

    @staticmethod
    def _sha256(path: Path) -> str:
        """Рассчитать SHA-256 файла."""

        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _render_markdown(report: Mapping[str, Any]) -> str:
        """Преобразовать акт приемки в русскоязычный Markdown."""

        lines = [
            "# Акт финальной приемки программного контура главы 6",
            "",
            "## 1. Итоговый статус",
            "",
            f"- Этап: `{report['stage']}`.",
            f"- Техническая приемка: `{report['accepted']}`.",
            f"- Сценариев: `{report['row_count']}`.",
            f"- Выполнено проверок: `{report['passed_check_count']}` "
            f"из `{report['check_count']}`.",
            f"- Статус основной гипотезы: `{report['hypothesis_status']}`.",
            "",
            "## 2. Приемочные проверки",
            "",
            "| Проверка | Статус |",
            "|---|---:|",
        ]
        for item in report["checks"]:
            status = "пройдена" if item["passed"] else "не пройдена"
            lines.append(f"| {item['description']} | {status} |")

        methodology = report["methodological_checks"]
        lines.extend(
            [
                "",
                "## 3. Методическая корректность",
                "",
                "| Условие | Значение |",
                "|---|---:|",
            ]
        )
        for name, value in methodology.items():
            lines.append(f"| `{name}` | `{value}` |")

        conclusion = report["scientific_conclusion"]
        lines.extend(
            [
                "",
                "## 4. Научный статус",
                "",
                conclusion.get("recommended_claim") or "Статус не сформулирован.",
                "",
                conclusion["acceptance_note"],
                "",
                "## 5. Ограничения",
                "",
            ]
        )
        limitations = conclusion.get("limitations", [])
        if limitations:
            lines.extend(f"- {item}" for item in limitations)
        else:
            lines.append("- Дополнительные ограничения не зафиксированы.")

        lines.extend(
            [
                "",
                "## 6. Решение",
                "",
                (
                    "Программный контур главы 6 принят и допускается к переносу "
                    "результатов в текст диссертации."
                    if report["accepted"]
                    else "Программный контур главы 6 не принят."
                ),
                "",
            ]
        )
        return "\n".join(lines)
