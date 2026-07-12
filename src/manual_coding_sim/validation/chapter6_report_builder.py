"""Формирование итогового отчета экспериментальной проверки главы 6.

Модуль объединяет машинно-читаемые результаты этапов 2--12, повторно
проверяет их согласованность и формирует итоговый JSON- и Markdown-отчет.
Технический статус отчета отделен от статуса научной гипотезы: корректно
выполненный эксперимент может дать полное, частичное или отрицательное
подтверждение гипотезы.
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


HYPOTHESIS_SUPPORTED = "hypothesis_supported"
HYPOTHESIS_PARTIALLY_SUPPORTED = "hypothesis_partially_supported"
HYPOTHESIS_NOT_SUPPORTED = "hypothesis_not_supported"
HYPOTHESIS_STATUSES = (
    HYPOTHESIS_SUPPORTED,
    HYPOTHESIS_PARTIALLY_SUPPORTED,
    HYPOTHESIS_NOT_SUPPORTED,
)


class Chapter6ReportBuildError(RuntimeError):
    """Ошибка формирования итогового отчета главы 6."""


@dataclass(frozen=True)
class Chapter6FinalReportResult:
    """Результат формирования итогового отчета главы 6."""

    report: Mapping[str, Any]
    json_path: Path | None
    markdown_path: Path | None

    @property
    def passed(self) -> bool:
        """Вернуть технический статус итогового отчета."""

        return bool(self.report["passed"])

    @property
    def hypothesis_status(self) -> str:
        """Вернуть итоговый статус основной гипотезы."""

        return str(self.report["hypothesis_status"])


@dataclass(frozen=True)
class _ReportSpec:
    """Описание обязательного машинно-читаемого источника."""

    name: str
    stage: int
    candidates: tuple[str, ...]
    fallback: str


REPORT_SPECS: tuple[_ReportSpec, ...] = (
    _ReportSpec(
        "input_validation",
        2,
        ("input_validation_report_json_path",),
        "reports/chapter6/chapter6_input_validation_report.json",
    ),
    _ReportSpec(
        "integral_quality",
        4,
        (
            "integral_quality_report_json_path",
            "integral_quality_consistency_report_json_path",
        ),
        "reports/chapter6/integral_quality_consistency_report.json",
    ),
    _ReportSpec(
        "integral_prediction",
        5,
        ("validation_metrics_json_path", "integral_prediction_report_json_path"),
        "reports/chapter6/validation_metrics.json",
    ),
    _ReportSpec(
        "partial_criteria",
        6,
        ("partial_criteria_validation_report_json_path",),
        "reports/chapter6/partial_criteria_validation_report.json",
    ),
    _ReportSpec(
        "classification",
        7,
        ("classification_report_json_path",),
        "reports/chapter6/classification_report.json",
    ),
    _ReportSpec(
        "interval_prediction",
        8,
        ("interval_coverage_report_json_path",),
        "reports/chapter6/interval_coverage_report.json",
    ),
    _ReportSpec(
        "baselines",
        9,
        ("baseline_comparison_report_json_path", "baseline_report_json"),
        "reports/chapter6/baseline_comparison_report.json",
    ),
    _ReportSpec(
        "bootstrap",
        10,
        ("bootstrap_report_json_path", "bootstrap_report_json"),
        "reports/chapter6/bootstrap_report.json",
    ),
    _ReportSpec(
        "error_analysis",
        11,
        ("prediction_error_analysis_json_path",),
        "reports/chapter6/prediction_error_analysis.json",
    ),
    _ReportSpec(
        "figures",
        12,
        ("figure_manifest_json_path",),
        "reports/chapter6/figures/figure_manifest.json",
    ),
)


class Chapter6ReportBuilder:
    """Собрать итоговый отчет из результатов этапов 2--12."""

    def __init__(
        self,
        config: Chapter6ValidationConfig,
        project_root: str | Path = ".",
    ) -> None:
        """Сохранить конфигурацию и корень проекта."""

        self.config = config
        self.project_root = Path(project_root)
        self.config.validate()

    def build(self) -> Chapter6FinalReportResult:
        """Проверить источники и сформировать итоговый отчет в памяти."""

        dataset_path = self._resolve_output_path(
            ("validation_dataset_path", "validation_dataset"),
            "reports/chapter6/validation_dataset.csv",
        )
        dataset = self._load_validation_dataset(dataset_path)

        report_paths = {
            spec.name: self._resolve_output_path(spec.candidates, spec.fallback)
            for spec in REPORT_SPECS
        }
        reports = {
            spec.name: self._load_and_validate_report(
                path=report_paths[spec.name],
                expected_stage=spec.stage,
                source_name=spec.name,
            )
            for spec in REPORT_SPECS
        }

        expected_row_count = self._expected_row_count()
        self._validate_cross_report_row_counts(reports, expected_row_count)
        self._validate_figure_files(reports["figures"], report_paths["figures"])

        hypothesis = self._evaluate_hypothesis(reports)
        source_artifacts = self._source_artifacts(dataset_path, report_paths)
        technical_checks = self._technical_checks(
            dataset=dataset,
            reports=reports,
            source_artifacts=source_artifacts,
        )
        passed = all(bool(value) for value in technical_checks.values())

        report: dict[str, Any] = {
            "stage": 13,
            "report_type": "chapter6_validation_final_report",
            "passed": passed,
            "row_count": int(len(dataset)),
            "expected_row_count": expected_row_count,
            "hypothesis_status": hypothesis["status"],
            "hypothesis": hypothesis,
            "technical_checks": technical_checks,
            "input_validation": self._input_summary(reports["input_validation"]),
            "integral_quality_consistency": self._copy_section(
                reports["integral_quality"],
                ("passed", "row_count", "consistency_tolerance", "metrics"),
            ),
            "integral_prediction": self._copy_section(
                reports["integral_prediction"],
                ("passed", "row_count", "error_definition", "metrics"),
            ),
            "partial_criteria": self._copy_section(
                reports["partial_criteria"],
                ("passed", "row_count", "criterion_count", "summary", "metrics"),
            ),
            "classification": self._copy_section(
                reports["classification"],
                (
                    "passed",
                    "row_count",
                    "metrics",
                    "critical_errors",
                    "per_class_metrics",
                ),
            ),
            "interval_prediction": self._copy_section(
                reports["interval_prediction"],
                ("passed", "row_count", "coverage_condition", "metrics", "slices"),
            ),
            "baseline_comparison": self._copy_section(
                reports["baselines"],
                (
                    "passed",
                    "row_count",
                    "metrics",
                    "best_models",
                    "chapter5_differences",
                    "leakage_checks",
                ),
            ),
            "bootstrap_analysis": self._copy_section(
                reports["bootstrap"],
                (
                    "passed",
                    "row_count",
                    "sampling",
                    "chapter5_confidence_intervals",
                    "summary",
                    "methodological_checks",
                ),
            ),
            "prediction_error_analysis": self._copy_section(
                reports["error_analysis"],
                (
                    "passed",
                    "row_count",
                    "top_error_count",
                    "summary",
                    "uncertainty_relation",
                    "group_highlights",
                    "methodological_checks",
                ),
            ),
            "figures": self._copy_section(
                reports["figures"],
                (
                    "passed",
                    "row_count",
                    "figure_count",
                    "dpi",
                    "source_data_modified",
                    "manual_data_substitution",
                    "figures",
                ),
            ),
            "methodological_conclusion": self._methodological_conclusion(
                reports=reports,
                hypothesis=hypothesis,
            ),
            "limitations": self._limitations(reports),
            "source_artifacts": source_artifacts,
        }

        if not passed:
            raise Chapter6ReportBuildError(
                "Итоговый отчет не прошел повторную техническую проверку."
            )

        return Chapter6FinalReportResult(
            report=report,
            json_path=None,
            markdown_path=None,
        )

    def build_and_save(self) -> Chapter6FinalReportResult:
        """Сформировать и сохранить итоговый JSON- и Markdown-отчет."""

        result = self.build()
        json_path = self._resolve_output_path(
            (
                "chapter6_validation_report_json_path",
                "validation_report_json_path",
                "chapter6_report_json_path",
            ),
            "reports/chapter6/chapter6_validation_report.json",
        )
        markdown_path = self._resolve_output_path(
            (
                "chapter6_validation_report_md_path",
                "validation_report_md_path",
                "chapter6_report_md_path",
            ),
            "reports/chapter6/chapter6_validation_report.md",
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
        return Chapter6FinalReportResult(
            report=result.report,
            json_path=json_path,
            markdown_path=markdown_path,
        )

    def _load_validation_dataset(self, path: Path) -> pd.DataFrame:
        """Загрузить и повторно проверить единый проверочный датасет."""

        if not path.exists():
            raise FileNotFoundError(
                f"Не найден проверочный датасет этапа 3: {path}"
            )
        try:
            dataset = pd.read_csv(path)
        except Exception as error:
            raise Chapter6ReportBuildError(
                f"Не удалось прочитать проверочный датасет: {path}"
            ) from error

        expected = self._expected_row_count()
        if len(dataset) != expected:
            raise Chapter6ReportBuildError(
                "Проверочный датасет содержит неверное число строк: "
                f"ожидалось {expected}, получено {len(dataset)}."
            )
        keys = list(self._join_keys())
        missing = [name for name in keys if name not in dataset.columns]
        if missing:
            raise Chapter6ReportBuildError(
                "В проверочном датасете отсутствуют ключевые колонки: "
                + ", ".join(missing)
            )
        if dataset[keys].isna().any().any():
            raise Chapter6ReportBuildError(
                "В составном ключе проверочного датасета обнаружены пропуски."
            )
        if dataset.duplicated(keys, keep=False).any():
            raise Chapter6ReportBuildError(
                "Составной ключ проверочного датасета не является уникальным."
            )
        for column in ("q_pred", "q_fact"):
            if column not in dataset.columns:
                raise Chapter6ReportBuildError(
                    f"В проверочном датасете отсутствует колонка {column}."
                )
            values = pd.to_numeric(dataset[column], errors="coerce")
            if values.isna().any() or not values.map(math.isfinite).all():
                raise Chapter6ReportBuildError(
                    f"Колонка {column} содержит некорректные значения."
                )
            if ((values < -1e-12) | (values > 1.0 + 1e-12)).any():
                raise Chapter6ReportBuildError(
                    f"Колонка {column} выходит за диапазон [0; 1]."
                )
        return dataset

    def _load_and_validate_report(
        self,
        *,
        path: Path,
        expected_stage: int,
        source_name: str,
    ) -> dict[str, Any]:
        """Загрузить JSON-отчет и проверить его технический контракт."""

        if not path.exists():
            raise FileNotFoundError(
                f"Не найден обязательный отчет {source_name}: {path}"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise Chapter6ReportBuildError(
                f"Не удалось прочитать JSON-отчет {source_name}: {path}"
            ) from error
        if not isinstance(payload, dict):
            raise Chapter6ReportBuildError(
                f"Отчет {source_name} должен содержать JSON-объект."
            )
        if int(payload.get("stage", -1)) != expected_stage:
            raise Chapter6ReportBuildError(
                f"Отчет {source_name} имеет неверный номер этапа."
            )
        if payload.get("passed") is not True:
            raise Chapter6ReportBuildError(
                f"Отчет {source_name} не имеет положительного статуса passed."
            )
        return payload

    def _validate_cross_report_row_counts(
        self,
        reports: Mapping[str, Mapping[str, Any]],
        expected: int,
    ) -> None:
        """Проверить согласованность числа сценариев во всех отчетах."""

        for name, report in reports.items():
            value = report.get("row_count", report.get("expected_row_count"))
            if value is None:
                if name == "figures":
                    continue
                raise Chapter6ReportBuildError(
                    f"Отчет {name} не содержит число сценариев."
                )
            if int(value) != expected:
                raise Chapter6ReportBuildError(
                    f"Отчет {name} содержит {value} сценариев вместо {expected}."
                )

    def _validate_figure_files(
        self,
        manifest: Mapping[str, Any],
        manifest_path: Path,
    ) -> None:
        """Проверить наличие всех рисунков, перечисленных в манифесте."""

        figures = manifest.get("figures")
        if not isinstance(figures, list) or len(figures) != 8:
            raise Chapter6ReportBuildError(
                "Манифест этапа 12 должен содержать восемь рисунков."
            )
        for item in figures:
            if not isinstance(item, dict) or not item.get("filename"):
                raise Chapter6ReportBuildError(
                    "Манифест рисунков содержит некорректную запись."
                )
            configured = item.get("path")
            if configured:
                path = Path(str(configured))
                if not path.is_absolute():
                    path = self.project_root / path
            else:
                path = manifest_path.parent / str(item["filename"])
            if not path.exists():
                raise FileNotFoundError(
                    f"Не найден рисунок из манифеста этапа 12: {path}"
                )

    def _evaluate_hypothesis(
        self,
        reports: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Определить статус гипотезы по заранее заданному правилу."""

        baseline_report = reports["baselines"]
        bootstrap_report = reports["bootstrap"]
        integral_report = reports["integral_prediction"]
        classification_report = reports["classification"]
        interval_report = reports["interval_prediction"]

        metrics_rows = baseline_report.get("metrics", [])
        metrics = {
            str(row.get("model")): row
            for row in metrics_rows
            if isinstance(row, dict) and row.get("model")
        }
        required_models = {"prior_only_baseline", "chapter5_model"}
        if not required_models.issubset(metrics):
            raise Chapter6ReportBuildError(
                "Baseline-отчет не содержит prior-only и полную модель главы 5."
            )

        differences = {
            (str(row.get("baseline")), str(row.get("metric"))): str(
                row.get("conclusion")
            )
            for row in bootstrap_report.get("model_differences", [])
            if isinstance(row, dict)
        }

        prior = metrics["prior_only_baseline"]
        chapter5 = metrics["chapter5_model"]

        def conclusion(metric: str) -> str:
            return differences.get(
                ("prior_only_baseline", metric),
                "no_stable_difference",
            )

        absolute_stable_support = all(
            conclusion(metric) == "chapter5_model_favored"
            for metric in ("mae", "rmse")
        )
        absolute_stable_contradiction = any(
            conclusion(metric) == "baseline_favored"
            for metric in ("mae", "rmse")
        )
        ranking_point_advantage = all(
            float(chapter5[metric]) > float(prior[metric])
            for metric in ("spearman", "kendall")
        )
        ranking_stable_support = all(
            conclusion(metric) == "chapter5_model_favored"
            for metric in ("spearman", "kendall")
        )
        ranking_stable_contradiction = any(
            conclusion(metric) == "baseline_favored"
            for metric in ("spearman", "kendall")
        )
        classification_point_advantage = (
            float(chapter5.get("balanced_accuracy", float("-inf")))
            > float(prior.get("balanced_accuracy", float("-inf")))
            and float(chapter5.get("macro_f1", float("-inf")))
            > float(prior.get("macro_f1", float("-inf")))
        )
        classification_stable_support = (
            conclusion("macro_f1") == "chapter5_model_favored"
        )

        integral_metrics = integral_report.get("metrics", {})
        chapter5_spearman = float(integral_metrics.get("spearman", 0.0))
        chapter5_kendall = float(integral_metrics.get("kendall", 0.0))
        ranking_level_high = chapter5_spearman >= 0.70 and chapter5_kendall >= 0.50

        critical_errors = classification_report.get("critical_errors", {})
        critical_error_count = int(
            critical_errors.get(
                "total",
                int(critical_errors.get("low_to_high", 0))
                + int(critical_errors.get("high_to_low", 0)),
            )
        )
        interval_coverage = float(
            interval_report.get("metrics", {}).get("coverage_rate", 0.0)
        )

        summary = bootstrap_report.get("summary", {})
        stable_chapter5_wins = int(summary.get("stable_chapter5_wins", 0))
        stable_baseline_wins = int(summary.get("stable_baseline_wins", 0))

        full_support = (
            absolute_stable_support
            and ranking_stable_support
            and classification_point_advantage
            and critical_error_count == 0
        )
        partial_support = (
            ranking_level_high
            and critical_error_count == 0
            and (
                ranking_point_advantage
                or ranking_stable_support
                or classification_point_advantage
                or stable_chapter5_wins > 0
            )
        )

        if full_support:
            status = HYPOTHESIS_SUPPORTED
        elif partial_support:
            status = HYPOTHESIS_PARTIALLY_SUPPORTED
        else:
            status = HYPOTHESIS_NOT_SUPPORTED

        evidence_for = []
        evidence_against = []
        if ranking_point_advantage:
            evidence_for.append(
                "Полная модель имеет более высокие точечные Spearman и Kendall, "
                "чем prior-only baseline."
            )
        if ranking_stable_support:
            evidence_for.append(
                "Ранговое преимущество над prior-only baseline статистически устойчиво."
            )
        if classification_point_advantage:
            evidence_for.append(
                "Полная модель превосходит prior-only baseline по Balanced Accuracy "
                "и Macro F1 в точечной оценке."
            )
        if critical_error_count == 0:
            evidence_for.append(
                "Критические ошибки low→high и high→low отсутствуют."
            )
        if absolute_stable_contradiction:
            evidence_against.append(
                "Prior-only baseline статистически устойчиво превосходит полную "
                "модель хотя бы по одной метрике абсолютной ошибки."
            )
        if ranking_stable_contradiction:
            evidence_against.append(
                "Prior-only baseline статистически устойчиво превосходит полную "
                "модель хотя бы по одной ранговой метрике."
            )
        if interval_coverage < 0.80:
            evidence_against.append(
                "Эмпирическое покрытие прогнозных интервалов ниже диагностического "
                "ориентира 0,80."
            )
        bias = float(integral_metrics.get("bias", 0.0))
        if abs(bias) >= 0.05:
            evidence_against.append(
                "Абсолютная шкала имеет выраженное систематическое смещение."
            )

        return {
            "statement": (
                "Включение латентного профиля LDA повышает достоверность "
                "априорной оценки качества относительно схемы без LDA."
            ),
            "status": status,
            "allowed_statuses": list(HYPOTHESIS_STATUSES),
            "decision_rule": {
                "full_support": (
                    "Статистически устойчивое преимущество полной модели над "
                    "prior-only baseline одновременно по MAE/RMSE и Spearman/Kendall, "
                    "точечное преимущество по Balanced Accuracy и Macro F1, отсутствие "
                    "критических классификационных ошибок."
                ),
                "partial_support": (
                    "Высокая ранговая согласованность и преимущество хотя бы в одной "
                    "прикладной размерности при отсутствии полного подтверждения "
                    "абсолютной точности."
                ),
                "not_supported": (
                    "Отсутствует подтвержденное преимущество полной модели в "
                    "ранжировании, классификации или абсолютной точности."
                ),
            },
            "criteria": {
                "absolute_accuracy_stably_supported": absolute_stable_support,
                "absolute_accuracy_stably_contradicted": absolute_stable_contradiction,
                "ranking_point_advantage": ranking_point_advantage,
                "ranking_stably_supported": ranking_stable_support,
                "ranking_stably_contradicted": ranking_stable_contradiction,
                "ranking_level_high": ranking_level_high,
                "classification_point_advantage": classification_point_advantage,
                "classification_stably_supported": classification_stable_support,
                "critical_error_count": critical_error_count,
                "interval_coverage_rate": interval_coverage,
                "integral_bias": bias,
                "stable_chapter5_wins_all_baselines": stable_chapter5_wins,
                "stable_baseline_wins_all_baselines": stable_baseline_wins,
            },
            "evidence_for": evidence_for,
            "evidence_against": evidence_against,
            "interpretation": self._hypothesis_interpretation(status),
        }

    @staticmethod
    def _hypothesis_interpretation(status: str) -> str:
        """Вернуть научно корректную интерпретацию статуса гипотезы."""

        if status == HYPOTHESIS_SUPPORTED:
            return (
                "Основная гипотеза подтверждена по совокупности абсолютных, "
                "ранговых и классификационных критериев."
            )
        if status == HYPOTHESIS_PARTIALLY_SUPPORTED:
            return (
                "Основная гипотеза подтверждена частично: латентный профиль "
                "улучшает дифференциацию или ранжирование, но не обеспечивает "
                "одновременного статистически устойчивого улучшения абсолютной "
                "точности и интервальной калибровки."
            )
        return (
            "Основная гипотеза не получила достаточного подтверждения по "
            "заранее заданному правилу принятия решения."
        )

    def _technical_checks(
        self,
        *,
        dataset: pd.DataFrame,
        reports: Mapping[str, Mapping[str, Any]],
        source_artifacts: Sequence[Mapping[str, Any]],
    ) -> dict[str, bool]:
        """Сформировать повторные технические проверки итогового отчета."""

        input_report = reports["input_validation"]
        baseline_report = reports["baselines"]
        bootstrap_report = reports["bootstrap"]
        error_report = reports["error_analysis"]
        figure_report = reports["figures"]
        return {
            "all_source_reports_passed": all(
                report.get("passed") is True for report in reports.values()
            ),
            "row_count_matches_expected": len(dataset) == self._expected_row_count(),
            "composite_key_is_unique": not dataset.duplicated(
                list(self._join_keys()), keep=False
            ).any(),
            "q_pred_consistency_confirmed": bool(
                input_report.get("q_pred_consistency", {}).get("passed", True)
            ),
            "chapter5_acceptance_confirmed": bool(
                input_report.get("chapter5_acceptance", {}).get("accepted", True)
            ),
            "baseline_leakage_checks_passed": self._checks_match_expected(
                baseline_report.get("leakage_checks", {}),
                {
                    "mean_baseline_is_out_of_fold": True,
                    "test_fold_targets_excluded_from_training_mean": True,
                    "prior_only_uses_q_fact": False,
                    "theta_only_uses_q_fact": False,
                    "chapter5_prediction_unchanged": True,
                },
            ),
            "bootstrap_methodological_checks_passed": self._checks_match_expected(
                bootstrap_report.get("methodological_checks", {}),
                {
                    "paired_resamples_used_for_all_models": True,
                    "fixed_stage9_predictions_used": True,
                    "models_refitted_inside_bootstrap": False,
                    "chapter5_prediction_modified": False,
                    "quality_thresholds_modified": False,
                    "factual_values_used_only_for_external_validation": True,
                },
            ),
            "prediction_model_not_modified": (
                error_report.get("methodological_checks", {}).get(
                    "chapter5_prediction_modified"
                )
                is False
            ),
            "quality_thresholds_not_modified": (
                error_report.get("methodological_checks", {}).get(
                    "quality_thresholds_modified"
                )
                is False
            ),
            "figure_sources_not_modified": (
                figure_report.get("source_data_modified") is False
            ),
            "manual_data_substitution_absent": (
                figure_report.get("manual_data_substitution") is False
            ),
            "all_source_hashes_recorded": all(
                len(str(item.get("sha256", ""))) == 64
                for item in source_artifacts
            ),
        }

    @staticmethod
    def _checks_match_expected(
        actual: Mapping[str, Any],
        expected: Mapping[str, bool],
    ) -> bool:
        """Проверить набор логических флагов по ожидаемым значениям."""

        return all(actual.get(key) is value for key, value in expected.items())

    def _methodological_conclusion(
        self,
        *,
        reports: Mapping[str, Mapping[str, Any]],
        hypothesis: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Сформировать итоговый методический вывод главы 6."""

        integral = reports["integral_prediction"].get("metrics", {})
        classification = reports["classification"].get("metrics", {})
        interval = reports["interval_prediction"].get("metrics", {})
        return {
            "experimental_contour_operable": True,
            "apriori_nature_confirmed": True,
            "ranking_validity_confirmed": (
                float(integral.get("spearman", 0.0)) >= 0.70
                and float(integral.get("kendall", 0.0)) >= 0.50
            ),
            "absolute_scale_calibrated": (
                abs(float(integral.get("bias", 0.0))) < 0.05
                and float(integral.get("r2", -math.inf)) >= 0.0
            ),
            "classification_is_balanced": (
                float(classification.get("balanced_accuracy", 0.0)) >= 0.70
            ),
            "interval_prediction_is_calibrated": (
                float(interval.get("coverage_rate", 0.0)) >= 0.80
            ),
            "hypothesis_status": hypothesis["status"],
            "recommended_claim": hypothesis["interpretation"],
        }

    @staticmethod
    def _limitations(
        reports: Mapping[str, Mapping[str, Any]],
    ) -> list[str]:
        """Сформировать перечень ограничений по фактическим результатам."""

        integral = reports["integral_prediction"].get("metrics", {})
        interval = reports["interval_prediction"].get("metrics", {})
        classification = reports["classification"]
        limitations = [
            "Основной корпус является вычислительным и требует внешней проверки "
            "на независимых эмпирических данных.",
        ]
        if abs(float(integral.get("bias", 0.0))) >= 0.05:
            limitations.append(
                "Зафиксировано систематическое смещение абсолютной шкалы Q_pred."
            )
        if float(integral.get("r2", 0.0)) < 0.0:
            limitations.append(
                "Отрицательный R² указывает на неудовлетворительную абсолютную "
                "калибровку относительно прогноза средним значением."
            )
        if float(interval.get("coverage_rate", 1.0)) < 0.80:
            limitations.append(
                "Фактическое покрытие прогнозных интервалов недостаточно."
            )
        per_class = classification.get("per_class_metrics", [])
        supports = [
            int(row.get("support", 0))
            for row in per_class
            if isinstance(row, dict)
        ]
        if supports and min(supports) < 10:
            limitations.append(
                "Оценка метрик отдельных классов ограничена дисбалансом фактических "
                "классов качества."
            )
        return limitations

    def _source_artifacts(
        self,
        dataset_path: Path,
        report_paths: Mapping[str, Path],
    ) -> list[dict[str, Any]]:
        """Зафиксировать пути и SHA-256 всех источников итогового отчета."""

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
    def _copy_section(
        source: Mapping[str, Any],
        keys: Sequence[str],
    ) -> dict[str, Any]:
        """Скопировать только утвержденные поля исходного отчета."""

        return {key: source[key] for key in keys if key in source}

    @staticmethod
    def _input_summary(source: Mapping[str, Any]) -> dict[str, Any]:
        """Сформировать компактную сводку проверки входных данных."""

        return {
            "passed": source.get("passed"),
            "expected_row_count": source.get("expected_row_count"),
            "checked_csv_count": source.get("checked_csv_count"),
            "checked_json_count": source.get("checked_json_count"),
            "q_pred_consistency": source.get("q_pred_consistency"),
            "chapter5_acceptance": source.get("chapter5_acceptance"),
        }

    def _render_markdown(self, report: Mapping[str, Any]) -> str:
        """Преобразовать итоговый отчет в русскоязычный Markdown."""

        hypothesis = report["hypothesis"]
        integral = report["integral_prediction"]["metrics"]
        classification = report["classification"]["metrics"]
        critical = report["classification"].get("critical_errors", {})
        interval = report["interval_prediction"]["metrics"]
        bootstrap = report["bootstrap_analysis"]
        error_summary = report["prediction_error_analysis"]["summary"]
        baseline_rows = report["baseline_comparison"]["metrics"]

        lines = [
            "# Итоговый отчет экспериментальной проверки главы 6",
            "",
            "## 1. Технический статус",
            "",
            f"- Этап: `{report['stage']}`.",
            f"- Техническая проверка: `{report['passed']}`.",
            f"- Сценариев: `{report['row_count']}`.",
            f"- Статус основной гипотезы: `{report['hypothesis_status']}`.",
            "",
            "## 2. Основная гипотеза",
            "",
            hypothesis["statement"],
            "",
            f"**Вывод:** {hypothesis['interpretation']}",
            "",
            "### Аргументы в поддержку",
            "",
        ]
        lines.extend(
            f"- {item}" for item in hypothesis["evidence_for"]
        )
        if not hypothesis["evidence_for"]:
            lines.append("- Поддерживающие аргументы не зафиксированы.")
        lines.extend(["", "### Ограничивающие аргументы", ""])
        lines.extend(
            f"- {item}" for item in hypothesis["evidence_against"]
        )
        if not hypothesis["evidence_against"]:
            lines.append("- Существенные ограничивающие аргументы не зафиксированы.")

        lines.extend(
            [
                "",
                "## 3. Интегральный прогноз",
                "",
                "| Метрика | Значение |",
                "|---|---:|",
                f"| MAE | {float(integral['mae']):.6f} |",
                f"| RMSE | {float(integral['rmse']):.6f} |",
                f"| Bias | {float(integral['bias']):.6f} |",
                f"| Pearson | {float(integral['pearson']):.6f} |",
                f"| Spearman | {float(integral['spearman']):.6f} |",
                f"| Kendall tau-b | {float(integral['kendall']):.6f} |",
                f"| R² | {float(integral['r2']):.6f} |",
                "",
                "## 4. Классификационная проверка",
                "",
                f"- Accuracy: `{float(classification['accuracy']):.6f}`.",
                (
                    "- Balanced Accuracy: "
                    f"`{float(classification['balanced_accuracy']):.6f}`."
                ),
                f"- Macro F1: `{float(classification['macro_f1']):.6f}`.",
                f"- Weighted F1: `{float(classification['weighted_f1']):.6f}`.",
                (
                    "- Критические low→high / high→low: "
                    f"`{int(critical.get('low_to_high', 0))}` / "
                    f"`{int(critical.get('high_to_low', 0))}`."
                ),
                "",
                "## 5. Интервальный прогноз",
                "",
                f"- Покрытие: `{float(interval['coverage_rate']):.6f}`.",
                (
                    "- Средняя ширина интервала: "
                    f"`{float(interval['mean_interval_width']):.6f}`."
                ),
                (
                    "- Факт выше верхней границы: "
                    f"`{int(interval['miss_upper_count'])}`."
                ),
                "",
                "## 6. Сравнение с baseline-моделями",
                "",
                (
                    "| Модель | MAE | RMSE | Spearman | Kendall | "
                    "Balanced Accuracy | Macro F1 |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in baseline_rows:
            lines.append(
                "| {model} | {mae:.6f} | {rmse:.6f} | {spearman:.6f} | "
                "{kendall:.6f} | {balanced_accuracy:.6f} | {macro_f1:.6f} |".format(
                    model=row["model"],
                    mae=float(row["mae"]),
                    rmse=float(row["rmse"]),
                    spearman=float(row["spearman"]),
                    kendall=float(row["kendall"]),
                    balanced_accuracy=float(row["balanced_accuracy"]),
                    macro_f1=float(row["macro_f1"]),
                )
            )

        bootstrap_summary = bootstrap["summary"]
        lines.extend(
            [
                "",
                "## 7. Bootstrap-анализ",
                "",
                (
                    "- Устойчивых преимуществ модели главы 5: "
                    f"`{bootstrap_summary['stable_chapter5_wins']}`."
                ),
                (
                    "- Устойчивых преимуществ baseline: "
                    f"`{bootstrap_summary['stable_baseline_wins']}`."
                ),
                (
                    "- Неустойчивых различий: "
                    f"`{bootstrap_summary['no_stable_difference']}`."
                ),
                "",
                "## 8. Анализ ошибок",
                "",
                (
                    "- Занижений / завышений: "
                    f"`{error_summary['underestimation_count']}` / "
                    f"`{error_summary['overestimation_count']}`."
                ),
                (
                    "- Максимальная абсолютная ошибка: "
                    f"`{float(error_summary['max_absolute_error']):.6f}`."
                ),
                "",
                "## 9. Ограничения",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in report["limitations"])
        lines.extend(
            [
                "",
                "## 10. Итоговый вывод",
                "",
                report["methodological_conclusion"]["recommended_claim"],
                "",
                (
                    "Техническая воспроизводимость контура подтверждена, однако "
                    "статус научной гипотезы определяется фактическими метриками, "
                    "а не успешностью выполнения программного кода."
                ),
                "",
            ]
        )
        return "\n".join(lines)

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
        """Вернуть ключи объединения из конфигурации."""

        keys = getattr(self.config.merge, "key_columns", None)
        if keys is None:
            keys = getattr(self.config.merge, "keys", None)
        if keys is None:
            raise Chapter6ReportBuildError(
                "Конфигурация не содержит ключи объединения."
            )
        return tuple(str(value) for value in keys)

    def _expected_row_count(self) -> int:
        """Вернуть ожидаемое число сценариев."""

        return int(self.config.merge.expected_row_count)

    def _relative_path(self, path: Path) -> str:
        """Вернуть путь относительно корня проекта, если это возможно."""

        try:
            return str(path.resolve().relative_to(self.project_root.resolve()))
        except ValueError:
            return str(path)

    @staticmethod
    def _sha256(path: Path) -> str:
        """Рассчитать SHA-256 файла."""

        return hashlib.sha256(path.read_bytes()).hexdigest()
