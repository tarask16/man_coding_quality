"""Формирование итогового отчета главы 5.

Модуль собирает итоговый JSON- и Markdown-отчет по априорному прогнозу
качества. Источниками являются только артефакты расчетного контура главы 5:
проверка утечки, нормировка, латентная компонента, частные критерии,
интегральный ``Q_pred`` и интервальная неопределенность. Фактические значения
качества и целевые метки в отчет не загружаются.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from manual_coding_sim.prediction.chapter5_config import (
    Chapter5PredictionConfig,
    QUALITY_CRITERIA,
)
from manual_coding_sim.prediction.paths import resolve_project_path


class Chapter5ReportBuildError(RuntimeError):
    """Ошибка формирования итогового отчета главы 5."""


@dataclass(frozen=True)
class Chapter5FinalReport:
    """JSON-совместимое представление итогового отчета главы 5."""

    stage: int
    report_type: str
    row_count: int
    source_artifacts: dict[str, str]
    method_safety: dict[str, object]
    normalization_summary: dict[str, object]
    latent_quality_summary: dict[str, object]
    partial_criteria_summary: dict[str, object]
    integral_quality_summary: dict[str, object]
    uncertainty_summary: dict[str, object]
    quality_class_counts: dict[str, int]
    conclusion: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Преобразовать отчет в обычный словарь для JSON-сохранения."""

        return asdict(self)


class Chapter5ReportBuilder:
    """Формирует итоговые JSON- и Markdown-отчеты главы 5."""

    def build_report(
        self,
        *,
        config: Chapter5PredictionConfig,
        project_root: Path,
    ) -> Chapter5FinalReport:
        """Собрать итоговый отчет из сохраненных артефактов главы 5."""

        artifacts = self._artifact_paths(config, project_root)
        self._require_artifacts(artifacts)

        q_pred = pd.read_csv(artifacts["q_pred"])
        components = pd.read_csv(artifacts["q_pred_components"])
        uncertainty = pd.read_csv(artifacts["prediction_uncertainty"])

        q_pred_report = self._read_json(artifacts["q_pred_report"])
        components_report = self._read_json(artifacts["q_pred_components_report"])
        uncertainty_report = self._read_json(artifacts["prediction_uncertainty_report"])
        normalization_report = self._read_json(artifacts["normalization_report"])
        latent_report = self._read_json(artifacts["latent_quality_component_report"])
        leakage_report = self._read_json(artifacts["leakage_report"])
        pipeline_report = self._read_json(artifacts["pipeline_run_report"])

        self._validate_row_alignment(q_pred, components, uncertainty)
        quality_class_counts = self._quality_class_counts(q_pred, config)

        return Chapter5FinalReport(
            stage=11,
            report_type="chapter5_prediction_final_report",
            row_count=int(q_pred.shape[0]),
            source_artifacts={name: str(path) for name, path in artifacts.items()},
            method_safety=self._method_safety_summary(leakage_report, pipeline_report),
            normalization_summary=self._normalization_summary(normalization_report),
            latent_quality_summary=self._latent_summary(latent_report),
            partial_criteria_summary=self._partial_summary(components, components_report),
            integral_quality_summary=self._integral_summary(q_pred_report),
            uncertainty_summary=self._uncertainty_summary(uncertainty_report),
            quality_class_counts=quality_class_counts,
            conclusion=self._conclusion(q_pred_report, uncertainty_report, quality_class_counts),
        )

    def save_json_report(self, report: Chapter5FinalReport, report_path: Path) -> None:
        """Сохранить итоговый отчет главы 5 в JSON."""

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_markdown_report(self, report: Chapter5FinalReport, report_path: Path) -> None:
        """Сохранить итоговый отчет главы 5 в Markdown."""

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(self.to_markdown(report), encoding="utf-8")

    def save_outputs(
        self,
        report: Chapter5FinalReport,
        *,
        json_report_path: Path,
        markdown_report_path: Path,
    ) -> None:
        """Сохранить итоговый JSON- и Markdown-отчет."""

        self.save_json_report(report, json_report_path)
        self.save_markdown_report(report, markdown_report_path)

    def to_markdown(self, report: Chapter5FinalReport) -> str:
        """Преобразовать итоговый отчет в человекочитаемый Markdown."""

        q_summary = report.integral_quality_summary
        u_summary = report.uncertainty_summary
        latent = report.latent_quality_summary
        norm = report.normalization_summary
        safety = report.method_safety
        class_counts = report.quality_class_counts
        partial = report.partial_criteria_summary
        conclusion = report.conclusion

        lines = [
            "# Итоговый отчет главы 5",
            "",
            "## 1. Назначение отчета",
            "",
            (
                "Отчет фиксирует результат априорного прогнозирования качества "
                "ручного кодирования на основе априорных признаков и латентного "
                "профиля LDA_prior. Фактические критерии качества при расчете не "
                "используются."
            ),
            "",
            "## 2. Методическая безопасность",
            "",
            f"- Априорный режим расчета: `{safety['apriori_only']}`.",
            f"- Проверка утечки пройдена: `{safety['leakage_check_passed']}`.",
            f"- Запрещенных колонок обнаружено: `{safety['forbidden_column_count']}`.",
            f"- Полный CLI-контур выполнен: `{safety['full_pipeline_completed']}`.",
            "",
            "## 3. Нормировка априорных признаков",
            "",
            f"- Строк входной таблицы: `{norm['row_count']}`.",
            f"- Нормированных числовых признаков: `{norm['normalized_feature_count']}`.",
            f"- Пропущенных нечисловых признаков: `{norm['skipped_feature_count']}`.",
            f"- Неизвестных входных признаков: `{len(norm['unknown_input_features'])}`.",
            "",
            "## 4. Латентная компонента качества",
            "",
            f"- Строк: `{latent['row_count']}`.",
            f"- q_latent min: `{latent['q_latent_min']:.6f}`.",
            f"- q_latent max: `{latent['q_latent_max']:.6f}`.",
            f"- q_latent mean: `{latent['q_latent_mean']:.6f}`.",
            f"- Доминирующие факторы: `{latent['dominant_topic_counts']}`.",
            "",
            "## 5. Частные прогнозные критерии",
            "",
            f"- Строк: `{partial['row_count']}`.",
            f"- Критериев: `{partial['criteria']}`.",
            "",
            "| Критерий | min | max | mean |",
            "|---|---:|---:|---:|",
        ]
        for criterion, values in partial["criterion_ranges"].items():
            lines.append(
                f"| {criterion} | {values['min']:.6f} | {values['max']:.6f} | {values['mean']:.6f} |"
            )

        lines.extend(
            [
                "",
                "## 6. Интегральный прогнозный показатель Q_pred",
                "",
                f"- Строк: `{q_summary['row_count']}`.",
                f"- Сумма весов частных критериев: `{q_summary['weight_sum']:.6f}`.",
                f"- Q_pred min: `{q_summary['q_pred_min']:.6f}`.",
                f"- Q_pred max: `{q_summary['q_pred_max']:.6f}`.",
                f"- Q_pred mean: `{q_summary['q_pred_mean']:.6f}`.",
                f"- Q_pred std: `{q_summary['q_pred_std']:.6f}`.",
                "",
                "## 7. Классы прогнозного качества",
                "",
                f"- Низкое качество: `{class_counts['low']}`.",
                f"- Среднее качество: `{class_counts['medium']}`.",
                f"- Высокое качество: `{class_counts['high']}`.",
                "",
                "## 8. Неопределенность прогноза",
                "",
                f"- uncertainty_score min: `{u_summary['uncertainty_score_min']:.6f}`.",
                f"- uncertainty_score max: `{u_summary['uncertainty_score_max']:.6f}`.",
                f"- uncertainty_score mean: `{u_summary['uncertainty_score_mean']:.6f}`.",
                f"- interval_radius min: `{u_summary['interval_radius_min']:.6f}`.",
                f"- interval_radius max: `{u_summary['interval_radius_max']:.6f}`.",
                f"- interval_radius mean: `{u_summary['interval_radius_mean']:.6f}`.",
                "",
                "## 9. Заключение",
                "",
                f"- Основной вывод: {conclusion['summary']}.",
                f"- Рекомендованное использование: {conclusion['recommended_use']}.",
                "",
            ]
        )
        return "\n".join(lines)

    def _artifact_paths(
        self,
        config: Chapter5PredictionConfig,
        project_root: Path,
    ) -> dict[str, Path]:
        """Вернуть пути к артефактам, необходимым для итогового отчета."""

        paths: Mapping[str, Path] = {
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
        return {name: resolve_project_path(project_root, path) for name, path in paths.items()}

    def _require_artifacts(self, artifacts: Mapping[str, Path]) -> None:
        """Проверить наличие всех входных артефактов итогового отчета."""

        missing = [str(path) for path in artifacts.values() if not path.exists()]
        if missing:
            joined = "; ".join(missing)
            msg = f"Не найдены артефакты для итогового отчета главы 5: {joined}."
            raise Chapter5ReportBuildError(msg)

    def _read_json(self, path: Path) -> dict[str, Any]:
        """Прочитать JSON-артефакт как словарь."""

        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            msg = f"JSON-артефакт должен содержать словарь: {path}."
            raise Chapter5ReportBuildError(msg)
        return payload

    def _validate_row_alignment(
        self,
        q_pred: pd.DataFrame,
        components: pd.DataFrame,
        uncertainty: pd.DataFrame,
    ) -> None:
        """Проверить согласованность числа строк ключевых таблиц."""

        row_counts = {
            "q_pred": int(q_pred.shape[0]),
            "q_pred_components": int(components.shape[0]),
            "prediction_uncertainty": int(uncertainty.shape[0]),
        }
        if len(set(row_counts.values())) != 1:
            msg = f"Ключевые таблицы главы 5 имеют разное число строк: {row_counts}."
            raise Chapter5ReportBuildError(msg)
        if "q_pred" not in q_pred.columns:
            msg = "В q_pred.csv отсутствует обязательная колонка q_pred."
            raise Chapter5ReportBuildError(msg)
        if not q_pred["q_pred"].between(0.0, 1.0).all():
            msg = "Колонка q_pred содержит значения вне диапазона [0, 1]."
            raise Chapter5ReportBuildError(msg)

    def _quality_class_counts(
        self,
        q_pred: pd.DataFrame,
        config: Chapter5PredictionConfig,
    ) -> dict[str, int]:
        """Посчитать распределение сценариев по прогнозным классам качества."""

        low_max = config.decision_thresholds.low_max
        high_min = config.decision_thresholds.high_min
        low_count = int((q_pred["q_pred"] <= low_max).sum())
        high_count = int((q_pred["q_pred"] >= high_min).sum())
        medium_count = int(q_pred.shape[0] - low_count - high_count)
        return {"low": low_count, "medium": medium_count, "high": high_count}

    def _method_safety_summary(
        self,
        leakage_report: Mapping[str, Any],
        pipeline_report: Mapping[str, Any],
    ) -> dict[str, object]:
        """Сформировать сводку методической безопасности расчета."""

        completed_steps = pipeline_report.get("completed_steps", {})
        return {
            "apriori_only": True,
            "leakage_check_passed": bool(leakage_report.get("is_safe", False)),
            "forbidden_column_count": len(leakage_report.get("forbidden_columns", [])),
            "forbidden_columns": list(leakage_report.get("forbidden_columns", [])),
            "full_pipeline_completed": bool(completed_steps) and all(completed_steps.values()),
        }

    def _normalization_summary(self, report: Mapping[str, Any]) -> dict[str, object]:
        """Сформировать сводку нормировки априорных признаков."""

        return {
            "row_count": int(report["row_count"]),
            "input_column_count": int(report["input_column_count"]),
            "normalized_feature_count": int(report["normalized_feature_count"]),
            "skipped_feature_count": int(report["skipped_feature_count"]),
            "non_numeric_features": list(report.get("non_numeric_features", [])),
            "unknown_input_features": list(report.get("unknown_input_features", [])),
            "missing_dictionary_features": list(report.get("missing_dictionary_features", [])),
        }

    def _latent_summary(self, report: Mapping[str, Any]) -> dict[str, object]:
        """Сформировать сводку латентной компоненты качества."""

        return {
            "row_count": int(report["row_count"]),
            "theta_columns": list(report["theta_columns"]),
            "factor_directions": dict(report["factor_directions"]),
            "q_latent_min": float(report["q_latent_min"]),
            "q_latent_max": float(report["q_latent_max"]),
            "q_latent_mean": float(report["q_latent_mean"]),
            "dominant_topic_counts": dict(report["dominant_topic_counts"]),
        }

    def _partial_summary(
        self,
        components: pd.DataFrame,
        report: Mapping[str, Any],
    ) -> dict[str, object]:
        """Сформировать сводку частных прогнозных критериев."""

        ranges: dict[str, dict[str, float]] = {}
        for criterion in QUALITY_CRITERIA:
            column = f"{criterion}_pred"
            if column not in components.columns:
                msg = f"В q_pred_components.csv отсутствует колонка {column}."
                raise Chapter5ReportBuildError(msg)
            ranges[criterion] = {
                "min": float(components[column].min()),
                "max": float(components[column].max()),
                "mean": float(components[column].mean()),
            }
        return {
            "row_count": int(report["row_count"]),
            "criteria": list(report["criteria"]),
            "criterion_ranges": ranges,
        }

    def _integral_summary(self, report: Mapping[str, Any]) -> dict[str, object]:
        """Сформировать сводку интегрального прогноза качества."""

        return {
            "row_count": int(report["row_count"]),
            "criteria": list(report["criteria"]),
            "weights": dict(report["weights"]),
            "weight_sum": float(report["weight_sum"]),
            "q_pred_min": float(report["q_pred_min"]),
            "q_pred_max": float(report["q_pred_max"]),
            "q_pred_mean": float(report["q_pred_mean"]),
            "q_pred_std": float(report["q_pred_std"]),
        }

    def _uncertainty_summary(self, report: Mapping[str, Any]) -> dict[str, object]:
        """Сформировать сводку интервальной неопределенности прогноза."""

        return {
            "row_count": int(report["row_count"]),
            "weight_sum": float(report["weight_sum"]),
            "delta": float(report["delta"]),
            "mean_stability": float(report["mean_stability"]),
            "input_missing_share": float(report["input_missing_share"]),
            "uncertainty_score_min": float(report["uncertainty_score_min"]),
            "uncertainty_score_max": float(report["uncertainty_score_max"]),
            "uncertainty_score_mean": float(report["uncertainty_score_mean"]),
            "interval_radius_min": float(report["interval_radius_min"]),
            "interval_radius_max": float(report["interval_radius_max"]),
            "interval_radius_mean": float(report["interval_radius_mean"]),
            "q_pred_lower_min": float(report["q_pred_lower_min"]),
            "q_pred_upper_max": float(report["q_pred_upper_max"]),
        }

    def _conclusion(
        self,
        q_pred_report: Mapping[str, Any],
        uncertainty_report: Mapping[str, Any],
        quality_class_counts: Mapping[str, int],
    ) -> dict[str, object]:
        """Сформировать краткий вывод итогового отчета."""

        q_min = float(q_pred_report["q_pred_min"])
        q_max = float(q_pred_report["q_pred_max"])
        q_mean = float(q_pred_report["q_pred_mean"])
        u_mean = float(uncertainty_report["uncertainty_score_mean"])
        summary = (
            "Априорный прогноз рассчитан для всех сценариев; значения Q_pred "
            f"находятся в диапазоне [{q_min:.6f}; {q_max:.6f}], среднее значение "
            f"равно {q_mean:.6f}."
        )
        return {
            "summary": summary,
            "recommended_use": (
                "ранжирование альтернатив ручного кодирования до получения "
                "фактических результатов и выделение сценариев с повышенной "
                "неопределенностью"
            ),
            "mean_uncertainty_score": u_mean,
            "quality_class_counts": dict(quality_class_counts),
        }
