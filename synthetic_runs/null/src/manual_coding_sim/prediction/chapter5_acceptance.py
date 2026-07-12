"""Финальная приемка программного блока главы 5.

Модуль выполняет техническую приемку сохраненных артефактов главы 5 после
построения итогового отчета. Проверяются наличие файлов, согласованность числа
строк, диапазоны прогнозных показателей, интервальные ограничения и признаки
методической безопасности априорного расчета.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from manual_coding_sim.prediction.chapter5_config import Chapter5PredictionConfig
from manual_coding_sim.prediction.paths import resolve_project_path


class Chapter5AcceptanceError(RuntimeError):
    """Ошибка финальной приемки артефактов главы 5."""


@dataclass(frozen=True)
class Chapter5AcceptanceReport:
    """Сводный отчет финальной приемки главы 5."""

    stage: int
    report_type: str
    accepted: bool
    checks: dict[str, bool]
    row_counts: dict[str, int]
    quality_ranges: dict[str, dict[str, float]]
    method_safety: dict[str, object]
    artifact_paths: dict[str, str]
    acceptance_notes: list[str]

    def to_dict(self) -> dict[str, object]:
        """Преобразовать отчет приемки в словарь для JSON-сохранения."""

        return asdict(self)


class Chapter5AcceptanceValidator:
    """Выполняет финальную приемку артефактов главы 5."""

    def validate(
        self,
        *,
        config: Chapter5PredictionConfig,
        project_root: Path,
    ) -> Chapter5AcceptanceReport:
        """Проверить итоговое состояние главы 5 и вернуть отчет приемки."""

        artifacts = self._artifact_paths(config, project_root)
        self._require_artifacts(artifacts)

        prior_features = pd.read_csv(artifacts["prior_features"])
        theta_prior = pd.read_csv(artifacts["theta_prior"])
        normalized = pd.read_csv(artifacts["normalized_prior_features"])
        latent = pd.read_csv(artifacts["latent_quality_component"])
        components = pd.read_csv(artifacts["q_pred_components"])
        q_pred = pd.read_csv(artifacts["q_pred"])
        uncertainty = pd.read_csv(artifacts["prediction_uncertainty"])

        leakage_report = self._read_json(artifacts["leakage_report"])
        pipeline_report = self._read_json(artifacts["pipeline_run_report"])
        final_report = self._read_json(artifacts["final_report_json"])

        row_counts = {
            "prior_features": int(prior_features.shape[0]),
            "theta_prior": int(theta_prior.shape[0]),
            "normalized_prior_features": int(normalized.shape[0]),
            "latent_quality_component": int(latent.shape[0]),
            "q_pred_components": int(components.shape[0]),
            "q_pred": int(q_pred.shape[0]),
            "prediction_uncertainty": int(uncertainty.shape[0]),
            "final_report": int(final_report.get("row_count", -1)),
        }
        expected_row_count = 150
        checks = {
            "artifact_existence": True,
            "prior_features_row_count": row_counts["prior_features"] == expected_row_count,
            "prior_features_has_protocol_id": "protocol_id" in prior_features.columns,
            "prior_features_no_scenario_duplicates": int(
                prior_features["scenario_id"].duplicated().sum()
            )
            == 0,
            "theta_prior_row_count": row_counts["theta_prior"] == expected_row_count,
            "theta_key_alignment": self._keys_are_aligned(prior_features, theta_prior),
            "all_output_row_counts": all(value == expected_row_count for value in row_counts.values()),
            "q_pred_range": self._series_in_range(q_pred["q_pred"]),
            "partial_criteria_range": self._partial_columns_in_range(components),
            "uncertainty_range": self._series_in_range(uncertainty["uncertainty_score"]),
            "interval_radius_range": self._series_in_range(uncertainty["interval_radius"]),
            "interval_order": bool(
                (
                    (uncertainty["q_pred_lower"] <= uncertainty["q_pred"])
                    & (uncertainty["q_pred"] <= uncertainty["q_pred_upper"])
                ).all()
            ),
            "no_missing_core_values": self._no_missing_core_values(
                components,
                q_pred,
                uncertainty,
            ),
            "leakage_check_passed": bool(leakage_report.get("is_safe", False)),
            "no_forbidden_columns": len(leakage_report.get("forbidden_columns", [])) == 0,
            "full_pipeline_completed": all(
                bool(value) for value in pipeline_report.get("completed_steps", {}).values()
            ),
            "final_report_stage": final_report.get("stage") == 11,
            "final_report_apriori_only": bool(
                final_report.get("method_safety", {}).get("apriori_only", False)
            ),
        }
        accepted = all(checks.values())
        method_safety = {
            "apriori_only": bool(final_report.get("method_safety", {}).get("apriori_only", False)),
            "leakage_check_passed": bool(leakage_report.get("is_safe", False)),
            "forbidden_column_count": len(leakage_report.get("forbidden_columns", [])),
            "full_pipeline_completed": checks["full_pipeline_completed"],
        }
        return Chapter5AcceptanceReport(
            stage=12,
            report_type="chapter5_final_acceptance_report",
            accepted=accepted,
            checks=checks,
            row_counts=row_counts,
            quality_ranges={
                "q_pred": self._range(q_pred["q_pred"]),
                "uncertainty_score": self._range(uncertainty["uncertainty_score"]),
                "interval_radius": self._range(uncertainty["interval_radius"]),
            },
            method_safety=method_safety,
            artifact_paths={name: str(path) for name, path in artifacts.items()},
            acceptance_notes=self._acceptance_notes(checks),
        )

    def save_json_report(self, report: Chapter5AcceptanceReport, report_path: Path) -> None:
        """Сохранить отчет приемки в JSON."""

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_markdown_report(self, report: Chapter5AcceptanceReport, report_path: Path) -> None:
        """Сохранить отчет приемки в Markdown."""

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(self.to_markdown(report), encoding="utf-8")

    def save_outputs(
        self,
        report: Chapter5AcceptanceReport,
        *,
        json_report_path: Path,
        markdown_report_path: Path,
    ) -> None:
        """Сохранить оба отчета финальной приемки."""

        self.save_json_report(report, json_report_path)
        self.save_markdown_report(report, markdown_report_path)

    def to_markdown(self, report: Chapter5AcceptanceReport) -> str:
        """Преобразовать отчет приемки в Markdown."""

        status = "пройдена" if report.accepted else "не пройдена"
        lines = [
            "# Финальная приемка главы 5",
            "",
            "## 1. Статус приемки",
            "",
            f"- Статус: **{status}**.",
            f"- Тип отчета: `{report.report_type}`.",
            f"- Этап: `{report.stage}`.",
            "",
            "## 2. Проверки",
            "",
            "| Проверка | Статус |",
            "|---|---:|",
        ]
        for name, value in report.checks.items():
            lines.append(f"| {name} | `{value}` |")
        lines.extend(
            [
                "",
                "## 3. Число строк",
                "",
                "| Артефакт | Строк |",
                "|---|---:|",
            ]
        )
        for name, value in report.row_counts.items():
            lines.append(f"| {name} | `{value}` |")
        lines.extend(
            [
                "",
                "## 4. Диапазоны качества",
                "",
                "| Показатель | min | max |",
                "|---|---:|---:|",
            ]
        )
        for name, value in report.quality_ranges.items():
            lines.append(f"| {name} | `{value['min']:.6f}` | `{value['max']:.6f}` |")
        lines.extend(
            [
                "",
                "## 5. Методическая безопасность",
                "",
                f"- Априорный режим: `{report.method_safety['apriori_only']}`.",
                f"- Проверка утечки: `{report.method_safety['leakage_check_passed']}`.",
                (
                    "- Число запрещенных колонок: "
                    f"`{report.method_safety['forbidden_column_count']}`."
                ),
                (
                    "- Полный контур выполнен: "
                    f"`{report.method_safety['full_pipeline_completed']}`."
                ),
                "",
                "## 6. Заключение",
                "",
            ]
        )
        lines.extend(f"- {note}" for note in report.acceptance_notes)
        lines.append("")
        return "\n".join(lines)

    def _artifact_paths(
        self,
        config: Chapter5PredictionConfig,
        project_root: Path,
    ) -> dict[str, Path]:
        """Вернуть пути к артефактам, проверяемым при приемке."""

        reports_dir = config.outputs.reports_dir
        paths: Mapping[str, Path] = {
            "prior_features": config.inputs.prior_features_path,
            "theta_prior": config.inputs.theta_prior_path,
            "leakage_report": reports_dir / "chapter5_leakage_report.json",
            "normalized_prior_features": config.outputs.normalized_prior_features_path,
            "latent_quality_component": config.outputs.latent_quality_component_path,
            "q_pred_components": config.outputs.q_pred_components_path,
            "q_pred": config.outputs.q_pred_path,
            "prediction_uncertainty": config.outputs.prediction_uncertainty_path,
            "pipeline_run_report": config.outputs.pipeline_run_report_path,
            "final_report_json": config.outputs.report_json_path,
            "final_report_md": config.outputs.report_md_path,
        }
        return {name: resolve_project_path(project_root, path) for name, path in paths.items()}

    def _require_artifacts(self, artifacts: Mapping[str, Path]) -> None:
        """Проверить наличие всех обязательных артефактов приемки."""

        missing = [str(path) for path in artifacts.values() if not path.exists()]
        if missing:
            joined = "; ".join(missing)
            msg = f"Не найдены артефакты для финальной приемки главы 5: {joined}."
            raise Chapter5AcceptanceError(msg)

    def _read_json(self, path: Path) -> dict[str, Any]:
        """Прочитать JSON-артефакт приемки."""

        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            msg = f"JSON-артефакт должен содержать словарь: {path}."
            raise Chapter5AcceptanceError(msg)
        return payload

    def _keys_are_aligned(self, prior_features: pd.DataFrame, theta_prior: pd.DataFrame) -> bool:
        """Проверить совпадение ключей ``prior_features`` и ``theta_prior``."""

        key_columns = ["scenario_id", "protocol_id"]
        if any(column not in prior_features.columns for column in key_columns):
            return False
        if any(column not in theta_prior.columns for column in key_columns):
            return False
        merged = prior_features[key_columns].merge(
            theta_prior[key_columns],
            on=key_columns,
            how="outer",
            indicator=True,
        )
        return bool((merged["_merge"] == "both").all())

    def _series_in_range(self, series: pd.Series, low: float = 0.0, high: float = 1.0) -> bool:
        """Проверить диапазон числовой серии."""

        return bool(series.notna().all() and (series >= low).all() and (series <= high).all())

    def _partial_columns_in_range(self, components: pd.DataFrame) -> bool:
        """Проверить диапазоны всех частных прогнозных критериев."""

        columns = [column for column in components.columns if column.endswith("_pred")]
        if len(columns) != 6:
            return False
        return bool(components[columns].notna().all().all() and self._series_in_range(components[columns].stack()))

    def _no_missing_core_values(
        self,
        components: pd.DataFrame,
        q_pred: pd.DataFrame,
        uncertainty: pd.DataFrame,
    ) -> bool:
        """Проверить отсутствие пропусков в ключевых расчетных колонках."""

        partial_columns = [column for column in components.columns if column.endswith("_pred")]
        uncertainty_columns = ["q_pred", "q_pred_lower", "q_pred_upper"]
        return bool(
            components[partial_columns].isna().sum().sum() == 0
            and q_pred["q_pred"].isna().sum() == 0
            and uncertainty[uncertainty_columns].isna().sum().sum() == 0
        )

    def _range(self, series: pd.Series) -> dict[str, float]:
        """Вернуть минимум и максимум серии."""

        return {"min": float(series.min()), "max": float(series.max())}

    def _acceptance_notes(self, checks: Mapping[str, bool]) -> list[str]:
        """Сформировать текстовое заключение по результату приемки."""

        failed_checks = [name for name, value in checks.items() if not value]
        if not failed_checks:
            return [
                "Все артефакты главы 5 согласованы и готовы к фиксации.",
                "Расчетный контур использует только априорные признаки и LDA_prior.",
                "Финальные pytest-проверки и хеширование выполняются внешней процедурой этапа 12.",
            ]
        joined = ", ".join(failed_checks)
        return [f"Приемка не пройдена; проблемные проверки: {joined}."]
