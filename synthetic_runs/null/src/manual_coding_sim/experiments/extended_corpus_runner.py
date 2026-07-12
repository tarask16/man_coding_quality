"""Генерация расширенного экспериментального корпуса главы 3.

Runner создает стандартные CSV-артефакты главы 3, которые уже используются
модулем главы 4: ``protocols.csv``, ``prior_features.csv``,
``diagnostic_features.csv``, ``fact_features.csv``, ``quality_targets.csv`` и
``all_features.csv``. Значения формируются из параметров сценариев, а не путем
копирования существующих строк.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from manual_coding_sim.experiments.extended_corpus_plan import (
    ExtendedCorpusPlanBuilder,
    ExtendedCorpusPlanConfig,
    ExtendedScenario,
)


@dataclass(frozen=True)
class ExtendedCorpusOutputPaths:
    """Пути выходных CSV-артефактов расширенного корпуса."""

    data_dir: Path = Path("data/processed")
    reports_dir: Path = Path("reports/chapter3")
    protocols_path: Path = Path("data/processed/protocols.csv")
    prior_features_path: Path = Path("data/processed/prior_features.csv")
    diagnostic_features_path: Path = Path("data/processed/diagnostic_features.csv")
    fact_features_path: Path = Path("data/processed/fact_features.csv")
    quality_targets_path: Path = Path("data/processed/quality_targets.csv")
    all_features_path: Path = Path("data/processed/all_features.csv")
    summary_json_path: Path = Path("reports/chapter3/extended_corpus_summary.json")
    summary_md_path: Path = Path("reports/chapter3/extended_corpus_summary.md")


@dataclass(frozen=True)
class ExtendedCorpusRunnerConfig:
    """Конфигурация генерации расширенного корпуса главы 3."""

    plan: ExtendedCorpusPlanConfig = ExtendedCorpusPlanConfig()
    outputs: ExtendedCorpusOutputPaths = ExtendedCorpusOutputPaths()
    overwrite: bool = True
    encoding: str = "utf-8"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        """Проверить корректность конфигурации runner-а."""

        self.plan.validate()
        if not self.encoding:
            msg = "encoding не должен быть пустым."
            raise ValueError(msg)


@dataclass(frozen=True)
class ExtendedCorpusGenerationResult:
    """Результат генерации расширенного корпуса."""

    document_count: int
    unique_scenario_count: int
    unique_protocol_count: int
    prior_feature_count: int
    diagnostic_feature_count: int
    fact_feature_count: int
    quality_target_count: int
    protocols_path: Path
    prior_features_path: Path
    diagnostic_features_path: Path
    fact_features_path: Path
    quality_targets_path: Path
    all_features_path: Path
    summary_json_path: Path
    summary_md_path: Path

    def to_dict(self) -> dict[str, object]:
        """Преобразовать результат в JSON-совместимый словарь."""

        return {
            "document_count": self.document_count,
            "unique_scenario_count": self.unique_scenario_count,
            "unique_protocol_count": self.unique_protocol_count,
            "prior_feature_count": self.prior_feature_count,
            "diagnostic_feature_count": self.diagnostic_feature_count,
            "fact_feature_count": self.fact_feature_count,
            "quality_target_count": self.quality_target_count,
            "artifacts": {
                "protocols": str(self.protocols_path),
                "prior_features": str(self.prior_features_path),
                "diagnostic_features": str(self.diagnostic_features_path),
                "fact_features": str(self.fact_features_path),
                "quality_targets": str(self.quality_targets_path),
                "all_features": str(self.all_features_path),
                "summary_json": str(self.summary_json_path),
                "summary_md": str(self.summary_md_path),
            },
        }


class ExtendedCorpusRunner:
    """Создает расширенный корпус главы 3 в стандартной файловой структуре."""

    def __init__(self, config: ExtendedCorpusRunnerConfig | None = None) -> None:
        """Создать runner генерации расширенного корпуса."""

        self.config = config or ExtendedCorpusRunnerConfig()
        self.config.validate()

    def run(self, project_root: str | Path = ".") -> ExtendedCorpusGenerationResult:
        """Сформировать расширенный корпус и сохранить CSV-артефакты."""

        root = Path(project_root)
        scenarios = ExtendedCorpusPlanBuilder(self.config.plan).build()
        outputs = self._resolve_outputs(root)
        self._ensure_can_write(outputs)

        protocol_rows = [self._protocol_row(scenario) for scenario in scenarios]
        prior_rows = [self._prior_row(scenario) for scenario in scenarios]
        diagnostic_rows = [self._diagnostic_row(scenario) for scenario in scenarios]
        fact_rows = [self._fact_row(scenario) for scenario in scenarios]
        quality_rows = [self._quality_row(scenario, fact_rows[index]) for index, scenario in enumerate(scenarios)]
        all_rows = self._merge_all_rows(
            prior_rows=prior_rows,
            diagnostic_rows=diagnostic_rows,
            fact_rows=fact_rows,
            quality_rows=quality_rows,
        )

        self._write_csv(outputs.protocols_path, protocol_rows)
        self._write_csv(outputs.prior_features_path, prior_rows)
        self._write_csv(outputs.diagnostic_features_path, diagnostic_rows)
        self._write_csv(outputs.fact_features_path, fact_rows)
        self._write_csv(outputs.quality_targets_path, quality_rows)
        self._write_csv(outputs.all_features_path, all_rows)

        result = ExtendedCorpusGenerationResult(
            document_count=len(scenarios),
            unique_scenario_count=ExtendedCorpusPlanBuilder.unique_scenario_count(scenarios),
            unique_protocol_count=ExtendedCorpusPlanBuilder.unique_protocol_count(scenarios),
            prior_feature_count=len(prior_rows[0]) - 4,
            diagnostic_feature_count=len(diagnostic_rows[0]) - 3,
            fact_feature_count=len(fact_rows[0]) - 3,
            quality_target_count=len(quality_rows[0]) - 3,
            protocols_path=outputs.protocols_path,
            prior_features_path=outputs.prior_features_path,
            diagnostic_features_path=outputs.diagnostic_features_path,
            fact_features_path=outputs.fact_features_path,
            quality_targets_path=outputs.quality_targets_path,
            all_features_path=outputs.all_features_path,
            summary_json_path=outputs.summary_json_path,
            summary_md_path=outputs.summary_md_path,
        )
        self._write_json(outputs.summary_json_path, self._summary_payload(result, scenarios))
        self._write_summary_md(outputs.summary_md_path, result)
        return result

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "ExtendedCorpusRunner":
        """Создать runner из YAML-конфигурации."""

        path = Path(config_path)
        with path.open("r", encoding="utf-8") as file_obj:
            payload = yaml.safe_load(file_obj) or {}
        if not isinstance(payload, dict):
            msg = "YAML-конфигурация расширенного корпуса должна быть словарем."
            raise ValueError(msg)

        section = payload.get("extended_corpus", payload)
        if not isinstance(section, dict):
            msg = "Секция extended_corpus должна быть словарем."
            raise ValueError(msg)

        plan_section = section.get("plan", {}) or {}
        output_section = section.get("output", {}) or {}
        if not isinstance(plan_section, dict) or not isinstance(output_section, dict):
            msg = "Секции plan и output должны быть словарями."
            raise ValueError(msg)

        data_dir = Path(str(output_section.get("data_dir", "data/processed")))
        reports_dir = Path(str(output_section.get("reports_dir", "reports/chapter3")))
        outputs = ExtendedCorpusOutputPaths(
            data_dir=data_dir,
            reports_dir=reports_dir,
            protocols_path=Path(str(output_section.get("protocols", data_dir / "protocols.csv"))),
            prior_features_path=Path(str(output_section.get("prior_features", data_dir / "prior_features.csv"))),
            diagnostic_features_path=Path(str(output_section.get("diagnostic_features", data_dir / "diagnostic_features.csv"))),
            fact_features_path=Path(str(output_section.get("fact_features", data_dir / "fact_features.csv"))),
            quality_targets_path=Path(str(output_section.get("quality_targets", data_dir / "quality_targets.csv"))),
            all_features_path=Path(str(output_section.get("all_features", data_dir / "all_features.csv"))),
            summary_json_path=Path(str(output_section.get("summary_json", reports_dir / "extended_corpus_summary.json"))),
            summary_md_path=Path(str(output_section.get("summary_md", reports_dir / "extended_corpus_summary.md"))),
        )
        config = ExtendedCorpusRunnerConfig(
            plan=ExtendedCorpusPlanConfig(
                document_count=int(plan_section.get("document_count", 150)),
                random_seed=int(plan_section.get("random_seed", 20260709)),
                protocols_per_scenario=int(plan_section.get("protocols_per_scenario", 1)),
            ),
            outputs=outputs,
            overwrite=bool(section.get("overwrite", True)),
            metadata=section.get("metadata", {}) if isinstance(section.get("metadata", {}), dict) else {},
        )
        return cls(config)

    def _resolve_outputs(self, project_root: Path) -> ExtendedCorpusOutputPaths:
        """Разрешить относительные пути относительно корня проекта."""

        outputs = self.config.outputs
        return ExtendedCorpusOutputPaths(
            data_dir=self._resolve_path(project_root, outputs.data_dir),
            reports_dir=self._resolve_path(project_root, outputs.reports_dir),
            protocols_path=self._resolve_path(project_root, outputs.protocols_path),
            prior_features_path=self._resolve_path(project_root, outputs.prior_features_path),
            diagnostic_features_path=self._resolve_path(project_root, outputs.diagnostic_features_path),
            fact_features_path=self._resolve_path(project_root, outputs.fact_features_path),
            quality_targets_path=self._resolve_path(project_root, outputs.quality_targets_path),
            all_features_path=self._resolve_path(project_root, outputs.all_features_path),
            summary_json_path=self._resolve_path(project_root, outputs.summary_json_path),
            summary_md_path=self._resolve_path(project_root, outputs.summary_md_path),
        )

    def _resolve_path(self, project_root: Path, path: Path) -> Path:
        """Вернуть абсолютный путь или путь относительно корня проекта."""

        if path.is_absolute():
            return path
        return project_root / path

    def _protocol_row(self, scenario: ExtendedScenario) -> dict[str, object]:
        """Сформировать строку протокола применения."""

        return {
            "run_id": scenario.run_id,
            "scenario_id": scenario.scenario_id,
            "protocol_id": scenario.protocol_id,
            "alternative_id": scenario.alternative_id,
            "coding_tool_type": scenario.coding_tool_type,
            "message_length": scenario.message_length,
            "message_complexity": scenario.message_complexity,
            "message_criticality": scenario.message_criticality,
            "condition_profile": scenario.condition_profile,
            "protocol_status": "generated_extended",
        }

    def _prior_row(self, scenario: ExtendedScenario) -> dict[str, object]:
        """Сформировать априорные признаки ``X_prior`` для одной строки."""

        total_nominal_time = self._nominal_time(scenario)
        operator_time = self._operator_estimated_time(scenario, total_nominal_time)
        condition_time = self._condition_adjusted_time(scenario, operator_time)
        expected_error = self._expected_error_probability(scenario)
        return {
            "run_id": scenario.run_id,
            "protocol_id": scenario.protocol_id,
            "scenario_id": scenario.scenario_id,
            "alternative_id": scenario.alternative_id,
            "prior_message_length": scenario.message_length,
            "prior_mean_complexity": scenario.message_complexity,
            "prior_mean_message_criticality": scenario.message_criticality,
            "prior_digit_ratio": scenario.digit_ratio,
            "prior_control_marker_ratio": scenario.control_marker_ratio,
            "prior_procedure_steps": scenario.procedure_steps,
            "prior_total_nominal_time": round(total_nominal_time, 4),
            "prior_operator_skill": scenario.operator_skill,
            "prior_operator_fatigue": scenario.operator_fatigue,
            "prior_operator_attention": scenario.operator_attention,
            "prior_operator_total_estimated_time": round(operator_time, 4),
            "prior_condition_noise_level": scenario.noise_level,
            "prior_condition_time_pressure": scenario.time_pressure,
            "prior_condition_mean_adjusted_attention": round(self._adjusted_attention(scenario), 4),
            "prior_condition_total_adjusted_time": round(condition_time, 4),
            "prior_control_intensity": scenario.control_intensity,
            "prior_expected_error_probability": round(expected_error, 6),
            "prior_coding_tool_type": scenario.coding_tool_type,
            "prior_condition_profile": scenario.condition_profile,
            "prior_symbol_group_count": self._symbol_group_count(scenario),
            "prior_alpha_ratio": round(max(0.0, 1.0 - scenario.digit_ratio - scenario.control_marker_ratio), 4),
            "prior_message_entropy_estimate": round(self._message_entropy_estimate(scenario), 4),
            "prior_operator_error_susceptibility": round(self._operator_error_susceptibility(scenario), 4),
            "prior_control_expected_detection_rate": round(self._control_detection_rate(scenario), 4),
            "prior_procedure_branch_count": self._procedure_branch_count(scenario),
            "prior_memory_load_index": round(self._memory_load_index(scenario), 4),
            "prior_time_pressure_index": round(scenario.time_pressure / 3, 4),
            "prior_condition_noise_adjustment": round(1.0 + scenario.noise_level * 0.12, 4),
            "prior_attention_deficit": round(max(0.0, 5.0 - self._adjusted_attention(scenario)), 4),
            "prior_manual_operation_count": self._manual_operation_count(scenario),
            "prior_verification_required": 1 if scenario.message_criticality >= 4 else 0,
            "prior_repetition_expected_count": max(0, scenario.operator_fatigue + scenario.time_pressure - 3),
        }

    def _diagnostic_row(self, scenario: ExtendedScenario) -> dict[str, object]:
        """Сформировать диагностические признаки без целевых показателей качества."""

        expected_error = self._expected_error_probability(scenario)
        return {
            "run_id": scenario.run_id,
            "protocol_id": scenario.protocol_id,
            "scenario_id": scenario.scenario_id,
            "diagnostic_expected_error_probability": round(expected_error, 6),
            "diagnostic_control_load": round(scenario.control_intensity * scenario.procedure_steps / 10, 4),
            "diagnostic_operator_stress": round((scenario.operator_fatigue + scenario.time_pressure + scenario.noise_level) / 12, 4),
            "diagnostic_risk_class": self._risk_class(expected_error),
            "diagnostic_condition_profile": scenario.condition_profile,
        }

    def _fact_row(self, scenario: ExtendedScenario) -> dict[str, object]:
        """Сформировать постфактум-признаки результата выполнения."""

        expected_error = self._expected_error_probability(scenario)
        predicted_errors = max(0, int(round(expected_error * scenario.message_length / 2)))
        duration = self._condition_adjusted_time(scenario, self._operator_estimated_time(scenario, self._nominal_time(scenario)))
        reject_count = 1 if expected_error > 0.18 else 0
        return {
            "run_id": scenario.run_id,
            "protocol_id": scenario.protocol_id,
            "scenario_id": scenario.scenario_id,
            "fact_error_count": predicted_errors,
            "fact_duration_sec": round(duration * (1.0 + expected_error), 4),
            "fact_recheck_count": max(0, scenario.control_intensity - 1),
            "fact_reject_count": reject_count,
            "fact_success": 0 if predicted_errors > 3 or reject_count else 1,
        }

    def _quality_row(
        self,
        scenario: ExtendedScenario,
        fact_row: Mapping[str, object],
    ) -> dict[str, object]:
        """Сформировать целевые показатели качества для глав 5–6."""

        error_count = float(fact_row["fact_error_count"])
        duration = float(fact_row["fact_duration_sec"])
        nominal = self._nominal_time(scenario)
        q_acc = max(0.0, 1.0 - error_count / max(1.0, scenario.message_length / 4))
        q_time = max(0.0, min(1.0, nominal / max(nominal, duration)))
        q_effort = max(0.0, 1.0 - (scenario.procedure_steps + scenario.control_intensity) / 18)
        q_res = 1.0 if int(fact_row["fact_success"]) == 1 else 0.35
        q_rep = max(0.0, 1.0 - scenario.operator_fatigue / 6)
        q_fit = max(0.0, 1.0 - self._expected_error_probability(scenario))
        integral = (0.30 * q_acc) + (0.20 * q_time) + (0.15 * q_effort) + (0.15 * q_res) + (0.10 * q_rep) + (0.10 * q_fit)
        return {
            "run_id": scenario.run_id,
            "protocol_id": scenario.protocol_id,
            "scenario_id": scenario.scenario_id,
            "q_acc": round(q_acc, 6),
            "q_time": round(q_time, 6),
            "q_effort": round(q_effort, 6),
            "q_res": round(q_res, 6),
            "q_rep": round(q_rep, 6),
            "q_fit": round(q_fit, 6),
            "integral_quality": round(integral, 6),
        }

    def _merge_all_rows(
        self,
        prior_rows: Sequence[Mapping[str, object]],
        diagnostic_rows: Sequence[Mapping[str, object]],
        fact_rows: Sequence[Mapping[str, object]],
        quality_rows: Sequence[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        """Сформировать объединенный датасет для аудита и последующих глав."""

        merged: list[dict[str, object]] = []
        for prior_row, diagnostic_row, fact_row, quality_row in zip(
            prior_rows,
            diagnostic_rows,
            fact_rows,
            quality_rows,
            strict=True,
        ):
            row = dict(prior_row)
            row.update({key: value for key, value in diagnostic_row.items() if key not in row})
            row.update({key: value for key, value in fact_row.items() if key not in row})
            row.update({key: value for key, value in quality_row.items() if key not in row})
            merged.append(row)
        return merged

    def _symbol_group_count(self, scenario: ExtendedScenario) -> int:
        """Оценить число групп символов в сообщении."""

        return max(1, int(round(scenario.message_length / 4)))

    def _message_entropy_estimate(self, scenario: ExtendedScenario) -> float:
        """Рассчитать априорную оценку информационной неоднородности сообщения."""

        return (
            1.0
            + scenario.message_complexity * 0.35
            + scenario.digit_ratio * 0.80
            + scenario.control_marker_ratio * 0.55
        )

    def _operator_error_susceptibility(self, scenario: ExtendedScenario) -> float:
        """Рассчитать априорную восприимчивость оператора к ошибкам."""

        susceptibility = (
            0.20
            + scenario.operator_fatigue * 0.12
            + scenario.time_pressure * 0.10
            + scenario.noise_level * 0.08
            - scenario.operator_skill * 0.06
        )
        return max(0.0, min(1.0, susceptibility))

    def _control_detection_rate(self, scenario: ExtendedScenario) -> float:
        """Оценить вероятность обнаружения ошибки контрольными действиями."""

        detection = 0.15 + scenario.control_intensity * 0.18 + scenario.control_marker_ratio * 0.75
        return max(0.0, min(0.95, detection))

    def _procedure_branch_count(self, scenario: ExtendedScenario) -> int:
        """Оценить число условных ветвлений процедуры."""

        return max(1, scenario.message_complexity + scenario.control_intensity + scenario.time_pressure)

    def _memory_load_index(self, scenario: ExtendedScenario) -> float:
        """Рассчитать индекс нагрузки на оперативную память оператора."""

        load = (
            scenario.message_length / 128
            + scenario.message_complexity / 5
            + scenario.procedure_steps / 12
            - scenario.control_marker_ratio
        ) / 3
        return max(0.0, min(1.0, load))

    def _manual_operation_count(self, scenario: ExtendedScenario) -> int:
        """Оценить число ручных операций кодирования и проверки."""

        return scenario.procedure_steps + max(1, int(round(scenario.message_length / 16))) + scenario.control_intensity

    def _nominal_time(self, scenario: ExtendedScenario) -> float:
        """Рассчитать номинальное время процедуры."""

        return (
            4.0
            + scenario.message_length * 0.35
            + scenario.procedure_steps * 1.8
            + scenario.message_complexity * 2.2
            + scenario.control_marker_ratio * 15.0
        )

    def _operator_estimated_time(
        self,
        scenario: ExtendedScenario,
        nominal_time: float,
    ) -> float:
        """Рассчитать оценочное время оператора."""

        skill_factor = 1.18 - scenario.operator_skill * 0.055
        fatigue_factor = 1.0 + scenario.operator_fatigue * 0.045
        return nominal_time * skill_factor * fatigue_factor

    def _condition_adjusted_time(
        self,
        scenario: ExtendedScenario,
        operator_time: float,
    ) -> float:
        """Рассчитать время с учетом условий выполнения."""

        condition_factor = 1.0 + scenario.noise_level * 0.055 + scenario.time_pressure * 0.07
        return operator_time * condition_factor

    def _adjusted_attention(self, scenario: ExtendedScenario) -> float:
        """Рассчитать внимание оператора с учетом внешних условий."""

        attention = scenario.operator_attention - scenario.noise_level * 0.35 - scenario.time_pressure * 0.25
        return max(1.0, min(5.0, attention))

    def _expected_error_probability(self, scenario: ExtendedScenario) -> float:
        """Рассчитать априорную вероятность ошибки для сценария."""

        risk = (
            0.025
            + scenario.message_complexity * 0.018
            + scenario.message_criticality * 0.010
            + scenario.digit_ratio * 0.055
            + scenario.operator_fatigue * 0.018
            + scenario.noise_level * 0.020
            + scenario.time_pressure * 0.022
            - scenario.control_intensity * 0.012
            - scenario.operator_skill * 0.010
            - scenario.control_marker_ratio * 0.035
        )
        return max(0.005, min(0.35, risk))

    def _risk_class(self, expected_error: float) -> str:
        """Преобразовать вероятность ошибки в категорию риска."""

        if expected_error >= 0.18:
            return "high"
        if expected_error >= 0.10:
            return "mid"
        return "low"

    def _ensure_can_write(self, outputs: ExtendedCorpusOutputPaths) -> None:
        """Проверить возможность записи всех выходных файлов."""

        paths = [
            outputs.protocols_path,
            outputs.prior_features_path,
            outputs.diagnostic_features_path,
            outputs.fact_features_path,
            outputs.quality_targets_path,
            outputs.all_features_path,
            outputs.summary_json_path,
            outputs.summary_md_path,
        ]
        if not self.config.overwrite:
            for path in paths:
                if path.exists():
                    msg = f"Файл уже существует и overwrite=False: {path}"
                    raise FileExistsError(msg)
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)

    def _write_csv(self, path: Path, rows: Sequence[Mapping[str, object]]) -> None:
        """Записать строки в CSV-файл."""

        if not rows:
            msg = f"Нельзя записать пустой CSV-файл: {path}"
            raise ValueError(msg)
        fieldnames = list(rows[0].keys())
        with path.open("w", encoding=self.config.encoding, newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_json(self, path: Path, payload: Mapping[str, object]) -> None:
        """Записать JSON-файл."""

        with path.open("w", encoding=self.config.encoding) as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2)

    def _summary_payload(
        self,
        result: ExtendedCorpusGenerationResult,
        scenarios: Sequence[ExtendedScenario],
    ) -> dict[str, object]:
        """Сформировать машинный отчет по расширенному корпусу."""

        complexity_values = sorted({scenario.message_complexity for scenario in scenarios})
        criticality_values = sorted({scenario.message_criticality for scenario in scenarios})
        condition_profiles = sorted({scenario.condition_profile for scenario in scenarios})
        payload = result.to_dict()
        payload["coverage"] = {
            "message_complexity_values": complexity_values,
            "message_criticality_values": criticality_values,
            "condition_profiles": condition_profiles,
            "tool_types": sorted({scenario.coding_tool_type for scenario in scenarios}),
        }
        payload["methodological_note"] = (
            "Корпус сформирован параметрическим планом сценариев главы 3; "
            "строки не являются копиями исходного малого корпуса."
        )
        return payload

    def _write_summary_md(
        self,
        path: Path,
        result: ExtendedCorpusGenerationResult,
    ) -> None:
        """Записать Markdown-отчет по расширенному корпусу."""

        lines = [
            "# Расширенный корпус главы 3",
            "",
            "Статус: generated.",
            "",
            "## Основные показатели",
            "",
            "| Показатель | Значение |",
            "|---|---:|",
            f"| Документы | {result.document_count} |",
            f"| Уникальные сценарии | {result.unique_scenario_count} |",
            f"| Уникальные протоколы | {result.unique_protocol_count} |",
            f"| Априорные признаки | {result.prior_feature_count} |",
            f"| Диагностические признаки | {result.diagnostic_feature_count} |",
            f"| Фактические признаки | {result.fact_feature_count} |",
            f"| Целевые показатели качества | {result.quality_target_count} |",
            "",
            "## Артефакты",
            "",
            f"- protocols.csv: `{result.protocols_path}`",
            f"- prior_features.csv: `{result.prior_features_path}`",
            f"- diagnostic_features.csv: `{result.diagnostic_features_path}`",
            f"- fact_features.csv: `{result.fact_features_path}`",
            f"- quality_targets.csv: `{result.quality_targets_path}`",
            f"- all_features.csv: `{result.all_features_path}`",
            "",
            "Методическое ограничение: фактические признаки и целевые показатели "
            "качества предназначены для глав 5–6 и не должны использоваться "
            "при обучении `LDA_prior`.",
        ]
        path.write_text("\n".join(lines) + "\n", encoding=self.config.encoding)
