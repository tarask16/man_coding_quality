"""Итоговая отчетность программной реализации главы 3.

Модуль фиксирует состав исследовательского симулятора процессов
ручного кодирования, проверяет наличие исходных файлов, тестов,
табличных артефактов и отчета воспроизводимого эксперимента.

Отчет используется для подтверждения того, что глава 3 имеет
программную реализацию, способную воспроизводимо формировать данные
для адаптированной LDA-модели, метода априорной оценки качества и
последующей проверки достоверности прогноза.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from manual_coding_sim.experiment_runner import (
    ExperimentRunner,
    ExperimentRunnerConfig,
)


EXPECTED_SOURCE_FILES: tuple[str, ...] = (
    "src/manual_coding_sim/__init__.py",
    "src/manual_coding_sim/config.py",
    "src/manual_coding_sim/types.py",
    "src/manual_coding_sim/message_model.py",
    "src/manual_coding_sim/procedure_model.py",
    "src/manual_coding_sim/operator_model.py",
    "src/manual_coding_sim/condition_model.py",
    "src/manual_coding_sim/error_model.py",
    "src/manual_coding_sim/control_model.py",
    "src/manual_coding_sim/protocol_simulator.py",
    "src/manual_coding_sim/feature_extractor.py",
    "src/manual_coding_sim/quality_calculator.py",
    "src/manual_coding_sim/dataset_builder.py",
    "src/manual_coding_sim/experiment_runner.py",
    "src/manual_coding_sim/chapter3_report.py",
)

EXPECTED_TEST_FILES: tuple[str, ...] = (
    "tests/test_stage1_package_structure.py",
    "tests/test_stage2_message_model.py",
    "tests/test_stage3_procedure_model.py",
    "tests/test_stage4_operator_model.py",
    "tests/test_stage5_condition_model.py",
    "tests/test_stage6_error_model.py",
    "tests/test_stage7_control_model.py",
    "tests/test_stage8_protocol_simulator.py",
    "tests/test_stage9_feature_extractor.py",
    "tests/test_stage10_quality_calculator.py",
    "tests/test_stage11_dataset_builder.py",
    "tests/test_stage12_experiment_runner.py",
    "tests/test_stage13_chapter3_report.py",
)

EXPECTED_DATASET_KEYS: tuple[str, ...] = (
    "protocols",
    "prior_features",
    "fact_features",
    "diagnostic_features",
    "quality_targets",
    "all_features",
    "summary",
)

SCIENTIFIC_COMPONENTS: tuple[dict[str, str], ...] = (
    {
        "component": "G",
        "module": "message_model.py",
        "meaning": "модель класса исходных сообщений M",
    },
    {
        "component": "S",
        "module": "procedure_model.py",
        "meaning": "модель средства ручного кодирования",
    },
    {
        "component": "O",
        "module": "operator_model.py",
        "meaning": "модель оператора ручного кодирования",
    },
    {
        "component": "U",
        "module": "condition_model.py",
        "meaning": "модель условий применения",
    },
    {
        "component": "ErrorModel",
        "module": "error_model.py",
        "meaning": "вероятностная модель ошибок",
    },
    {
        "component": "K",
        "module": "control_model.py",
        "meaning": "модель контрольных процедур",
    },
    {
        "component": "ProtocolSimulator",
        "module": "protocol_simulator.py",
        "meaning": "интегральный симулятор протоколов",
    },
    {
        "component": "FeatureExtractor",
        "module": "feature_extractor.py",
        "meaning": "разделение X_prior, X_fact и X_diag",
    },
    {
        "component": "QualityCalculator",
        "module": "quality_calculator.py",
        "meaning": "расчет q(A)",
    },
    {
        "component": "DatasetBuilder",
        "module": "dataset_builder.py",
        "meaning": "формирование табличного датасета",
    },
    {
        "component": "ExperimentRunner",
        "module": "experiment_runner.py",
        "meaning": "воспроизводимый запуск вычислительного эксперимента",
    },
)


@dataclass(frozen=True)
class Chapter3ImplementationReportConfig:
    """Конфигурация формирования итогового отчета главы 3."""

    project_root: Path | str = Path.cwd()
    output_json: Path | str = Path("reports") / "chapter3" / "chapter3_implementation_report.json"
    output_markdown: Path | str = Path("reports") / "chapter3" / "chapter3_implementation_report.md"
    run_experiment: bool = True
    experiment_run_count: int = 5
    random_seed: int = 42
    scenario_id: str = "A_001"
    overwrite: bool = True

    def validate(self) -> None:
        """Проверяет корректность параметров итогового отчета."""
        root = Path(self.project_root)
        if not root.exists():
            raise FileNotFoundError(f"Корень проекта не найден: {root}")

        if self.experiment_run_count <= 0:
            raise ValueError("experiment_run_count должен быть положительным.")

        if self.random_seed < 0:
            raise ValueError("random_seed не должен быть отрицательным.")

        if not self.scenario_id:
            raise ValueError("scenario_id не должен быть пустым.")


@dataclass(frozen=True)
class Chapter3ImplementationReport:
    """Итоговый отчет о программной реализации главы 3."""

    summary: dict[str, Any]
    source_file_status: dict[str, bool]
    test_file_status: dict[str, bool]
    dataset_artifact_status: dict[str, bool]
    experiment_summary: dict[str, Any]
    json_path: Path
    markdown_path: Path

    @property
    def all_required_files_present(self) -> bool:
        """Возвращает признак наличия всех обязательных файлов."""
        return all(self.source_file_status.values()) and all(
            self.test_file_status.values(),
        )

    @property
    def all_dataset_artifacts_present(self) -> bool:
        """Возвращает признак наличия всех артефактов датасета."""
        return all(self.dataset_artifact_status.values())


def build_chapter3_implementation_report(
    config: Chapter3ImplementationReportConfig | None = None,
) -> Chapter3ImplementationReport:
    """Формирует итоговый отчет по программной реализации главы 3."""
    report_config = config or Chapter3ImplementationReportConfig()
    report_config.validate()
    project_root = Path(report_config.project_root)

    source_status = collect_file_status(project_root, EXPECTED_SOURCE_FILES)
    test_status = collect_file_status(project_root, EXPECTED_TEST_FILES)
    experiment_summary: dict[str, Any] = {}
    dataset_status: dict[str, bool] = {
        key: False for key in EXPECTED_DATASET_KEYS
    }

    if report_config.run_experiment:
        experiment_summary = _run_control_experiment(
            project_root=project_root,
            report_config=report_config,
        )
        dataset_status = collect_dataset_artifact_status(experiment_summary)

    summary = {
        "title": "Итоговый отчет программной реализации главы 3",
        "scientific_chain": "S, O, U, G, K → M → E_h → C → D_h → M'",
        "implemented_scope": (
            "компьютерное моделирование процессов ручного кодирования, "
            "разделение признаков и формирование датасета"
        ),
        "not_implemented_in_chapter3": [
            "адаптированная LDA-модель латентных факторов качества",
            "построение априорной прогнозной оценки качества",
            "проверка достоверности прогноза по train/val/test-разбиению",
        ],
        "source_files_total": len(source_status),
        "source_files_present": sum(source_status.values()),
        "test_files_total": len(test_status),
        "test_files_present": sum(test_status.values()),
        "dataset_artifacts_total": len(dataset_status),
        "dataset_artifacts_present": sum(dataset_status.values()),
        "scientific_components": list(SCIENTIFIC_COMPONENTS),
    }

    json_path = project_root / Path(report_config.output_json)
    markdown_path = project_root / Path(report_config.output_markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": summary,
        "source_file_status": source_status,
        "test_file_status": test_status,
        "dataset_artifact_status": dataset_status,
        "experiment_summary": experiment_summary,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_report_markdown(payload),
        encoding="utf-8",
    )

    return Chapter3ImplementationReport(
        summary=summary,
        source_file_status=source_status,
        test_file_status=test_status,
        dataset_artifact_status=dataset_status,
        experiment_summary=experiment_summary,
        json_path=json_path,
        markdown_path=markdown_path,
    )


def collect_file_status(
    project_root: Path,
    relative_paths: tuple[str, ...],
) -> dict[str, bool]:
    """Проверяет наличие файлов относительно корня проекта."""
    return {
        path: (project_root / path).exists()
        for path in relative_paths
    }


def collect_dataset_artifact_status(
    experiment_summary: dict[str, Any],
) -> dict[str, bool]:
    """Проверяет наличие табличных артефактов эксперимента.

    Пустая строка не должна считаться существующим путем. В Python выражение
    Path("").exists() проверяет текущий каталог и обычно возвращает True, что
    недопустимо для контроля CSV/JSON-артефактов датасета главы 3.
    """
    saved_files = experiment_summary.get("saved_files", {})
    if not isinstance(saved_files, dict) or not saved_files:
        return {key: False for key in EXPECTED_DATASET_KEYS}

    artifact_status: dict[str, bool] = {}
    for key in EXPECTED_DATASET_KEYS:
        raw_path = saved_files.get(key)
        if not isinstance(raw_path, str) or not raw_path.strip():
            artifact_status[key] = False
            continue

        artifact_status[key] = Path(raw_path).exists()

    return artifact_status


def render_report_markdown(payload: dict[str, Any]) -> str:
    """Формирует Markdown-представление итогового отчета."""
    summary = payload["summary"]
    experiment_summary = payload.get("experiment_summary", {})
    source_status = payload["source_file_status"]
    test_status = payload["test_file_status"]
    dataset_status = payload["dataset_artifact_status"]

    lines = [
        "# Итоговый отчет программной реализации главы 3",
        "",
        "## Назначение",
        "",
        (
            "Программная реализация подтверждает возможность "
            "воспроизводимого компьютерного моделирования процессов "
            "ручного кодирования и формирования данных для дальнейшей "
            "априорной оценки качества."
        ),
        "",
        "## Реализованная цепочка",
        "",
        f"`{summary['scientific_chain']}`",
        "",
        "## Состав научных компонентов",
        "",
        "| Компонент | Модуль | Смысл |",
        "|---|---|---|",
    ]

    for item in summary["scientific_components"]:
        lines.append(
            f"| {item['component']} | {item['module']} | {item['meaning']} |",
        )

    lines.extend(
        [
            "",
            "## Контроль файлов",
            "",
            f"Исходные файлы: {summary['source_files_present']} "
            f"из {summary['source_files_total']}.",
            f"Тестовые файлы: {summary['test_files_present']} "
            f"из {summary['test_files_total']}.",
            f"Артефакты датасета: {summary['dataset_artifacts_present']} "
            f"из {summary['dataset_artifacts_total']}.",
            "",
            "## Исходные файлы",
            "",
        ],
    )
    lines.extend(_status_lines(source_status))
    lines.extend(["", "## Тестовые файлы", ""])
    lines.extend(_status_lines(test_status))
    lines.extend(["", "## Артефакты датасета", ""])
    lines.extend(_status_lines(dataset_status))

    if experiment_summary:
        lines.extend(
            [
                "",
                "## Воспроизводимый эксперимент",
                "",
                f"Эксперимент: `{experiment_summary.get('experiment_name')}`",
                f"Сценарий: `{experiment_summary.get('scenario_id')}`",
                f"Число прогонов: `{experiment_summary.get('run_count')}`",
                f"random_seed: `{experiment_summary.get('random_seed')}`",
                (
                    "Контроль воспроизводимости: "
                    f"`{experiment_summary.get('reproducibility_ok')}`"
                ),
                (
                    "Контрольный хеш: "
                    f"`{experiment_summary.get('reproducibility_hash')}`"
                ),
            ],
        )

    lines.extend(
        [
            "",
            "## Ограничение области реализации",
            "",
        ],
    )
    for item in summary["not_implemented_in_chapter3"]:
        lines.append(f"- {item}")

    return "\n".join(lines) + "\n"


def _status_lines(status: dict[str, bool]) -> list[str]:
    """Преобразует словарь статусов в Markdown-строки."""
    return [
        f"- [{'OK' if exists else 'MISSING'}] `{path}`"
        for path, exists in status.items()
    ]


def _run_control_experiment(
    project_root: Path,
    report_config: Chapter3ImplementationReportConfig,
) -> dict[str, Any]:
    """Запускает малый контрольный эксперимент для итогового отчета."""
    experiment_config = ExperimentRunnerConfig(
        experiment_name="chapter3_final_control_experiment",
        random_seed=report_config.random_seed,
        run_count=report_config.experiment_run_count,
        scenario_id=report_config.scenario_id,
        output_dir=project_root / "data" / "processed",
        reports_dir=project_root / "reports" / "chapter3",
        overwrite=report_config.overwrite,
        check_reproducibility=True,
    )
    result = ExperimentRunner(experiment_config).run()
    return result.summary


def main() -> None:
    """CLI-точка входа для формирования итогового отчета главы 3."""
    report = build_chapter3_implementation_report()
    print("Итоговый отчет программной реализации главы 3 сформирован.")
    print(f"JSON: {report.json_path}")
    print(f"Markdown: {report.markdown_path}")
    print(f"Все обязательные файлы: {report.all_required_files_present}")
    print(f"Все артефакты датасета: {report.all_dataset_artifacts_present}")


if __name__ == "__main__":
    main()
