"""Исправление предупреждения RuntimeWarning при запуске отчета главы 3.

Скрипт удаляет преждевременный импорт модуля `chapter3_report` из
`src/manual_coding_sim/__init__.py`. Для пакета с layout `src/` отчетный модуль
должен запускаться командой `python -m manual_coding_sim.chapter3_report`, но
не должен импортироваться автоматически при обычной загрузке пакета
`manual_coding_sim`.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


PROJECT_ROOT = Path.cwd()
PACKAGE_INIT_PATH = PROJECT_ROOT / "src" / "manual_coding_sim" / "__init__.py"
REPORT_PATH = PROJECT_ROOT / "reports" / "chapter3" / "stage13_runtime_warning_fix_report.json"

INIT_CONTENT = '''"""Исследовательский симулятор ручного кодирования для главы 3 диссертации.

Пакет содержит программные модели компонентов сценария A = {S, O, U, G, K},
а также средства формирования протоколов, признаков и показателей качества.
Служебные отчетные модули запускаются отдельно и не импортируются здесь,
чтобы не нарушать корректный запуск через `python -m`.
"""

from manual_coding_sim.condition_model import (
    ConditionModel,
    ConditionModelConfig,
    ConditionPlanEstimate,
    ConditionProfile,
    ConditionStepEstimate,
    condition_estimates_to_rows,
    summarize_condition_estimate,
)
from manual_coding_sim.config import load_experiment_config
from manual_coding_sim.control_model import (
    ControlModel,
    ControlModelConfig,
    ControlProfile,
    ControlProtocol,
    ControlStepOutcome,
    control_protocols_to_rows,
    summarize_control_protocol,
)
from manual_coding_sim.dataset_builder import (
    DatasetBuilder,
    DatasetBuilderConfig,
    DatasetBuildResult,
    build_dataset,
)
from manual_coding_sim.error_model import (
    ErrorModel,
    ErrorModelConfig,
    ErrorProtocol,
    ErrorStepOutcome,
    error_protocols_to_rows,
    summarize_error_protocol,
)
from manual_coding_sim.experiment_runner import (
    ExperimentRunResult,
    ExperimentRunner,
    ExperimentRunnerConfig,
    hash_dataset_result,
    run_experiment,
    run_experiment_from_yaml,
)
from manual_coding_sim.feature_extractor import (
    FeatureExtractor,
    FeatureExtractorConfig,
    feature_group_to_flat_row,
    feature_groups_to_rows,
    summarize_feature_group,
    validate_feature_group,
)
from manual_coding_sim.message_model import (
    MessageGenerationConfig,
    MessageModel,
    messages_to_rows,
    summarize_message,
)
from manual_coding_sim.operator_model import (
    OperatorModel,
    OperatorModelConfig,
    OperatorPlanEstimate,
    OperatorProfile,
    OperatorState,
    OperatorStepEstimate,
    operator_estimates_to_rows,
    summarize_operator_estimate,
)
from manual_coding_sim.procedure_model import (
    CodingOperationRule,
    ProcedureModel,
    ProcedureModelConfig,
    ProcedurePlan,
    ProcedureStep,
    procedure_plans_to_rows,
    summarize_procedure_plan,
)
from manual_coding_sim.protocol_simulator import (
    ProtocolSimulator,
    ProtocolSimulatorConfig,
    SimulationResult,
    simulation_results_to_rows,
    summarize_simulation_result,
)
from manual_coding_sim.quality_calculator import (
    QualityAssessment,
    QualityCalculator,
    QualityCalculatorConfig,
    quality_assessments_to_rows,
    quality_vector_to_dict,
    summarize_quality_assessment,
)
from manual_coding_sim.types import (
    FeatureGroup,
    GeneratedMessage,
    MessageElement,
    QualityVector,
    ScenarioParameters,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "load_experiment_config",
    "ScenarioParameters",
    "MessageElement",
    "GeneratedMessage",
    "QualityVector",
    "FeatureGroup",
    "MessageGenerationConfig",
    "MessageModel",
    "summarize_message",
    "messages_to_rows",
    "CodingOperationRule",
    "ProcedureModelConfig",
    "ProcedureStep",
    "ProcedurePlan",
    "ProcedureModel",
    "summarize_procedure_plan",
    "procedure_plans_to_rows",
    "OperatorProfile",
    "OperatorModelConfig",
    "OperatorState",
    "OperatorStepEstimate",
    "OperatorPlanEstimate",
    "OperatorModel",
    "summarize_operator_estimate",
    "operator_estimates_to_rows",
    "ConditionProfile",
    "ConditionModelConfig",
    "ConditionStepEstimate",
    "ConditionPlanEstimate",
    "ConditionModel",
    "summarize_condition_estimate",
    "condition_estimates_to_rows",
    "ErrorModelConfig",
    "ErrorStepOutcome",
    "ErrorProtocol",
    "ErrorModel",
    "summarize_error_protocol",
    "error_protocols_to_rows",
    "ControlProfile",
    "ControlModelConfig",
    "ControlStepOutcome",
    "ControlProtocol",
    "ControlModel",
    "summarize_control_protocol",
    "control_protocols_to_rows",
    "ProtocolSimulatorConfig",
    "SimulationResult",
    "ProtocolSimulator",
    "summarize_simulation_result",
    "simulation_results_to_rows",
    "FeatureExtractorConfig",
    "FeatureExtractor",
    "validate_feature_group",
    "summarize_feature_group",
    "feature_group_to_flat_row",
    "feature_groups_to_rows",
    "QualityCalculatorConfig",
    "QualityAssessment",
    "QualityCalculator",
    "quality_vector_to_dict",
    "summarize_quality_assessment",
    "quality_assessments_to_rows",
    "DatasetBuilderConfig",
    "DatasetBuildResult",
    "DatasetBuilder",
    "build_dataset",
    "ExperimentRunnerConfig",
    "ExperimentRunResult",
    "ExperimentRunner",
    "hash_dataset_result",
    "run_experiment",
    "run_experiment_from_yaml",
]
'''


def main() -> None:
    """Перезаписывает `__init__.py` безопасной версией без импортов отчета."""
    if not PACKAGE_INIT_PATH.exists():
        raise FileNotFoundError(f"Не найден файл пакета: {PACKAGE_INIT_PATH}")

    PACKAGE_INIT_PATH.write_text(INIT_CONTENT, encoding="utf-8")
    ast.parse(PACKAGE_INIT_PATH.read_text(encoding="utf-8"))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "stage": 13,
        "task": "fix_runtime_warning_for_chapter3_report",
        "fixed_file": str(PACKAGE_INIT_PATH),
        "removed_automatic_chapter3_report_import": True,
        "recommended_check_commands": [
            "python -m pytest",
            "python -W error::RuntimeWarning -m manual_coding_sim.chapter3_report",
        ],
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Исправление RuntimeWarning подготовлено.")
    print(f"[OK] Обновлен файл: {PACKAGE_INIT_PATH}")
    print(f"[OK] Отчет: {REPORT_PATH}")
    print("\nТеперь выполните команды:")
    print("python -m pytest")
    print("python -W error::RuntimeWarning -m manual_coding_sim.chapter3_report")


if __name__ == "__main__":
    main()
