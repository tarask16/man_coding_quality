"""Исследовательский симулятор ручного кодирования для главы 3 диссертации.

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
