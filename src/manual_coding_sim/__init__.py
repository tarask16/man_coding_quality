"""
Базовый пакет исследовательского симулятора ручного кодирования.

Пакет предназначен для программной реализации главы 3 диссертации:
компьютерного моделирования процессов ручного кодирования и декодирования
при априорной оценке качества ручных средств кодирования информации.
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
from manual_coding_sim.error_model import (
    ErrorModel,
    ErrorModelConfig,
    ErrorProtocol,
    ErrorStepOutcome,
    error_protocols_to_rows,
    summarize_error_protocol,
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
from manual_coding_sim.types import (
    FeatureGroup,
    GeneratedMessage,
    MessageElement,
    QualityVector,
    ScenarioParameters,
)

__version__ = "0.1.0"

__all__ = [
    "CodingOperationRule",
    "ConditionModel",
    "ConditionModelConfig",
    "ConditionPlanEstimate",
    "ConditionProfile",
    "ConditionStepEstimate",
    "ControlModel",
    "ControlModelConfig",
    "ControlProfile",
    "ControlProtocol",
    "ControlStepOutcome",
    "ErrorModel",
    "ErrorModelConfig",
    "ErrorProtocol",
    "ErrorStepOutcome",
    "FeatureExtractor",
    "FeatureExtractorConfig",
    "FeatureGroup",
    "GeneratedMessage",
    "MessageElement",
    "MessageGenerationConfig",
    "MessageModel",
    "OperatorModel",
    "OperatorModelConfig",
    "OperatorPlanEstimate",
    "OperatorProfile",
    "OperatorState",
    "OperatorStepEstimate",
    "ProcedureModel",
    "ProcedureModelConfig",
    "ProcedurePlan",
    "ProcedureStep",
    "ProtocolSimulator",
    "ProtocolSimulatorConfig",
    "QualityVector",
    "ScenarioParameters",
    "SimulationResult",
    "condition_estimates_to_rows",
    "control_protocols_to_rows",
    "error_protocols_to_rows",
    "feature_group_to_flat_row",
    "feature_groups_to_rows",
    "load_experiment_config",
    "messages_to_rows",
    "operator_estimates_to_rows",
    "procedure_plans_to_rows",
    "simulation_results_to_rows",
    "summarize_condition_estimate",
    "summarize_control_protocol",
    "summarize_error_protocol",
    "summarize_feature_group",
    "summarize_message",
    "summarize_operator_estimate",
    "summarize_procedure_plan",
    "summarize_simulation_result",
    "validate_feature_group",
]
