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
    "QualityVector",
    "ScenarioParameters",
    "condition_estimates_to_rows",
    "load_experiment_config",
    "messages_to_rows",
    "operator_estimates_to_rows",
    "procedure_plans_to_rows",
    "summarize_condition_estimate",
    "summarize_message",
    "summarize_operator_estimate",
    "summarize_procedure_plan",
]
