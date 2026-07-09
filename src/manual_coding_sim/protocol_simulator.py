"""Интегральный симулятор протоколов ручного кодирования.

Модуль относится к главе 3 диссертации и объединяет ранее созданные
модели класса сообщений G, средства ручного кодирования S, оператора O,
условий применения U, вероятностной модели ошибок и контрольных
процедур K.

Симулятор формирует воспроизводимый протокол моделирования цепочки:

    S, O, U, G, K -> M -> план S -> оценка O -> оценка U -> ошибки -> контроль

На данном этапе не выполняются декодирование D_h, восстановление M'
и итоговый расчет вектора частных показателей качества q(A). Эти
операции будут добавлены на следующих этапах после появления моделей
извлечения признаков и расчета качества.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from manual_coding_sim.condition_model import (
    ConditionModel,
    ConditionModelConfig,
    ConditionPlanEstimate,
)
from manual_coding_sim.control_model import (
    ControlModel,
    ControlModelConfig,
    ControlProtocol,
)
from manual_coding_sim.error_model import (
    ErrorModel,
    ErrorModelConfig,
    ErrorProtocol,
)
from manual_coding_sim.message_model import (
    MessageGenerationConfig,
    MessageModel,
)
from manual_coding_sim.operator_model import (
    OperatorModel,
    OperatorModelConfig,
    OperatorPlanEstimate,
)
from manual_coding_sim.procedure_model import (
    ProcedureModel,
    ProcedureModelConfig,
    ProcedurePlan,
)
from manual_coding_sim.types import GeneratedMessage, ScenarioParameters


@dataclass(frozen=True)
class ProtocolSimulatorConfig:
    """Конфигурация интегрального симулятора протоколов.

    Конфигурация задает параметры сценария A = {S, O, U, G, K} и
    фиксированные зерна генераторов случайных чисел. Разделение зерен
    позволяет независимо воспроизводить генерацию сообщений M,
    возникновение ошибок и работу контрольных процедур K.
    """

    scenario_id: str = "A_001"
    message_config: MessageGenerationConfig = field(
        default_factory=MessageGenerationConfig,
    )
    procedure_config: ProcedureModelConfig = field(
        default_factory=ProcedureModelConfig,
    )
    operator_config: OperatorModelConfig = field(
        default_factory=OperatorModelConfig,
    )
    condition_config: ConditionModelConfig = field(
        default_factory=ConditionModelConfig,
    )
    error_config: ErrorModelConfig = field(default_factory=ErrorModelConfig)
    control_config: ControlModelConfig = field(
        default_factory=ControlModelConfig,
    )
    message_random_seed: int = 42
    error_random_seed: int = 1042
    control_random_seed: int = 2042

    def validate(self) -> None:
        """Проверяет корректность конфигурации симулятора."""
        if not self.scenario_id:
            raise ValueError("Идентификатор сценария A не задан.")

        if self.message_random_seed < 0:
            raise ValueError("message_random_seed не должен быть отрицательным.")

        if self.error_random_seed < 0:
            raise ValueError("error_random_seed не должен быть отрицательным.")

        if self.control_random_seed < 0:
            raise ValueError("control_random_seed не должен быть отрицательным.")

        self.message_config.validate()
        self.procedure_config.validate()
        self.operator_config.validate()
        self.condition_config.validate()
        self.error_config.validate()
        self.control_config.validate()


@dataclass(frozen=True)
class SimulationResult:
    """Результат одного полного прогона симулятора протоколов.

    Объект сохраняет все промежуточные артефакты моделирования, чтобы
    обеспечить трассируемость перехода от априорного сценария
    A = {S, O, U, G, K} к фактическому протоколу ошибок и контроля.
    """

    scenario: ScenarioParameters
    message: GeneratedMessage
    procedure_plan: ProcedurePlan
    operator_estimate: OperatorPlanEstimate
    condition_estimate: ConditionPlanEstimate
    error_protocol: ErrorProtocol
    control_protocol: ControlProtocol
    metadata: dict[str, int | float | str]


class ProtocolSimulator:
    """Интегральный симулятор протоколов ручного кодирования.

    Симулятор не раскрывает конкретные прикладные способы скрытой связи.
    Он оперирует абстрактными сообщениями, нормативными операциями,
    априорными профилями оператора, условий и контроля, а также
    вероятностным протоколом ошибок.
    """

    def __init__(self, config: ProtocolSimulatorConfig | None = None) -> None:
        """Инициализирует все модели, входящие в сценарий A."""
        self.config = config or ProtocolSimulatorConfig()
        self.config.validate()

        self.message_model = MessageModel(
            config=self.config.message_config,
            random_seed=self.config.message_random_seed,
        )
        self.procedure_model = ProcedureModel(self.config.procedure_config)
        self.operator_model = OperatorModel(self.config.operator_config)
        self.condition_model = ConditionModel(self.config.condition_config)
        self.error_model = ErrorModel(
            config=self.config.error_config,
            random_seed=self.config.error_random_seed,
        )
        self.control_model = ControlModel(
            config=self.config.control_config,
            random_seed=self.config.control_random_seed,
        )

    def simulate_once(self, message_id: str | None = None) -> SimulationResult:
        """Выполняет один воспроизводимый прогон моделирования."""
        message = self.message_model.generate_message(message_id=message_id)
        procedure_plan = self.procedure_model.build_plan(message)
        operator_estimate = self.operator_model.estimate_plan(procedure_plan)
        condition_estimate = self.condition_model.estimate_plan(operator_estimate)
        error_protocol = self.error_model.generate_protocol(condition_estimate)
        control_protocol = self.control_model.generate_protocol(error_protocol)

        scenario = self._build_scenario()
        metadata = self._build_metadata(
            message=message,
            procedure_plan=procedure_plan,
            condition_estimate=condition_estimate,
            error_protocol=error_protocol,
            control_protocol=control_protocol,
        )

        return SimulationResult(
            scenario=scenario,
            message=message,
            procedure_plan=procedure_plan,
            operator_estimate=operator_estimate,
            condition_estimate=condition_estimate,
            error_protocol=error_protocol,
            control_protocol=control_protocol,
            metadata=metadata,
        )

    def simulate_batch(self, count: int) -> tuple[SimulationResult, ...]:
        """Выполняет пакет прогонов для серии сообщений M."""
        if count <= 0:
            raise ValueError("Число прогонов моделирования должно быть положительным.")

        return tuple(self.simulate_once() for _ in range(count))

    def reset(self) -> None:
        """Возвращает псевдослучайные модели к начальному состоянию."""
        self.message_model.reset()
        self.error_model.reset()
        self.control_model.reset()

    def _build_scenario(self) -> ScenarioParameters:
        """Формирует описание сценария A = {S, O, U, G, K}."""
        return ScenarioParameters(
            scenario_id=self.config.scenario_id,
            coding_tool_id=self.config.procedure_config.procedure_id,
            operator_id=self.config.operator_config.profile.operator_id,
            condition_id=self.config.condition_config.profile.condition_id,
            message_class_id=self.config.message_config.message_class_id,
            control_procedure_id=self.config.control_config.profile.control_id,
        )

    def _build_metadata(
        self,
        message: GeneratedMessage,
        procedure_plan: ProcedurePlan,
        condition_estimate: ConditionPlanEstimate,
        error_protocol: ErrorProtocol,
        control_protocol: ControlProtocol,
    ) -> dict[str, int | float | str]:
        """Формирует контрольную метаинформацию одного прогона."""
        step_count = int(procedure_plan.metadata["step_count"])
        error_count = int(error_protocol.metadata["error_count"])
        residual_error_count = int(
            control_protocol.metadata["residual_error_count"],
        )
        adjusted_time = sum(
            step.adjusted_time for step in condition_estimate.step_estimates
        )

        return {
            "scenario_id": self.config.scenario_id,
            "message_id": message.message_id,
            "step_count": step_count,
            "message_length": len(message.elements),
            "original_error_count": error_count,
            "residual_error_count": residual_error_count,
            "nominal_time": float(
                procedure_plan.metadata["total_nominal_time"],
            ),
            "condition_adjusted_time": round(adjusted_time, 6),
            "message_random_seed": self.config.message_random_seed,
            "error_random_seed": self.config.error_random_seed,
            "control_random_seed": self.config.control_random_seed,
        }


def summarize_simulation_result(
    result: SimulationResult,
) -> dict[str, int | float | str]:
    """Возвращает контрольную сводку по одному прогону симулятора."""
    if not result.procedure_plan.steps:
        raise ValueError("Результат моделирования не содержит шагов процедуры.")

    adjusted_time = sum(
        step.adjusted_time for step in result.condition_estimate.step_estimates
    )
    total_control_effort = float(
        result.control_protocol.metadata["total_control_effort"],
    )

    return {
        "scenario_id": result.scenario.scenario_id,
        "message_id": result.message.message_id,
        "message_length": len(result.message.elements),
        "step_count": len(result.procedure_plan.steps),
        "nominal_time": float(
            result.procedure_plan.metadata["total_nominal_time"],
        ),
        "condition_adjusted_time": round(adjusted_time, 6),
        "original_error_count": int(
            result.control_protocol.metadata["original_error_count"],
        ),
        "detected_error_count": int(
            result.control_protocol.metadata["detected_error_count"],
        ),
        "corrected_error_count": int(
            result.control_protocol.metadata["corrected_error_count"],
        ),
        "residual_error_count": int(
            result.control_protocol.metadata["residual_error_count"],
        ),
        "total_control_effort": round(total_control_effort, 6),
    }


def simulation_results_to_rows(
    results: Iterable[SimulationResult],
) -> list[dict[str, int | float | str]]:
    """Преобразует результаты моделирования в строки контрольной таблицы."""
    return [summarize_simulation_result(result) for result in results]
