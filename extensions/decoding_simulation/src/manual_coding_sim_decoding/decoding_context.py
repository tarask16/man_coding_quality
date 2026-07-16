"""Модель декодирующего оператора и условий выполнения обратной процедуры.

Модуль этапа 4 оценивает выполнение формального плана ``D_h(C)`` отдельным
декодирующим оператором в заданных условиях. Все расчеты детерминированы и
формируют только оценки времени, внимания, утомления, усилия и устойчивости.
Ошибки декодирования, контроль декодирования и восстановленное сообщение
``M'`` на этом этапе не создаются.

Модель принимает только :class:`DecodingPlan`. Поля исходного сообщения и
трассировочное ``source_value`` материального сообщения C ей недоступны.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from manual_coding_sim_decoding.config import (
    DecodingConditionConfig,
    DecodingOperatorConfig,
)
from manual_coding_sim_decoding.decoding_procedure import DecodingPlan, DecodingStep


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Ограничить числовое значение заданным интервалом."""
    return min(upper, max(lower, value))


@dataclass(frozen=True)
class DecodingOperatorState:
    """Расчетное состояние декодирующего оператора на одном шаге."""

    operation_index: int
    completed_operations: int
    attention: float
    fatigue: float
    accumulated_effort: float


@dataclass(frozen=True)
class DecodingOperatorStepEstimate:
    """Оценка выполнения одного нормативного шага D_h(C) оператором."""

    decoding_step_id: str
    encoded_element_id: str
    encoded_position: int
    operator_id: str
    baseline_nominal_time: float
    estimated_time: float
    attention: float
    fatigue: float
    effort: float
    complexity_pressure: float
    reference_pressure: float
    ambiguity_pressure: float
    interpretation_skill: float
    token_recognized: bool


@dataclass(frozen=True)
class DecodingOperatorPlanEstimate:
    """Детерминированная оценка выполнения полного плана декодирования."""

    operator_id: str
    decoding_procedure_id: str
    encoded_message_id: str
    source_message_id: str
    step_estimates: tuple[DecodingOperatorStepEstimate, ...]
    metadata: dict[str, int | float | str | bool]

    def to_dict(self) -> dict[str, Any]:
        """Преобразовать оценку оператора в JSON-совместимый словарь."""
        return asdict(self)


@dataclass(frozen=True)
class DecodingConditionStepEstimate:
    """Оценка влияния условий на один шаг декодирования."""

    decoding_step_id: str
    encoded_position: int
    operator_id: str
    condition_id: str
    baseline_time: float
    adjusted_time: float
    baseline_attention: float
    adjusted_attention: float
    environmental_load: float
    time_pressure: float
    instruction_pressure: float
    stability_index: float
    ambiguity_pressure: float


@dataclass(frozen=True)
class DecodingConditionPlanEstimate:
    """Оценка выполнения D_h(C) оператором в заданных условиях."""

    condition_id: str
    operator_id: str
    decoding_procedure_id: str
    encoded_message_id: str
    source_message_id: str
    step_estimates: tuple[DecodingConditionStepEstimate, ...]
    metadata: dict[str, int | float | str | bool]

    def to_dict(self) -> dict[str, Any]:
        """Преобразовать оценку условий в JSON-совместимый словарь."""
        return asdict(self)


@dataclass(frozen=True)
class DecodingExecutionContext:
    """Объединенная детерминированная оценка оператора и условий."""

    operator_estimate: DecodingOperatorPlanEstimate
    condition_estimate: DecodingConditionPlanEstimate
    metadata: dict[str, int | float | str | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Преобразовать контекст выполнения в JSON-совместимый словарь."""
        return asdict(self)


class DecodingOperatorModel:
    """Детерминированная модель отдельного декодирующего оператора."""

    def __init__(self, config: DecodingOperatorConfig | None = None) -> None:
        """Инициализировать модель профилем декодирующего оператора."""
        self.config = config or DecodingOperatorConfig()
        self.config.validate()
        self.profile = self.config.profile

    def estimate_step(
        self,
        step: DecodingStep,
        operation_index: int,
    ) -> DecodingOperatorStepEstimate:
        """Рассчитать оценку выполнения одного шага D_h(C)."""
        if operation_index < 1:
            raise ValueError("Индекс операции декодирования должен начинаться с 1.")

        state = self._calculate_state(step, operation_index)
        reference_pressure = self._calculate_reference_pressure(step)
        ambiguity_pressure = self._calculate_ambiguity_pressure(step)
        complexity_pressure = self._calculate_complexity_pressure(step)

        time_multiplier = (
            1.0
            + complexity_pressure
            + reference_pressure
            + ambiguity_pressure
            + state.fatigue * self.config.fatigue_time_weight
        ) / self.profile.work_rate
        estimated_time = step.nominal_time * time_multiplier
        effort = step.complexity * (
            1.0 + state.fatigue + reference_pressure + ambiguity_pressure
        )

        return DecodingOperatorStepEstimate(
            decoding_step_id=step.decoding_step_id,
            encoded_element_id=step.encoded_element_id,
            encoded_position=step.encoded_position,
            operator_id=self.profile.operator_id,
            baseline_nominal_time=round(step.nominal_time, 6),
            estimated_time=round(estimated_time, 6),
            attention=round(state.attention, 6),
            fatigue=round(state.fatigue, 6),
            effort=round(effort, 6),
            complexity_pressure=round(complexity_pressure, 6),
            reference_pressure=round(reference_pressure, 6),
            ambiguity_pressure=round(ambiguity_pressure, 6),
            interpretation_skill=self.profile.interpretation_skill,
            token_recognized=step.token_recognized,
        )

    def estimate_plan(self, plan: DecodingPlan) -> DecodingOperatorPlanEstimate:
        """Рассчитать оценку выполнения полного формального плана."""
        if not plan.steps:
            raise ValueError("План декодирования не содержит шагов для оценки.")

        step_estimates = tuple(
            self.estimate_step(step, index)
            for index, step in enumerate(plan.steps, start=1)
        )
        total_estimated_time = sum(item.estimated_time for item in step_estimates)
        total_effort = sum(item.effort for item in step_estimates)
        mean_attention = sum(item.attention for item in step_estimates) / len(
            step_estimates
        )
        final_fatigue = step_estimates[-1].fatigue
        mean_ambiguity_pressure = sum(
            item.ambiguity_pressure for item in step_estimates
        ) / len(step_estimates)

        return DecodingOperatorPlanEstimate(
            operator_id=self.profile.operator_id,
            decoding_procedure_id=plan.decoding_procedure_id,
            encoded_message_id=plan.encoded_message_id,
            source_message_id=plan.source_message_id,
            step_estimates=step_estimates,
            metadata={
                "step_count": len(step_estimates),
                "total_estimated_time": round(total_estimated_time, 6),
                "total_effort": round(total_effort, 6),
                "mean_attention": round(mean_attention, 6),
                "final_fatigue": round(final_fatigue, 6),
                "mean_ambiguity_pressure": round(mean_ambiguity_pressure, 6),
                "preparation_level": self.profile.preparation_level,
                "experience_level": self.profile.experience_level,
                "interpretation_skill": self.profile.interpretation_skill,
                "reference_skill": self.profile.reference_skill,
                "decoder_profile_is_independent": True,
                "uses_source_value_for_decoding": False,
                "generates_decoding_errors": False,
            },
        )

    def estimate_batch(
        self,
        plans: Iterable[DecodingPlan],
    ) -> tuple[DecodingOperatorPlanEstimate, ...]:
        """Рассчитать оценки декодирующего оператора для пакета планов."""
        return tuple(self.estimate_plan(plan) for plan in plans)

    def _calculate_state(
        self,
        step: DecodingStep,
        operation_index: int,
    ) -> DecodingOperatorState:
        """Рассчитать внимание и утомленность декодирующего оператора."""
        completed_operations = operation_index - 1
        ambiguity_pressure = self._calculate_ambiguity_pressure(step)
        fatigue = _clip(
            completed_operations * self.profile.fatigue_rate
            + step.complexity * 0.05
            + ambiguity_pressure * 0.05
        )
        reference_pressure = self._calculate_reference_pressure(step)
        attention_loss = (
            fatigue
            + step.complexity * 0.10
            + reference_pressure * 0.25
            + ambiguity_pressure * 0.40
        )
        attention = max(
            self.profile.min_attention,
            _clip(self.profile.base_attention - attention_loss),
        )
        accumulated_effort = completed_operations * step.complexity
        return DecodingOperatorState(
            operation_index=operation_index,
            completed_operations=completed_operations,
            attention=attention,
            fatigue=fatigue,
            accumulated_effort=accumulated_effort,
        )

    def _calculate_complexity_pressure(self, step: DecodingStep) -> float:
        """Рассчитать давление сложности с учетом навыков оператора."""
        aggregate_skill = (
            self.profile.preparation_level
            + self.profile.experience_level
            + self.profile.interpretation_skill
        ) / 3.0
        return _clip(
            step.complexity
            * (1.0 - aggregate_skill)
            * self.config.complexity_weight
        )

    def _calculate_reference_pressure(self, step: DecodingStep) -> float:
        """Рассчитать нагрузку обращения к таблице или инструкции."""
        if not step.reference_required:
            return 0.0
        return _clip(
            self.config.reference_penalty * (1.0 - self.profile.reference_skill)
        )

    def _calculate_ambiguity_pressure(self, step: DecodingStep) -> float:
        """Рассчитать нагрузку неразрешенного входного токена без его исправления."""
        if step.token_recognized:
            return 0.0
        return self.config.unresolved_token_penalty


class DecodingConditionModel:
    """Детерминированная модель условий выполнения обратной процедуры."""

    def __init__(self, config: DecodingConditionConfig | None = None) -> None:
        """Инициализировать модель профилем условий декодирования."""
        self.config = config or DecodingConditionConfig()
        self.config.validate()
        self.profile = self.config.profile

    def estimate_step(
        self,
        step: DecodingOperatorStepEstimate,
        time_pressure: float,
    ) -> DecodingConditionStepEstimate:
        """Рассчитать влияние условий на один оцененный шаг."""
        if not 0.0 <= time_pressure <= 1.0:
            raise ValueError("time_pressure должен находиться в диапазоне [0; 1].")

        instruction_pressure = _clip(1.0 - self.profile.instruction_access)
        environmental_load = self._calculate_environmental_load(
            instruction_pressure
        )
        time_multiplier = (
            1.0
            + environmental_load * self.config.time_expansion_weight
            + time_pressure * self.config.time_pressure_weight
            + step.ambiguity_pressure * self.config.ambiguity_time_weight
        )
        attention_loss = (
            environmental_load * self.config.environment_attention_weight
            + time_pressure * self.config.time_attention_weight
            + step.ambiguity_pressure * self.config.ambiguity_attention_weight
        )
        adjusted_attention = _clip(step.attention * (1.0 - attention_loss))
        stability_index = 1.0 - _clip(
            (
                environmental_load
                + time_pressure
                + step.ambiguity_pressure
            )
            / 3.0
        )

        return DecodingConditionStepEstimate(
            decoding_step_id=step.decoding_step_id,
            encoded_position=step.encoded_position,
            operator_id=step.operator_id,
            condition_id=self.profile.condition_id,
            baseline_time=round(step.estimated_time, 6),
            adjusted_time=round(step.estimated_time * time_multiplier, 6),
            baseline_attention=round(step.attention, 6),
            adjusted_attention=round(adjusted_attention, 6),
            environmental_load=round(environmental_load, 6),
            time_pressure=round(time_pressure, 6),
            instruction_pressure=round(instruction_pressure, 6),
            stability_index=round(stability_index, 6),
            ambiguity_pressure=round(step.ambiguity_pressure, 6),
        )

    def estimate_plan(
        self,
        operator_estimate: DecodingOperatorPlanEstimate,
    ) -> DecodingConditionPlanEstimate:
        """Рассчитать влияние условий на оценку полного плана."""
        if not operator_estimate.step_estimates:
            raise ValueError("Оценка декодирующего оператора не содержит шагов.")

        baseline_total_time = float(
            operator_estimate.metadata["total_estimated_time"]
        )
        time_pressure = self._calculate_time_pressure(baseline_total_time)
        step_estimates = tuple(
            self.estimate_step(step, time_pressure)
            for step in operator_estimate.step_estimates
        )
        total_adjusted_time = sum(item.adjusted_time for item in step_estimates)
        mean_adjusted_attention = sum(
            item.adjusted_attention for item in step_estimates
        ) / len(step_estimates)
        mean_environmental_load = sum(
            item.environmental_load for item in step_estimates
        ) / len(step_estimates)
        mean_stability_index = sum(
            item.stability_index for item in step_estimates
        ) / len(step_estimates)

        return DecodingConditionPlanEstimate(
            condition_id=self.profile.condition_id,
            operator_id=operator_estimate.operator_id,
            decoding_procedure_id=operator_estimate.decoding_procedure_id,
            encoded_message_id=operator_estimate.encoded_message_id,
            source_message_id=operator_estimate.source_message_id,
            step_estimates=step_estimates,
            metadata={
                "step_count": len(step_estimates),
                "baseline_total_time": round(baseline_total_time, 6),
                "total_adjusted_time": round(total_adjusted_time, 6),
                "mean_adjusted_attention": round(mean_adjusted_attention, 6),
                "mean_environmental_load": round(mean_environmental_load, 6),
                "mean_stability_index": round(mean_stability_index, 6),
                "time_pressure": round(time_pressure, 6),
                "noise_level": self.profile.noise_level,
                "workload_level": self.profile.workload_level,
                "instruction_access": self.profile.instruction_access,
                "decoding_conditions_are_independent": True,
                "generates_decoding_errors": False,
            },
        )

    def estimate_batch(
        self,
        operator_estimates: Iterable[DecodingOperatorPlanEstimate],
    ) -> tuple[DecodingConditionPlanEstimate, ...]:
        """Рассчитать влияние условий для пакета оценок декодирования."""
        return tuple(self.estimate_plan(item) for item in operator_estimates)

    def _calculate_time_pressure(self, baseline_total_time: float) -> float:
        """Рассчитать давление дефицита времени."""
        if baseline_total_time <= 0:
            raise ValueError("Базовое расчетное время должно быть положительным.")
        if self.profile.time_limit_seconds is None:
            return 0.0
        if baseline_total_time <= self.profile.time_limit_seconds:
            return 0.0
        return _clip(
            (baseline_total_time - self.profile.time_limit_seconds)
            / baseline_total_time
        )

    def _calculate_environmental_load(self, instruction_pressure: float) -> float:
        """Рассчитать интегральную внешнюю нагрузку условий декодирования."""
        lighting_pressure = 1.0 - self.profile.lighting_quality
        return _clip(
            self.profile.noise_level * self.config.noise_weight
            + self.profile.workload_level * self.config.workload_weight
            + instruction_pressure * self.config.instruction_weight
            + self.profile.interruption_rate * self.config.interruption_weight
            + lighting_pressure * self.config.lighting_weight
        )


class DecodingExecutionContextModel:
    """Оркестратор детерминированной оценки оператора и условий."""

    def __init__(
        self,
        operator_config: DecodingOperatorConfig | None = None,
        condition_config: DecodingConditionConfig | None = None,
    ) -> None:
        """Инициализировать независимые модели O_d и U_d."""
        self.operator_model = DecodingOperatorModel(operator_config)
        self.condition_model = DecodingConditionModel(condition_config)

    def estimate_plan(self, plan: DecodingPlan) -> DecodingExecutionContext:
        """Сформировать единый контекст выполнения обратной процедуры."""
        operator_estimate = self.operator_model.estimate_plan(plan)
        condition_estimate = self.condition_model.estimate_plan(operator_estimate)
        return DecodingExecutionContext(
            operator_estimate=operator_estimate,
            condition_estimate=condition_estimate,
            metadata={
                "operator_id": operator_estimate.operator_id,
                "condition_id": condition_estimate.condition_id,
                "decoding_procedure_id": plan.decoding_procedure_id,
                "encoded_message_id": plan.encoded_message_id,
                "source_message_id": plan.source_message_id,
                "uses_only_decoding_plan": True,
                "uses_source_value_for_decoding": False,
                "generates_decoding_errors": False,
                "creates_decoded_message": False,
            },
        )


def summarize_decoding_context(
    context: DecodingExecutionContext,
) -> dict[str, int | float | str | bool]:
    """Вернуть компактную сводку детерминированного контекста этапа 4."""
    operator_metadata = context.operator_estimate.metadata
    condition_metadata = context.condition_estimate.metadata
    return {
        "operator_id": context.operator_estimate.operator_id,
        "condition_id": context.condition_estimate.condition_id,
        "step_count": int(operator_metadata["step_count"]),
        "total_estimated_time": float(operator_metadata["total_estimated_time"]),
        "total_adjusted_time": float(condition_metadata["total_adjusted_time"]),
        "mean_attention": float(operator_metadata["mean_attention"]),
        "mean_adjusted_attention": float(
            condition_metadata["mean_adjusted_attention"]
        ),
        "time_pressure": float(condition_metadata["time_pressure"]),
        "uses_source_value_for_decoding": False,
        "generates_decoding_errors": False,
    }
