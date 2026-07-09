"""Модель условий применения U в процессе ручного кодирования.

Модуль относится к главе 3 диссертации и описывает влияние внешних
условий применения на априорную оценку выполнения нормативного плана
оператором O. Условия U учитывают дефицит времени, шум, доступность
инструкции, рабочую нагрузку, частоту прерываний и качество освещения.

На данном этапе модель не генерирует ошибки, не выполняет контрольные
процедуры K и не формирует фактический результат M'. Она только
преобразует оценку O + ProcedurePlan в оценку O + U + ProcedurePlan,
пригодную для дальнейшего формирования априорных признаков X_prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from manual_coding_sim.operator_model import OperatorPlanEstimate, OperatorStepEstimate


def _validate_unit_interval(value: float, field_name: str) -> None:
    """Проверяет принадлежность значения диапазону [0; 1]."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Поле {field_name} должно находиться в диапазоне [0; 1].")


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Ограничивает числовое значение заданным интервалом."""
    return min(upper, max(lower, value))


@dataclass(frozen=True)
class ConditionProfile:
    """
    Априорный профиль условий применения U.

    Профиль описывает внешние факторы сценария A = {S, O, U, G, K},
    которые известны до фактического выполнения процедуры E_h и могут
    использоваться при априорной оценке качества ручного кодирования.
    """

    condition_id: str = "U_001"
    time_limit_seconds: float | None = None
    noise_level: float = 0.20
    instruction_access: float = 0.90
    workload_level: float = 0.30
    interruption_rate: float = 0.10
    lighting_quality: float = 0.90

    def validate(self) -> None:
        """Проверяет корректность профиля условий применения U."""
        if not self.condition_id:
            raise ValueError("Идентификатор условий применения U не задан.")

        if self.time_limit_seconds is not None and self.time_limit_seconds <= 0:
            raise ValueError("Лимит времени должен быть положительным.")

        _validate_unit_interval(self.noise_level, "noise_level")
        _validate_unit_interval(self.instruction_access, "instruction_access")
        _validate_unit_interval(self.workload_level, "workload_level")
        _validate_unit_interval(self.interruption_rate, "interruption_rate")
        _validate_unit_interval(self.lighting_quality, "lighting_quality")


@dataclass(frozen=True)
class ConditionModelConfig:
    """Конфигурация модели условий применения U."""

    profile: ConditionProfile = field(default_factory=ConditionProfile)
    noise_weight: float = 0.25
    workload_weight: float = 0.30
    instruction_weight: float = 0.20
    interruption_weight: float = 0.15
    lighting_weight: float = 0.10
    time_pressure_weight: float = 0.30
    time_expansion_weight: float = 0.45

    def validate(self) -> None:
        """Проверяет параметры модели условий применения U."""
        self.profile.validate()
        _validate_unit_interval(self.noise_weight, "noise_weight")
        _validate_unit_interval(self.workload_weight, "workload_weight")
        _validate_unit_interval(self.instruction_weight, "instruction_weight")
        _validate_unit_interval(self.interruption_weight, "interruption_weight")
        _validate_unit_interval(self.lighting_weight, "lighting_weight")
        _validate_unit_interval(self.time_pressure_weight, "time_pressure_weight")
        _validate_unit_interval(self.time_expansion_weight, "time_expansion_weight")


@dataclass(frozen=True)
class ConditionStepEstimate:
    """
    Оценка влияния условий U на один шаг нормативного плана.

    Оценка содержит только априорные расчетные величины. Она не является
    фактическим протоколом применения средства S и не содержит признака
    возникновения ошибки.
    """

    step_id: str
    source_position: int
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


@dataclass(frozen=True)
class ConditionPlanEstimate:
    """
    Оценка выполнения плана при заданных условиях применения U.

    Объект связывает априорную оценку оператора O с условиями U и
    формирует характеристики, которые позже могут быть включены в
    априорное пространство признаков X_prior.
    """

    condition_id: str
    operator_id: str
    procedure_id: str
    message_id: str
    step_estimates: tuple[ConditionStepEstimate, ...]
    metadata: dict[str, int | float | str]


class ConditionModel:
    """
    Модель условий применения U.

    Модель рассчитывает влияние внешней обстановки на время выполнения,
    внимание и устойчивость нормативного процесса ручного кодирования.
    На этом этапе влияние условий является детерминированной априорной
    оценкой и не приводит к генерации ошибок.
    """

    def __init__(self, config: ConditionModelConfig | None = None) -> None:
        """Инициализирует модель условий применения U."""
        self.config = config or ConditionModelConfig()
        self.config.validate()
        self.profile = self.config.profile

    def estimate_step(
        self,
        step: OperatorStepEstimate,
        time_pressure: float,
    ) -> ConditionStepEstimate:
        """Рассчитывает влияние условий U на один оцененный шаг оператора O."""
        _validate_unit_interval(time_pressure, "time_pressure")

        instruction_pressure = self._calculate_instruction_pressure()
        environmental_load = self._calculate_environmental_load(
            instruction_pressure=instruction_pressure,
        )
        time_multiplier = (
            1.0
            + environmental_load * self.config.time_expansion_weight
            + time_pressure * self.config.time_pressure_weight
        )
        attention_modifier = self._calculate_attention_modifier(
            environmental_load=environmental_load,
            time_pressure=time_pressure,
        )
        adjusted_time = step.estimated_time * time_multiplier
        adjusted_attention = step.attention * attention_modifier
        stability_index = 1.0 - _clip((environmental_load + time_pressure) / 2.0)

        return ConditionStepEstimate(
            step_id=step.step_id,
            source_position=step.source_position,
            operator_id=step.operator_id,
            condition_id=self.profile.condition_id,
            baseline_time=round(step.estimated_time, 6),
            adjusted_time=round(adjusted_time, 6),
            baseline_attention=round(step.attention, 6),
            adjusted_attention=round(adjusted_attention, 6),
            environmental_load=round(environmental_load, 6),
            time_pressure=round(time_pressure, 6),
            instruction_pressure=round(instruction_pressure, 6),
            stability_index=round(stability_index, 6),
        )

    def estimate_plan(
        self,
        operator_estimate: OperatorPlanEstimate,
    ) -> ConditionPlanEstimate:
        """Рассчитывает влияние условий U на оценку выполнения плана оператором O."""
        if not operator_estimate.step_estimates:
            raise ValueError("Оценка оператора O не содержит шагов.")

        baseline_total_time = float(
            operator_estimate.metadata["total_estimated_time"]
        )
        time_pressure = self._calculate_time_pressure(baseline_total_time)
        step_estimates = tuple(
            self.estimate_step(step=step, time_pressure=time_pressure)
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

        return ConditionPlanEstimate(
            condition_id=self.profile.condition_id,
            operator_id=operator_estimate.operator_id,
            procedure_id=operator_estimate.procedure_id,
            message_id=operator_estimate.message_id,
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
            },
        )

    def estimate_batch(
        self,
        operator_estimates: Iterable[OperatorPlanEstimate],
    ) -> tuple[ConditionPlanEstimate, ...]:
        """Рассчитывает влияние условий U для пакета оценок оператора O."""
        return tuple(self.estimate_plan(estimate) for estimate in operator_estimates)

    def _calculate_time_pressure(self, baseline_total_time: float) -> float:
        """Рассчитывает давление дефицита времени для заданного плана."""
        if baseline_total_time <= 0:
            raise ValueError("Базовое расчетное время должно быть положительным.")

        if self.profile.time_limit_seconds is None:
            return 0.0

        if baseline_total_time <= self.profile.time_limit_seconds:
            return 0.0

        deficit = baseline_total_time - self.profile.time_limit_seconds
        return _clip(deficit / baseline_total_time)

    def _calculate_instruction_pressure(self) -> float:
        """Рассчитывает давление от ограниченной доступности инструкции."""
        return _clip(1.0 - self.profile.instruction_access)

    def _calculate_environmental_load(self, instruction_pressure: float) -> float:
        """Рассчитывает интегральную нагрузку условий применения U."""
        lighting_deficit = 1.0 - self.profile.lighting_quality
        load = (
            self.profile.noise_level * self.config.noise_weight
            + self.profile.workload_level * self.config.workload_weight
            + instruction_pressure * self.config.instruction_weight
            + self.profile.interruption_rate * self.config.interruption_weight
            + lighting_deficit * self.config.lighting_weight
        )
        return _clip(load)

    def _calculate_attention_modifier(
        self,
        environmental_load: float,
        time_pressure: float,
    ) -> float:
        """Рассчитывает множитель внимания под воздействием условий U."""
        attention_loss = _clip(
            environmental_load * 0.60 + time_pressure * 0.35,
        )
        return _clip(1.0 - attention_loss, lower=0.10, upper=1.0)


def summarize_condition_estimate(
    estimate: ConditionPlanEstimate,
) -> dict[str, int | float | str]:
    """Возвращает контрольную сводку по оценке условий применения U."""
    if not estimate.step_estimates:
        raise ValueError("Оценка условий применения U не содержит шагов.")

    return {
        "condition_id": estimate.condition_id,
        "operator_id": estimate.operator_id,
        "procedure_id": estimate.procedure_id,
        "message_id": estimate.message_id,
        "step_count": int(estimate.metadata["step_count"]),
        "baseline_total_time": float(estimate.metadata["baseline_total_time"]),
        "total_adjusted_time": float(estimate.metadata["total_adjusted_time"]),
        "mean_adjusted_attention": float(
            estimate.metadata["mean_adjusted_attention"]
        ),
        "mean_environmental_load": float(
            estimate.metadata["mean_environmental_load"]
        ),
        "mean_stability_index": float(
            estimate.metadata["mean_stability_index"]
        ),
        "time_pressure": float(estimate.metadata["time_pressure"]),
    }


def condition_estimates_to_rows(
    estimates: Iterable[ConditionPlanEstimate],
) -> list[dict[str, int | float | str]]:
    """Преобразует оценки условий U в строки контрольной таблицы."""
    return [summarize_condition_estimate(estimate) for estimate in estimates]
