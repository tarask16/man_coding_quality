"""Модель оператора O в процессе ручного кодирования.

Модуль относится к главе 3 диссертации и описывает априорные
характеристики оператора: подготовленность, опыт, базовое внимание,
утомляемость и навык контроля. На данном этапе модель не генерирует
ошибки и не изменяет результат кодирования. Она только рассчитывает
детерминированные оценки времени, внимания и трудоемкости для шагов
нормативного плана средства S.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from manual_coding_sim.procedure_model import ProcedurePlan, ProcedureStep


def _validate_unit_interval(value: float, field_name: str) -> None:
    """Проверяет принадлежность значения диапазону [0; 1]."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Поле {field_name} должно находиться в диапазоне [0; 1].")


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Ограничивает числовое значение заданным интервалом."""
    return min(upper, max(lower, value))


@dataclass(frozen=True)
class OperatorProfile:
    """
    Априорный профиль оператора O.

    Профиль описывает характеристики человека-оператора до начала
    процедуры E_h. Эти характеристики могут использоваться как часть
    априорного описания сценария A = {S, O, U, G, K}.
    """

    operator_id: str = "O_001"
    preparation_level: float = 0.75
    experience_level: float = 0.65
    base_attention: float = 0.85
    fatigue_rate: float = 0.015
    control_skill: float = 0.70
    work_rate: float = 1.00
    min_attention: float = 0.25

    def validate(self) -> None:
        """Проверяет корректность априорного профиля оператора O."""
        if not self.operator_id:
            raise ValueError("Идентификатор оператора O не задан.")

        _validate_unit_interval(self.preparation_level, "preparation_level")
        _validate_unit_interval(self.experience_level, "experience_level")
        _validate_unit_interval(self.base_attention, "base_attention")
        _validate_unit_interval(self.fatigue_rate, "fatigue_rate")
        _validate_unit_interval(self.control_skill, "control_skill")
        _validate_unit_interval(self.min_attention, "min_attention")

        if self.work_rate <= 0:
            raise ValueError("Скорость работы оператора должна быть положительной.")

        if self.min_attention > self.base_attention:
            raise ValueError(
                "Минимальное внимание не должно превышать базовое внимание."
            )


@dataclass(frozen=True)
class OperatorModelConfig:
    """Конфигурация модели оператора O."""

    profile: OperatorProfile = field(default_factory=OperatorProfile)
    reference_penalty: float = 0.08
    control_marker_penalty: float = 0.05
    complexity_weight: float = 0.35
    fatigue_time_weight: float = 0.45

    def validate(self) -> None:
        """Проверяет параметры модели оператора."""
        self.profile.validate()
        _validate_unit_interval(self.reference_penalty, "reference_penalty")
        _validate_unit_interval(
            self.control_marker_penalty,
            "control_marker_penalty",
        )
        _validate_unit_interval(self.complexity_weight, "complexity_weight")
        _validate_unit_interval(self.fatigue_time_weight, "fatigue_time_weight")


@dataclass(frozen=True)
class OperatorState:
    """
    Расчетное состояние оператора на шаге процедуры.

    Состояние используется для детерминированной оценки внимания и
    накопленной утомленности. Оно не является фактическим протоколом
    ошибок и не содержит результата кодирования.
    """

    operation_index: int
    completed_operations: int
    attention: float
    fatigue: float
    accumulated_effort: float


@dataclass(frozen=True)
class OperatorStepEstimate:
    """
    Оценка влияния оператора O на один шаг нормативного плана S.

    Оценка содержит априорные расчетные величины: ожидаемое время,
    внимание, утомленность и трудоемкость. Она не указывает, возникла
    ли ошибка, и не формирует кодированное представление C.
    """

    step_id: str
    source_position: int
    operator_id: str
    estimated_time: float
    attention: float
    fatigue: float
    effort: float
    complexity_pressure: float
    reference_pressure: float
    control_skill: float


@dataclass(frozen=True)
class OperatorPlanEstimate:
    """
    Оценка выполнения оператором нормативного плана ручного кодирования.

    Данный объект связывает профиль оператора O с планом средства S и
    формирует априорные характеристики, которые позже будут включены
    в пространство признаков X_prior.
    """

    operator_id: str
    procedure_id: str
    message_id: str
    step_estimates: tuple[OperatorStepEstimate, ...]
    metadata: dict[str, int | float | str]


class OperatorModel:
    """
    Модель оператора O.

    Модель оценивает, как подготовленность, опыт, внимание и
    утомляемость оператора влияют на длительность и трудоемкость
    выполнения нормативных шагов средства ручного кодирования S.
    """

    def __init__(self, config: OperatorModelConfig | None = None) -> None:
        """Инициализирует модель оператора O."""
        self.config = config or OperatorModelConfig()
        self.config.validate()
        self.profile = self.config.profile

    def estimate_step(
        self,
        step: ProcedureStep,
        operation_index: int,
    ) -> OperatorStepEstimate:
        """Рассчитывает влияние оператора O на один нормативный шаг."""
        if operation_index < 1:
            raise ValueError("Индекс операции должен начинаться с 1.")

        state = self._calculate_state(step=step, operation_index=operation_index)
        reference_pressure = self._calculate_reference_pressure(step)
        complexity_pressure = self._calculate_complexity_pressure(step)

        skill_deficit = 1.0 - (
            self.profile.preparation_level + self.profile.experience_level
        ) / 2.0
        time_multiplier = (
            1.0
            + complexity_pressure * skill_deficit
            + reference_pressure
            + state.fatigue * self.config.fatigue_time_weight
        ) / self.profile.work_rate

        estimated_time = round(step.nominal_time * time_multiplier, 6)
        effort = round(step.complexity * (1.0 + state.fatigue), 6)

        return OperatorStepEstimate(
            step_id=step.step_id,
            source_position=step.source_position,
            operator_id=self.profile.operator_id,
            estimated_time=estimated_time,
            attention=round(state.attention, 6),
            fatigue=round(state.fatigue, 6),
            effort=effort,
            complexity_pressure=round(complexity_pressure, 6),
            reference_pressure=round(reference_pressure, 6),
            control_skill=self.profile.control_skill,
        )

    def estimate_plan(self, plan: ProcedurePlan) -> OperatorPlanEstimate:
        """Рассчитывает априорную оценку выполнения плана оператором O."""
        if not plan.steps:
            raise ValueError("План процедуры не содержит шагов для оценки.")

        step_estimates = tuple(
            self.estimate_step(step=step, operation_index=index)
            for index, step in enumerate(plan.steps, start=1)
        )
        total_estimated_time = sum(item.estimated_time for item in step_estimates)
        total_effort = sum(item.effort for item in step_estimates)
        mean_attention = sum(item.attention for item in step_estimates) / len(
            step_estimates
        )
        final_fatigue = step_estimates[-1].fatigue

        return OperatorPlanEstimate(
            operator_id=self.profile.operator_id,
            procedure_id=plan.procedure_id,
            message_id=plan.message_id,
            step_estimates=step_estimates,
            metadata={
                "step_count": len(step_estimates),
                "total_estimated_time": round(total_estimated_time, 6),
                "total_effort": round(total_effort, 6),
                "mean_attention": round(mean_attention, 6),
                "final_fatigue": round(final_fatigue, 6),
                "preparation_level": self.profile.preparation_level,
                "experience_level": self.profile.experience_level,
                "control_skill": self.profile.control_skill,
            },
        )

    def estimate_batch(
        self,
        plans: Iterable[ProcedurePlan],
    ) -> tuple[OperatorPlanEstimate, ...]:
        """Рассчитывает оценки оператора O для пакета планов средства S."""
        return tuple(self.estimate_plan(plan) for plan in plans)

    def _calculate_state(
        self,
        step: ProcedureStep,
        operation_index: int,
    ) -> OperatorState:
        """Рассчитывает внимание и утомленность оператора на шаге."""
        completed_operations = operation_index - 1
        fatigue = _clip(
            completed_operations * self.profile.fatigue_rate
            + step.complexity * 0.05,
        )
        attention_loss = (
            fatigue
            + step.complexity * 0.10
            + self._calculate_reference_pressure(step) * 0.25
        )
        attention = max(
            self.profile.min_attention,
            _clip(self.profile.base_attention - attention_loss),
        )
        accumulated_effort = step.complexity * (1.0 + fatigue)

        return OperatorState(
            operation_index=operation_index,
            completed_operations=completed_operations,
            attention=attention,
            fatigue=fatigue,
            accumulated_effort=accumulated_effort,
        )

    def _calculate_reference_pressure(self, step: ProcedureStep) -> float:
        """Рассчитывает нагрузку от обращения к справочной части средства S."""
        pressure = 0.0
        if step.reference_required:
            pressure += self.config.reference_penalty
        if step.control_marker_required:
            pressure += self.config.control_marker_penalty
        return _clip(pressure)

    def _calculate_complexity_pressure(self, step: ProcedureStep) -> float:
        """Рассчитывает нагрузку от сложности нормативной операции."""
        return _clip(step.complexity * self.config.complexity_weight)


def summarize_operator_estimate(
    estimate: OperatorPlanEstimate,
) -> dict[str, int | float | str]:
    """Возвращает контрольную сводку по априорной оценке оператора O."""
    if not estimate.step_estimates:
        raise ValueError("Оценка оператора не содержит шагов.")

    return {
        "operator_id": estimate.operator_id,
        "procedure_id": estimate.procedure_id,
        "message_id": estimate.message_id,
        "step_count": int(estimate.metadata["step_count"]),
        "total_estimated_time": float(
            estimate.metadata["total_estimated_time"]
        ),
        "total_effort": float(estimate.metadata["total_effort"]),
        "mean_attention": float(estimate.metadata["mean_attention"]),
        "final_fatigue": float(estimate.metadata["final_fatigue"]),
        "control_skill": float(estimate.metadata["control_skill"]),
    }


def operator_estimates_to_rows(
    estimates: Iterable[OperatorPlanEstimate],
) -> list[dict[str, int | float | str]]:
    """Преобразует оценки оператора O в строки контрольной таблицы."""
    return [summarize_operator_estimate(estimate) for estimate in estimates]
