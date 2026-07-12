"""Вероятностная модель ошибок при ручном кодировании.

Модуль относится к главе 3 диссертации и описывает переход от
априорной оценки S + O + U к вероятностному протоколу ошибок. На этом
этапе не моделируются контрольные процедуры K, обнаружение и исправление
ошибок, декодирование D_h и восстановленное сообщение M'.

Модель формирует только событие возникновения ошибки на шаге ручного
кодирования и тип абстрактной ошибки. Полученный протокол далее будет
использоваться как основа для моделирования контроля K и расчета
фактических признаков X_fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Iterable

from manual_coding_sim.condition_model import (
    ConditionPlanEstimate,
    ConditionStepEstimate,
)


def _validate_unit_interval(value: float, field_name: str) -> None:
    """Проверяет принадлежность значения диапазону [0; 1]."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Поле {field_name} должно находиться в диапазоне [0; 1].")


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Ограничивает числовое значение заданным интервалом."""
    return min(upper, max(lower, value))


@dataclass(frozen=True)
class ErrorModelConfig:
    """
    Конфигурация вероятностной модели ошибок.

    Параметры задают априорное преобразование расчетных характеристик
    S + O + U в вероятность ошибки на шаге процедуры E_h. Конфигурация
    не содержит фактических результатов декодирования и не использует M'.
    """

    error_model_id: str = "ERR_001"
    base_error_rate: float = 0.035
    attention_deficit_weight: float = 0.45
    environmental_load_weight: float = 0.30
    time_pressure_weight: float = 0.25
    instruction_pressure_weight: float = 0.20
    max_error_probability: float = 0.85
    default_error_weight: float = 1.0
    error_types: tuple[str, ...] = (
        "abstract_substitution_error",
        "abstract_omission_error",
        "abstract_position_error",
        "abstract_reference_error",
        "abstract_service_marker_error",
    )

    def validate(self) -> None:
        """Проверяет корректность параметров вероятностной модели ошибок."""
        if not self.error_model_id:
            raise ValueError("Идентификатор модели ошибок не задан.")

        _validate_unit_interval(self.base_error_rate, "base_error_rate")
        _validate_unit_interval(
            self.attention_deficit_weight,
            "attention_deficit_weight",
        )
        _validate_unit_interval(
            self.environmental_load_weight,
            "environmental_load_weight",
        )
        _validate_unit_interval(self.time_pressure_weight, "time_pressure_weight")
        _validate_unit_interval(
            self.instruction_pressure_weight,
            "instruction_pressure_weight",
        )
        _validate_unit_interval(
            self.max_error_probability,
            "max_error_probability",
        )

        if self.default_error_weight <= 0:
            raise ValueError("Вес ошибки должен быть положительным.")

        if not self.error_types:
            raise ValueError("Список типов абстрактных ошибок не должен быть пустым.")

        if self.base_error_rate > self.max_error_probability:
            raise ValueError(
                "Базовая вероятность ошибки не должна превышать верхний предел."
            )


@dataclass(frozen=True)
class ErrorStepOutcome:
    """
    Вероятностный результат одного шага ручного кодирования.

    Объект фиксирует вероятность ошибки и факт ее возникновения в рамках
    моделируемого прогона. Контроль, обнаружение и исправление ошибки
    на этом этапе не выполняются.
    """

    step_id: str
    source_position: int
    operator_id: str
    condition_id: str
    error_model_id: str
    error_probability: float
    random_draw: float
    error_occurred: bool
    error_type: str | None
    error_weight: float
    attention_deficit: float
    environmental_load: float
    time_pressure: float
    instruction_pressure: float


@dataclass(frozen=True)
class ErrorProtocol:
    """
    Протокол вероятностного моделирования ошибок.

    Протокол является промежуточным артефактом между априорными оценками
    S + O + U и последующим моделированием контрольных процедур K. Он
    еще не является итоговым фактическим качеством q(A), поскольку не
    учитывает обнаружение, исправление и декодирование.
    """

    error_model_id: str
    condition_id: str
    operator_id: str
    procedure_id: str
    message_id: str
    step_outcomes: tuple[ErrorStepOutcome, ...]
    metadata: dict[str, int | float | str]


class ErrorModel:
    """
    Вероятностная модель ошибок ручного кодирования.

    Модель рассчитывает вероятность ошибки на каждом шаге по априорным
    характеристикам внимания, нагрузки условий U, дефицита времени и
    доступности инструкции. Фактические события ошибок генерируются
    псевдослучайно и воспроизводимы при фиксированном random_seed.
    """

    def __init__(
        self,
        config: ErrorModelConfig | None = None,
        random_seed: int = 42,
    ) -> None:
        """Инициализирует модель ошибок с фиксированным зерном генератора."""
        self.config = config or ErrorModelConfig()
        self.config.validate()
        self.random_seed = random_seed
        self._rng = Random(random_seed)

    def generate_protocol(
        self,
        condition_estimate: ConditionPlanEstimate,
    ) -> ErrorProtocol:
        """Формирует протокол ошибок для оценки S + O + U."""
        if not condition_estimate.step_estimates:
            raise ValueError("Оценка условий U не содержит шагов.")

        outcomes = tuple(
            self.generate_step_outcome(step)
            for step in condition_estimate.step_estimates
        )

        error_count = sum(1 for outcome in outcomes if outcome.error_occurred)
        weighted_error_sum = sum(outcome.error_weight for outcome in outcomes)
        mean_error_probability = sum(
            outcome.error_probability for outcome in outcomes
        ) / len(outcomes)

        return ErrorProtocol(
            error_model_id=self.config.error_model_id,
            condition_id=condition_estimate.condition_id,
            operator_id=condition_estimate.operator_id,
            procedure_id=condition_estimate.procedure_id,
            message_id=condition_estimate.message_id,
            step_outcomes=outcomes,
            metadata={
                "step_count": len(outcomes),
                "error_count": error_count,
                "error_rate": round(error_count / len(outcomes), 6),
                "weighted_error_sum": round(weighted_error_sum, 6),
                "mean_error_probability": round(mean_error_probability, 6),
                "random_seed": self.random_seed,
            },
        )

    def generate_step_outcome(
        self,
        step: ConditionStepEstimate,
    ) -> ErrorStepOutcome:
        """Формирует вероятностный результат одного шага."""
        error_probability = self.calculate_error_probability(step)
        random_draw = self._rng.random()
        error_occurred = random_draw < error_probability
        error_type = self._choose_error_type() if error_occurred else None
        error_weight = self._calculate_error_weight(step) if error_occurred else 0.0
        attention_deficit = _clip(1.0 - step.adjusted_attention)

        return ErrorStepOutcome(
            step_id=step.step_id,
            source_position=step.source_position,
            operator_id=step.operator_id,
            condition_id=step.condition_id,
            error_model_id=self.config.error_model_id,
            error_probability=round(error_probability, 6),
            random_draw=round(random_draw, 6),
            error_occurred=error_occurred,
            error_type=error_type,
            error_weight=round(error_weight, 6),
            attention_deficit=round(attention_deficit, 6),
            environmental_load=round(step.environmental_load, 6),
            time_pressure=round(step.time_pressure, 6),
            instruction_pressure=round(step.instruction_pressure, 6),
        )

    def generate_batch(
        self,
        condition_estimates: Iterable[ConditionPlanEstimate],
    ) -> tuple[ErrorProtocol, ...]:
        """Формирует протоколы ошибок для пакета оценок условий U."""
        return tuple(
            self.generate_protocol(estimate) for estimate in condition_estimates
        )

    def calculate_error_probability(self, step: ConditionStepEstimate) -> float:
        """Рассчитывает вероятность ошибки для шага процедуры E_h."""
        attention_deficit = _clip(1.0 - step.adjusted_attention)
        pressure = (
            attention_deficit * self.config.attention_deficit_weight
            + step.environmental_load * self.config.environmental_load_weight
            + step.time_pressure * self.config.time_pressure_weight
            + step.instruction_pressure * self.config.instruction_pressure_weight
        )
        probability = self.config.base_error_rate * (1.0 + pressure)
        return _clip(probability, upper=self.config.max_error_probability)

    def reset(self) -> None:
        """Возвращает генератор случайных чисел к начальному состоянию."""
        self._rng = Random(self.random_seed)

    def _choose_error_type(self) -> str:
        """Выбирает тип абстрактной ошибки без раскрытия прикладного способа кодирования."""
        return self._rng.choice(self.config.error_types)

    def _calculate_error_weight(self, step: ConditionStepEstimate) -> float:
        """Рассчитывает условный вес ошибки для последующей оценки качества."""
        pressure_factor = 1.0 + 0.5 * step.time_pressure + 0.3 * step.environmental_load
        return self.config.default_error_weight * pressure_factor


def summarize_error_protocol(
    protocol: ErrorProtocol,
) -> dict[str, int | float | str]:
    """Возвращает контрольную сводку по протоколу ошибок."""
    if not protocol.step_outcomes:
        raise ValueError("Протокол ошибок не содержит шагов.")

    return {
        "error_model_id": protocol.error_model_id,
        "condition_id": protocol.condition_id,
        "operator_id": protocol.operator_id,
        "procedure_id": protocol.procedure_id,
        "message_id": protocol.message_id,
        "step_count": int(protocol.metadata["step_count"]),
        "error_count": int(protocol.metadata["error_count"]),
        "error_rate": float(protocol.metadata["error_rate"]),
        "weighted_error_sum": float(protocol.metadata["weighted_error_sum"]),
        "mean_error_probability": float(
            protocol.metadata["mean_error_probability"]
        ),
        "random_seed": int(protocol.metadata["random_seed"]),
    }


def error_protocols_to_rows(
    protocols: Iterable[ErrorProtocol],
) -> list[dict[str, int | float | str]]:
    """Преобразует протоколы ошибок в строки контрольной таблицы."""
    return [summarize_error_protocol(protocol) for protocol in protocols]
