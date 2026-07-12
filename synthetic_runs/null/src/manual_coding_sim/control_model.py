"""Модель контрольных процедур K при ручном кодировании.

Модуль относится к главе 3 диссертации и описывает переход от
вероятностного протокола ошибок к протоколу контроля. Контрольные
процедуры K моделируют обнаружение и исправление ошибок, возникших
на шагах ручного кодирования.

На данном этапе не выполняются декодирование D_h, восстановление M'
и итоговый расчет фактического качества q(A). Результат ControlModel
является промежуточным фактическим протоколом, который будет использован
при построении симулятора протоколов и извлечении фактических признаков
X_fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Iterable

from manual_coding_sim.error_model import ErrorProtocol, ErrorStepOutcome


def _validate_unit_interval(value: float, field_name: str) -> None:
    """Проверяет принадлежность значения диапазону [0; 1]."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Поле {field_name} должно находиться в диапазоне [0; 1].")


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Ограничивает числовое значение заданным интервалом."""
    return min(upper, max(lower, value))


@dataclass(frozen=True)
class ControlProfile:
    """
    Априорный профиль контрольных процедур K.

    Профиль описывает контроль, который применяется после выполнения
    ручных операций кодирования. Параметры профиля известны до
    фактического выполнения E_h и определяют вероятность обнаружения
    и исправления возникших ошибок.
    """

    control_id: str = "K_001"
    detection_skill: float = 0.75
    correction_skill: float = 0.70
    reference_check_coverage: float = 0.80
    repeated_check_coverage: float = 0.55
    critical_error_priority: float = 0.70

    def validate(self) -> None:
        """Проверяет корректность профиля контрольных процедур K."""
        if not self.control_id:
            raise ValueError("Идентификатор контрольных процедур K не задан.")

        _validate_unit_interval(self.detection_skill, "detection_skill")
        _validate_unit_interval(self.correction_skill, "correction_skill")
        _validate_unit_interval(
            self.reference_check_coverage,
            "reference_check_coverage",
        )
        _validate_unit_interval(
            self.repeated_check_coverage,
            "repeated_check_coverage",
        )
        _validate_unit_interval(
            self.critical_error_priority,
            "critical_error_priority",
        )


@dataclass(frozen=True)
class ControlModelConfig:
    """Конфигурация модели контрольных процедур K."""

    profile: ControlProfile = field(default_factory=ControlProfile)
    base_detection_rate: float = 0.35
    base_correction_rate: float = 0.45
    detection_skill_weight: float = 0.30
    correction_skill_weight: float = 0.35
    coverage_weight: float = 0.25
    criticality_weight: float = 0.20
    attention_penalty_weight: float = 0.18
    environmental_penalty_weight: float = 0.12
    time_pressure_penalty_weight: float = 0.15
    max_detection_probability: float = 0.95
    max_correction_probability: float = 0.90
    base_control_effort: float = 1.0

    def validate(self) -> None:
        """Проверяет параметры модели контрольных процедур K."""
        self.profile.validate()

        _validate_unit_interval(self.base_detection_rate, "base_detection_rate")
        _validate_unit_interval(self.base_correction_rate, "base_correction_rate")
        _validate_unit_interval(
            self.detection_skill_weight,
            "detection_skill_weight",
        )
        _validate_unit_interval(
            self.correction_skill_weight,
            "correction_skill_weight",
        )
        _validate_unit_interval(self.coverage_weight, "coverage_weight")
        _validate_unit_interval(self.criticality_weight, "criticality_weight")
        _validate_unit_interval(
            self.attention_penalty_weight,
            "attention_penalty_weight",
        )
        _validate_unit_interval(
            self.environmental_penalty_weight,
            "environmental_penalty_weight",
        )
        _validate_unit_interval(
            self.time_pressure_penalty_weight,
            "time_pressure_penalty_weight",
        )
        _validate_unit_interval(
            self.max_detection_probability,
            "max_detection_probability",
        )
        _validate_unit_interval(
            self.max_correction_probability,
            "max_correction_probability",
        )

        if self.base_control_effort <= 0:
            raise ValueError("Базовая трудоемкость контроля должна быть положительной.")

        if self.base_detection_rate > self.max_detection_probability:
            raise ValueError(
                "Базовая вероятность обнаружения не должна превышать верхний предел."
            )

        if self.base_correction_rate > self.max_correction_probability:
            raise ValueError(
                "Базовая вероятность исправления не должна превышать верхний предел."
            )


@dataclass(frozen=True)
class ControlStepOutcome:
    """
    Результат контроля одного шага ручного кодирования.

    Объект связывает возникшую ошибку с процедурами ее обнаружения и
    исправления. Остаточная ошибка соответствует ошибке, которая
    сохранилась после применения контрольных процедур K.
    """

    step_id: str
    source_position: int
    control_id: str
    error_model_id: str
    error_occurred: bool
    error_type: str | None
    error_weight: float
    detection_probability: float
    detection_draw: float | None
    error_detected: bool
    correction_probability: float
    correction_draw: float | None
    error_corrected: bool
    residual_error: bool
    control_effort: float


@dataclass(frozen=True)
class ControlProtocol:
    """
    Протокол применения контрольных процедур K.

    Протокол является результатом обработки ErrorProtocol и содержит
    сведения о выявленных, исправленных и остаточных ошибках. Он еще
    не является восстановленным сообщением M' и не завершает расчет
    фактического качества q(A).
    """

    control_id: str
    error_model_id: str
    condition_id: str
    operator_id: str
    procedure_id: str
    message_id: str
    step_outcomes: tuple[ControlStepOutcome, ...]
    metadata: dict[str, int | float | str]


class ControlModel:
    """
    Модель контрольных процедур K.

    Модель рассчитывает вероятность обнаружения и исправления ошибок,
    возникших на шагах ручного кодирования. Генерация событий контроля
    является псевдослучайной и воспроизводимой при фиксированном
    random_seed.
    """

    def __init__(
        self,
        config: ControlModelConfig | None = None,
        random_seed: int = 42,
    ) -> None:
        """Инициализирует модель контроля с фиксированным зерном генератора."""
        self.config = config or ControlModelConfig()
        self.config.validate()
        self.profile = self.config.profile
        self.random_seed = random_seed
        self._rng = Random(random_seed)

    def generate_protocol(self, error_protocol: ErrorProtocol) -> ControlProtocol:
        """Формирует протокол контроля K для протокола ошибок."""
        if not error_protocol.step_outcomes:
            raise ValueError("Протокол ошибок не содержит шагов для контроля.")

        outcomes = tuple(
            self.generate_step_outcome(outcome)
            for outcome in error_protocol.step_outcomes
        )

        original_error_count = sum(
            1 for outcome in outcomes if outcome.error_occurred
        )
        detected_error_count = sum(
            1 for outcome in outcomes if outcome.error_detected
        )
        corrected_error_count = sum(
            1 for outcome in outcomes if outcome.error_corrected
        )
        residual_error_count = sum(
            1 for outcome in outcomes if outcome.residual_error
        )
        total_control_effort = sum(outcome.control_effort for outcome in outcomes)
        mean_detection_probability = sum(
            outcome.detection_probability for outcome in outcomes
        ) / len(outcomes)

        detection_rate = self._safe_ratio(
            numerator=detected_error_count,
            denominator=original_error_count,
        )
        correction_rate = self._safe_ratio(
            numerator=corrected_error_count,
            denominator=detected_error_count,
        )
        residual_error_rate = self._safe_ratio(
            numerator=residual_error_count,
            denominator=original_error_count,
        )

        return ControlProtocol(
            control_id=self.profile.control_id,
            error_model_id=error_protocol.error_model_id,
            condition_id=error_protocol.condition_id,
            operator_id=error_protocol.operator_id,
            procedure_id=error_protocol.procedure_id,
            message_id=error_protocol.message_id,
            step_outcomes=outcomes,
            metadata={
                "step_count": len(outcomes),
                "original_error_count": original_error_count,
                "detected_error_count": detected_error_count,
                "corrected_error_count": corrected_error_count,
                "residual_error_count": residual_error_count,
                "detection_rate": round(detection_rate, 6),
                "correction_rate": round(correction_rate, 6),
                "residual_error_rate": round(residual_error_rate, 6),
                "total_control_effort": round(total_control_effort, 6),
                "mean_detection_probability": round(
                    mean_detection_probability,
                    6,
                ),
                "random_seed": self.random_seed,
            },
        )

    def generate_step_outcome(
        self,
        error_outcome: ErrorStepOutcome,
    ) -> ControlStepOutcome:
        """Формирует результат контроля одного шага."""
        detection_probability = self.calculate_detection_probability(
            error_outcome,
        )
        correction_probability = self.calculate_correction_probability(
            error_outcome,
            detection_probability=detection_probability,
        )

        detection_draw = None
        correction_draw = None
        error_detected = False
        error_corrected = False

        if error_outcome.error_occurred:
            detection_draw = self._rng.random()
            error_detected = detection_draw < detection_probability

            if error_detected:
                correction_draw = self._rng.random()
                error_corrected = correction_draw < correction_probability

        residual_error = error_outcome.error_occurred and not error_corrected
        control_effort = self._calculate_control_effort(
            error_outcome=error_outcome,
            error_detected=error_detected,
            error_corrected=error_corrected,
        )

        return ControlStepOutcome(
            step_id=error_outcome.step_id,
            source_position=error_outcome.source_position,
            control_id=self.profile.control_id,
            error_model_id=error_outcome.error_model_id,
            error_occurred=error_outcome.error_occurred,
            error_type=error_outcome.error_type,
            error_weight=error_outcome.error_weight,
            detection_probability=round(detection_probability, 6),
            detection_draw=(
                round(detection_draw, 6)
                if detection_draw is not None
                else None
            ),
            error_detected=error_detected,
            correction_probability=round(correction_probability, 6),
            correction_draw=(
                round(correction_draw, 6)
                if correction_draw is not None
                else None
            ),
            error_corrected=error_corrected,
            residual_error=residual_error,
            control_effort=round(control_effort, 6),
        )

    def generate_batch(
        self,
        error_protocols: Iterable[ErrorProtocol],
    ) -> tuple[ControlProtocol, ...]:
        """Формирует протоколы контроля для пакета протоколов ошибок."""
        return tuple(
            self.generate_protocol(protocol) for protocol in error_protocols
        )

    def calculate_detection_probability(
        self,
        error_outcome: ErrorStepOutcome,
    ) -> float:
        """Рассчитывает вероятность обнаружения ошибки контрольной процедурой K."""
        if not error_outcome.error_occurred:
            return 0.0

        coverage = (
            self.profile.reference_check_coverage
            + self.profile.repeated_check_coverage
        ) / 2.0
        criticality = _clip(error_outcome.error_weight / 2.0)
        positive_part = (
            self.config.base_detection_rate
            + self.profile.detection_skill * self.config.detection_skill_weight
            + coverage * self.config.coverage_weight
            + criticality
            * self.profile.critical_error_priority
            * self.config.criticality_weight
        )
        penalty = (
            error_outcome.attention_deficit
            * self.config.attention_penalty_weight
            + error_outcome.environmental_load
            * self.config.environmental_penalty_weight
            + error_outcome.time_pressure
            * self.config.time_pressure_penalty_weight
        )
        probability = positive_part - penalty
        return _clip(probability, upper=self.config.max_detection_probability)

    def calculate_correction_probability(
        self,
        error_outcome: ErrorStepOutcome,
        detection_probability: float | None = None,
    ) -> float:
        """Рассчитывает вероятность исправления обнаруженной ошибки."""
        if not error_outcome.error_occurred:
            return 0.0

        if detection_probability is None:
            detection_probability = self.calculate_detection_probability(
                error_outcome,
            )

        coverage = (
            self.profile.reference_check_coverage
            + self.profile.repeated_check_coverage
        ) / 2.0
        penalty = (
            error_outcome.environmental_load * 0.10
            + error_outcome.time_pressure * 0.12
        )
        probability = (
            self.config.base_correction_rate
            + self.profile.correction_skill * self.config.correction_skill_weight
            + coverage * 0.15
            + detection_probability * 0.10
            - penalty
        )
        return _clip(probability, upper=self.config.max_correction_probability)

    def reset(self) -> None:
        """Возвращает генератор случайных чисел к начальному состоянию."""
        self._rng = Random(self.random_seed)

    @staticmethod
    def _safe_ratio(numerator: int, denominator: int) -> float:
        """Безопасно рассчитывает отношение для контрольных метрик."""
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _calculate_control_effort(
        self,
        error_outcome: ErrorStepOutcome,
        error_detected: bool,
        error_corrected: bool,
    ) -> float:
        """Рассчитывает условную трудоемкость контрольной процедуры."""
        effort = self.config.base_control_effort
        if error_outcome.error_occurred:
            effort += 0.5 + error_outcome.error_weight * 0.25
        if error_detected:
            effort += 0.5
        if error_corrected:
            effort += 0.4
        return effort


def summarize_control_protocol(
    protocol: ControlProtocol,
) -> dict[str, int | float | str]:
    """Возвращает контрольную сводку по протоколу K."""
    if not protocol.step_outcomes:
        raise ValueError("Протокол контроля не содержит шагов.")

    return {
        "control_id": protocol.control_id,
        "error_model_id": protocol.error_model_id,
        "condition_id": protocol.condition_id,
        "operator_id": protocol.operator_id,
        "procedure_id": protocol.procedure_id,
        "message_id": protocol.message_id,
        "step_count": int(protocol.metadata["step_count"]),
        "original_error_count": int(
            protocol.metadata["original_error_count"]
        ),
        "detected_error_count": int(
            protocol.metadata["detected_error_count"]
        ),
        "corrected_error_count": int(
            protocol.metadata["corrected_error_count"]
        ),
        "residual_error_count": int(
            protocol.metadata["residual_error_count"]
        ),
        "detection_rate": float(protocol.metadata["detection_rate"]),
        "correction_rate": float(protocol.metadata["correction_rate"]),
        "residual_error_rate": float(
            protocol.metadata["residual_error_rate"]
        ),
        "total_control_effort": float(
            protocol.metadata["total_control_effort"]
        ),
        "mean_detection_probability": float(
            protocol.metadata["mean_detection_probability"]
        ),
        "random_seed": int(protocol.metadata["random_seed"]),
    }


def control_protocols_to_rows(
    protocols: Iterable[ControlProtocol],
) -> list[dict[str, int | float | str]]:
    """Преобразует протоколы контроля в строки контрольной таблицы."""
    return [summarize_control_protocol(protocol) for protocol in protocols]
