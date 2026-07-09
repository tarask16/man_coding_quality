"""
Создание модели контрольных процедур ControlModel для исследовательского симулятора.

Скрипт относится к этапу 7 программной реализации главы 3 диссертации.
Он создает модуль ControlModel, который по вероятностному протоколу ошибок
ErrorProtocol моделирует контрольные процедуры K: обнаружение и исправление
ошибок ручного кодирования. На данном этапе не реализуются декодирование D_h,
восстановленное сообщение M' и итоговый расчет фактического качества q(A).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import textwrap


ROOT = Path.cwd()
SRC_DIR = ROOT / "src" / "manual_coding_sim"
TESTS_DIR = ROOT / "tests"
REPORTS_DIR = ROOT / "reports" / "chapter3"


def write_text_file(path: Path, content: str) -> None:
    """Записывает текстовый файл в кодировке UTF-8 без лишних начальных отступов."""
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_content = textwrap.dedent(content).lstrip("\n")
    path.write_text(normalized_content, encoding="utf-8")


def check_python_syntax(path: Path) -> dict[str, str]:
    """Проверяет синтаксическую корректность Python-файла."""
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        return {"status": "OK", "message": "Синтаксис корректен"}
    except SyntaxError as error:
        return {"status": "ERROR", "message": str(error)}


def main() -> None:
    """Создает модуль ControlModel, тесты и отчет этапа 7."""
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    write_text_file(
        SRC_DIR / "control_model.py",
        '''
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
        ''',
    )

    write_text_file(
        SRC_DIR / "__init__.py",
        '''
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
            "ControlModel",
            "ControlModelConfig",
            "ControlProfile",
            "ControlProtocol",
            "ControlStepOutcome",
            "ErrorModel",
            "ErrorModelConfig",
            "ErrorProtocol",
            "ErrorStepOutcome",
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
            "control_protocols_to_rows",
            "error_protocols_to_rows",
            "load_experiment_config",
            "messages_to_rows",
            "operator_estimates_to_rows",
            "procedure_plans_to_rows",
            "summarize_condition_estimate",
            "summarize_control_protocol",
            "summarize_error_protocol",
            "summarize_message",
            "summarize_operator_estimate",
            "summarize_procedure_plan",
        ]
        ''',
    )

    write_text_file(
        TESTS_DIR / "test_stage7_control_model.py",
        '''
        """Тесты модели контрольных процедур ControlModel."""

        from dataclasses import replace

        import pytest

        from manual_coding_sim.condition_model import ConditionModel
        from manual_coding_sim.control_model import (
            ControlModel,
            ControlModelConfig,
            ControlProfile,
            ControlProtocol,
            control_protocols_to_rows,
            summarize_control_protocol,
        )
        from manual_coding_sim.error_model import ErrorModel, ErrorModelConfig
        from manual_coding_sim.message_model import MessageGenerationConfig, MessageModel
        from manual_coding_sim.operator_model import OperatorModel
        from manual_coding_sim.procedure_model import ProcedureModel


        def _make_condition_estimate(message_id: str = "M_CONTROL"):
            """Формирует оценку S + O + U для тестов контроля K."""
            message_config = MessageGenerationConfig(min_length=10, max_length=10)
            message = MessageModel(config=message_config, random_seed=42).generate_message(
                message_id=message_id,
            )
            plan = ProcedureModel().build_plan(message)
            operator_estimate = OperatorModel().estimate_plan(plan)
            return ConditionModel().estimate_plan(operator_estimate)


        def _make_error_protocol(
            message_id: str = "M_CONTROL",
            force_errors: bool = True,
        ):
            """Формирует протокол ошибок для проверки контрольных процедур K."""
            condition_estimate = _make_condition_estimate(message_id=message_id)
            if force_errors:
                error_config = ErrorModelConfig(
                    base_error_rate=1.0,
                    max_error_probability=1.0,
                )
            else:
                error_config = ErrorModelConfig(
                    base_error_rate=0.0,
                    max_error_probability=1.0,
                )
            return ErrorModel(config=error_config, random_seed=123).generate_protocol(
                condition_estimate,
            )


        def test_control_model_generates_protocol_for_error_protocol() -> None:
            """Модель K должна формировать протокол контроля для ErrorProtocol."""
            error_protocol = _make_error_protocol()
            control_protocol = ControlModel(random_seed=77).generate_protocol(
                error_protocol,
            )

            assert isinstance(control_protocol, ControlProtocol)
            assert control_protocol.control_id == "K_001"
            assert control_protocol.error_model_id == error_protocol.error_model_id
            assert control_protocol.condition_id == error_protocol.condition_id
            assert control_protocol.operator_id == error_protocol.operator_id
            assert control_protocol.procedure_id == error_protocol.procedure_id
            assert control_protocol.message_id == error_protocol.message_id
            assert len(control_protocol.step_outcomes) == len(
                error_protocol.step_outcomes,
            )


        def test_control_step_outcomes_have_valid_ranges() -> None:
            """Вероятности контроля и расчетные показатели должны быть допустимыми."""
            protocol = ControlModel(random_seed=77).generate_protocol(
                _make_error_protocol(),
            )

            for outcome in protocol.step_outcomes:
                assert 0.0 <= outcome.detection_probability <= 1.0
                assert 0.0 <= outcome.correction_probability <= 1.0
                assert outcome.control_effort > 0.0
                if outcome.detection_draw is not None:
                    assert 0.0 <= outcome.detection_draw <= 1.0
                if outcome.correction_draw is not None:
                    assert 0.0 <= outcome.correction_draw <= 1.0
                if not outcome.error_occurred:
                    assert outcome.error_detected is False
                    assert outcome.error_corrected is False
                    assert outcome.residual_error is False


        def test_control_model_is_reproducible_with_same_seed() -> None:
            """Одинаковый random_seed должен давать одинаковый протокол контроля."""
            error_protocol = _make_error_protocol()

            first = ControlModel(random_seed=555).generate_protocol(error_protocol)
            second = ControlModel(random_seed=555).generate_protocol(error_protocol)

            first_trace = [
                (
                    outcome.detection_draw,
                    outcome.error_detected,
                    outcome.correction_draw,
                    outcome.error_corrected,
                    outcome.residual_error,
                )
                for outcome in first.step_outcomes
            ]
            second_trace = [
                (
                    outcome.detection_draw,
                    outcome.error_detected,
                    outcome.correction_draw,
                    outcome.error_corrected,
                    outcome.residual_error,
                )
                for outcome in second.step_outcomes
            ]

            assert first_trace == second_trace


        def test_control_model_reset_restores_random_sequence() -> None:
            """Сброс модели K должен восстанавливать последовательность контроля."""
            error_protocol = _make_error_protocol()
            model = ControlModel(random_seed=555)

            first = model.generate_protocol(error_protocol)
            model.generate_protocol(error_protocol)
            model.reset()
            after_reset = model.generate_protocol(error_protocol)

            assert [item.detection_draw for item in first.step_outcomes] == [
                item.detection_draw for item in after_reset.step_outcomes
            ]
            assert [item.error_corrected for item in first.step_outcomes] == [
                item.error_corrected for item in after_reset.step_outcomes
            ]


        def test_stronger_control_profile_increases_detection_probability() -> None:
            """Более сильный профиль K должен повышать вероятность обнаружения ошибок."""
            error_protocol = _make_error_protocol()
            weak_profile = ControlProfile(
                control_id="K_WEAK",
                detection_skill=0.0,
                correction_skill=0.0,
                reference_check_coverage=0.0,
                repeated_check_coverage=0.0,
                critical_error_priority=0.0,
            )
            strong_profile = ControlProfile(
                control_id="K_STRONG",
                detection_skill=1.0,
                correction_skill=1.0,
                reference_check_coverage=1.0,
                repeated_check_coverage=1.0,
                critical_error_priority=1.0,
            )

            weak_protocol = ControlModel(
                config=ControlModelConfig(profile=weak_profile),
                random_seed=1,
            ).generate_protocol(error_protocol)
            strong_protocol = ControlModel(
                config=ControlModelConfig(profile=strong_profile),
                random_seed=1,
            ).generate_protocol(error_protocol)

            assert (
                strong_protocol.metadata["mean_detection_probability"]
                > weak_protocol.metadata["mean_detection_probability"]
            )


        def test_residual_error_count_matches_step_flags() -> None:
            """Число остаточных ошибок должно соответствовать флагам шагов."""
            protocol = ControlModel(random_seed=7).generate_protocol(
                _make_error_protocol(),
            )

            expected_residual = sum(
                1 for outcome in protocol.step_outcomes if outcome.residual_error
            )
            expected_detected = sum(
                1 for outcome in protocol.step_outcomes if outcome.error_detected
            )
            expected_corrected = sum(
                1 for outcome in protocol.step_outcomes if outcome.error_corrected
            )

            assert protocol.metadata["residual_error_count"] == expected_residual
            assert protocol.metadata["detected_error_count"] == expected_detected
            assert protocol.metadata["corrected_error_count"] == expected_corrected


        def test_control_model_handles_protocol_without_errors() -> None:
            """При отсутствии ошибок K не должен формировать обнаружения и исправления."""
            protocol = ControlModel(random_seed=9).generate_protocol(
                _make_error_protocol(force_errors=False),
            )

            assert protocol.metadata["original_error_count"] == 0
            assert protocol.metadata["detected_error_count"] == 0
            assert protocol.metadata["corrected_error_count"] == 0
            assert protocol.metadata["residual_error_count"] == 0
            assert protocol.metadata["detection_rate"] == 0.0
            assert protocol.metadata["correction_rate"] == 0.0


        def test_control_model_generates_batch_and_rows() -> None:
            """Пакетная обработка должна формировать строки контрольной таблицы."""
            protocols = (
                _make_error_protocol(message_id="M_CONTROL_1"),
                _make_error_protocol(message_id="M_CONTROL_2"),
            )
            control_protocols = ControlModel(random_seed=42).generate_batch(protocols)
            rows = control_protocols_to_rows(control_protocols)

            assert len(control_protocols) == 2
            assert len(rows) == 2
            assert rows[0]["message_id"] == "M_CONTROL_1"
            assert rows[1]["message_id"] == "M_CONTROL_2"
            assert "residual_error_count" in rows[0]


        def test_invalid_control_model_config_is_rejected() -> None:
            """Некорректная конфигурация K должна отклоняться."""
            with pytest.raises(ValueError):
                ControlProfile(detection_skill=-0.1).validate()

            with pytest.raises(ValueError):
                ControlProfile(reference_check_coverage=1.5).validate()

            with pytest.raises(ValueError):
                ControlModelConfig(base_control_effort=0.0).validate()

            with pytest.raises(ValueError):
                ControlModelConfig(
                    base_detection_rate=0.9,
                    max_detection_probability=0.5,
                ).validate()

            with pytest.raises(ValueError):
                ControlModelConfig(
                    base_correction_rate=0.9,
                    max_correction_probability=0.5,
                ).validate()


        def test_empty_error_protocol_is_rejected() -> None:
            """Пустой протокол ошибок не должен использоваться для контроля."""
            error_protocol = _make_error_protocol()
            empty_protocol = replace(error_protocol, step_outcomes=())

            with pytest.raises(ValueError):
                ControlModel().generate_protocol(empty_protocol)


        def test_summarize_control_protocol_returns_control_summary() -> None:
            """Сводка должна содержать ключевые показатели протокола K."""
            protocol = ControlModel(random_seed=77).generate_protocol(
                _make_error_protocol(),
            )
            summary = summarize_control_protocol(protocol)

            assert summary["control_id"] == protocol.control_id
            assert summary["message_id"] == protocol.message_id
            assert summary["step_count"] == protocol.metadata["step_count"]
            assert summary["residual_error_count"] == protocol.metadata[
                "residual_error_count"
            ]
            assert summary["random_seed"] == 77
        ''',
    )

    syntax_targets = [
        SRC_DIR / "control_model.py",
        SRC_DIR / "__init__.py",
        TESTS_DIR / "test_stage7_control_model.py",
    ]
    syntax_report = {str(path): check_python_syntax(path) for path in syntax_targets}

    report = {
        "stage": "Этап 7",
        "task": "Модель контрольных процедур ControlModel",
        "created_files": [str(path) for path in syntax_targets],
        "syntax_report": syntax_report,
        "next_command": "python -m pytest",
    }
    report_path = REPORTS_DIR / "stage7_control_model_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("ЭТАП 7. МОДЕЛЬ КОНТРОЛЬНЫХ ПРОЦЕДУР K")
    print("=" * 56)
    for path in syntax_targets:
        status = syntax_report[str(path)]["status"]
        print(f"[{status}] {path.relative_to(ROOT)}")
    print(f"[OK] Отчет: {report_path}")
    print("\nТеперь выполните команду:")
    print("python -m pytest")


if __name__ == "__main__":
    main()
