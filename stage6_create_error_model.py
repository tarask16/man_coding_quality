"""
Создание вероятностной модели ошибок ErrorModel для исследовательского симулятора.

Скрипт относится к этапу 6 программной реализации главы 3 диссертации.
Он создает модуль ErrorModel, который по априорной оценке S + O + U
формирует вероятностный протокол ошибок ручного кодирования без реализации
контрольных процедур K, декодирования D_h и восстановленного сообщения M'.
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
    """Создает модуль ErrorModel, тесты и отчет этапа 6."""
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    write_text_file(
        SRC_DIR / "error_model.py",
        '''
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
            "error_protocols_to_rows",
            "load_experiment_config",
            "messages_to_rows",
            "operator_estimates_to_rows",
            "procedure_plans_to_rows",
            "summarize_condition_estimate",
            "summarize_error_protocol",
            "summarize_message",
            "summarize_operator_estimate",
            "summarize_procedure_plan",
        ]
        ''',
    )

    write_text_file(
        TESTS_DIR / "test_stage6_error_model.py",
        '''
        """Тесты вероятностной модели ошибок ErrorModel."""

        from dataclasses import replace

        import pytest

        from manual_coding_sim.condition_model import (
            ConditionModel,
            ConditionProfile,
        )
        from manual_coding_sim.error_model import (
            ErrorModel,
            ErrorModelConfig,
            ErrorProtocol,
            error_protocols_to_rows,
            summarize_error_protocol,
        )
        from manual_coding_sim.message_model import MessageGenerationConfig, MessageModel
        from manual_coding_sim.operator_model import OperatorModel
        from manual_coding_sim.procedure_model import ProcedureModel


        def _make_operator_estimate(message_id: str = "M_ERROR"):
            """Формирует воспроизводимую оценку O + S для проверки ошибок."""
            message_config = MessageGenerationConfig(min_length=10, max_length=10)
            message = MessageModel(config=message_config, random_seed=42).generate_message(
                message_id=message_id,
            )
            plan = ProcedureModel().build_plan(message)
            return OperatorModel().estimate_plan(plan)


        def _make_condition_estimate(
            profile: ConditionProfile | None = None,
            message_id: str = "M_ERROR",
        ):
            """Формирует оценку S + O + U для вероятностной модели ошибок."""
            operator_estimate = _make_operator_estimate(message_id=message_id)
            condition_model = ConditionModel()
            if profile is not None:
                condition_model = ConditionModel(
                    config=replace(condition_model.config, profile=profile),
                )
            return condition_model.estimate_plan(operator_estimate)


        def test_error_model_generates_protocol_for_condition_estimate() -> None:
            """Модель ошибок должна формировать протокол для оценки S + O + U."""
            condition_estimate = _make_condition_estimate()
            model = ErrorModel(random_seed=123)

            protocol = model.generate_protocol(condition_estimate)

            assert isinstance(protocol, ErrorProtocol)
            assert protocol.error_model_id == "ERR_001"
            assert protocol.condition_id == condition_estimate.condition_id
            assert protocol.operator_id == condition_estimate.operator_id
            assert protocol.procedure_id == condition_estimate.procedure_id
            assert protocol.message_id == condition_estimate.message_id
            assert len(protocol.step_outcomes) == len(condition_estimate.step_estimates)
            assert protocol.metadata["step_count"] == len(condition_estimate.step_estimates)


        def test_error_step_outcomes_have_valid_ranges() -> None:
            """Вероятности и расчетные давления должны находиться в диапазоне [0; 1]."""
            protocol = ErrorModel(random_seed=123).generate_protocol(
                _make_condition_estimate(),
            )

            for outcome in protocol.step_outcomes:
                assert 0.0 <= outcome.error_probability <= 1.0
                assert 0.0 <= outcome.random_draw <= 1.0
                assert 0.0 <= outcome.attention_deficit <= 1.0
                assert 0.0 <= outcome.environmental_load <= 1.0
                assert 0.0 <= outcome.time_pressure <= 1.0
                assert 0.0 <= outcome.instruction_pressure <= 1.0
                if outcome.error_occurred:
                    assert outcome.error_type is not None
                    assert outcome.error_weight > 0
                else:
                    assert outcome.error_type is None
                    assert outcome.error_weight == 0.0


        def test_error_model_is_reproducible_with_same_seed() -> None:
            """Одинаковый random_seed должен давать одинаковые протоколы ошибок."""
            condition_estimate = _make_condition_estimate()

            first = ErrorModel(random_seed=777).generate_protocol(condition_estimate)
            second = ErrorModel(random_seed=777).generate_protocol(condition_estimate)

            first_trace = [
                (
                    outcome.error_probability,
                    outcome.random_draw,
                    outcome.error_occurred,
                    outcome.error_type,
                )
                for outcome in first.step_outcomes
            ]
            second_trace = [
                (
                    outcome.error_probability,
                    outcome.random_draw,
                    outcome.error_occurred,
                    outcome.error_type,
                )
                for outcome in second.step_outcomes
            ]
            assert first_trace == second_trace


        def test_error_model_reset_restores_random_sequence() -> None:
            """Сброс модели должен восстанавливать последовательность событий ошибок."""
            condition_estimate = _make_condition_estimate()
            model = ErrorModel(random_seed=555)

            first = model.generate_protocol(condition_estimate)
            model.generate_protocol(condition_estimate)
            model.reset()
            after_reset = model.generate_protocol(condition_estimate)

            assert [item.random_draw for item in first.step_outcomes] == [
                item.random_draw for item in after_reset.step_outcomes
            ]
            assert [item.error_occurred for item in first.step_outcomes] == [
                item.error_occurred for item in after_reset.step_outcomes
            ]


        def test_adverse_conditions_increase_mean_error_probability() -> None:
            """Неблагоприятные условия U должны повышать среднюю вероятность ошибки."""
            mild_profile = ConditionProfile(
                condition_id="U_MILD",
                time_limit_seconds=None,
                noise_level=0.0,
                instruction_access=1.0,
                workload_level=0.0,
                interruption_rate=0.0,
                lighting_quality=1.0,
            )
            adverse_profile = ConditionProfile(
                condition_id="U_ADVERSE",
                time_limit_seconds=1.0,
                noise_level=1.0,
                instruction_access=0.0,
                workload_level=1.0,
                interruption_rate=1.0,
                lighting_quality=0.0,
            )

            mild_protocol = ErrorModel(random_seed=1).generate_protocol(
                _make_condition_estimate(profile=mild_profile),
            )
            adverse_protocol = ErrorModel(random_seed=1).generate_protocol(
                _make_condition_estimate(profile=adverse_profile),
            )

            assert (
                adverse_protocol.metadata["mean_error_probability"]
                > mild_protocol.metadata["mean_error_probability"]
            )


        def test_error_count_matches_step_flags() -> None:
            """Число ошибок в metadata должно соответствовать флагам шагов."""
            protocol = ErrorModel(
                config=ErrorModelConfig(base_error_rate=0.50, max_error_probability=0.90),
                random_seed=2,
            ).generate_protocol(_make_condition_estimate())

            expected_count = sum(1 for item in protocol.step_outcomes if item.error_occurred)
            expected_weight = sum(item.error_weight for item in protocol.step_outcomes)

            assert protocol.metadata["error_count"] == expected_count
            assert protocol.metadata["weighted_error_sum"] == round(expected_weight, 6)


        def test_error_model_generates_batch_and_rows() -> None:
            """Пакетная обработка должна формировать строки контрольной таблицы."""
            estimates = (
                _make_condition_estimate(message_id="M_ERROR_1"),
                _make_condition_estimate(message_id="M_ERROR_2"),
            )
            protocols = ErrorModel(random_seed=42).generate_batch(estimates)
            rows = error_protocols_to_rows(protocols)

            assert len(protocols) == 2
            assert len(rows) == 2
            assert rows[0]["message_id"] == "M_ERROR_1"
            assert rows[1]["message_id"] == "M_ERROR_2"
            assert "mean_error_probability" in rows[0]


        def test_invalid_error_model_config_is_rejected() -> None:
            """Некорректная конфигурация модели ошибок должна отклоняться."""
            with pytest.raises(ValueError):
                ErrorModelConfig(base_error_rate=-0.1).validate()

            with pytest.raises(ValueError):
                ErrorModelConfig(default_error_weight=0.0).validate()

            with pytest.raises(ValueError):
                ErrorModelConfig(error_types=()).validate()

            with pytest.raises(ValueError):
                ErrorModelConfig(
                    base_error_rate=0.9,
                    max_error_probability=0.5,
                ).validate()


        def test_empty_condition_estimate_is_rejected() -> None:
            """Пустая оценка условий U не должна использоваться для протокола ошибок."""
            condition_estimate = _make_condition_estimate()
            empty_estimate = replace(condition_estimate, step_estimates=())

            with pytest.raises(ValueError):
                ErrorModel().generate_protocol(empty_estimate)


        def test_summarize_error_protocol_returns_control_summary() -> None:
            """Сводка должна содержать ключевые показатели протокола ошибок."""
            protocol = ErrorModel(random_seed=123).generate_protocol(
                _make_condition_estimate(),
            )
            summary = summarize_error_protocol(protocol)

            assert summary["error_model_id"] == protocol.error_model_id
            assert summary["message_id"] == protocol.message_id
            assert summary["step_count"] == protocol.metadata["step_count"]
            assert summary["error_count"] == protocol.metadata["error_count"]
            assert summary["random_seed"] == 123
        ''',
    )

    syntax_targets = [
        SRC_DIR / "error_model.py",
        SRC_DIR / "__init__.py",
        TESTS_DIR / "test_stage6_error_model.py",
    ]
    syntax_report = {str(path): check_python_syntax(path) for path in syntax_targets}

    report = {
        "stage": "Этап 6",
        "task": "Вероятностная модель ошибок ErrorModel",
        "created_files": [str(path) for path in syntax_targets],
        "syntax_report": syntax_report,
        "next_command": "python -m pytest",
    }
    report_path = REPORTS_DIR / "stage6_error_model_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("ЭТАП 6. ВЕРОЯТНОСТНАЯ МОДЕЛЬ ОШИБОК")
    print("=" * 56)
    for path in syntax_targets:
        status = syntax_report[str(path)]["status"]
        print(f"[{status}] {path.relative_to(ROOT)}")
    print(f"[OK] Отчет: {report_path}")
    print("\nТеперь выполните команду:")
    print("python -m pytest")


if __name__ == "__main__":
    main()
