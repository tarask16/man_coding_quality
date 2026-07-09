"""
Создание модели оператора O для исследовательского симулятора.

Скрипт относится к этапу 4 программной реализации главы 3 диссертации.
Он создает модуль OperatorModel, который оценивает влияние параметров
оператора O на нормативный план ручного кодирования без моделирования
случайных ошибок, контрольных процедур и фактического результата M'.
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
    """Создает модуль OperatorModel, тесты и отчет этапа 4."""
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    write_text_file(
        SRC_DIR / "operator_model.py",
        '''
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
            "load_experiment_config",
            "messages_to_rows",
            "operator_estimates_to_rows",
            "procedure_plans_to_rows",
            "summarize_message",
            "summarize_operator_estimate",
            "summarize_procedure_plan",
        ]
        ''',
    )

    write_text_file(
        TESTS_DIR / "test_stage4_operator_model.py",
        '''
        """Тесты модели оператора O."""

        import pytest

        from manual_coding_sim.message_model import MessageGenerationConfig, MessageModel
        from manual_coding_sim.operator_model import (
            OperatorModel,
            OperatorModelConfig,
            OperatorProfile,
            operator_estimates_to_rows,
            summarize_operator_estimate,
        )
        from manual_coding_sim.procedure_model import ProcedureModel


        def _make_test_plan():
            """Формирует воспроизводимый нормативный план S для тестов оператора O."""
            message_config = MessageGenerationConfig(min_length=8, max_length=8)
            message = MessageModel(config=message_config, random_seed=42).generate_message(
                message_id="M_OPERATOR",
            )
            return ProcedureModel().build_plan(message)


        def test_operator_model_estimates_plan() -> None:
            """Модель O должна рассчитывать оценку для каждого шага плана S."""
            plan = _make_test_plan()
            model = OperatorModel()

            estimate = model.estimate_plan(plan)

            assert estimate.operator_id == "O_001"
            assert estimate.procedure_id == plan.procedure_id
            assert estimate.message_id == plan.message_id
            assert len(estimate.step_estimates) == len(plan.steps)
            assert estimate.metadata["step_count"] == len(plan.steps)
            assert estimate.metadata["total_estimated_time"] > 0
            assert estimate.metadata["total_effort"] > 0


        def test_attention_and_fatigue_are_in_valid_range() -> None:
            """Внимание и утомленность оператора должны оставаться в диапазоне [0; 1]."""
            plan = _make_test_plan()
            model = OperatorModel()

            estimate = model.estimate_plan(plan)

            for step_estimate in estimate.step_estimates:
                assert 0.0 <= step_estimate.attention <= 1.0
                assert 0.0 <= step_estimate.fatigue <= 1.0
                assert step_estimate.effort >= 0.0


        def test_attention_decreases_or_stays_with_fatigue() -> None:
            """К концу однотипной процедуры внимание не должно возрастать."""
            plan = _make_test_plan()
            model = OperatorModel()

            estimate = model.estimate_plan(plan)
            first_attention = estimate.step_estimates[0].attention
            last_attention = estimate.step_estimates[-1].attention

            assert last_attention <= first_attention


        def test_estimated_time_depends_on_operator_profile() -> None:
            """Менее подготовленный оператор должен иметь большую расчетную длительность."""
            plan = _make_test_plan()
            strong_operator = OperatorModel(
                OperatorModelConfig(
                    profile=OperatorProfile(
                        operator_id="O_STRONG",
                        preparation_level=0.95,
                        experience_level=0.95,
                        base_attention=0.95,
                        fatigue_rate=0.005,
                        control_skill=0.90,
                        work_rate=1.20,
                    )
                )
            )
            weak_operator = OperatorModel(
                OperatorModelConfig(
                    profile=OperatorProfile(
                        operator_id="O_WEAK",
                        preparation_level=0.30,
                        experience_level=0.25,
                        base_attention=0.70,
                        fatigue_rate=0.030,
                        control_skill=0.40,
                        work_rate=0.80,
                    )
                )
            )

            strong_time = strong_operator.estimate_plan(plan).metadata[
                "total_estimated_time"
            ]
            weak_time = weak_operator.estimate_plan(plan).metadata[
                "total_estimated_time"
            ]

            assert weak_time > strong_time


        def test_control_skill_is_preserved_in_step_estimates() -> None:
            """Навык контроля оператора должен переноситься в оценки шагов."""
            plan = _make_test_plan()
            model = OperatorModel(
                OperatorModelConfig(
                    profile=OperatorProfile(operator_id="O_CTRL", control_skill=0.82)
                )
            )

            estimate = model.estimate_plan(plan)

            assert estimate.operator_id == "O_CTRL"
            assert {step.control_skill for step in estimate.step_estimates} == {0.82}
            assert estimate.metadata["control_skill"] == 0.82


        def test_estimate_batch_creates_estimate_for_each_plan() -> None:
            """Пакет планов S должен преобразовываться в пакет оценок оператора O."""
            message_model = MessageModel(
                config=MessageGenerationConfig(min_length=4, max_length=4),
                random_seed=42,
            )
            messages = message_model.generate_batch(3)
            plans = ProcedureModel().build_batch(messages)
            operator_model = OperatorModel()

            estimates = operator_model.estimate_batch(plans)

            assert len(estimates) == 3
            assert [estimate.message_id for estimate in estimates] == [
                "M_000001",
                "M_000002",
                "M_000003",
            ]


        def test_summary_and_rows() -> None:
            """Сводка оценки O должна содержать априорные характеристики выполнения."""
            plan = _make_test_plan()
            estimate = OperatorModel().estimate_plan(plan)

            summary = summarize_operator_estimate(estimate)
            rows = operator_estimates_to_rows((estimate,))

            assert summary["operator_id"] == "O_001"
            assert summary["message_id"] == "M_OPERATOR"
            assert summary["step_count"] == 8
            assert summary["total_estimated_time"] > 0
            assert summary["total_effort"] > 0
            assert 0.0 <= summary["mean_attention"] <= 1.0
            assert rows == [summary]


        def test_invalid_operator_profile_is_rejected() -> None:
            """Профиль оператора O должен отклонять некорректные параметры."""
            invalid_config = OperatorModelConfig(
                profile=OperatorProfile(preparation_level=1.5)
            )

            with pytest.raises(ValueError, match="preparation_level"):
                OperatorModel(config=invalid_config)
        ''',
    )

    python_files = [
        SRC_DIR / "__init__.py",
        SRC_DIR / "operator_model.py",
        TESTS_DIR / "test_stage4_operator_model.py",
    ]
    syntax_report = {
        str(path.relative_to(ROOT)): check_python_syntax(path) for path in python_files
    }

    report = {
        "stage": "stage_4_operator_model",
        "status": "OK" if all(item["status"] == "OK" for item in syntax_report.values()) else "ERROR",
        "created_or_rewritten_files": [str(path.relative_to(ROOT)) for path in python_files],
        "scientific_scope": (
            "Реализована модель оператора O как априорное описание подготовленности, "
            "опыта, внимания, утомляемости и навыка контроля. Модель рассчитывает "
            "ожидаемое время и трудоемкость выполнения нормативного плана S, но не "
            "генерирует ошибки, не моделирует условия U, контрольные процедуры K и "
            "не формирует восстановленное сообщение M'."
        ),
        "syntax_report": syntax_report,
        "next_command": "python -m pytest",
    }

    report_path = REPORTS_DIR / "stage4_operator_model_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("ЭТАП 4. МОДЕЛЬ ОПЕРАТОРА O")
    print("=" * 40)
    for file_path in report["created_or_rewritten_files"]:
        print(f"[OK] {file_path}")
    print(f"[OK] Отчет: {report_path}")
    print("\nТеперь выполните команду:")
    print("python -m pytest")


if __name__ == "__main__":
    main()
