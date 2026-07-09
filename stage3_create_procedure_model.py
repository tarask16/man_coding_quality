"""
Создание модели средства ручного кодирования S для исследовательского симулятора.

Скрипт относится к этапу 3 программной реализации главы 3 диссертации.
Он создает модуль ProcedureModel, который формирует нормативный план
ручного кодирования для исходного сообщения M без моделирования ошибок,
условий применения, оператора и контрольных процедур.
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
    """Создает модуль ProcedureModel, тесты и отчет этапа 3."""
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    write_text_file(
        SRC_DIR / "procedure_model.py",
        '''
        """Модель средства ручного кодирования S.

        Модуль относится к главе 3 диссертации и реализует нормативное
        описание ручной процедуры кодирования. На данном этапе средство S
        не является конкретной прикладной системой скрытой связи: оно задает
        только абстрактные операции, необходимые для дальнейшего моделирования
        трудоемкости, длительности и потенциальной ошибкоопасности процесса.
        """

        from __future__ import annotations

        from dataclasses import dataclass, field
        from typing import Iterable

        from manual_coding_sim.types import GeneratedMessage, MessageElement


        @dataclass(frozen=True)
        class CodingOperationRule:
            """
            Правило выбора нормативной операции ручного кодирования.

            Правило связывает тип элемента исходного сообщения M с абстрактной
            операцией средства S. Оно не раскрывает конкретный способ скрытой
            передачи информации, а задает только параметры, влияющие на
            вычислительное моделирование качества ручного кодирования.
            """

            element_type: str
            operation_type: str
            nominal_time: float
            complexity: float
            reference_required: bool = False
            control_marker_required: bool = False

            def validate(self) -> None:
                """Проверяет корректность параметров правила средства S."""
                if not self.element_type:
                    raise ValueError("Тип элемента сообщения не задан.")

                if not self.operation_type:
                    raise ValueError("Тип операции ручного кодирования не задан.")

                if self.nominal_time <= 0:
                    raise ValueError("Нормативное время операции должно быть положительным.")

                if not 0.0 <= self.complexity <= 1.0:
                    raise ValueError(
                        "Сложность операции должна находиться в диапазоне [0; 1]."
                    )


        @dataclass(frozen=True)
        class ProcedureModelConfig:
            """
            Конфигурация средства ручного кодирования S.

            Параметры конфигурации являются априорными: они известны до начала
            выполнения процедуры E_h и могут использоваться для последующего
            формирования признаков X_prior.
            """

            procedure_id: str = "S_001"
            procedure_name: str = "abstract_manual_coding_procedure"
            operation_rules: tuple[CodingOperationRule, ...] = field(
                default_factory=lambda: (
                    CodingOperationRule(
                        element_type="symbol",
                        operation_type="abstract_substitution",
                        nominal_time=1.20,
                        complexity=0.45,
                    ),
                    CodingOperationRule(
                        element_type="digit",
                        operation_type="abstract_numeric_mapping",
                        nominal_time=1.45,
                        complexity=0.55,
                        reference_required=True,
                    ),
                    CodingOperationRule(
                        element_type="service",
                        operation_type="abstract_service_marking",
                        nominal_time=1.80,
                        complexity=0.65,
                        reference_required=True,
                        control_marker_required=True,
                    ),
                )
            )
            default_rule: CodingOperationRule = field(
                default_factory=lambda: CodingOperationRule(
                    element_type="default",
                    operation_type="abstract_copying",
                    nominal_time=1.00,
                    complexity=0.35,
                )
            )

            def validate(self) -> None:
                """Проверяет корректность конфигурации средства S."""
                if not self.procedure_id:
                    raise ValueError("Идентификатор средства ручного кодирования S не задан.")

                if not self.procedure_name:
                    raise ValueError("Название процедуры ручного кодирования не задано.")

                if not self.operation_rules:
                    raise ValueError("Набор правил операций средства S не должен быть пустым.")

                seen_element_types: set[str] = set()
                for rule in self.operation_rules:
                    rule.validate()
                    if rule.element_type in seen_element_types:
                        raise ValueError(
                            "Для одного типа элемента задано несколько правил: "
                            f"{rule.element_type}."
                        )
                    seen_element_types.add(rule.element_type)

                self.default_rule.validate()


        @dataclass(frozen=True)
        class ProcedureStep:
            """
            Нормативный шаг ручного кодирования одного элемента m_j.

            Шаг описывает то, что должно быть выполнено средством S для элемента
            исходного сообщения M. Ошибки оператора и фактический результат
            выполнения здесь не моделируются.
            """

            step_id: str
            source_position: int
            source_value: str
            source_element_type: str
            operation_type: str
            abstract_token: str
            nominal_time: float
            complexity: float
            reference_required: bool
            control_marker_required: bool


        @dataclass(frozen=True)
        class ProcedurePlan:
            """
            Нормативный план применения средства ручного кодирования S к сообщению M.

            План является промежуточным артефактом между исходным сообщением M
            и последующим моделированием выполнения процедуры E_h. Он нужен для
            расчета априорных характеристик: числа операций, нормативного времени,
            средней сложности и доли операций, требующих справочного обращения.
            """

            procedure_id: str
            message_id: str
            steps: tuple[ProcedureStep, ...]
            metadata: dict[str, int | float | str]


        class ProcedureModel:
            """
            Модель средства ручного кодирования S.

            Модель строит нормативный план кодирования для сообщения M. Она не
            выполняет вероятностное моделирование ошибок и не формирует итоговую
            оценку качества q(A). Эти функции будут добавлены на следующих этапах.
            """

            def __init__(self, config: ProcedureModelConfig | None = None) -> None:
                """Инициализирует средство S и индексирует правила операций."""
                self.config = config or ProcedureModelConfig()
                self.config.validate()
                self._rules_by_type = {
                    rule.element_type: rule for rule in self.config.operation_rules
                }

            def build_plan(self, message: GeneratedMessage) -> ProcedurePlan:
                """Строит нормативный план ручного кодирования для сообщения M."""
                if not message.elements:
                    raise ValueError("Невозможно построить план для пустого сообщения M.")

                steps = tuple(
                    self._build_step(element=element, step_index=index)
                    for index, element in enumerate(message.elements, start=1)
                )

                total_nominal_time = sum(step.nominal_time for step in steps)
                mean_complexity = sum(step.complexity for step in steps) / len(steps)
                reference_steps = sum(1 for step in steps if step.reference_required)
                control_marker_steps = sum(
                    1 for step in steps if step.control_marker_required
                )

                return ProcedurePlan(
                    procedure_id=self.config.procedure_id,
                    message_id=message.message_id,
                    steps=steps,
                    metadata={
                        "procedure_name": self.config.procedure_name,
                        "step_count": len(steps),
                        "total_nominal_time": round(total_nominal_time, 4),
                        "mean_complexity": round(mean_complexity, 4),
                        "reference_step_count": reference_steps,
                        "control_marker_step_count": control_marker_steps,
                    },
                )

            def build_batch(self, messages: Iterable[GeneratedMessage]) -> tuple[ProcedurePlan, ...]:
                """Строит планы ручного кодирования для пакета сообщений M."""
                return tuple(self.build_plan(message) for message in messages)

            def get_rule_for_element(self, element: MessageElement) -> CodingOperationRule:
                """Возвращает правило операции S для элемента сообщения m_j."""
                return self._rules_by_type.get(
                    element.element_type,
                    self.config.default_rule,
                )

            def _build_step(self, element: MessageElement, step_index: int) -> ProcedureStep:
                """Формирует нормативный шаг ручного кодирования."""
                rule = self.get_rule_for_element(element)
                abstract_token = self._make_abstract_token(rule, element)

                return ProcedureStep(
                    step_id=f"STEP_{step_index:04d}",
                    source_position=element.position,
                    source_value=element.value,
                    source_element_type=element.element_type,
                    operation_type=rule.operation_type,
                    abstract_token=abstract_token,
                    nominal_time=rule.nominal_time,
                    complexity=rule.complexity,
                    reference_required=rule.reference_required,
                    control_marker_required=rule.control_marker_required,
                )

            def _make_abstract_token(
                self,
                rule: CodingOperationRule,
                element: MessageElement,
            ) -> str:
                """
                Формирует абстрактный токен нормативного кодирования.

                Токен нужен только для трассировки шага в протоколе моделирования.
                Он не является описанием конкретного способа скрытой связи.
                """
                safe_value = str(element.value).replace(" ", "_")
                return f"TOK_{rule.operation_type}_{element.position:04d}_{safe_value}"


        def summarize_procedure_plan(
            plan: ProcedurePlan,
        ) -> dict[str, int | float | str]:
            """Возвращает контрольную сводку по нормативному плану средства S."""
            if not plan.steps:
                raise ValueError("План процедуры не содержит шагов.")

            return {
                "procedure_id": plan.procedure_id,
                "message_id": plan.message_id,
                "step_count": len(plan.steps),
                "total_nominal_time": float(plan.metadata["total_nominal_time"]),
                "mean_complexity": float(plan.metadata["mean_complexity"]),
                "reference_step_count": int(plan.metadata["reference_step_count"]),
                "control_marker_step_count": int(
                    plan.metadata["control_marker_step_count"]
                ),
            }


        def procedure_plans_to_rows(
            plans: Iterable[ProcedurePlan],
        ) -> list[dict[str, int | float | str]]:
            """Преобразует планы средства S в строки контрольной таблицы."""
            return [summarize_procedure_plan(plan) for plan in plans]
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
            "ProcedureModel",
            "ProcedureModelConfig",
            "ProcedurePlan",
            "ProcedureStep",
            "QualityVector",
            "ScenarioParameters",
            "load_experiment_config",
            "messages_to_rows",
            "procedure_plans_to_rows",
            "summarize_message",
            "summarize_procedure_plan",
        ]
        ''',
    )

    write_text_file(
        TESTS_DIR / "test_stage3_procedure_model.py",
        '''
        """Тесты модели средства ручного кодирования S."""

        import pytest

        from manual_coding_sim.message_model import MessageGenerationConfig, MessageModel
        from manual_coding_sim.procedure_model import (
            CodingOperationRule,
            ProcedureModel,
            ProcedureModelConfig,
            procedure_plans_to_rows,
            summarize_procedure_plan,
        )


        def _make_test_message():
            """Формирует воспроизводимое сообщение M для тестов средства S."""
            message_config = MessageGenerationConfig(min_length=6, max_length=6)
            message_model = MessageModel(config=message_config, random_seed=42)
            return message_model.generate_message(message_id="M_PROC")


        def test_procedure_model_builds_plan_for_message() -> None:
            """Средство S должно строить план с одним шагом на каждый элемент M."""
            message = _make_test_message()
            model = ProcedureModel()

            plan = model.build_plan(message)

            assert plan.procedure_id == "S_001"
            assert plan.message_id == "M_PROC"
            assert len(plan.steps) == len(message.elements)
            assert plan.metadata["step_count"] == len(message.elements)
            assert plan.metadata["total_nominal_time"] > 0
            assert 0.0 <= plan.metadata["mean_complexity"] <= 1.0


        def test_procedure_steps_preserve_source_positions() -> None:
            """Шаги процедуры должны сохранять позиции элементов m_j."""
            message = _make_test_message()
            model = ProcedureModel()

            plan = model.build_plan(message)

            for index, step in enumerate(plan.steps):
                source_element = message.elements[index]
                assert step.step_id == f"STEP_{index + 1:04d}"
                assert step.source_position == source_element.position
                assert step.source_value == source_element.value
                assert step.source_element_type == source_element.element_type
                assert step.abstract_token


        def test_rule_selection_uses_element_type() -> None:
            """Выбор нормативной операции должен зависеть от типа элемента m_j."""
            custom_config = ProcedureModelConfig(
                operation_rules=(
                    CodingOperationRule(
                        element_type="symbol",
                        operation_type="test_symbol_operation",
                        nominal_time=1.0,
                        complexity=0.2,
                    ),
                    CodingOperationRule(
                        element_type="digit",
                        operation_type="test_digit_operation",
                        nominal_time=2.0,
                        complexity=0.4,
                    ),
                    CodingOperationRule(
                        element_type="service",
                        operation_type="test_service_operation",
                        nominal_time=3.0,
                        complexity=0.6,
                    ),
                )
            )
            message_config = MessageGenerationConfig(
                min_length=3,
                max_length=3,
                element_types=("symbol",),
            )
            message = MessageModel(config=message_config, random_seed=1).generate_message()
            model = ProcedureModel(config=custom_config)

            plan = model.build_plan(message)

            assert {step.operation_type for step in plan.steps} == {
                "test_symbol_operation"
            }


        def test_default_rule_handles_unknown_element_type() -> None:
            """Для неизвестного типа элемента должно применяться правило по умолчанию."""
            message = _make_test_message()
            unknown_element = message.elements[0].__class__(
                value="X",
                element_type="unknown_type",
                position=0,
                criticality=0.5,
            )
            model = ProcedureModel()

            rule = model.get_rule_for_element(unknown_element)

            assert rule.operation_type == "abstract_copying"


        def test_build_batch_creates_plan_for_each_message() -> None:
            """Пакет сообщений M должен преобразовываться в пакет планов S."""
            message_model = MessageModel(
                config=MessageGenerationConfig(min_length=4, max_length=4),
                random_seed=42,
            )
            messages = message_model.generate_batch(3)
            procedure_model = ProcedureModel()

            plans = procedure_model.build_batch(messages)

            assert len(plans) == 3
            assert [plan.message_id for plan in plans] == [
                "M_000001",
                "M_000002",
                "M_000003",
            ]


        def test_plan_summary_and_rows() -> None:
            """Сводка плана должна содержать контрольные априорные характеристики."""
            message = _make_test_message()
            model = ProcedureModel()
            plan = model.build_plan(message)

            summary = summarize_procedure_plan(plan)
            rows = procedure_plans_to_rows((plan,))

            assert summary["procedure_id"] == "S_001"
            assert summary["message_id"] == "M_PROC"
            assert summary["step_count"] == 6
            assert summary["total_nominal_time"] > 0
            assert 0.0 <= summary["mean_complexity"] <= 1.0
            assert rows == [summary]


        def test_empty_message_rejected() -> None:
            """Пустое сообщение M не должно иметь нормативного плана кодирования."""
            message = _make_test_message()
            empty_message = message.__class__(
                message_id="M_EMPTY",
                elements=(),
                metadata=message.metadata,
            )
            model = ProcedureModel()

            with pytest.raises(ValueError, match="пустого сообщения"):
                model.build_plan(empty_message)


        def test_procedure_config_validation_rejects_invalid_rule() -> None:
            """Конфигурация средства S должна отклонять некорректные правила."""
            invalid_config = ProcedureModelConfig(
                operation_rules=(
                    CodingOperationRule(
                        element_type="symbol",
                        operation_type="bad_operation",
                        nominal_time=0.0,
                        complexity=0.5,
                    ),
                )
            )

            with pytest.raises(ValueError, match="Нормативное время"):
                ProcedureModel(config=invalid_config)
        ''',
    )

    python_files = [
        SRC_DIR / "__init__.py",
        SRC_DIR / "procedure_model.py",
        TESTS_DIR / "test_stage3_procedure_model.py",
    ]
    syntax_report = {str(path.relative_to(ROOT)): check_python_syntax(path) for path in python_files}

    report = {
        "stage": "stage_3_procedure_model",
        "status": "OK" if all(item["status"] == "OK" for item in syntax_report.values()) else "ERROR",
        "created_or_rewritten_files": [str(path.relative_to(ROOT)) for path in python_files],
        "scientific_scope": (
            "Реализована модель средства ручного кодирования S как нормативный план "
            "абстрактных операций над элементами сообщения M. Оператор O, условия U, "
            "вероятностные ошибки, контрольные процедуры K и фактическое декодирование "
            "M' на этом этапе не реализуются."
        ),
        "syntax_report": syntax_report,
        "next_command": "python -m pytest",
    }

    report_path = REPORTS_DIR / "stage3_procedure_model_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("ЭТАП 3. МОДЕЛЬ СРЕДСТВА РУЧНОГО КОДИРОВАНИЯ S")
    print("=" * 64)
    for file_path in report["created_or_rewritten_files"]:
        print(f"[OK] {file_path}")
    print(f"[OK] Отчет: {report_path}")
    print("\nТеперь выполните команду:")
    print("python -m pytest")


if __name__ == "__main__":
    main()
