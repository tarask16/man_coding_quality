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
