"""Материальное представление кодированного сообщения C.

Модуль преобразует нормативный :class:`ProcedurePlan` базового пакета в
явную последовательность кодированных элементов. Остаточные ошибки после
контроля применяются только к новой копии последовательности C. Исходные
объекты базового пакета остаются неизменными.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from manual_coding_sim.control_model import ControlProtocol, ControlStepOutcome
from manual_coding_sim.procedure_model import ProcedurePlan, ProcedureStep

from manual_coding_sim_decoding.config import MaterialEncodingConfig


@dataclass(frozen=True)
class EncodedElement:
    """Один фактически материализованный элемент кодированного сообщения C."""

    element_id: str
    encoded_position: int
    source_step_id: str
    source_position: int
    source_value: str
    source_element_type: str
    operation_type: str
    normative_token: str
    token: str
    residual_error: bool
    error_type: str | None
    mutation_kind: str


@dataclass(frozen=True)
class EncodedMessage:
    """Материальное кодированное сообщение C как упорядоченная последовательность."""

    encoded_message_id: str
    source_message_id: str
    procedure_id: str
    elements: tuple[EncodedElement, ...]
    metadata: dict[str, int | float | str | bool]

    @property
    def tokens(self) -> tuple[str, ...]:
        """Вернуть фактическую последовательность токенов C."""
        return tuple(element.token for element in self.elements)


@dataclass(frozen=True)
class EncodingTraceStep:
    """Трасса перехода от нормативного шага к фактическому элементу C."""

    source_step_id: str
    source_position: int
    normative_position: int
    normative_token: str
    materialized: bool
    encoded_position: int | None
    actual_token: str | None
    residual_error: bool
    error_type: str | None
    mutation_kind: str
    position_changed: bool


@dataclass(frozen=True)
class EncodingProtocol:
    """Полный протокол материализации нормативного плана в сообщение C."""

    encoded_message_id: str
    source_message_id: str
    procedure_id: str
    control_id: str
    trace_steps: tuple[EncodingTraceStep, ...]
    metadata: dict[str, int | float | str | bool]


@dataclass(frozen=True)
class MaterialEncodingResult:
    """Связанный результат формирования кодированного сообщения и трассы."""

    encoded_message: EncodedMessage
    protocol: EncodingProtocol

    def to_dict(self) -> dict[str, Any]:
        """Преобразовать результат в JSON-совместимый словарь."""
        return asdict(self)


@dataclass
class _WorkingElement:
    """Внутреннее изменяемое представление до фиксации результата."""

    element_id: str
    step: ProcedureStep
    normative_position: int
    actual_token: str
    materialized: bool = True
    residual_error: bool = False
    error_type: str | None = None
    mutation_kind: str = "none"
    encoded_position: int | None = None
    position_changed: bool = False


class EncodedMessageBuilder:
    """Строитель материального сообщения C по плану и протоколу контроля."""

    def __init__(self, config: MaterialEncodingConfig | None = None) -> None:
        """Инициализировать правила абстрактной материализации ошибок."""
        self.config = config or MaterialEncodingConfig()
        self.config.validate()

    def build(
        self,
        procedure_plan: ProcedurePlan,
        control_protocol: ControlProtocol,
    ) -> MaterialEncodingResult:
        """Сформировать C и трассу, не изменяя входные объекты.

        Учитываются только остаточные ошибки ``residual_error=True``. Ошибки,
        исправленные базовой моделью контроля, не изменяют фактический токен.
        """
        self._validate_inputs(procedure_plan, control_protocol)
        outcomes = self._index_control_outcomes(control_protocol.step_outcomes)
        working = [
            _WorkingElement(
                element_id=f"ENC_{index:04d}",
                step=step,
                normative_position=index - 1,
                actual_token=step.abstract_token,
            )
            for index, step in enumerate(procedure_plan.steps, start=1)
        ]

        position_error_ids: list[str] = []
        for item in working:
            outcome = outcomes[item.step.step_id]
            if not outcome.residual_error:
                continue
            self._apply_residual_error(item, outcome)
            if item.mutation_kind == "position":
                position_error_ids.append(item.element_id)

        materialized = [item for item in working if item.materialized]
        self._apply_position_errors(materialized, position_error_ids)
        for encoded_position, item in enumerate(materialized):
            item.encoded_position = encoded_position

        encoded_message_id = (
            f"{self.config.encoded_message_prefix}_{procedure_plan.message_id}"
        )
        elements = tuple(self._freeze_element(item) for item in materialized)
        traces = tuple(self._freeze_trace(item) for item in working)
        metadata = self._build_metadata(procedure_plan.steps, elements, traces)

        encoded_message = EncodedMessage(
            encoded_message_id=encoded_message_id,
            source_message_id=procedure_plan.message_id,
            procedure_id=procedure_plan.procedure_id,
            elements=elements,
            metadata=dict(metadata),
        )
        protocol = EncodingProtocol(
            encoded_message_id=encoded_message_id,
            source_message_id=procedure_plan.message_id,
            procedure_id=procedure_plan.procedure_id,
            control_id=control_protocol.control_id,
            trace_steps=traces,
            metadata=dict(metadata),
        )
        return MaterialEncodingResult(
            encoded_message=encoded_message,
            protocol=protocol,
        )

    def build_from_simulation_result(
        self,
        simulation_result: Any,
    ) -> MaterialEncodingResult:
        """Сформировать C из публичных артефактов базового SimulationResult."""
        if not hasattr(simulation_result, "procedure_plan"):
            raise TypeError("Результат симуляции не содержит procedure_plan.")
        if not hasattr(simulation_result, "control_protocol"):
            raise TypeError("Результат симуляции не содержит control_protocol.")
        return self.build(
            procedure_plan=simulation_result.procedure_plan,
            control_protocol=simulation_result.control_protocol,
        )

    def _validate_inputs(
        self,
        procedure_plan: ProcedurePlan,
        control_protocol: ControlProtocol,
    ) -> None:
        """Проверить идентификаторы и взаимно однозначное покрытие шагов."""
        if not procedure_plan.steps:
            raise ValueError("Нормативный план не содержит шагов кодирования.")
        if not control_protocol.step_outcomes:
            raise ValueError("Протокол контроля не содержит результатов шагов.")
        if procedure_plan.message_id != control_protocol.message_id:
            raise ValueError("message_id плана и протокола контроля не совпадают.")
        if procedure_plan.procedure_id != control_protocol.procedure_id:
            raise ValueError("procedure_id плана и протокола контроля не совпадают.")

        plan_keys = [
            (step.step_id, step.source_position)
            for step in procedure_plan.steps
        ]
        control_keys = [
            (outcome.step_id, outcome.source_position)
            for outcome in control_protocol.step_outcomes
        ]
        if len(set(plan_keys)) != len(plan_keys):
            raise ValueError("Нормативный план содержит дублирующиеся шаги.")
        if len(set(control_keys)) != len(control_keys):
            raise ValueError("Протокол контроля содержит дублирующиеся шаги.")
        if set(plan_keys) != set(control_keys):
            raise ValueError(
                "Шаги нормативного плана и протокола контроля не образуют "
                "взаимно однозначное соответствие."
            )

    @staticmethod
    def _index_control_outcomes(
        outcomes: Iterable[ControlStepOutcome],
    ) -> dict[str, ControlStepOutcome]:
        """Индексировать результаты контроля по идентификатору шага."""
        return {outcome.step_id: outcome for outcome in outcomes}

    def _apply_residual_error(
        self,
        item: _WorkingElement,
        outcome: ControlStepOutcome,
    ) -> None:
        """Применить одну остаточную ошибку к рабочей копии элемента."""
        if not outcome.error_type:
            raise ValueError(
                f"Для остаточной ошибки шага {outcome.step_id} не задан error_type."
            )

        item.residual_error = True
        item.error_type = outcome.error_type
        error_type = outcome.error_type

        if error_type == "abstract_omission_error":
            item.materialized = False
            item.mutation_kind = "omission"
            return
        if error_type == "abstract_position_error":
            item.mutation_kind = "position"
            return
        if error_type == "abstract_substitution_error":
            item.mutation_kind = "substitution"
            item.actual_token = self._mutated_token(
                self.config.substitution_prefix,
                item,
            )
            return
        if error_type == "abstract_reference_error":
            item.mutation_kind = "reference"
            item.actual_token = self._mutated_token(
                self.config.reference_prefix,
                item,
            )
            return
        if error_type == "abstract_service_marker_error":
            item.mutation_kind = "service_marker"
            item.actual_token = self._mutated_token(
                self.config.service_marker_prefix,
                item,
            )
            return

        item.mutation_kind = "unknown"
        item.actual_token = self._mutated_token(
            self.config.unknown_error_prefix,
            item,
        )

    def _apply_position_errors(
        self,
        materialized: list[_WorkingElement],
        element_ids: list[str],
    ) -> None:
        """Сместить ошибочные элементы в рабочей последовательности C."""
        if len(materialized) < 2:
            return
        for element_id in element_ids:
            current_index = next(
                (
                    index
                    for index, item in enumerate(materialized)
                    if item.element_id == element_id
                ),
                None,
            )
            if current_index is None:
                continue
            forward_index = current_index + self.config.position_shift
            if forward_index < len(materialized):
                target_index = forward_index
            else:
                target_index = max(0, current_index - self.config.position_shift)
            if target_index == current_index:
                continue
            item = materialized.pop(current_index)
            materialized.insert(target_index, item)
            item.position_changed = True

    @staticmethod
    def _mutated_token(prefix: str, item: _WorkingElement) -> str:
        """Сформировать детерминированный абстрактный ошибочный токен."""
        return f"{prefix}_{item.step.source_position:04d}_{item.step.abstract_token}"

    @staticmethod
    def _freeze_element(item: _WorkingElement) -> EncodedElement:
        """Зафиксировать материализованный элемент после всех преобразований."""
        if item.encoded_position is None:
            raise RuntimeError("Для материализованного элемента не задана позиция.")
        return EncodedElement(
            element_id=item.element_id,
            encoded_position=item.encoded_position,
            source_step_id=item.step.step_id,
            source_position=item.step.source_position,
            source_value=item.step.source_value,
            source_element_type=item.step.source_element_type,
            operation_type=item.step.operation_type,
            normative_token=item.step.abstract_token,
            token=item.actual_token,
            residual_error=item.residual_error,
            error_type=item.error_type,
            mutation_kind=item.mutation_kind,
        )

    @staticmethod
    def _freeze_trace(item: _WorkingElement) -> EncodingTraceStep:
        """Зафиксировать трассу, включая нематериализованные пропуски."""
        return EncodingTraceStep(
            source_step_id=item.step.step_id,
            source_position=item.step.source_position,
            normative_position=item.normative_position,
            normative_token=item.step.abstract_token,
            materialized=item.materialized,
            encoded_position=item.encoded_position,
            actual_token=item.actual_token if item.materialized else None,
            residual_error=item.residual_error,
            error_type=item.error_type,
            mutation_kind=item.mutation_kind,
            position_changed=item.position_changed,
        )

    @staticmethod
    def _build_metadata(
        steps: tuple[ProcedureStep, ...],
        elements: tuple[EncodedElement, ...],
        traces: tuple[EncodingTraceStep, ...],
    ) -> dict[str, int | float | str | bool]:
        """Рассчитать контрольные показатели материализации C."""
        normative_tokens = tuple(step.abstract_token for step in steps)
        actual_tokens = tuple(element.token for element in elements)
        residual_error_count = sum(trace.residual_error for trace in traces)
        omission_count = sum(
            trace.mutation_kind == "omission" for trace in traces
        )
        moved_count = sum(trace.position_changed for trace in traces)
        altered_token_count = sum(
            trace.materialized
            and trace.actual_token != trace.normative_token
            for trace in traces
        )
        return {
            "normative_element_count": len(steps),
            "materialized_element_count": len(elements),
            "residual_error_count": residual_error_count,
            "omission_count": omission_count,
            "moved_element_count": moved_count,
            "altered_token_count": altered_token_count,
            "is_normative_equivalent": actual_tokens == normative_tokens,
        }


def summarize_material_encoding(
    result: MaterialEncodingResult,
) -> dict[str, int | float | str | bool]:
    """Вернуть компактную сводку результата формирования C."""
    message = result.encoded_message
    return {
        "encoded_message_id": message.encoded_message_id,
        "source_message_id": message.source_message_id,
        "procedure_id": message.procedure_id,
        **message.metadata,
    }
