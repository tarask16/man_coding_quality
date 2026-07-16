"""Формальная обратная процедура декодирования кодированного сообщения C.

Модуль строит нормативный план декодирования ``D_h(C)`` по фактической
последовательности токенов :class:`EncodedMessage`. На этапе 3 не моделируются
ошибки декодирующего оператора, контроль декодирования и восстановленное
сообщение ``M'``. Результатом является только проверяемая последовательность
обратных операций.

Принципиальное ограничение: модель не использует ``source_value`` элементов C
для определения результата. Кандидатное значение извлекается только из
распознанного абстрактного токена. Поэтому поврежденный токен остается
неразрешенным и не восстанавливается по скрытой трассировочной информации.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from manual_coding_sim_decoding.config import (
    DecodingOperationRule,
    FormalDecodingConfig,
)
from manual_coding_sim_decoding.encoded_message import EncodedMessage


@dataclass(frozen=True)
class TokenParseResult:
    """Результат синтаксического разбора одного фактического токена C."""

    token: str
    recognized: bool
    encoding_operation_type: str | None
    declared_source_position: int | None
    candidate_value: str | None
    inferred_element_type: str | None
    failure_reason: str | None


@dataclass(frozen=True)
class DecodingStep:
    """Один нормативный шаг обратной процедуры декодирования."""

    decoding_step_id: str
    encoded_element_id: str
    encoded_position: int
    source_step_id: str
    observed_token: str
    token_recognized: bool
    encoding_operation_type: str | None
    decoding_operation_type: str
    declared_source_position: int | None
    candidate_value: str | None
    inferred_element_type: str | None
    nominal_time: float
    complexity: float
    reference_required: bool
    input_residual_encoding_error: bool
    input_error_type: str | None
    unresolved_reason: str | None


@dataclass(frozen=True)
class DecodingPlan:
    """Нормативный план ``D_h(C)`` без моделирования выполнения оператором."""

    decoding_procedure_id: str
    encoded_message_id: str
    source_message_id: str
    coding_procedure_id: str
    steps: tuple[DecodingStep, ...]
    metadata: dict[str, int | float | str | bool]

    @property
    def candidate_values(self) -> tuple[str | None, ...]:
        """Вернуть кандидатные значения в фактическом порядке элементов C."""
        return tuple(step.candidate_value for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Преобразовать план в JSON-совместимый словарь."""
        return asdict(self)


class DecodingProcedureModel:
    """Построитель формальной обратной процедуры по сообщению C."""

    def __init__(self, config: FormalDecodingConfig | None = None) -> None:
        """Инициализировать модель и индекс правил обратных операций."""
        self.config = config or FormalDecodingConfig()
        self.config.validate()
        self._rules_by_encoding_operation = {
            rule.encoding_operation_type: rule
            for rule in self.config.operation_rules
        }
        position_width = self.config.token_position_width
        token_prefix = re.escape(self.config.token_prefix)
        self._token_pattern = re.compile(
            rf"^{token_prefix}_(?P<operation>.+)_"
            rf"(?P<position>\d{{{position_width}}})_(?P<value>.+)$"
        )

    def build_plan(self, encoded_message: EncodedMessage) -> DecodingPlan:
        """Построить нормативный план декодирования фактического сообщения C.

        Порядок шагов совпадает с ``encoded_position``. Поле
        ``declared_source_position``, извлеченное из токена, сохраняется только
        для диагностики и не используется для перестановки элементов.
        """
        self._validate_encoded_message(encoded_message)
        steps = tuple(
            self._build_step(element, index)
            for index, element in enumerate(encoded_message.elements, start=1)
        )
        recognized_count = sum(step.token_recognized for step in steps)
        unresolved_count = len(steps) - recognized_count
        reference_count = sum(step.reference_required for step in steps)
        total_nominal_time = sum(step.nominal_time for step in steps)
        mean_complexity = (
            sum(step.complexity for step in steps) / len(steps)
            if steps
            else 0.0
        )

        return DecodingPlan(
            decoding_procedure_id=self.config.decoding_procedure_id,
            encoded_message_id=encoded_message.encoded_message_id,
            source_message_id=encoded_message.source_message_id,
            coding_procedure_id=encoded_message.procedure_id,
            steps=steps,
            metadata={
                "input_element_count": len(encoded_message.elements),
                "recognized_token_count": recognized_count,
                "unresolved_token_count": unresolved_count,
                "reference_step_count": reference_count,
                "total_nominal_time": round(total_nominal_time, 6),
                "mean_complexity": round(mean_complexity, 6),
                "preserved_encoded_order": True,
                "uses_source_value_for_decoding": False,
                "decoding_plan_complete": unresolved_count == 0,
            },
        )

    def parse_token(self, token: str) -> TokenParseResult:
        """Разобрать нормативный абстрактный токен без доступа к исходному M."""
        if not token or not token.strip():
            return self._unrecognized(token, "Пустой токен кодированного элемента.")

        match = self._token_pattern.fullmatch(token)
        if match is None:
            return self._unrecognized(
                token,
                "Токен не соответствует нормативному формату обратной процедуры.",
            )

        operation_type = match.group("operation")
        if operation_type not in self._rules_by_encoding_operation:
            return TokenParseResult(
                token=token,
                recognized=False,
                encoding_operation_type=operation_type,
                declared_source_position=int(match.group("position")),
                candidate_value=None,
                inferred_element_type=None,
                failure_reason=(
                    "Для операции кодирования не задано правило обратного "
                    f"преобразования: {operation_type}."
                ),
            )

        encoded_value = match.group("value")
        candidate_value = encoded_value.replace("_", " ")
        return TokenParseResult(
            token=token,
            recognized=True,
            encoding_operation_type=operation_type,
            declared_source_position=int(match.group("position")),
            candidate_value=candidate_value,
            inferred_element_type=self._infer_element_type(candidate_value),
            failure_reason=None,
        )

    def _build_step(self, element, index: int) -> DecodingStep:
        """Сформировать один шаг D_h по фактическому токену элемента C."""
        parsed = self.parse_token(element.token)
        rule = self._resolve_rule(parsed)
        if self.config.fail_on_unknown_token and not parsed.recognized:
            raise ValueError(
                f"Невозможно построить шаг для токена {element.token}: "
                f"{parsed.failure_reason}"
            )

        return DecodingStep(
            decoding_step_id=f"DSTEP_{index:04d}",
            encoded_element_id=element.element_id,
            encoded_position=element.encoded_position,
            source_step_id=element.source_step_id,
            observed_token=element.token,
            token_recognized=parsed.recognized,
            encoding_operation_type=parsed.encoding_operation_type,
            decoding_operation_type=rule.decoding_operation_type,
            declared_source_position=parsed.declared_source_position,
            candidate_value=parsed.candidate_value,
            inferred_element_type=parsed.inferred_element_type,
            nominal_time=rule.nominal_time,
            complexity=rule.complexity,
            reference_required=rule.reference_required,
            input_residual_encoding_error=element.residual_error,
            input_error_type=element.error_type,
            unresolved_reason=parsed.failure_reason,
        )

    def _resolve_rule(self, parsed: TokenParseResult) -> DecodingOperationRule:
        """Выбрать обратное правило или безопасное правило неразрешенного шага."""
        if parsed.recognized and parsed.encoding_operation_type is not None:
            return self._rules_by_encoding_operation[parsed.encoding_operation_type]
        return self.config.unresolved_rule

    def _infer_element_type(self, value: str) -> str:
        """Определить абстрактный тип восстановимого значения."""
        if value in self.config.service_values:
            return "service"
        if value.isdigit():
            return "digit"
        return "symbol"

    @staticmethod
    def _unrecognized(token: str, reason: str) -> TokenParseResult:
        """Сформировать единообразный результат неуспешного разбора."""
        return TokenParseResult(
            token=token,
            recognized=False,
            encoding_operation_type=None,
            declared_source_position=None,
            candidate_value=None,
            inferred_element_type=None,
            failure_reason=reason,
        )

    @staticmethod
    def _validate_encoded_message(encoded_message: EncodedMessage) -> None:
        """Проверить идентификаторы и линейный порядок элементов C."""
        if not encoded_message.encoded_message_id:
            raise ValueError("Не задан encoded_message_id сообщения C.")
        if not encoded_message.source_message_id:
            raise ValueError("Не задан source_message_id сообщения C.")
        if not encoded_message.procedure_id:
            raise ValueError("Не задан procedure_id сообщения C.")
        if not encoded_message.elements:
            raise ValueError("Кодированное сообщение C не содержит элементов.")

        element_ids = [element.element_id for element in encoded_message.elements]
        if len(set(element_ids)) != len(element_ids):
            raise ValueError("Сообщение C содержит дублирующиеся element_id.")

        positions = [
            element.encoded_position for element in encoded_message.elements
        ]
        if positions != list(range(len(encoded_message.elements))):
            raise ValueError(
                "encoded_position должен образовывать непрерывный порядок 0..n-1."
            )
        if any(not element.token for element in encoded_message.elements):
            raise ValueError("Все элементы C должны содержать непустой token.")


def summarize_decoding_plan(plan: DecodingPlan) -> dict[str, int | float | str | bool]:
    """Вернуть компактную сводку нормативного плана декодирования."""
    return {
        "decoding_procedure_id": plan.decoding_procedure_id,
        "encoded_message_id": plan.encoded_message_id,
        "source_message_id": plan.source_message_id,
        "input_element_count": int(plan.metadata["input_element_count"]),
        "recognized_token_count": int(plan.metadata["recognized_token_count"]),
        "unresolved_token_count": int(plan.metadata["unresolved_token_count"]),
        "total_nominal_time": float(plan.metadata["total_nominal_time"]),
        "mean_complexity": float(plan.metadata["mean_complexity"]),
        "decoding_plan_complete": bool(plan.metadata["decoding_plan_complete"]),
    }
