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
