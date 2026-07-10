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
