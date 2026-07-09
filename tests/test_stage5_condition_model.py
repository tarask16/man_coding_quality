"""Тесты модели условий применения U."""

import pytest

from manual_coding_sim.condition_model import (
    ConditionModel,
    ConditionModelConfig,
    ConditionProfile,
    condition_estimates_to_rows,
    summarize_condition_estimate,
)
from manual_coding_sim.message_model import MessageGenerationConfig, MessageModel
from manual_coding_sim.operator_model import OperatorModel
from manual_coding_sim.procedure_model import ProcedureModel


def _make_operator_estimate(message_id: str = "M_CONDITION"):
    """Формирует воспроизводимую оценку O + S для проверки условий U."""
    message_config = MessageGenerationConfig(min_length=8, max_length=8)
    message = MessageModel(config=message_config, random_seed=42).generate_message(
        message_id=message_id,
    )
    plan = ProcedureModel().build_plan(message)
    return OperatorModel().estimate_plan(plan)


def test_condition_model_estimates_operator_plan() -> None:
    """Модель U должна рассчитывать оценку для каждого шага плана O + S."""
    operator_estimate = _make_operator_estimate()
    model = ConditionModel()

    estimate = model.estimate_plan(operator_estimate)

    assert estimate.condition_id == "U_001"
    assert estimate.operator_id == operator_estimate.operator_id
    assert estimate.procedure_id == operator_estimate.procedure_id
    assert estimate.message_id == operator_estimate.message_id
    assert len(estimate.step_estimates) == len(operator_estimate.step_estimates)
    assert estimate.metadata["step_count"] == len(operator_estimate.step_estimates)
    assert estimate.metadata["total_adjusted_time"] > 0


def test_condition_estimates_have_valid_ranges() -> None:
    """Расчетные показатели условий U должны оставаться в допустимых диапазонах."""
    estimate = ConditionModel().estimate_plan(_make_operator_estimate())

    for step in estimate.step_estimates:
        assert step.adjusted_time >= step.baseline_time
        assert 0.0 <= step.baseline_attention <= 1.0
        assert 0.0 <= step.adjusted_attention <= 1.0
        assert 0.0 <= step.environmental_load <= 1.0
        assert 0.0 <= step.time_pressure <= 1.0
        assert 0.0 <= step.instruction_pressure <= 1.0
        assert 0.0 <= step.stability_index <= 1.0


def test_harsh_conditions_increase_adjusted_time() -> None:
    """Неблагоприятные условия U должны увеличивать расчетную длительность."""
    operator_estimate = _make_operator_estimate()
    mild_model = ConditionModel(
        ConditionModelConfig(
            profile=ConditionProfile(
                condition_id="U_MILD",
                noise_level=0.0,
                instruction_access=1.0,
                workload_level=0.0,
                interruption_rate=0.0,
                lighting_quality=1.0,
            )
        )
    )
    harsh_model = ConditionModel(
        ConditionModelConfig(
            profile=ConditionProfile(
                condition_id="U_HARSH",
                time_limit_seconds=1.0,
                noise_level=1.0,
                instruction_access=0.0,
                workload_level=1.0,
                interruption_rate=1.0,
                lighting_quality=0.0,
            )
        )
    )

    mild_time = mild_model.estimate_plan(operator_estimate).metadata[
        "total_adjusted_time"
    ]
    harsh_time = harsh_model.estimate_plan(operator_estimate).metadata[
        "total_adjusted_time"
    ]

    assert harsh_time > mild_time


def test_time_limit_creates_time_pressure() -> None:
    """Жесткий лимит времени должен создавать ненулевое временное давление."""
    operator_estimate = _make_operator_estimate()
    model = ConditionModel(
        ConditionModelConfig(
            profile=ConditionProfile(time_limit_seconds=1.0)
        )
    )

    estimate = model.estimate_plan(operator_estimate)

    assert estimate.metadata["time_pressure"] > 0.0
    assert {step.time_pressure for step in estimate.step_estimates} == {
        estimate.metadata["time_pressure"]
    }


def test_good_conditions_preserve_time_pressure_zero() -> None:
    """При отсутствии лимита времени временное давление должно быть нулевым."""
    estimate = ConditionModel().estimate_plan(_make_operator_estimate())

    assert estimate.metadata["time_pressure"] == 0.0
    assert {step.time_pressure for step in estimate.step_estimates} == {0.0}


def test_estimate_batch_creates_estimate_for_each_operator_plan() -> None:
    """Пакет оценок O должен преобразовываться в пакет оценок U."""
    message_model = MessageModel(
        config=MessageGenerationConfig(min_length=4, max_length=4),
        random_seed=42,
    )
    messages = message_model.generate_batch(3)
    plans = ProcedureModel().build_batch(messages)
    operator_estimates = OperatorModel().estimate_batch(plans)

    condition_estimates = ConditionModel().estimate_batch(operator_estimates)

    assert len(condition_estimates) == 3
    assert [estimate.message_id for estimate in condition_estimates] == [
        "M_000001",
        "M_000002",
        "M_000003",
    ]


def test_summary_and_rows() -> None:
    """Сводка условий U должна содержать априорные характеристики среды."""
    estimate = ConditionModel().estimate_plan(_make_operator_estimate())

    summary = summarize_condition_estimate(estimate)
    rows = condition_estimates_to_rows((estimate,))

    assert summary["condition_id"] == "U_001"
    assert summary["message_id"] == "M_CONDITION"
    assert summary["step_count"] == 8
    assert summary["baseline_total_time"] > 0
    assert summary["total_adjusted_time"] >= summary["baseline_total_time"]
    assert 0.0 <= summary["mean_adjusted_attention"] <= 1.0
    assert 0.0 <= summary["mean_environmental_load"] <= 1.0
    assert 0.0 <= summary["mean_stability_index"] <= 1.0
    assert rows == [summary]


def test_invalid_condition_profile_is_rejected() -> None:
    """Профиль условий применения U должен отклонять некорректные параметры."""
    invalid_config = ConditionModelConfig(
        profile=ConditionProfile(noise_level=1.5)
    )

    with pytest.raises(ValueError, match="noise_level"):
        ConditionModel(config=invalid_config)
