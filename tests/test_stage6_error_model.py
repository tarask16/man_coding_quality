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
