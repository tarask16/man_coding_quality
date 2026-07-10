"""Тесты модели контрольных процедур ControlModel."""

from dataclasses import replace

import pytest

from manual_coding_sim.condition_model import ConditionModel
from manual_coding_sim.control_model import (
    ControlModel,
    ControlModelConfig,
    ControlProfile,
    ControlProtocol,
    control_protocols_to_rows,
    summarize_control_protocol,
)
from manual_coding_sim.error_model import ErrorModel, ErrorModelConfig
from manual_coding_sim.message_model import MessageGenerationConfig, MessageModel
from manual_coding_sim.operator_model import OperatorModel
from manual_coding_sim.procedure_model import ProcedureModel


def _make_condition_estimate(message_id: str = "M_CONTROL"):
    """Формирует оценку S + O + U для тестов контроля K."""
    message_config = MessageGenerationConfig(min_length=10, max_length=10)
    message = MessageModel(config=message_config, random_seed=42).generate_message(
        message_id=message_id,
    )
    plan = ProcedureModel().build_plan(message)
    operator_estimate = OperatorModel().estimate_plan(plan)
    return ConditionModel().estimate_plan(operator_estimate)


def _make_error_protocol(
    message_id: str = "M_CONTROL",
    force_errors: bool = True,
):
    """Формирует протокол ошибок для проверки контрольных процедур K."""
    condition_estimate = _make_condition_estimate(message_id=message_id)
    if force_errors:
        error_config = ErrorModelConfig(
            base_error_rate=1.0,
            max_error_probability=1.0,
        )
    else:
        error_config = ErrorModelConfig(
            base_error_rate=0.0,
            max_error_probability=1.0,
        )
    return ErrorModel(config=error_config, random_seed=123).generate_protocol(
        condition_estimate,
    )


def test_control_model_generates_protocol_for_error_protocol() -> None:
    """Модель K должна формировать протокол контроля для ErrorProtocol."""
    error_protocol = _make_error_protocol()
    control_protocol = ControlModel(random_seed=77).generate_protocol(
        error_protocol,
    )

    assert isinstance(control_protocol, ControlProtocol)
    assert control_protocol.control_id == "K_001"
    assert control_protocol.error_model_id == error_protocol.error_model_id
    assert control_protocol.condition_id == error_protocol.condition_id
    assert control_protocol.operator_id == error_protocol.operator_id
    assert control_protocol.procedure_id == error_protocol.procedure_id
    assert control_protocol.message_id == error_protocol.message_id
    assert len(control_protocol.step_outcomes) == len(
        error_protocol.step_outcomes,
    )


def test_control_step_outcomes_have_valid_ranges() -> None:
    """Вероятности контроля и расчетные показатели должны быть допустимыми."""
    protocol = ControlModel(random_seed=77).generate_protocol(
        _make_error_protocol(),
    )

    for outcome in protocol.step_outcomes:
        assert 0.0 <= outcome.detection_probability <= 1.0
        assert 0.0 <= outcome.correction_probability <= 1.0
        assert outcome.control_effort > 0.0
        if outcome.detection_draw is not None:
            assert 0.0 <= outcome.detection_draw <= 1.0
        if outcome.correction_draw is not None:
            assert 0.0 <= outcome.correction_draw <= 1.0
        if not outcome.error_occurred:
            assert outcome.error_detected is False
            assert outcome.error_corrected is False
            assert outcome.residual_error is False


def test_control_model_is_reproducible_with_same_seed() -> None:
    """Одинаковый random_seed должен давать одинаковый протокол контроля."""
    error_protocol = _make_error_protocol()

    first = ControlModel(random_seed=555).generate_protocol(error_protocol)
    second = ControlModel(random_seed=555).generate_protocol(error_protocol)

    first_trace = [
        (
            outcome.detection_draw,
            outcome.error_detected,
            outcome.correction_draw,
            outcome.error_corrected,
            outcome.residual_error,
        )
        for outcome in first.step_outcomes
    ]
    second_trace = [
        (
            outcome.detection_draw,
            outcome.error_detected,
            outcome.correction_draw,
            outcome.error_corrected,
            outcome.residual_error,
        )
        for outcome in second.step_outcomes
    ]

    assert first_trace == second_trace


def test_control_model_reset_restores_random_sequence() -> None:
    """Сброс модели K должен восстанавливать последовательность контроля."""
    error_protocol = _make_error_protocol()
    model = ControlModel(random_seed=555)

    first = model.generate_protocol(error_protocol)
    model.generate_protocol(error_protocol)
    model.reset()
    after_reset = model.generate_protocol(error_protocol)

    assert [item.detection_draw for item in first.step_outcomes] == [
        item.detection_draw for item in after_reset.step_outcomes
    ]
    assert [item.error_corrected for item in first.step_outcomes] == [
        item.error_corrected for item in after_reset.step_outcomes
    ]


def test_stronger_control_profile_increases_detection_probability() -> None:
    """Более сильный профиль K должен повышать вероятность обнаружения ошибок."""
    error_protocol = _make_error_protocol()
    weak_profile = ControlProfile(
        control_id="K_WEAK",
        detection_skill=0.0,
        correction_skill=0.0,
        reference_check_coverage=0.0,
        repeated_check_coverage=0.0,
        critical_error_priority=0.0,
    )
    strong_profile = ControlProfile(
        control_id="K_STRONG",
        detection_skill=1.0,
        correction_skill=1.0,
        reference_check_coverage=1.0,
        repeated_check_coverage=1.0,
        critical_error_priority=1.0,
    )

    weak_protocol = ControlModel(
        config=ControlModelConfig(profile=weak_profile),
        random_seed=1,
    ).generate_protocol(error_protocol)
    strong_protocol = ControlModel(
        config=ControlModelConfig(profile=strong_profile),
        random_seed=1,
    ).generate_protocol(error_protocol)

    assert (
        strong_protocol.metadata["mean_detection_probability"]
        > weak_protocol.metadata["mean_detection_probability"]
    )


def test_residual_error_count_matches_step_flags() -> None:
    """Число остаточных ошибок должно соответствовать флагам шагов."""
    protocol = ControlModel(random_seed=7).generate_protocol(
        _make_error_protocol(),
    )

    expected_residual = sum(
        1 for outcome in protocol.step_outcomes if outcome.residual_error
    )
    expected_detected = sum(
        1 for outcome in protocol.step_outcomes if outcome.error_detected
    )
    expected_corrected = sum(
        1 for outcome in protocol.step_outcomes if outcome.error_corrected
    )

    assert protocol.metadata["residual_error_count"] == expected_residual
    assert protocol.metadata["detected_error_count"] == expected_detected
    assert protocol.metadata["corrected_error_count"] == expected_corrected


def test_control_model_handles_protocol_without_errors() -> None:
    """При отсутствии ошибок K не должен формировать обнаружения и исправления."""
    protocol = ControlModel(random_seed=9).generate_protocol(
        _make_error_protocol(force_errors=False),
    )

    assert protocol.metadata["original_error_count"] == 0
    assert protocol.metadata["detected_error_count"] == 0
    assert protocol.metadata["corrected_error_count"] == 0
    assert protocol.metadata["residual_error_count"] == 0
    assert protocol.metadata["detection_rate"] == 0.0
    assert protocol.metadata["correction_rate"] == 0.0


def test_control_model_generates_batch_and_rows() -> None:
    """Пакетная обработка должна формировать строки контрольной таблицы."""
    protocols = (
        _make_error_protocol(message_id="M_CONTROL_1"),
        _make_error_protocol(message_id="M_CONTROL_2"),
    )
    control_protocols = ControlModel(random_seed=42).generate_batch(protocols)
    rows = control_protocols_to_rows(control_protocols)

    assert len(control_protocols) == 2
    assert len(rows) == 2
    assert rows[0]["message_id"] == "M_CONTROL_1"
    assert rows[1]["message_id"] == "M_CONTROL_2"
    assert "residual_error_count" in rows[0]


def test_invalid_control_model_config_is_rejected() -> None:
    """Некорректная конфигурация K должна отклоняться."""
    with pytest.raises(ValueError):
        ControlProfile(detection_skill=-0.1).validate()

    with pytest.raises(ValueError):
        ControlProfile(reference_check_coverage=1.5).validate()

    with pytest.raises(ValueError):
        ControlModelConfig(base_control_effort=0.0).validate()

    with pytest.raises(ValueError):
        ControlModelConfig(
            base_detection_rate=0.9,
            max_detection_probability=0.5,
        ).validate()

    with pytest.raises(ValueError):
        ControlModelConfig(
            base_correction_rate=0.9,
            max_correction_probability=0.5,
        ).validate()


def test_empty_error_protocol_is_rejected() -> None:
    """Пустой протокол ошибок не должен использоваться для контроля."""
    error_protocol = _make_error_protocol()
    empty_protocol = replace(error_protocol, step_outcomes=())

    with pytest.raises(ValueError):
        ControlModel().generate_protocol(empty_protocol)


def test_summarize_control_protocol_returns_control_summary() -> None:
    """Сводка должна содержать ключевые показатели протокола K."""
    protocol = ControlModel(random_seed=77).generate_protocol(
        _make_error_protocol(),
    )
    summary = summarize_control_protocol(protocol)

    assert summary["control_id"] == protocol.control_id
    assert summary["message_id"] == protocol.message_id
    assert summary["step_count"] == protocol.metadata["step_count"]
    assert summary["residual_error_count"] == protocol.metadata[
        "residual_error_count"
    ]
    assert summary["random_seed"] == 77
