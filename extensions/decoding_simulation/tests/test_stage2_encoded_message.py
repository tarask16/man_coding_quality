"""Тесты этапа 2: материальное представление кодированного сообщения C."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXTENSION_SRC = PROJECT_ROOT / "extensions/decoding_simulation/src"
BASE_SRC = PROJECT_ROOT / "src"
for source_path in (EXTENSION_SRC, BASE_SRC):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from manual_coding_sim.control_model import ControlProtocol, ControlStepOutcome
from manual_coding_sim.procedure_model import ProcedurePlan, ProcedureStep
from manual_coding_sim_decoding.config import load_decoding_extension_config
from manual_coding_sim_decoding.encoded_message import EncodedMessageBuilder
from manual_coding_sim_decoding.runner import main


CONFIG_PATH = (
    PROJECT_ROOT
    / "extensions/decoding_simulation/configs/decoding_stage2.yaml"
)


def _make_plan(count: int = 5) -> ProcedurePlan:
    """Сформировать компактный нормативный план для тестов материализации."""
    steps = tuple(
        ProcedureStep(
            step_id=f"STEP_{index + 1:04d}",
            source_position=index,
            source_value=f"V{index}",
            source_element_type="symbol",
            operation_type="replace",
            abstract_token=f"TOK_replace_{index:04d}_V{index}",
            nominal_time=1.0,
            complexity=0.2,
            reference_required=False,
            control_marker_required=index == count - 1,
        )
        for index in range(count)
    )
    return ProcedurePlan(
        procedure_id="S_TEST",
        message_id="M_TEST",
        steps=steps,
        metadata={
            "step_count": count,
            "total_nominal_time": float(count),
            "mean_complexity": 0.2,
        },
    )


def _make_control(
    plan: ProcedurePlan,
    errors: dict[int, str] | None = None,
    corrected_positions: set[int] | None = None,
) -> ControlProtocol:
    """Сформировать протокол контроля с заданными остаточными ошибками."""
    errors = errors or {}
    corrected_positions = corrected_positions or set()
    outcomes = []
    for step in plan.steps:
        error_type = errors.get(step.source_position)
        error_occurred = error_type is not None
        error_corrected = step.source_position in corrected_positions
        outcomes.append(
            ControlStepOutcome(
                step_id=step.step_id,
                source_position=step.source_position,
                control_id="K_TEST",
                error_model_id="ERR_TEST",
                error_occurred=error_occurred,
                error_type=error_type,
                error_weight=1.0 if error_occurred else 0.0,
                detection_probability=1.0 if error_occurred else 0.0,
                detection_draw=0.0 if error_occurred else None,
                error_detected=error_occurred,
                correction_probability=1.0 if error_corrected else 0.0,
                correction_draw=0.0 if error_corrected else None,
                error_corrected=error_corrected,
                residual_error=error_occurred and not error_corrected,
                control_effort=0.1,
            )
        )
    return ControlProtocol(
        control_id="K_TEST",
        error_model_id="ERR_TEST",
        condition_id="U_TEST",
        operator_id="O_TEST",
        procedure_id=plan.procedure_id,
        message_id=plan.message_id,
        step_outcomes=tuple(outcomes),
        metadata={
            "step_count": len(outcomes),
            "original_error_count": len(errors),
            "residual_error_count": len(errors) - len(corrected_positions),
        },
    )


def test_stage2_without_errors_matches_normative_plan() -> None:
    """Проверить точное соответствие C нормативному плану без ошибок."""
    plan = _make_plan()
    control = _make_control(plan)
    result = EncodedMessageBuilder().build(plan, control)

    assert result.encoded_message.tokens == tuple(
        step.abstract_token for step in plan.steps
    )
    assert result.encoded_message.metadata["is_normative_equivalent"] is True
    assert result.encoded_message.metadata["residual_error_count"] == 0
    assert all(trace.materialized for trace in result.protocol.trace_steps)
    assert all(trace.mutation_kind == "none" for trace in result.protocol.trace_steps)


def test_stage2_residual_errors_change_only_material_copy() -> None:
    """Проверить все поддержанные мутации и неизменность базовых объектов."""
    plan = _make_plan()
    control = _make_control(
        plan,
        errors={
            0: "abstract_substitution_error",
            1: "abstract_omission_error",
            2: "abstract_position_error",
            3: "abstract_reference_error",
            4: "abstract_service_marker_error",
        },
    )
    plan_snapshot = copy.deepcopy(plan)
    control_snapshot = copy.deepcopy(control)

    result = EncodedMessageBuilder().build(plan, control)

    assert plan == plan_snapshot
    assert control == control_snapshot
    assert len(result.encoded_message.elements) == 4
    assert result.encoded_message.metadata["residual_error_count"] == 5
    assert result.encoded_message.metadata["omission_count"] == 1
    assert result.encoded_message.metadata["is_normative_equivalent"] is False
    assert any(
        element.token.startswith("ERR_SUB_")
        for element in result.encoded_message.elements
    )
    assert any(
        element.token.startswith("ERR_REF_")
        for element in result.encoded_message.elements
    )
    assert any(
        element.token.startswith("ERR_SERVICE_")
        for element in result.encoded_message.elements
    )
    omitted_trace = next(
        trace
        for trace in result.protocol.trace_steps
        if trace.mutation_kind == "omission"
    )
    assert omitted_trace.materialized is False
    assert omitted_trace.encoded_position is None
    position_trace = next(
        trace
        for trace in result.protocol.trace_steps
        if trace.mutation_kind == "position"
    )
    assert position_trace.position_changed is True


def test_stage2_corrected_error_does_not_modify_encoded_token() -> None:
    """Проверить, что исправленная контролем ошибка не переносится в C."""
    plan = _make_plan(count=3)
    control = _make_control(
        plan,
        errors={1: "abstract_substitution_error"},
        corrected_positions={1},
    )
    result = EncodedMessageBuilder().build(plan, control)

    assert result.encoded_message.tokens == tuple(
        step.abstract_token for step in plan.steps
    )
    assert result.encoded_message.metadata["residual_error_count"] == 0
    assert result.encoded_message.metadata["is_normative_equivalent"] is True


def test_stage2_rejects_inconsistent_identifiers() -> None:
    """Проверить остановку при несовпадении идентификаторов входных артефактов."""
    plan = _make_plan()
    control = _make_control(plan)
    inconsistent = ControlProtocol(
        control_id=control.control_id,
        error_model_id=control.error_model_id,
        condition_id=control.condition_id,
        operator_id=control.operator_id,
        procedure_id=control.procedure_id,
        message_id="M_OTHER",
        step_outcomes=control.step_outcomes,
        metadata=control.metadata,
    )

    with pytest.raises(ValueError, match="message_id"):
        EncodedMessageBuilder().build(plan, inconsistent)


def test_stage2_config_loads_material_encoding_rules() -> None:
    """Проверить загрузку новых правил без нарушения конфигурации этапа 1."""
    config = load_decoding_extension_config(CONFIG_PATH)

    assert config.material_encoding.encoded_message_prefix == "C"
    assert config.material_encoding.position_shift == 1
    assert config.material_encoding.substitution_prefix == "ERR_SUB"


def test_stage2_cli_writes_only_extension_artifact() -> None:
    """Проверить формирование демонстрационного JSON внутри расширения."""
    output_relative = "reports/stage2/pytest_encoded_message.json"
    output_path = (
        PROJECT_ROOT / "extensions/decoding_simulation" / output_relative
    )
    output_path.unlink(missing_ok=True)

    exit_code = main(
        [
            "--project-root",
            str(PROJECT_ROOT),
            "--config",
            str(CONFIG_PATH),
            "--check-base-contract",
            "--build-encoded-message",
            "--message-id",
            "M_PYTEST_STAGE2",
            "--output",
            output_relative,
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["encoded_message"]["source_message_id"] == "M_PYTEST_STAGE2"
    assert payload["encoded_message"]["encoded_message_id"] == "C_M_PYTEST_STAGE2"
    output_path.unlink()
