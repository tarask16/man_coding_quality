"""Тесты этапа 3: формальная обратная процедура декодирования D_h(C)."""

from __future__ import annotations

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

from manual_coding_sim_decoding.config import (
    FormalDecodingConfig,
    load_decoding_extension_config,
)
from manual_coding_sim_decoding.decoding_procedure import DecodingProcedureModel
from manual_coding_sim_decoding.encoded_message import EncodedElement, EncodedMessage
from manual_coding_sim_decoding.runner import main


CONFIG_PATH = (
    PROJECT_ROOT
    / "extensions/decoding_simulation/configs/decoding_stage3.yaml"
)


def _element(
    index: int,
    token: str,
    *,
    source_value: str = "HIDDEN_VALUE",
    residual_error: bool = False,
    error_type: str | None = None,
) -> EncodedElement:
    """Сформировать элемент C с управляемым токеном и скрытой трассой."""
    return EncodedElement(
        element_id=f"ENC_{index + 1:04d}",
        encoded_position=index,
        source_step_id=f"STEP_{index + 1:04d}",
        source_position=index,
        source_value=source_value,
        source_element_type="symbol",
        operation_type="trace_only_operation",
        normative_token=f"NORMATIVE_{index}",
        token=token,
        residual_error=residual_error,
        error_type=error_type,
        mutation_kind="none" if not residual_error else "substitution",
    )


def _message(elements: tuple[EncodedElement, ...]) -> EncodedMessage:
    """Сформировать компактное кодированное сообщение C."""
    return EncodedMessage(
        encoded_message_id="C_TEST",
        source_message_id="M_TEST",
        procedure_id="S_TEST",
        elements=elements,
        metadata={"materialized_element_count": len(elements)},
    )


def test_stage3_builds_inverse_steps_for_supported_operations() -> None:
    """Проверить правила для символа, цифры, служебного и копируемого элемента."""
    message = _message(
        (
            _element(0, "TOK_abstract_substitution_0000_A"),
            _element(1, "TOK_abstract_numeric_mapping_0001_7"),
            _element(2, "TOK_abstract_service_marking_0002_SEP"),
            _element(3, "TOK_abstract_copying_0003_Z"),
        )
    )

    plan = DecodingProcedureModel().build_plan(message)

    assert plan.candidate_values == ("A", "7", "SEP", "Z")
    assert [step.inferred_element_type for step in plan.steps] == [
        "symbol",
        "digit",
        "service",
        "symbol",
    ]
    assert [step.decoding_operation_type for step in plan.steps] == [
        "abstract_inverse_substitution",
        "abstract_inverse_numeric_mapping",
        "abstract_inverse_service_interpretation",
        "abstract_inverse_copying",
    ]
    assert plan.metadata["recognized_token_count"] == 4
    assert plan.metadata["unresolved_token_count"] == 0
    assert plan.metadata["decoding_plan_complete"] is True


def test_stage3_does_not_use_source_value_for_candidate() -> None:
    """Проверить отсутствие восстановления по скрытому source_value."""
    element = _element(
        0,
        "TOK_abstract_substitution_0000_VISIBLE",
        source_value="SECRET_SOURCE_VALUE",
    )

    plan = DecodingProcedureModel().build_plan(_message((element,)))

    assert plan.steps[0].candidate_value == "VISIBLE"
    assert plan.steps[0].candidate_value != element.source_value
    assert plan.metadata["uses_source_value_for_decoding"] is False


def test_stage3_marks_damaged_token_as_unresolved() -> None:
    """Проверить безопасную фиксацию поврежденного токена без подстановки M."""
    element = _element(
        0,
        "ERR_SUB_TOK_abstract_substitution_0000_A",
        source_value="A",
        residual_error=True,
        error_type="abstract_substitution_error",
    )

    plan = DecodingProcedureModel().build_plan(_message((element,)))
    step = plan.steps[0]

    assert step.token_recognized is False
    assert step.candidate_value is None
    assert step.decoding_operation_type == "abstract_unresolved_inverse_operation"
    assert step.input_residual_encoding_error is True
    assert plan.metadata["unresolved_token_count"] == 1
    assert plan.metadata["decoding_plan_complete"] is False


def test_stage3_strict_mode_rejects_unknown_token() -> None:
    """Проверить режим немедленной остановки на неизвестном токене."""
    config = FormalDecodingConfig(fail_on_unknown_token=True)
    model = DecodingProcedureModel(config)

    with pytest.raises(ValueError, match="Невозможно построить шаг"):
        model.build_plan(_message((_element(0, "BROKEN_TOKEN"),)))


def test_stage3_preserves_actual_encoded_order() -> None:
    """Проверить, что встроенная позиция токена не меняет фактический порядок C."""
    message = _message(
        (
            _element(0, "TOK_abstract_substitution_0001_B"),
            _element(1, "TOK_abstract_substitution_0000_A"),
        )
    )

    plan = DecodingProcedureModel().build_plan(message)

    assert plan.candidate_values == ("B", "A")
    assert [step.declared_source_position for step in plan.steps] == [1, 0]
    assert [step.encoded_position for step in plan.steps] == [0, 1]
    assert plan.metadata["preserved_encoded_order"] is True


def test_stage3_rejects_non_contiguous_encoded_positions() -> None:
    """Проверить остановку при нарушении линейного порядка C."""
    first = _element(0, "TOK_abstract_substitution_0000_A")
    second = EncodedElement(
        element_id="ENC_0002",
        encoded_position=3,
        source_step_id="STEP_0002",
        source_position=1,
        source_value="B",
        source_element_type="symbol",
        operation_type="abstract_substitution",
        normative_token="TOK_abstract_substitution_0001_B",
        token="TOK_abstract_substitution_0001_B",
        residual_error=False,
        error_type=None,
        mutation_kind="none",
    )

    with pytest.raises(ValueError, match="encoded_position"):
        DecodingProcedureModel().build_plan(_message((first, second)))


def test_stage3_config_loads_formal_decoding_rules() -> None:
    """Проверить загрузку правил D_h вместе с конфигурацией предыдущих этапов."""
    config = load_decoding_extension_config(CONFIG_PATH)

    assert config.extension.version == "0.1.0"
    assert config.formal_decoding.decoding_procedure_id == "D_001"
    assert len(config.formal_decoding.operation_rules) == 4
    assert config.formal_decoding.fail_on_unknown_token is False
    assert config.material_encoding.encoded_message_prefix == "C"


def test_stage3_cli_writes_plan_only_inside_extension() -> None:
    """Проверить формирование JSON плана D_h(C) в изолированной папке."""
    output_relative = "reports/stage3/pytest_decoding_plan.json"
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
            "--build-decoding-plan",
            "--message-id",
            "M_PYTEST_STAGE3",
            "--output",
            output_relative,
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["encoded_message"]["source_message_id"] == "M_PYTEST_STAGE3"
    assert payload["decoding_plan"]["encoded_message_id"] == "C_M_PYTEST_STAGE3"
    assert payload["decoding_plan"]["metadata"]["uses_source_value_for_decoding"] is False
    output_path.unlink()
