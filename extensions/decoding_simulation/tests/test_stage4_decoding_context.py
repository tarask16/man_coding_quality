"""Тесты этапа 4: декодирующий оператор и условия выполнения D_h(C)."""

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
    DecodingConditionConfig,
    DecodingConditionProfile,
    DecodingOperatorConfig,
    DecodingOperatorProfile,
    load_decoding_extension_config,
)
from manual_coding_sim_decoding.decoding_context import (
    DecodingConditionModel,
    DecodingExecutionContextModel,
    DecodingOperatorModel,
)
from manual_coding_sim_decoding.decoding_procedure import DecodingPlan, DecodingStep
from manual_coding_sim_decoding.runner import main


CONFIG_PATH = (
    PROJECT_ROOT
    / "extensions/decoding_simulation/configs/decoding_stage4.yaml"
)


def _step(
    index: int,
    *,
    recognized: bool = True,
    reference_required: bool = False,
    complexity: float = 0.4,
) -> DecodingStep:
    """Сформировать управляемый нормативный шаг декодирования."""
    return DecodingStep(
        decoding_step_id=f"DSTEP_{index + 1:04d}",
        encoded_element_id=f"ENC_{index + 1:04d}",
        encoded_position=index,
        source_step_id=f"STEP_{index + 1:04d}",
        observed_token=(
            f"TOK_abstract_substitution_{index:04d}_A"
            if recognized
            else "BROKEN_TOKEN"
        ),
        token_recognized=recognized,
        encoding_operation_type="abstract_substitution" if recognized else None,
        decoding_operation_type=(
            "abstract_inverse_substitution"
            if recognized
            else "abstract_unresolved_inverse_operation"
        ),
        declared_source_position=index if recognized else None,
        candidate_value="A" if recognized else None,
        inferred_element_type="symbol" if recognized else None,
        nominal_time=1.2,
        complexity=complexity,
        reference_required=reference_required,
        input_residual_encoding_error=not recognized,
        input_error_type=(
            None if recognized else "abstract_substitution_error"
        ),
        unresolved_reason=None if recognized else "Поврежденный токен.",
    )


def _plan(*steps: DecodingStep) -> DecodingPlan:
    """Сформировать компактный формальный план D_h(C)."""
    return DecodingPlan(
        decoding_procedure_id="D_TEST",
        encoded_message_id="C_TEST",
        source_message_id="M_TEST",
        coding_procedure_id="S_TEST",
        steps=tuple(steps),
        metadata={
            "input_element_count": len(steps),
            "recognized_token_count": sum(step.token_recognized for step in steps),
            "unresolved_token_count": sum(
                not step.token_recognized for step in steps
            ),
            "uses_source_value_for_decoding": False,
        },
    )


def test_stage4_config_loads_independent_operator_and_conditions() -> None:
    """Проверить загрузку отдельных профилей O_d и U_d."""
    config = load_decoding_extension_config(CONFIG_PATH)

    assert config.decoding_operator.profile.operator_id == "OD_001"
    assert config.decoding_conditions.profile.condition_id == "UD_001"
    assert config.decoding_operator.profile.interpretation_skill == 0.72
    assert config.decoding_conditions.profile.time_limit_seconds is None
    assert config.extension.version == "0.1.0"


def test_stage4_operator_estimate_is_deterministic_and_independent() -> None:
    """Проверить детерминированную оценку отдельного декодирующего оператора."""
    plan = _plan(_step(0), _step(1, reference_required=True, complexity=0.6))
    model = DecodingOperatorModel()

    first = model.estimate_plan(plan)
    second = model.estimate_plan(plan)

    assert first == second
    assert first.operator_id == "OD_001"
    assert len(first.step_estimates) == 2
    assert first.metadata["decoder_profile_is_independent"] is True
    assert first.metadata["uses_source_value_for_decoding"] is False
    assert first.metadata["generates_decoding_errors"] is False
    assert first.step_estimates[1].reference_pressure > 0.0


def test_stage4_operator_does_not_create_decoding_errors() -> None:
    """Проверить, что этап 4 оценивает нагрузку, но не генерирует исходы ошибок."""
    estimate = DecodingOperatorModel().estimate_plan(_plan(_step(0), _step(1)))
    payload = estimate.to_dict()

    assert payload["metadata"]["generates_decoding_errors"] is False
    for step in payload["step_estimates"]:
        assert "error_occurred" not in step
        assert "actual_value" not in step
        assert "decoded_value" not in step


def test_stage4_unresolved_token_increases_ambiguity_without_resolution() -> None:
    """Проверить нагрузку неизвестного токена без автоматического исправления."""
    estimate = DecodingOperatorModel().estimate_plan(
        _plan(_step(0, recognized=True), _step(1, recognized=False))
    )

    recognized, unresolved = estimate.step_estimates
    assert recognized.ambiguity_pressure == 0.0
    assert unresolved.ambiguity_pressure > 0.0
    assert unresolved.estimated_time > recognized.estimated_time
    assert unresolved.attention < recognized.attention


def test_stage4_conditions_adjust_time_and_attention() -> None:
    """Проверить детерминированное влияние внешних условий U_d."""
    operator_estimate = DecodingOperatorModel().estimate_plan(
        _plan(_step(0), _step(1, reference_required=True))
    )
    conditions = DecodingConditionModel().estimate_plan(operator_estimate)

    assert conditions.condition_id == "UD_001"
    assert conditions.metadata["total_adjusted_time"] > conditions.metadata[
        "baseline_total_time"
    ]
    assert conditions.metadata["mean_adjusted_attention"] < operator_estimate.metadata[
        "mean_attention"
    ]
    assert conditions.metadata["generates_decoding_errors"] is False


def test_stage4_time_limit_produces_time_pressure() -> None:
    """Проверить давление времени при отдельном лимите декодирования."""
    profile = DecodingConditionProfile(time_limit_seconds=0.5)
    model = DecodingConditionModel(DecodingConditionConfig(profile=profile))
    operator_estimate = DecodingOperatorModel().estimate_plan(
        _plan(_step(0), _step(1))
    )

    result = model.estimate_plan(operator_estimate)

    assert result.metadata["time_pressure"] > 0.0
    assert result.metadata["total_adjusted_time"] > result.metadata[
        "baseline_total_time"
    ]


def test_stage4_context_uses_only_decoding_plan() -> None:
    """Проверить границы контекста до этапа генерации ошибок и M'."""
    context = DecodingExecutionContextModel().estimate_plan(
        _plan(_step(0), _step(1))
    )

    assert context.metadata["uses_only_decoding_plan"] is True
    assert context.metadata["uses_source_value_for_decoding"] is False
    assert context.metadata["generates_decoding_errors"] is False
    assert context.metadata["creates_decoded_message"] is False


def test_stage4_rejects_invalid_profiles() -> None:
    """Проверить валидацию профилей декодирующего оператора и условий."""
    with pytest.raises(ValueError, match="Минимальное внимание"):
        DecodingOperatorConfig(
            profile=DecodingOperatorProfile(
                base_attention=0.4,
                min_attention=0.7,
            )
        ).validate()

    with pytest.raises(ValueError, match="Лимит времени"):
        DecodingConditionConfig(
            profile=DecodingConditionProfile(time_limit_seconds=0.0)
        ).validate()


def test_stage4_cli_writes_context_only_inside_extension() -> None:
    """Проверить создание JSON-контекста в изолированной папке."""
    output_relative = "reports/stage4/pytest_decoding_context.json"
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
            "--estimate-decoding-context",
            "--message-id",
            "M_PYTEST_STAGE4",
            "--output",
            output_relative,
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["encoded_message"]["source_message_id"] == "M_PYTEST_STAGE4"
    assert payload["decoding_operator_estimate"]["operator_id"] == "OD_001"
    assert payload["decoding_condition_estimate"]["condition_id"] == "UD_001"
    assert payload["decoding_execution_context"]["generates_decoding_errors"] is False
    assert payload["decoding_execution_context"]["creates_decoded_message"] is False
    output_path.unlink()
