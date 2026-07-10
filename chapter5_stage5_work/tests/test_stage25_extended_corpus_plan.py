"""Тесты плана расширенного корпуса главы 3."""

import pytest

from manual_coding_sim.experiments.extended_corpus_plan import (
    ExtendedCorpusPlanBuilder,
    ExtendedCorpusPlanConfig,
)


def test_extended_plan_creates_required_number_of_scenarios() -> None:
    """План должен создавать заданное число сценариев и протоколов."""

    scenarios = ExtendedCorpusPlanBuilder(
        ExtendedCorpusPlanConfig(document_count=120, random_seed=11)
    ).build()

    assert len(scenarios) == 120
    assert ExtendedCorpusPlanBuilder.unique_scenario_count(scenarios) == 120
    assert ExtendedCorpusPlanBuilder.unique_protocol_count(scenarios) == 120


def test_extended_plan_is_reproducible_with_fixed_seed() -> None:
    """Фиксированный random_seed должен давать воспроизводимый план."""

    first = ExtendedCorpusPlanBuilder(
        ExtendedCorpusPlanConfig(document_count=20, random_seed=42)
    ).build()
    second = ExtendedCorpusPlanBuilder(
        ExtendedCorpusPlanConfig(document_count=20, random_seed=42)
    ).build()

    assert first == second


def test_extended_plan_covers_core_parameter_levels() -> None:
    """План должен покрывать несколько уровней ключевых параметров."""

    scenarios = ExtendedCorpusPlanBuilder(
        ExtendedCorpusPlanConfig(document_count=100, random_seed=77)
    ).build()

    assert len({item.message_complexity for item in scenarios}) >= 5
    assert len({item.message_criticality for item in scenarios}) >= 5
    assert len({item.operator_skill for item in scenarios}) >= 5
    assert len({item.condition_profile for item in scenarios}) >= 3
    assert len({item.coding_tool_type for item in scenarios}) >= 4


def test_extended_plan_supports_multiple_protocols_per_scenario() -> None:
    """План должен поддерживать несколько протоколов на один сценарий."""

    scenarios = ExtendedCorpusPlanBuilder(
        ExtendedCorpusPlanConfig(
            document_count=12,
            protocols_per_scenario=3,
            random_seed=11,
        )
    ).build()

    assert len(scenarios) == 12
    assert ExtendedCorpusPlanBuilder.unique_scenario_count(scenarios) == 4
    assert ExtendedCorpusPlanBuilder.unique_protocol_count(scenarios) == 12


def test_extended_plan_rejects_invalid_config() -> None:
    """Некорректные параметры плана должны отклоняться."""

    with pytest.raises(ValueError):
        ExtendedCorpusPlanConfig(document_count=0).validate()
    with pytest.raises(ValueError):
        ExtendedCorpusPlanConfig(document_count=1, protocols_per_scenario=2).validate()
