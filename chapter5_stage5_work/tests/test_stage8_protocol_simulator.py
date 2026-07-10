"""Тесты этапа 8: интегральный симулятор протоколов."""

from __future__ import annotations

import pytest

from manual_coding_sim import ProtocolSimulator
from manual_coding_sim.protocol_simulator import (
    ProtocolSimulatorConfig,
    SimulationResult,
    simulation_results_to_rows,
    summarize_simulation_result,
)


def test_protocol_simulator_imports() -> None:
    """Проверяет импортируемость интегрального симулятора."""
    simulator = ProtocolSimulator()

    assert isinstance(simulator.config, ProtocolSimulatorConfig)


def test_simulate_once_returns_complete_result() -> None:
    """Проверяет полный прогон от M до протокола контроля K."""
    simulator = ProtocolSimulator()
    result = simulator.simulate_once(message_id="M_TEST")

    assert isinstance(result, SimulationResult)
    assert result.message.message_id == "M_TEST"
    assert result.procedure_plan.message_id == "M_TEST"
    assert result.operator_estimate.message_id == "M_TEST"
    assert result.condition_estimate.message_id == "M_TEST"
    assert result.error_protocol.message_id == "M_TEST"
    assert result.control_protocol.message_id == "M_TEST"


def test_step_counts_are_consistent() -> None:
    """Проверяет согласованность числа шагов во всех артефактах."""
    result = ProtocolSimulator().simulate_once()
    step_count = len(result.message.elements)

    assert len(result.procedure_plan.steps) == step_count
    assert len(result.operator_estimate.step_estimates) == step_count
    assert len(result.condition_estimate.step_estimates) == step_count
    assert len(result.error_protocol.step_outcomes) == step_count
    assert len(result.control_protocol.step_outcomes) == step_count


def test_scenario_identifiers_are_preserved() -> None:
    """Проверяет сохранение компонентов сценария A = {S, O, U, G, K}."""
    config = ProtocolSimulatorConfig(scenario_id="A_TEST")
    result = ProtocolSimulator(config).simulate_once()

    assert result.scenario.scenario_id == "A_TEST"
    assert result.scenario.coding_tool_id == "S_001"
    assert result.scenario.operator_id == "O_001"
    assert result.scenario.condition_id == "U_001"
    assert result.scenario.message_class_id == "G_001"
    assert result.scenario.control_procedure_id == "K_001"


def test_reset_reproduces_first_result() -> None:
    """Проверяет воспроизводимость интегрального прогона после reset()."""
    simulator = ProtocolSimulator()
    first = summarize_simulation_result(simulator.simulate_once())
    simulator.reset()
    second = summarize_simulation_result(simulator.simulate_once())

    assert first == second


def test_equal_seeds_reproduce_protocols() -> None:
    """Проверяет воспроизводимость при одинаковой конфигурации."""
    config = ProtocolSimulatorConfig(
        message_random_seed=11,
        error_random_seed=22,
        control_random_seed=33,
    )
    first = summarize_simulation_result(ProtocolSimulator(config).simulate_once())
    second = summarize_simulation_result(ProtocolSimulator(config).simulate_once())

    assert first == second


def test_simulate_batch_returns_requested_count() -> None:
    """Проверяет пакетный запуск симулятора."""
    simulator = ProtocolSimulator()
    results = simulator.simulate_batch(3)

    assert len(results) == 3
    assert [result.message.message_id for result in results] == [
        "M_000001",
        "M_000002",
        "M_000003",
    ]


def test_simulate_batch_rejects_non_positive_count() -> None:
    """Проверяет отклонение некорректного числа прогонов."""
    simulator = ProtocolSimulator()

    with pytest.raises(ValueError):
        simulator.simulate_batch(0)


def test_summary_contains_expected_fields() -> None:
    """Проверяет контрольную сводку результата моделирования."""
    result = ProtocolSimulator().simulate_once(message_id="M_SUM")
    summary = summarize_simulation_result(result)

    assert summary["scenario_id"] == "A_001"
    assert summary["message_id"] == "M_SUM"
    assert summary["message_length"] == len(result.message.elements)
    assert summary["step_count"] == len(result.procedure_plan.steps)
    assert summary["condition_adjusted_time"] >= summary["nominal_time"]
    assert summary["residual_error_count"] <= summary["original_error_count"]


def test_result_metadata_matches_summary() -> None:
    """Проверяет согласованность метаданных результата и сводки."""
    result = ProtocolSimulator().simulate_once()
    summary = summarize_simulation_result(result)

    assert result.metadata["message_id"] == summary["message_id"]
    assert result.metadata["step_count"] == summary["step_count"]
    assert result.metadata["condition_adjusted_time"] == summary[
        "condition_adjusted_time"
    ]


def test_results_to_rows() -> None:
    """Проверяет преобразование результатов моделирования в таблицу."""
    results = ProtocolSimulator().simulate_batch(2)
    rows = simulation_results_to_rows(results)

    assert len(rows) == 2
    assert rows[0]["message_id"] == "M_000001"
    assert rows[1]["message_id"] == "M_000002"
    assert all("residual_error_count" in row for row in rows)
