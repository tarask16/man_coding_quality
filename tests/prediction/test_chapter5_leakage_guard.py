"""Тесты первичной защиты главы 5 от утечки признаков."""

import pytest

from manual_coding_sim.prediction import Chapter5LeakageError, Chapter5LeakageGuard


def test_leakage_guard_accepts_prior_columns() -> None:
    """Априорные признаки не должны считаться методической утечкой."""

    guard = Chapter5LeakageGuard()
    result = guard.check_columns(
        [
            "scenario_id",
            "protocol_id",
            "prior_total_nominal_time",
            "prior_operator_attention",
        ]
    )

    assert result.is_safe is True
    assert result.forbidden_columns == ()


def test_leakage_guard_rejects_fact_and_target_columns() -> None:
    """Фактические и целевые колонки должны блокироваться."""

    guard = Chapter5LeakageGuard()
    result = guard.check_columns(
        [
            "scenario_id",
            "fact_error_count",
            "actual_time_total",
            "q_acc",
            "integral_quality",
        ]
    )

    assert result.is_safe is False
    assert "fact_error_count" in result.forbidden_columns
    assert "actual_time_total" in result.forbidden_columns
    assert "q_acc" in result.forbidden_columns
    assert "integral_quality" in result.forbidden_columns


def test_leakage_guard_raises_russian_error() -> None:
    """При утечке должно формироваться русскоязычное сообщение об ошибке."""

    guard = Chapter5LeakageGuard()

    with pytest.raises(Chapter5LeakageError, match="методическая утечка"):
        guard.require_safe_columns(["scenario_id", "target_quality"])
