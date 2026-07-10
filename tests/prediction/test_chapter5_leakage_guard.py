"""Тесты защиты главы 5 от утечки признаков."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from manual_coding_sim.prediction import Chapter5LeakageError, Chapter5LeakageGuard


def test_leakage_guard_accepts_prior_columns() -> None:
    """Априорные признаки не должны считаться методической утечкой."""

    guard = Chapter5LeakageGuard()
    result = guard.check_columns(
        [
            "scenario_id",
            "protocol_id",
            "run_id",
            "alternative_id",
            "prior_total_nominal_time",
            "prior_operator_attention",
        ]
    )

    assert result.is_safe is True
    assert result.forbidden_columns == ()
    assert result.checked_column_count == 6
    assert result.prior_columns == (
        "prior_total_nominal_time",
        "prior_operator_attention",
    )
    assert "alternative_id" in result.service_columns


def test_leakage_guard_rejects_fact_and_target_columns() -> None:
    """Фактические и целевые колонки должны блокироваться."""

    guard = Chapter5LeakageGuard()
    result = guard.check_columns(
        [
            "scenario_id",
            "fact_error_count",
            "actual_time_total",
            "target_quality",
            "q_acc",
            "integral_quality",
        ]
    )

    assert result.is_safe is False
    assert "fact_error_count" in result.forbidden_columns
    assert "actual_time_total" in result.forbidden_columns
    assert "target_quality" in result.forbidden_columns
    assert "q_acc" in result.forbidden_columns
    assert "integral_quality" in result.forbidden_columns


def test_leakage_guard_rejects_case_insensitive_columns() -> None:
    """Проверка должна блокировать запрещенные признаки без учета регистра."""

    guard = Chapter5LeakageGuard()
    result = guard.check_columns(["Scenario_ID", "Fact_Error_Count", "TARGET_quality"])

    assert result.is_safe is False
    assert result.forbidden_columns == ("Fact_Error_Count", "TARGET_quality")


def test_leakage_guard_reports_non_prior_columns_without_blocking() -> None:
    """Нестандартные нецелевые колонки фиксируются в отчете, но не являются утечкой."""

    guard = Chapter5LeakageGuard()
    result = guard.check_columns(["scenario_id", "comment", "prior_message_length"])

    assert result.is_safe is True
    assert result.non_prior_columns == ("comment",)


def test_leakage_guard_raises_russian_error() -> None:
    """При утечке должно формироваться русскоязычное сообщение об ошибке."""

    guard = Chapter5LeakageGuard()

    with pytest.raises(Chapter5LeakageError, match="методическая утечка"):
        guard.require_safe_columns(["scenario_id", "target_quality"])


def test_leakage_guard_checks_dataframe() -> None:
    """Проверка должна работать с pandas DataFrame."""

    guard = Chapter5LeakageGuard()
    df = pd.DataFrame(
        {
            "scenario_id": ["scn_0001"],
            "prior_operator_attention": [0.8],
            "fact_success": [1],
        }
    )

    with pytest.raises(Chapter5LeakageError, match="fact_success"):
        guard.require_safe_dataframe(df, source_name="prior_features.csv")


def test_leakage_guard_saves_json_report(tmp_path: Path) -> None:
    """Отчет проверки утечки должен сохраняться в JSON-файл."""

    guard = Chapter5LeakageGuard()
    result = guard.check_columns(
        ["scenario_id", "prior_operator_attention"],
        source_name="prior_features.csv",
    )
    report_path = tmp_path / "chapter5_leakage_report.json"

    guard.save_json_report(report_path, result)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["is_safe"] is True
    assert payload["source_name"] == "prior_features.csv"
    assert payload["prior_columns"] == ["prior_operator_attention"]
