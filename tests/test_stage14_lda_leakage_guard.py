"""Тесты защиты LDA_prior от утечки фактических данных."""

from pathlib import Path

import pytest

from manual_coding_sim.lda.leakage_guard import LeakageGuard, LdaLeakageError


def test_leakage_guard_allows_prior_features_only() -> None:
    """Априорные признаки без целевых колонок должны проходить проверку."""

    guard = LeakageGuard()
    result = guard.validate_prior_input(
        source_paths=[Path("data/processed/prior_features.csv")],
        columns=[
            "scenario_id",
            "message_length_level",
            "procedure_complexity_level",
            "operator_training_level",
        ],
    )

    assert result.is_safe
    assert result.forbidden_sources == ()
    assert result.forbidden_columns == ()


def test_leakage_guard_rejects_fact_features_source() -> None:
    """Файл fact_features.csv запрещен для основной LDA_prior."""

    guard = LeakageGuard()

    with pytest.raises(LdaLeakageError, match="fact_features.csv"):
        guard.validate_prior_sources(["data/processed/fact_features.csv"])


def test_leakage_guard_rejects_quality_targets_source() -> None:
    """Файл quality_targets.csv запрещен для основной LDA_prior."""

    guard = LeakageGuard()

    with pytest.raises(LdaLeakageError, match="quality_targets.csv"):
        guard.validate_prior_sources(["data/processed/quality_targets.csv"])


def test_leakage_guard_rejects_all_features_source() -> None:
    """Объединенный файл all_features.csv нельзя подавать в LDA_prior напрямую."""

    guard = LeakageGuard()

    with pytest.raises(LdaLeakageError, match="all_features.csv"):
        guard.validate_prior_sources(["data/processed/all_features.csv"])


def test_leakage_guard_rejects_quality_columns() -> None:
    """Частные и интегральные показатели качества запрещены как признаки LDA_prior."""

    guard = LeakageGuard()

    with pytest.raises(LdaLeakageError, match="integral_quality"):
        guard.validate_prior_columns(["message_length", "integral_quality"])

    with pytest.raises(LdaLeakageError, match="q_acc"):
        guard.validate_prior_columns(["message_length", "q_acc"])


def test_leakage_guard_rejects_fact_and_target_prefixes() -> None:
    """Колонки с фактическими и целевыми префиксами запрещены."""

    guard = LeakageGuard()

    with pytest.raises(LdaLeakageError, match="fact_error_count"):
        guard.validate_prior_columns(["message_length", "fact_error_count"])

    with pytest.raises(LdaLeakageError, match="target_quality_class"):
        guard.validate_prior_columns(["message_length", "target_quality_class"])

    with pytest.raises(LdaLeakageError, match="quality_group"):
        guard.validate_prior_columns(["message_length", "quality_group"])


def test_leakage_check_result_reports_all_detected_violations() -> None:
    """Нестрогая проверка должна возвращать полный перечень нарушений."""

    guard = LeakageGuard()
    result = guard.check_prior_input(
        source_paths=[
            "data/processed/prior_features.csv",
            "data/processed/fact_features.csv",
            "data/processed/quality_targets.csv",
        ],
        columns=[
            "message_length",
            "fact_error_count",
            "q_time",
            "integral_quality",
        ],
    )

    assert not result.is_safe
    assert result.forbidden_sources == ("fact_features.csv", "quality_targets.csv")
    assert result.forbidden_columns == (
        "fact_error_count",
        "integral_quality",
        "q_time",
    )
    assert result.to_dict()["is_safe"] is False
