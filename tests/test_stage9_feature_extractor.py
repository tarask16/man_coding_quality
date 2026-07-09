"""Тесты этапа 9: извлечение признаков X_prior, X_fact и X_diag."""

from __future__ import annotations

import pytest

from manual_coding_sim import FeatureExtractor, ProtocolSimulator
from manual_coding_sim.feature_extractor import (
    FeatureExtractorConfig,
    feature_group_to_flat_row,
    feature_groups_to_rows,
    summarize_feature_group,
    validate_feature_group,
)
from manual_coding_sim.types import FeatureGroup


def _make_feature_group() -> FeatureGroup:
    """Формирует FeatureGroup для тестов этапа 9."""
    result = ProtocolSimulator().simulate_once(message_id="M_FEATURE")
    return FeatureExtractor().extract(result)


def test_feature_extractor_imports() -> None:
    """Проверяет импортируемость извлекателя признаков."""
    extractor = FeatureExtractor()

    assert isinstance(extractor.config, FeatureExtractorConfig)


def test_extract_returns_feature_group() -> None:
    """Проверяет получение FeatureGroup из результата моделирования."""
    feature_group = _make_feature_group()

    assert isinstance(feature_group, FeatureGroup)
    assert feature_group.scenario_id == "A_001"
    assert feature_group.prior_features
    assert feature_group.fact_features
    assert feature_group.diagnostic_features


def test_prior_features_do_not_contain_fact_prefix() -> None:
    """Проверяет отсутствие фактических признаков в X_prior."""
    feature_group = _make_feature_group()

    assert all(
        not name.startswith("fact_")
        for name in feature_group.prior_features
    )
    assert all(name.startswith("prior_") for name in feature_group.prior_features)


def test_feature_groups_have_disjoint_names() -> None:
    """Проверяет непересечение имен X_prior, X_fact и X_diag."""
    feature_group = _make_feature_group()
    prior_keys = set(feature_group.prior_features)
    fact_keys = set(feature_group.fact_features)
    diagnostic_keys = set(feature_group.diagnostic_features)

    assert not prior_keys & fact_keys
    assert not prior_keys & diagnostic_keys
    assert not fact_keys & diagnostic_keys


def test_prior_feature_values_are_consistent() -> None:
    """Проверяет отдельные априорные признаки сценария."""
    result = ProtocolSimulator().simulate_once(message_id="M_PRIOR")
    feature_group = FeatureExtractor().extract(result)

    assert feature_group.prior_features["prior_message_length"] == len(
        result.message.elements,
    )
    assert feature_group.prior_features["prior_step_count"] == len(
        result.procedure_plan.steps,
    )
    assert 0.0 <= feature_group.prior_features[
        "prior_condition_mean_environmental_load"
    ] <= 1.0


def test_fact_features_match_protocol_metadata() -> None:
    """Проверяет соответствие X_fact фактическому протоколу."""
    result = ProtocolSimulator().simulate_once(message_id="M_FACT")
    feature_group = FeatureExtractor().extract(result)

    assert feature_group.fact_features["fact_error_count"] == float(
        result.error_protocol.metadata["error_count"],
    )
    assert feature_group.fact_features["fact_residual_error_count"] == float(
        result.control_protocol.metadata["residual_error_count"],
    )
    assert feature_group.fact_features[
        "fact_residual_error_count"
    ] <= feature_group.fact_features["fact_error_count"]


def test_diagnostic_features_include_random_seeds() -> None:
    """Проверяет диагностические признаки воспроизводимости."""
    feature_group = _make_feature_group()

    assert feature_group.diagnostic_features["diag_message_random_seed"] == 42.0
    assert feature_group.diagnostic_features["diag_error_random_seed"] == 1042.0
    assert feature_group.diagnostic_features["diag_control_random_seed"] == 2042.0


def test_extract_batch_returns_requested_count() -> None:
    """Проверяет пакетное извлечение признаков."""
    results = ProtocolSimulator().simulate_batch(3)
    feature_groups = FeatureExtractor().extract_batch(results)

    assert len(feature_groups) == 3
    assert all(group.scenario_id == "A_001" for group in feature_groups)


def test_flat_row_can_exclude_fact_and_diagnostic_features() -> None:
    """Проверяет формирование строки только с X_prior."""
    feature_group = _make_feature_group()
    row = feature_group_to_flat_row(
        feature_group,
        include_fact=False,
        include_diagnostic=False,
    )

    assert "scenario_id" in row
    assert "prior_message_length" in row
    assert all(not name.startswith("fact_") for name in row)
    assert all(not name.startswith("diag_") for name in row)


def test_feature_groups_to_rows() -> None:
    """Проверяет преобразование набора FeatureGroup в строки таблицы."""
    results = ProtocolSimulator().simulate_batch(2)
    feature_groups = FeatureExtractor().extract_batch(results)
    rows = feature_groups_to_rows(feature_groups)

    assert len(rows) == 2
    assert all("prior_message_length" in row for row in rows)
    assert all("fact_error_count" in row for row in rows)
    assert all("diag_mean_error_probability" in row for row in rows)


def test_summarize_feature_group() -> None:
    """Проверяет сводку по группам признаков."""
    feature_group = _make_feature_group()
    summary = summarize_feature_group(feature_group)

    assert summary["scenario_id"] == "A_001"
    assert summary["prior_feature_count"] == len(feature_group.prior_features)
    assert summary["fact_feature_count"] == len(feature_group.fact_features)
    assert summary["diagnostic_feature_count"] == len(
        feature_group.diagnostic_features,
    )


def test_validate_feature_group_rejects_overlap() -> None:
    """Проверяет отклонение пересекающихся имен признаков."""
    broken_group = FeatureGroup(
        scenario_id="A_BAD",
        prior_features={"same_name": 1.0},
        fact_features={"same_name": 2.0},
        diagnostic_features={"diag_value": 3.0},
    )

    with pytest.raises(ValueError):
        validate_feature_group(broken_group)


def test_invalid_config_is_rejected() -> None:
    """Проверяет отклонение некорректной конфигурации."""
    with pytest.raises(ValueError):
        FeatureExtractorConfig(round_digits=-1).validate()
