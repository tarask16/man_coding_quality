"""Тесты нормировки априорных признаков главы 5."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from manual_coding_sim.prediction import (
    PriorFeatureNormalizationError,
    PriorFeatureNormalizer,
)
from manual_coding_sim.prediction.chapter5_config import Chapter5PriorFeatureDictionary


def test_normalizer_direct_and_inverse_normalization() -> None:
    """Прямая и обратная нормировка должны давать значения в диапазоне [0, 1]."""

    df = pd.DataFrame(
        {
            "scenario_id": ["s1", "s2", "s3"],
            "protocol_id": ["p1", "p2", "p3"],
            "prior_risk": [0.0, 0.5, 1.0],
            "prior_skill": [1.0, 2.0, 3.0],
        }
    )
    dictionary = Chapter5PriorFeatureDictionary(
        features={
            "prior_risk": {"direction": "lower_is_better", "role": "prior"},
            "prior_skill": {"direction": "higher_is_better", "role": "prior"},
        }
    )

    result = PriorFeatureNormalizer(dictionary).normalize(df)

    assert result.normalized_features["prior_risk_norm"].tolist() == [1.0, 0.5, 0.0]
    assert result.normalized_features["prior_skill_norm"].tolist() == [0.0, 0.5, 1.0]
    assert result.report.normalized_feature_count == 2
    assert result.report.skipped_feature_count == 0


def test_normalizer_skips_non_numeric_prior_features() -> None:
    """Нечисловые априорные признаки должны фиксироваться в отчете и пропускаться."""

    df = pd.DataFrame(
        {
            "scenario_id": ["s1", "s2"],
            "protocol_id": ["p1", "p2"],
            "prior_profile": ["normal", "hard"],
            "prior_time": [10.0, 20.0],
        }
    )
    dictionary = {
        "prior_profile": {"direction": "neutral", "role": "prior"},
        "prior_time": {"direction": "lower_is_better", "role": "prior"},
    }

    result = PriorFeatureNormalizer(dictionary).normalize(df)

    assert "prior_profile_norm" not in result.normalized_features.columns
    assert "prior_profile" in result.report.non_numeric_features
    assert "prior_time_norm" in result.normalized_features.columns


def test_normalizer_fills_constant_features() -> None:
    """Константные признаки должны получать фиксированное нормированное значение."""

    df = pd.DataFrame(
        {
            "scenario_id": ["s1", "s2"],
            "protocol_id": ["p1", "p2"],
            "prior_constant": [5.0, 5.0],
        }
    )
    dictionary = {
        "prior_constant": {"direction": "lower_is_better", "role": "prior"},
    }

    result = PriorFeatureNormalizer(dictionary).normalize(df)

    assert result.normalized_features["prior_constant_norm"].tolist() == [1.0, 1.0]
    assert result.report.constant_features == ("prior_constant",)


def test_normalizer_reports_unknown_input_features() -> None:
    """Признак, отсутствующий в словаре, должен попадать в отчет как неизвестный."""

    df = pd.DataFrame(
        {
            "scenario_id": ["s1", "s2"],
            "protocol_id": ["p1", "p2"],
            "prior_known": [1.0, 2.0],
            "prior_unknown": [3.0, 4.0],
        }
    )
    dictionary = {
        "prior_known": {"direction": "higher_is_better", "role": "prior"},
    }

    result = PriorFeatureNormalizer(dictionary).normalize(df)

    assert "prior_unknown" in result.report.unknown_input_features
    assert "prior_unknown_norm" not in result.normalized_features.columns


def test_normalizer_rejects_missing_protocol_id() -> None:
    """Без protocol_id нормировка входов главы 5 должна прерываться."""

    df = pd.DataFrame({"scenario_id": ["s1"], "prior_value": [1.0]})
    dictionary = {"prior_value": {"direction": "higher_is_better", "role": "prior"}}

    with pytest.raises(PriorFeatureNormalizationError, match="protocol_id"):
        PriorFeatureNormalizer(dictionary).normalize(df)


def test_normalizer_saves_csv_and_json_report(tmp_path: Path) -> None:
    """Нормировщик должен сохранять таблицу и JSON-отчет."""

    df = pd.DataFrame(
        {
            "scenario_id": ["s1", "s2"],
            "protocol_id": ["p1", "p2"],
            "prior_value": [1.0, 2.0],
        }
    )
    dictionary = {"prior_value": {"direction": "higher_is_better", "role": "prior"}}
    normalizer = PriorFeatureNormalizer(dictionary)
    result = normalizer.normalize(df)
    csv_path = tmp_path / "normalized_prior_features.csv"
    report_path = tmp_path / "normalization_report.json"

    normalizer.save_outputs(
        result,
        normalized_features_path=csv_path,
        report_path=report_path,
    )

    saved_df = pd.read_csv(csv_path)
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_df.shape == (2, 3)
    assert saved_report["normalized_feature_count"] == 1
