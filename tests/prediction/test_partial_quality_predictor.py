"""Тесты расчета частных прогнозных критериев главы 5."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from manual_coding_sim.prediction import (
    Chapter5FeatureCriterionWeights,
    Chapter5FeatureWeights,
    PartialQualityPredictionError,
    PartialQualityPredictor,
)


def _feature_weights() -> Chapter5FeatureWeights:
    """Создать минимальные веса признаков для всех критериев."""

    criteria = {
        criterion: Chapter5FeatureCriterionWeights(
            observed_weight=0.6,
            features={"prior_good": 0.7, "prior_bad": 0.3},
        )
        for criterion in ("q_acc", "q_time", "q_effort", "q_res", "q_rep", "q_fit")
    }
    return Chapter5FeatureWeights(criteria=criteria)


def test_partial_quality_predictor_calculates_all_criteria() -> None:
    """Калькулятор должен рассчитать шесть частных прогнозных критериев."""

    normalized = pd.DataFrame(
        {
            "scenario_id": ["scn_1", "scn_2"],
            "protocol_id": ["prt_1", "prt_2"],
            "prior_good_norm": [1.0, 0.0],
            "prior_bad_norm": [0.0, 1.0],
        }
    )
    latent = pd.DataFrame(
        {
            "scenario_id": ["scn_1", "scn_2"],
            "protocol_id": ["prt_1", "prt_2"],
            "q_latent": [0.8, 0.2],
        }
    )
    predictor = PartialQualityPredictor(_feature_weights())

    result = predictor.predict(normalized, latent)

    assert result.components.shape[0] == 2
    assert result.components.loc[0, "q_acc_feature_component"] == pytest.approx(0.7)
    assert result.components.loc[0, "q_acc_pred"] == pytest.approx(0.74)
    assert result.components.loc[1, "q_acc_pred"] == pytest.approx(0.26)
    assert result.report.row_count == 2
    assert set(result.report.criteria) == {"q_acc", "q_time", "q_effort", "q_res", "q_rep", "q_fit"}


def test_partial_quality_predictor_rejects_missing_normalized_feature() -> None:
    """Отсутствующий нормированный признак должен блокировать расчет."""

    normalized = pd.DataFrame(
        {
            "scenario_id": ["scn_1"],
            "protocol_id": ["prt_1"],
            "prior_good_norm": [1.0],
        }
    )
    latent = pd.DataFrame(
        {
            "scenario_id": ["scn_1"],
            "protocol_id": ["prt_1"],
            "q_latent": [0.8],
        }
    )
    predictor = PartialQualityPredictor(_feature_weights())

    with pytest.raises(PartialQualityPredictionError, match="prior_bad_norm"):
        predictor.predict(normalized, latent)


def test_partial_quality_predictor_rejects_duplicate_keys() -> None:
    """Дубли ключей объединения должны быть запрещены."""

    normalized = pd.DataFrame(
        {
            "scenario_id": ["scn_1", "scn_1"],
            "protocol_id": ["prt_1", "prt_1"],
            "prior_good_norm": [1.0, 1.0],
            "prior_bad_norm": [0.0, 0.0],
        }
    )
    latent = pd.DataFrame(
        {
            "scenario_id": ["scn_1"],
            "protocol_id": ["prt_1"],
            "q_latent": [0.8],
        }
    )
    predictor = PartialQualityPredictor(_feature_weights())

    with pytest.raises(PartialQualityPredictionError, match="дубли ключей"):
        predictor.predict(normalized, latent)


def test_partial_quality_predictor_saves_outputs(tmp_path: Path) -> None:
    """Калькулятор должен сохранять CSV и JSON-отчет."""

    normalized = pd.DataFrame(
        {
            "scenario_id": ["scn_1"],
            "protocol_id": ["prt_1"],
            "prior_good_norm": [1.0],
            "prior_bad_norm": [0.0],
        }
    )
    latent = pd.DataFrame(
        {
            "scenario_id": ["scn_1"],
            "protocol_id": ["prt_1"],
            "q_latent": [0.8],
        }
    )
    predictor = PartialQualityPredictor(_feature_weights())
    result = predictor.predict(normalized, latent)
    components_path = tmp_path / "q_pred_components.csv"
    report_path = tmp_path / "q_pred_components_report.json"

    predictor.save_outputs(result, components_path=components_path, report_path=report_path)

    assert components_path.exists()
    assert report_path.exists()
    saved = pd.read_csv(components_path)
    assert saved.loc[0, "q_acc_pred"] == pytest.approx(0.74)
