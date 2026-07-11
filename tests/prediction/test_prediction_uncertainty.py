"""Тесты расчета неопределенности прогноза главы 5."""

import pandas as pd
import pytest

from manual_coding_sim.prediction.chapter5_config import Chapter5UncertaintyConfig
from manual_coding_sim.prediction.prediction_uncertainty import (
    PredictionUncertaintyError,
    PredictionUncertaintyEstimator,
)


def _q_pred() -> pd.DataFrame:
    """Создать минимальную таблицу интегрального прогноза."""

    return pd.DataFrame(
        {
            "scenario_id": ["S1", "S2"],
            "protocol_id": ["P1", "P2"],
            "run_id": ["R", "R"],
            "q_pred": [0.25, 0.75],
        }
    )


def _theta_prior() -> pd.DataFrame:
    """Создать минимальную таблицу латентных профилей."""

    return pd.DataFrame(
        {
            "scenario_id": ["S1", "S2"],
            "protocol_id": ["P1", "P2"],
            "theta_0": [0.8, 1.0 / 3.0],
            "theta_1": [0.1, 1.0 / 3.0],
            "theta_2": [0.1, 1.0 / 3.0],
        }
    )


def _normalized_features() -> pd.DataFrame:
    """Создать минимальную таблицу нормированных априорных признаков."""

    return pd.DataFrame(
        {
            "scenario_id": ["S1", "S2"],
            "protocol_id": ["P1", "P2"],
            "prior_a_norm": [0.1, 0.9],
            "prior_b_norm": [0.2, 0.8],
        }
    )


def test_prediction_uncertainty_estimator_builds_intervals() -> None:
    """Оцениватель должен формировать интервалы вокруг q_pred."""

    estimator = PredictionUncertaintyEstimator(Chapter5UncertaintyConfig())
    result = estimator.estimate(_q_pred(), _theta_prior(), _normalized_features())

    assert result.report.row_count == 2
    assert result.report.weight_sum == pytest.approx(1.0)
    assert result.uncertainty["uncertainty_score"].between(0.0, 1.0).all()
    assert result.uncertainty["q_pred_lower"].between(0.0, 1.0).all()
    assert result.uncertainty["q_pred_upper"].between(0.0, 1.0).all()
    assert (result.uncertainty["q_pred_lower"] <= result.uncertainty["q_pred"]).all()
    assert (result.uncertainty["q_pred"] <= result.uncertainty["q_pred_upper"]).all()


def test_prediction_uncertainty_entropy_is_higher_for_uniform_theta() -> None:
    """Равномерный theta-профиль должен иметь большую энтропию."""

    estimator = PredictionUncertaintyEstimator(Chapter5UncertaintyConfig())
    result = estimator.estimate(_q_pred(), _theta_prior(), _normalized_features())

    entropy_by_scenario = dict(
        zip(result.uncertainty["scenario_id"], result.uncertainty["theta_entropy"], strict=True)
    )
    assert entropy_by_scenario["S2"] > entropy_by_scenario["S1"]
    assert entropy_by_scenario["S2"] == pytest.approx(1.0)


def test_prediction_uncertainty_rejects_invalid_q_pred_range() -> None:
    """Значения q_pred вне [0, 1] должны приводить к ошибке."""

    q_pred = _q_pred()
    q_pred.loc[0, "q_pred"] = 1.2
    estimator = PredictionUncertaintyEstimator(Chapter5UncertaintyConfig())

    with pytest.raises(PredictionUncertaintyError, match="q_pred"):
        estimator.estimate(q_pred, _theta_prior(), _normalized_features())


def test_prediction_uncertainty_rejects_duplicate_keys() -> None:
    """Дубли ключей сценария должны запрещаться."""

    theta_prior = pd.concat([_theta_prior(), _theta_prior().iloc[[0]]], ignore_index=True)
    estimator = PredictionUncertaintyEstimator(Chapter5UncertaintyConfig())

    with pytest.raises(PredictionUncertaintyError, match="дубли"):
        estimator.estimate(_q_pred(), theta_prior, _normalized_features())
