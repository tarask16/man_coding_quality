"""Тесты расчета интегрального прогнозного показателя главы 5."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from manual_coding_sim.prediction import (
    Chapter5QualityWeights,
    IntegralQualityPredictionError,
    IntegralQualityPredictor,
)


def _quality_weights() -> Chapter5QualityWeights:
    """Создать равномерные веса частных критериев."""

    return Chapter5QualityWeights(
        weights={
            "q_acc": 1.0 / 6.0,
            "q_time": 1.0 / 6.0,
            "q_effort": 1.0 / 6.0,
            "q_res": 1.0 / 6.0,
            "q_rep": 1.0 / 6.0,
            "q_fit": 1.0 / 6.0,
        }
    )


def _components() -> pd.DataFrame:
    """Создать минимальную таблицу частных прогнозных критериев."""

    return pd.DataFrame(
        {
            "scenario_id": ["scn_1", "scn_2"],
            "protocol_id": ["prt_1", "prt_2"],
            "run_id": ["run_1", "run_1"],
            "alternative_id": ["alt_1", "alt_2"],
            "q_latent": [0.8, 0.2],
            "q_acc_pred": [0.9, 0.3],
            "q_time_pred": [0.8, 0.4],
            "q_effort_pred": [0.7, 0.5],
            "q_res_pred": [0.6, 0.6],
            "q_rep_pred": [0.5, 0.7],
            "q_fit_pred": [0.4, 0.8],
        }
    )


def test_integral_quality_predictor_calculates_q_pred() -> None:
    """Калькулятор должен агрегировать шесть частных критериев."""

    predictor = IntegralQualityPredictor(_quality_weights())

    result = predictor.predict(_components())

    assert result.q_pred.shape[0] == 2
    assert result.q_pred.loc[0, "q_pred"] == pytest.approx(0.65)
    assert result.q_pred.loc[1, "q_pred"] == pytest.approx(0.55)
    assert result.report.row_count == 2
    assert result.report.q_pred_min == pytest.approx(0.55)
    assert result.report.q_pred_max == pytest.approx(0.65)
    assert len(result.report.criterion_reports) == 6


def test_integral_quality_predictor_rejects_missing_prediction_column() -> None:
    """Отсутствие частного критерия должно блокировать интегральный расчет."""

    components = _components().drop(columns=["q_fit_pred"])
    predictor = IntegralQualityPredictor(_quality_weights())

    with pytest.raises(IntegralQualityPredictionError, match="q_fit_pred"):
        predictor.predict(components)


def test_integral_quality_predictor_rejects_out_of_range_values() -> None:
    """Частные прогнозы вне диапазона [0, 1] должны быть запрещены."""

    components = _components()
    components.loc[0, "q_acc_pred"] = 1.2
    predictor = IntegralQualityPredictor(_quality_weights())

    with pytest.raises(IntegralQualityPredictionError, match="q_acc_pred"):
        predictor.predict(components)


def test_integral_quality_predictor_rejects_duplicate_keys() -> None:
    """Дубли ключей сценариев должны быть запрещены."""

    components = _components()
    components.loc[1, "scenario_id"] = "scn_1"
    components.loc[1, "protocol_id"] = "prt_1"
    predictor = IntegralQualityPredictor(_quality_weights())

    with pytest.raises(IntegralQualityPredictionError, match="дубли ключей"):
        predictor.predict(components)


def test_integral_quality_predictor_saves_outputs(tmp_path: Path) -> None:
    """Калькулятор должен сохранять CSV и JSON-отчет."""

    predictor = IntegralQualityPredictor(_quality_weights())
    result = predictor.predict(_components())
    q_pred_path = tmp_path / "q_pred.csv"
    report_path = tmp_path / "q_pred_report.json"

    predictor.save_outputs(result, q_pred_path=q_pred_path, report_path=report_path)

    assert q_pred_path.exists()
    assert report_path.exists()
    saved = pd.read_csv(q_pred_path)
    assert saved.loc[0, "q_pred"] == pytest.approx(0.65)
