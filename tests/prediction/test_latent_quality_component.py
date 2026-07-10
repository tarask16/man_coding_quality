"""Тесты расчета латентной компоненты качества главы 5."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from manual_coding_sim.prediction import (
    LatentQualityComponentCalculator,
    LatentQualityComponentError,
)


def test_latent_component_calculates_q_latent_from_theta_profile() -> None:
    """Калькулятор должен преобразовывать theta-профиль в q_latent."""

    theta_prior = pd.DataFrame(
        {
            "run_id": ["run_001", "run_001"],
            "scenario_id": ["scn_0001", "scn_0002"],
            "protocol_id": ["prt_0001", "prt_0002"],
            "theta_0": [0.2, 0.1],
            "theta_1": [0.3, 0.2],
            "theta_2": [0.5, 0.7],
        }
    )
    calculator = LatentQualityComponentCalculator(
        {"theta_0": -1.0, "theta_1": -1.0, "theta_2": 1.0}
    )

    result = calculator.calculate(theta_prior)

    assert result.latent_quality.shape[0] == 2
    assert result.latent_quality["q_latent"].tolist() == pytest.approx([0.5, 0.7])
    assert result.latent_quality["theta_dominant_topic"].tolist() == ["theta_2", "theta_2"]
    assert result.report.q_latent_min == pytest.approx(0.5)
    assert result.report.q_latent_max == pytest.approx(0.7)


def test_latent_component_penalizes_negative_factors() -> None:
    """Высокие theta_0 и theta_1 должны снижать латентную компоненту качества."""

    theta_prior = pd.DataFrame(
        {
            "scenario_id": ["bad", "good"],
            "protocol_id": ["p_bad", "p_good"],
            "theta_0": [0.8, 0.1],
            "theta_1": [0.1, 0.1],
            "theta_2": [0.1, 0.8],
        }
    )
    calculator = LatentQualityComponentCalculator(
        {"theta_0": -1.0, "theta_1": -1.0, "theta_2": 1.0}
    )

    result = calculator.calculate(theta_prior)

    bad_value = result.latent_quality.loc[0, "q_latent"]
    good_value = result.latent_quality.loc[1, "q_latent"]
    assert bad_value < good_value
    assert bad_value == pytest.approx(0.1)
    assert good_value == pytest.approx(0.8)


def test_latent_component_rejects_wrong_theta_sum() -> None:
    """Сумма theta-компонент должна быть равна единице."""

    theta_prior = pd.DataFrame(
        {
            "scenario_id": ["scn_0001"],
            "protocol_id": ["prt_0001"],
            "theta_0": [0.2],
            "theta_1": [0.2],
            "theta_2": [0.2],
        }
    )
    calculator = LatentQualityComponentCalculator(
        {"theta_0": -1.0, "theta_1": -1.0, "theta_2": 1.0}
    )

    with pytest.raises(LatentQualityComponentError, match="Сумма theta"):
        calculator.calculate(theta_prior)


def test_latent_component_rejects_missing_direction() -> None:
    """Отсутствующее направление фактора должно блокировать расчет."""

    with pytest.raises(LatentQualityComponentError, match="theta_2"):
        LatentQualityComponentCalculator({"theta_0": -1.0, "theta_1": -1.0})


def test_latent_component_saves_csv_and_json_report(tmp_path: Path) -> None:
    """Калькулятор должен сохранять таблицу и отчет латентной компоненты."""

    theta_prior = pd.DataFrame(
        {
            "scenario_id": ["scn_0001"],
            "protocol_id": ["prt_0001"],
            "theta_0": [0.2],
            "theta_1": [0.3],
            "theta_2": [0.5],
        }
    )
    calculator = LatentQualityComponentCalculator(
        {"theta_0": -1.0, "theta_1": -1.0, "theta_2": 1.0}
    )
    result = calculator.calculate(theta_prior)
    csv_path = tmp_path / "latent_quality_component.csv"
    report_path = tmp_path / "latent_quality_component_report.json"

    calculator.save_outputs(result, latent_component_path=csv_path, report_path=report_path)

    assert csv_path.exists()
    assert report_path.exists()
    saved = pd.read_csv(csv_path)
    assert saved.loc[0, "q_latent"] == pytest.approx(0.5)
