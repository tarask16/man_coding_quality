"""Тесты этапа 10: расчет частных показателей качества q(A)."""

from __future__ import annotations

import pytest

from manual_coding_sim import FeatureExtractor, ProtocolSimulator
from manual_coding_sim.quality_calculator import (
    QualityAssessment,
    QualityCalculator,
    QualityCalculatorConfig,
    quality_assessments_to_rows,
    quality_vector_to_dict,
    summarize_quality_assessment,
)
from manual_coding_sim.types import FeatureGroup, QualityVector


def _make_feature_group() -> FeatureGroup:
    """Формирует FeatureGroup из полного результата моделирования."""
    result = ProtocolSimulator().simulate_once(message_id="M_QUALITY")
    return FeatureExtractor().extract(result)


def _manual_feature_group(residual_error_rate: float) -> FeatureGroup:
    """Создает управляемую группу признаков для проверки чувствительности q(A)."""
    return FeatureGroup(
        scenario_id="A_MANUAL",
        prior_features={
            "prior_step_count": 10.0,
            "prior_total_nominal_time": 10.0,
            "prior_operator_total_effort": 10.0,
            "prior_condition_total_adjusted_time": 12.0,
            "prior_condition_mean_adjusted_attention": 0.80,
            "prior_condition_mean_stability_index": 0.75,
            "prior_condition_time_pressure": 0.10,
        },
        fact_features={
            "fact_residual_error_count": residual_error_rate * 10.0,
            "fact_residual_error_rate": residual_error_rate,
            "fact_detection_rate": 0.80,
            "fact_correction_rate": 0.70,
        },
        diagnostic_features={"diag_mean_error_probability": 0.10},
    )


def test_quality_calculator_imports() -> None:
    """Проверяет импортируемость калькулятора качества."""
    calculator = QualityCalculator()

    assert isinstance(calculator.config, QualityCalculatorConfig)


def test_calculate_returns_quality_assessment() -> None:
    """Проверяет получение QualityAssessment для FeatureGroup."""
    assessment = QualityCalculator().calculate(_make_feature_group())

    assert isinstance(assessment, QualityAssessment)
    assert assessment.scenario_id == "A_001"
    assert isinstance(assessment.quality_vector, QualityVector)


def test_quality_indicators_are_unit_interval() -> None:
    """Проверяет диапазон всех частных показателей q(A)."""
    assessment = QualityCalculator().calculate(_make_feature_group())
    values = quality_vector_to_dict(assessment.quality_vector)

    assert all(0.0 <= value <= 1.0 for value in values.values())
    assert 0.0 <= assessment.integral_quality <= 1.0


def test_quality_vector_to_dict_contains_all_indicators() -> None:
    """Проверяет преобразование QualityVector в словарь."""
    assessment = QualityCalculator().calculate(_make_feature_group())
    row = quality_vector_to_dict(assessment.quality_vector)

    assert set(row) == {"q_acc", "q_time", "q_effort", "q_res", "q_rep", "q_fit"}


def test_summary_contains_quality_and_metadata() -> None:
    """Проверяет отчетную сводку по q(A)."""
    assessment = QualityCalculator().calculate(_make_feature_group())
    summary = summarize_quality_assessment(assessment)

    assert summary["scenario_id"] == "A_001"
    assert "q_acc" in summary
    assert "integral_quality" in summary
    assert summary["step_count"] > 0


def test_batch_calculation_returns_requested_count() -> None:
    """Проверяет пакетный расчет q(A)."""
    results = ProtocolSimulator().simulate_batch(3)
    feature_groups = FeatureExtractor().extract_batch(results)
    assessments = QualityCalculator().calculate_batch(feature_groups)

    assert len(assessments) == 3
    assert all(item.scenario_id == "A_001" for item in assessments)


def test_quality_assessments_to_rows() -> None:
    """Проверяет преобразование набора оценок качества в строки таблицы."""
    feature_groups = FeatureExtractor().extract_batch(
        ProtocolSimulator().simulate_batch(2),
    )
    assessments = QualityCalculator().calculate_batch(feature_groups)
    rows = quality_assessments_to_rows(assessments)

    assert len(rows) == 2
    assert all("q_fit" in row for row in rows)
    assert all("integral_quality" in row for row in rows)


def test_accuracy_decreases_when_residual_errors_increase() -> None:
    """Проверяет чувствительность q_acc к остаточным ошибкам."""
    calculator = QualityCalculator()
    low_error = calculator.calculate(_manual_feature_group(0.10))
    high_error = calculator.calculate(_manual_feature_group(0.60))

    assert low_error.quality_vector.q_acc > high_error.quality_vector.q_acc
    assert low_error.integral_quality > high_error.integral_quality


def test_invalid_config_is_rejected() -> None:
    """Проверяет отклонение некорректной конфигурации качества."""
    with pytest.raises(ValueError):
        QualityCalculatorConfig(max_effort_per_step=0.0).validate()

    with pytest.raises(ValueError):
        QualityCalculatorConfig(
            indicator_weights={
                "q_acc": 1.0,
                "q_time": 1.0,
                "q_effort": 1.0,
                "q_res": 1.0,
                "q_rep": 1.0,
                "q_fit": -1.0,
            },
        ).validate()


def test_missing_required_features_are_rejected() -> None:
    """Проверяет отклонение FeatureGroup без необходимых признаков."""
    broken_group = FeatureGroup(
        scenario_id="A_BAD",
        prior_features={"prior_step_count": 10.0},
        fact_features={"fact_residual_error_rate": 0.1},
        diagnostic_features={"diag_value": 1.0},
    )

    with pytest.raises(ValueError):
        QualityCalculator().calculate(broken_group)
