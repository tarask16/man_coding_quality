"""Тесты конфигурационного слоя главы 5."""

from pathlib import Path

import pytest

from manual_coding_sim.prediction import (
    Chapter5DecisionThresholds,
    Chapter5FactorDirections,
    Chapter5PredictionConfig,
    Chapter5QualityWeights,
    Chapter5UncertaintyConfig,
    load_chapter5_prediction_config,
)
from manual_coding_sim.prediction.paths import (
    DEFAULT_CHAPTER5_REPORTS_DIR,
    DEFAULT_PRIOR_FEATURES_PATH,
    DEFAULT_THETA_PRIOR_PATH,
    resolve_project_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_chapter5_config_accepts_defaults() -> None:
    """Конфигурация главы 5 по умолчанию должна быть валидной при заполненных секциях."""

    loaded_config = load_chapter5_prediction_config(project_root=PROJECT_ROOT)

    loaded_config.validate()

    assert loaded_config.expected_topic_count == 3
    assert loaded_config.inputs.prior_features_path == DEFAULT_PRIOR_FEATURES_PATH
    assert loaded_config.inputs.theta_prior_path == DEFAULT_THETA_PRIOR_PATH
    assert loaded_config.outputs.reports_dir == DEFAULT_CHAPTER5_REPORTS_DIR


def test_chapter5_config_rejects_invalid_topic_count() -> None:
    """Некорректное число латентных факторов должно давать русское сообщение."""

    config = Chapter5PredictionConfig(expected_topic_count=1)

    with pytest.raises(ValueError, match="Число латентных факторов"):
        config.validate()


def test_resolve_project_path_uses_project_root() -> None:
    """Относительный путь должен разрешаться от корня проекта."""

    project_root = Path("/project")
    relative_path = Path("reports/chapter5/q_pred.csv")

    assert resolve_project_path(project_root, relative_path) == project_root / relative_path


def test_loaded_quality_weights_are_normalized() -> None:
    """Веса частных критериев из YAML должны суммироваться в единицу."""

    config = load_chapter5_prediction_config(project_root=PROJECT_ROOT)

    assert sum(config.quality_weights.weights.values()) == pytest.approx(1.0)
    assert set(config.quality_weights.weights) == {
        "q_acc",
        "q_time",
        "q_effort",
        "q_res",
        "q_rep",
        "q_fit",
    }


def test_loaded_factor_directions_cover_all_theta_columns() -> None:
    """Направления факторов должны быть заданы для theta_0, theta_1 и theta_2."""

    config = load_chapter5_prediction_config(project_root=PROJECT_ROOT)

    assert config.factor_directions.directions == {
        "theta_0": -1.0,
        "theta_1": -1.0,
        "theta_2": 1.0,
    }


def test_loaded_uncertainty_weights_are_normalized() -> None:
    """Веса источников неопределенности должны суммироваться в единицу."""

    config = load_chapter5_prediction_config(project_root=PROJECT_ROOT)

    assert sum(config.uncertainty.weights.values()) == pytest.approx(1.0)
    assert config.uncertainty.mean_stability == pytest.approx(0.84885)


def test_loaded_feature_weights_are_valid() -> None:
    """Каждый частный критерий должен иметь нормированные веса признаков."""

    config = load_chapter5_prediction_config(project_root=PROJECT_ROOT)

    for criterion_name, criterion_weights in config.feature_weights.criteria.items():
        assert 0 <= criterion_weights.observed_weight <= 1, criterion_name
        assert sum(criterion_weights.features.values()) == pytest.approx(1.0)


def test_prior_feature_dictionary_contains_only_prior_features() -> None:
    """Словарь признаков должен содержать только априорные признаки."""

    config = load_chapter5_prediction_config(project_root=PROJECT_ROOT)

    assert "prior_total_nominal_time" in config.prior_feature_dictionary.features
    assert all(
        feature_name.startswith("prior_")
        for feature_name in config.prior_feature_dictionary.features
    )


def test_quality_weights_reject_missing_criterion() -> None:
    """Отсутствие обязательного критерия должно давать русское сообщение."""

    weights = Chapter5QualityWeights(weights={"q_acc": 1.0})

    with pytest.raises(ValueError, match="отсутствуют обязательные ключи"):
        weights.validate()


def test_factor_directions_reject_missing_theta() -> None:
    """Отсутствие направления theta должно обнаруживаться на этапе конфигурации."""

    directions = Chapter5FactorDirections(directions={"theta_0": -1.0, "theta_1": -1.0})

    with pytest.raises(ValueError, match="theta_2"):
        directions.validate()


def test_uncertainty_rejects_wrong_weight_sum() -> None:
    """Некорректная сумма весов неопределенности должна быть запрещена."""

    config = Chapter5UncertaintyConfig(
        weights={"theta_entropy": 0.5, "lda_stability": 0.5, "input_quality": 0.5}
    )

    with pytest.raises(ValueError, match="Сумма весов неопределенности"):
        config.validate()


def test_decision_thresholds_reject_wrong_order() -> None:
    """Порог low_max должен быть меньше high_min."""

    thresholds = Chapter5DecisionThresholds(low_max=0.8, high_min=0.7)

    with pytest.raises(ValueError, match="Пороги качества"):
        thresholds.validate()
