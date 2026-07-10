"""Тесты конфигурационного каркаса главы 5."""

from pathlib import Path

import pytest

from manual_coding_sim.prediction import Chapter5PredictionConfig
from manual_coding_sim.prediction.paths import (
    DEFAULT_CHAPTER5_REPORTS_DIR,
    DEFAULT_PRIOR_FEATURES_PATH,
    DEFAULT_THETA_PRIOR_PATH,
    resolve_project_path,
)


def test_chapter5_config_accepts_defaults() -> None:
    """Конфигурация главы 5 по умолчанию должна быть валидной."""

    config = Chapter5PredictionConfig()

    config.validate()

    assert config.expected_topic_count == 3
    assert config.inputs.prior_features_path == DEFAULT_PRIOR_FEATURES_PATH
    assert config.inputs.theta_prior_path == DEFAULT_THETA_PRIOR_PATH
    assert config.outputs.reports_dir == DEFAULT_CHAPTER5_REPORTS_DIR


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
