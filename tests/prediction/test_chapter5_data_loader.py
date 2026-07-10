"""Тесты каркаса загрузчика входных данных главы 5."""

import pytest

from manual_coding_sim.prediction import Chapter5DataLoader


def test_data_loader_describes_expected_inputs() -> None:
    """Загрузчик должен описывать обязательные входные файлы главы 5."""

    loader = Chapter5DataLoader()
    contract = loader.describe_expected_inputs()

    assert contract.prior_features_path.as_posix() == "data/processed/prior_features.csv"
    assert contract.theta_prior_path.as_posix() == "reports/chapter4/theta_prior.csv"
    assert (
        contract.topic_interpretation_path.as_posix()
        == "reports/chapter4/topic_interpretation.json"
    )


def test_data_loader_load_is_not_implemented_on_stage1() -> None:
    """Фактическая загрузка данных не должна быть скрыто реализована на этапе 1."""

    loader = Chapter5DataLoader()

    with pytest.raises(NotImplementedError, match="этапе 3"):
        loader.load()
