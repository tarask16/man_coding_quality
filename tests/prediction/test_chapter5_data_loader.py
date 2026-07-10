"""Тесты загрузки и объединения входных данных главы 5."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from manual_coding_sim.prediction import (
    Chapter5DataLoadError,
    Chapter5DataLoader,
    Chapter5InputPaths,
)


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


def test_data_loader_loads_and_merges_inputs_by_scenario_id(tmp_path: Path) -> None:
    """Загрузчик должен объединять prior_features и theta_prior по scenario_id."""

    paths = _write_input_files(tmp_path, include_protocol_in_prior=False)
    loader = Chapter5DataLoader(paths=paths, project_root=tmp_path, expected_topic_count=3)

    loaded_inputs = loader.load()

    assert loaded_inputs.validation_report.scenario_count == 2
    assert loaded_inputs.validation_report.theta_row_count == 2
    assert loaded_inputs.validation_report.prior_feature_row_count == 2
    assert loaded_inputs.validation_report.merge_key_columns == ("scenario_id",)
    assert loaded_inputs.validation_report.prior_has_protocol_id is False
    assert set(loaded_inputs.merged_data["protocol_id"]) == {"prt_0001", "prt_0002"}
    assert "prior_total_nominal_time" in loaded_inputs.merged_data.columns


def test_data_loader_loads_and_merges_inputs_by_scenario_and_protocol(tmp_path: Path) -> None:
    """Если protocol_id есть в prior_features, он должен входить в ключ объединения."""

    paths = _write_input_files(tmp_path, include_protocol_in_prior=True)
    loader = Chapter5DataLoader(paths=paths, project_root=tmp_path, expected_topic_count=3)

    loaded_inputs = loader.load()

    assert loaded_inputs.validation_report.merge_key_columns == ("scenario_id", "protocol_id")
    assert loaded_inputs.validation_report.prior_has_protocol_id is True
    assert loaded_inputs.merged_data.shape[0] == 2


def test_data_loader_rejects_duplicate_prior_keys(tmp_path: Path) -> None:
    """Дубли ключей в prior_features должны блокировать неоднозначное объединение."""

    paths = _write_input_files(tmp_path, include_protocol_in_prior=False)
    prior_path = tmp_path / paths.prior_features_path
    duplicated = pd.concat([pd.read_csv(prior_path), pd.read_csv(prior_path).iloc[[0]]])
    duplicated.to_csv(prior_path, index=False)
    loader = Chapter5DataLoader(paths=paths, project_root=tmp_path, expected_topic_count=3)

    with pytest.raises(Chapter5DataLoadError, match="дубли идентификаторов"):
        loader.load()


def test_data_loader_rejects_theta_sum_mismatch(tmp_path: Path) -> None:
    """Сумма theta-компонент должна быть близка к единице."""

    paths = _write_input_files(tmp_path, include_protocol_in_prior=False)
    theta_path = tmp_path / paths.theta_prior_path
    theta_df = pd.read_csv(theta_path)
    theta_df.loc[0, "theta_2"] = 0.4
    theta_df.to_csv(theta_path, index=False)
    loader = Chapter5DataLoader(paths=paths, project_root=tmp_path, expected_topic_count=3)

    with pytest.raises(Chapter5DataLoadError, match="Сумма компонентов theta"):
        loader.load()


def test_data_loader_rejects_topic_interpretation_not_allowed(tmp_path: Path) -> None:
    """Интерпретация тем должна быть разрешена для априорного прогноза."""

    paths = _write_input_files(tmp_path, include_protocol_in_prior=False)
    topic_path = tmp_path / paths.topic_interpretation_path
    payload = json.loads(topic_path.read_text(encoding="utf-8"))
    payload["allowed_for_apriori_forecast"] = False
    topic_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    loader = Chapter5DataLoader(paths=paths, project_root=tmp_path, expected_topic_count=3)

    with pytest.raises(Chapter5DataLoadError, match="не разрешена для априорного прогноза"):
        loader.load()


def test_data_loader_rejects_merge_mismatch(tmp_path: Path) -> None:
    """Несовпадение сценариев между theta_prior и prior_features должно выявляться."""

    paths = _write_input_files(tmp_path, include_protocol_in_prior=False)
    prior_path = tmp_path / paths.prior_features_path
    prior_df = pd.read_csv(prior_path)
    prior_df.loc[1, "scenario_id"] = "scn_absent"
    prior_df.to_csv(prior_path, index=False)
    loader = Chapter5DataLoader(paths=paths, project_root=tmp_path, expected_topic_count=3)

    with pytest.raises(Chapter5DataLoadError, match="Не удалось однозначно объединить"):
        loader.load()


def _write_input_files(tmp_path: Path, *, include_protocol_in_prior: bool) -> Chapter5InputPaths:
    """Создать минимальные согласованные входные файлы главы 5 для тестов."""

    prior_path = Path("data/processed/prior_features.csv")
    theta_path = Path("reports/chapter4/theta_prior.csv")
    topic_path = Path("reports/chapter4/topic_interpretation.json")
    (tmp_path / prior_path).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / theta_path).parent.mkdir(parents=True, exist_ok=True)

    prior_payload = {
        "run_id": ["run_001", "run_001"],
        "scenario_id": ["scn_0001", "scn_0002"],
        "prior_total_nominal_time": [10.0, 20.0],
        "prior_operator_attention": [0.8, 0.6],
    }
    if include_protocol_in_prior:
        prior_payload["protocol_id"] = ["prt_0001", "prt_0002"]
    pd.DataFrame(prior_payload).to_csv(tmp_path / prior_path, index=False)

    pd.DataFrame(
        {
            "document_index": [0, 1],
            "run_id": ["run_001", "run_001"],
            "protocol_id": ["prt_0001", "prt_0002"],
            "scenario_id": ["scn_0001", "scn_0002"],
            "theta_0": [0.2, 0.1],
            "theta_1": [0.3, 0.2],
            "theta_2": [0.5, 0.7],
            "selected_k": [3, 3],
            "random_state": [11, 11],
        }
    ).to_csv(tmp_path / theta_path, index=False)

    topic_payload = {
        "allowed_for_apriori_forecast": True,
        "model_name": "LDA_prior",
        "topic_count": 3,
        "topics": [
            {"topic_id": 0, "top_tokens": ["prior_a__level_high"]},
            {"topic_id": 1, "top_tokens": ["prior_b__level_high"]},
            {"topic_id": 2, "top_tokens": ["prior_c__level_low"]},
        ],
    }
    (tmp_path / topic_path).write_text(
        json.dumps(topic_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return Chapter5InputPaths(
        prior_features_path=prior_path,
        theta_prior_path=theta_path,
        topic_interpretation_path=topic_path,
    )
