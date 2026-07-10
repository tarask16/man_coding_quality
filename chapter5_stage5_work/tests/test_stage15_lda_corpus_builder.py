"""Тесты построителя LDA-корпуса по prior_features.csv."""

import csv
import json
from pathlib import Path

import pytest

from manual_coding_sim.lda.config import LdaTokenizationConfig
from manual_coding_sim.lda.corpus_builder import (
    LdaCorpusBuilder,
    LdaCorpusBuilderConfig,
)
from manual_coding_sim.lda.leakage_guard import LdaLeakageError


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Записать тестовый CSV-файл с априорными признаками."""

    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, object]:
    """Прочитать JSON-файл, сформированный построителем корпуса."""

    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _prior_rows() -> list[dict[str, object]]:
    """Вернуть минимальный набор априорных признаков для тестов."""

    return [
        {
            "run_id": "r001",
            "protocol_id": "p001",
            "scenario_id": "s001",
            "has_control": 1,
            "message_length": 10,
            "procedure_type": "simple",
        },
        {
            "run_id": "r002",
            "protocol_id": "p002",
            "scenario_id": "s002",
            "has_control": 0,
            "message_length": 20,
            "procedure_type": "complex",
        },
        {
            "run_id": "r003",
            "protocol_id": "p003",
            "scenario_id": "s003",
            "has_control": 1,
            "message_length": 30,
            "procedure_type": "simple",
        },
        {
            "run_id": "r004",
            "protocol_id": "p004",
            "scenario_id": "s004",
            "has_control": 0,
            "message_length": 40,
            "procedure_type": "complex",
        },
    ]


def test_lda_corpus_builder_creates_required_artifacts(tmp_path: Path) -> None:
    """Построитель должен создать корпус, словарь, карту токенов и отчеты."""

    prior_path = tmp_path / "prior_features.csv"
    output_dir = tmp_path / "lda"
    _write_csv(prior_path, _prior_rows())

    builder = LdaCorpusBuilder(
        config=LdaCorpusBuilderConfig(
            tokenization=LdaTokenizationConfig(df_min=1, df_max_ratio=1.0),
        )
    )
    result = builder.build_from_csv(prior_path, output_dir)

    assert result.document_count == 4
    assert result.token_count_before_filter > 0
    assert result.token_count_after_filter > 0
    assert result.dictionary_path.exists()
    assert result.token_map_path.exists()
    assert result.corpus_path.exists()
    assert result.metadata_path.exists()
    assert result.leakage_report_path.exists()
    assert len(result.corpus_hash) == 64


def test_lda_corpus_builder_writes_traceable_long_corpus(tmp_path: Path) -> None:
    """Строки корпуса должны сохранять связь с протоколом и сценарием."""

    prior_path = tmp_path / "prior_features.csv"
    output_dir = tmp_path / "lda"
    _write_csv(prior_path, _prior_rows())

    builder = LdaCorpusBuilder(
        config=LdaCorpusBuilderConfig(
            tokenization=LdaTokenizationConfig(df_min=1, df_max_ratio=1.0),
        )
    )
    result = builder.build_from_csv(prior_path, output_dir)

    with result.corpus_path.open("r", encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))

    assert rows
    assert set(rows[0]) == {
        "document_index",
        "run_id",
        "protocol_id",
        "scenario_id",
        "token_id",
        "token",
        "count",
    }
    assert {row["protocol_id"] for row in rows} == {"p001", "p002", "p003", "p004"}
    assert {row["scenario_id"] for row in rows} == {"s001", "s002", "s003", "s004"}


def test_lda_corpus_builder_filters_rare_and_common_tokens(tmp_path: Path) -> None:
    """Редкие и чрезмерно общие токены должны исключаться из словаря."""

    prior_path = tmp_path / "prior_features.csv"
    output_dir = tmp_path / "lda"
    rows = [
        {
            "protocol_id": "p001",
            "scenario_id": "s001",
            "common_feature": "same",
            "rare_feature": "unique",
            "group_feature": "a",
        },
        {
            "protocol_id": "p002",
            "scenario_id": "s002",
            "common_feature": "same",
            "rare_feature": "other",
            "group_feature": "a",
        },
        {
            "protocol_id": "p003",
            "scenario_id": "s003",
            "common_feature": "same",
            "rare_feature": "other",
            "group_feature": "b",
        },
        {
            "protocol_id": "p004",
            "scenario_id": "s004",
            "common_feature": "same",
            "rare_feature": "other",
            "group_feature": "b",
        },
    ]
    _write_csv(prior_path, rows)

    builder = LdaCorpusBuilder(
        config=LdaCorpusBuilderConfig(
            tokenization=LdaTokenizationConfig(df_min=2, df_max_ratio=0.75),
        )
    )
    result = builder.build_from_csv(prior_path, output_dir)
    dictionary = _read_json(result.dictionary_path)
    tokens = {item["token"] for item in dictionary["tokens"]}

    assert "common_feature__value_same" not in tokens
    assert "rare_feature__value_unique" not in tokens
    assert "group_feature__value_a" in tokens
    assert "group_feature__value_b" in tokens


def test_lda_corpus_builder_rejects_leaking_columns(tmp_path: Path) -> None:
    """Построитель корпуса должен отклонять целевые колонки качества."""

    prior_path = tmp_path / "prior_features.csv"
    output_dir = tmp_path / "lda"
    rows = [
        {"protocol_id": "p001", "message_length": 10, "q_acc": 0.95},
        {"protocol_id": "p002", "message_length": 20, "q_acc": 0.90},
    ]
    _write_csv(prior_path, rows)

    builder = LdaCorpusBuilder()

    with pytest.raises(LdaLeakageError, match="q_acc"):
        builder.build_from_csv(prior_path, output_dir)


def test_lda_corpus_builder_hash_is_stable_for_same_input(tmp_path: Path) -> None:
    """Одинаковые входные данные должны давать одинаковый хеш корпуса."""

    prior_path = tmp_path / "prior_features.csv"
    output_dir_1 = tmp_path / "lda_1"
    output_dir_2 = tmp_path / "lda_2"
    _write_csv(prior_path, _prior_rows())

    builder = LdaCorpusBuilder(
        config=LdaCorpusBuilderConfig(
            tokenization=LdaTokenizationConfig(df_min=1, df_max_ratio=1.0),
        )
    )
    result_1 = builder.build_from_csv(prior_path, output_dir_1)
    result_2 = builder.build_from_csv(prior_path, output_dir_2)

    assert result_1.corpus_hash == result_2.corpus_hash


def test_lda_corpus_builder_metadata_contains_tokenization_parameters(
    tmp_path: Path,
) -> None:
    """Метаданные корпуса должны фиксировать параметры токенизации."""

    prior_path = tmp_path / "prior_features.csv"
    output_dir = tmp_path / "lda"
    _write_csv(prior_path, _prior_rows())

    builder = LdaCorpusBuilder(
        config=LdaCorpusBuilderConfig(
            tokenization=LdaTokenizationConfig(
                df_min=1,
                df_max_ratio=1.0,
                numeric_strategy="uniform",
                numeric_bins=4,
            ),
        )
    )
    result = builder.build_from_csv(prior_path, output_dir)
    metadata = _read_json(result.metadata_path)

    assert metadata["document_count"] == 4
    assert metadata["df_min"] == 1
    assert metadata["df_max_ratio"] == 1.0
    assert metadata["numeric_strategy"] == "uniform"
    assert metadata["numeric_bins"] == 4
    assert metadata["corpus_hash"] == result.corpus_hash
