"""Тесты обучения основной модели LDA_prior."""

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from manual_coding_sim.lda.config import LdaTokenizationConfig
from manual_coding_sim.lda.corpus_builder import (
    LdaCorpusBuilder,
    LdaCorpusBuilderConfig,
)
from manual_coding_sim.lda.lda_prior_model import (
    LdaPriorModel,
    LdaPriorModelConfig,
)
from manual_coding_sim.lda.matrix_builder import LdaMatrixBuilder


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Записать тестовый CSV с априорными признаками."""

    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Прочитать CSV-артефакт в список словарей."""

    with path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def _read_json(path: Path) -> dict[str, object]:
    """Прочитать JSON-артефакт."""

    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _prior_rows() -> list[dict[str, object]]:
    """Вернуть корпус с двумя устойчивыми группами априорных признаков."""

    return [
        {
            "run_id": "r001",
            "protocol_id": "p001",
            "scenario_id": "s001",
            "has_control": 1,
            "message_length": 10,
            "procedure_type": "simple",
            "operator_skill": "high",
        },
        {
            "run_id": "r002",
            "protocol_id": "p002",
            "scenario_id": "s002",
            "has_control": 1,
            "message_length": 12,
            "procedure_type": "simple",
            "operator_skill": "high",
        },
        {
            "run_id": "r003",
            "protocol_id": "p003",
            "scenario_id": "s003",
            "has_control": 0,
            "message_length": 40,
            "procedure_type": "complex",
            "operator_skill": "low",
        },
        {
            "run_id": "r004",
            "protocol_id": "p004",
            "scenario_id": "s004",
            "has_control": 0,
            "message_length": 45,
            "procedure_type": "complex",
            "operator_skill": "low",
        },
        {
            "run_id": "r005",
            "protocol_id": "p005",
            "scenario_id": "s005",
            "has_control": 1,
            "message_length": 14,
            "procedure_type": "simple",
            "operator_skill": "medium",
        },
        {
            "run_id": "r006",
            "protocol_id": "p006",
            "scenario_id": "s006",
            "has_control": 0,
            "message_length": 50,
            "procedure_type": "complex",
            "operator_skill": "medium",
        },
    ]


def _build_test_corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Построить тестовый LDA-корпус через штатный построитель."""

    prior_path = tmp_path / "prior_features.csv"
    data_dir = tmp_path / "data" / "processed" / "lda"
    _write_csv(prior_path, _prior_rows())

    builder = LdaCorpusBuilder(
        config=LdaCorpusBuilderConfig(
            tokenization=LdaTokenizationConfig(df_min=1, df_max_ratio=1.0),
        )
    )
    result = builder.build_from_csv(prior_path, data_dir)
    return result.corpus_path, result.dictionary_path, result.metadata_path


def test_lda_matrix_builder_restores_document_token_matrix(tmp_path: Path) -> None:
    """Построитель матрицы должен восстановить размерность корпуса."""

    corpus_path, dictionary_path, metadata_path = _build_test_corpus(tmp_path)

    matrix_result = LdaMatrixBuilder().build_from_files(
        corpus_path=corpus_path,
        dictionary_path=dictionary_path,
        metadata_path=metadata_path,
    )

    assert matrix_result.matrix.shape[0] == 6
    assert matrix_result.matrix.shape[1] == len(matrix_result.vocabulary)
    assert matrix_result.matrix.sum() > 0
    assert matrix_result.documents[0].protocol_id == "p001"
    assert matrix_result.documents[0].scenario_id == "s001"
    assert matrix_result.corpus_hash is not None


def test_lda_prior_model_creates_required_artifacts(tmp_path: Path) -> None:
    """Обучение LDA_prior должно создать модель и основные отчеты."""

    corpus_path, dictionary_path, metadata_path = _build_test_corpus(tmp_path)
    model = LdaPriorModel(
        LdaPriorModelConfig(
            n_components=2,
            max_iter=5,
            random_state=42,
        )
    )

    result = model.fit_from_artifacts(
        corpus_path=corpus_path,
        dictionary_path=dictionary_path,
        metadata_path=metadata_path,
        models_dir=tmp_path / "models" / "lda",
        reports_dir=tmp_path / "reports" / "chapter4",
    )

    assert result.model_path.exists()
    assert result.theta_prior_path.exists()
    assert result.topic_word_path.exists()
    assert result.metadata_path.exists()
    assert result.document_count == 6
    assert result.token_count > 0
    assert result.n_components == 2
    assert len(result.model_hash) == 64
    assert joblib.load(result.model_path).n_components == 2


def test_theta_prior_rows_are_normalized(tmp_path: Path) -> None:
    """В каждой строке theta_prior сумма theta-компонент должна быть равна 1."""

    corpus_path, dictionary_path, metadata_path = _build_test_corpus(tmp_path)
    result = LdaPriorModel(
        LdaPriorModelConfig(n_components=2, max_iter=5, random_state=42)
    ).fit_from_artifacts(
        corpus_path=corpus_path,
        dictionary_path=dictionary_path,
        metadata_path=metadata_path,
        models_dir=tmp_path / "models" / "lda",
        reports_dir=tmp_path / "reports" / "chapter4",
    )

    rows = _read_csv(result.theta_prior_path)
    theta_columns = [column for column in rows[0] if column.startswith("theta_")]

    assert len(rows) == 6
    assert theta_columns == ["theta_0", "theta_1"]
    for row in rows:
        theta_sum = sum(float(row[column]) for column in theta_columns)
        assert theta_sum == pytest.approx(1.0, abs=1e-9)
        assert row["selected_k"] == "2"
        assert row["random_state"] == "42"


def test_topic_word_rows_are_normalized_by_topic(tmp_path: Path) -> None:
    """Распределение phi_k по каждому топику должно суммироваться в 1."""

    corpus_path, dictionary_path, metadata_path = _build_test_corpus(tmp_path)
    result = LdaPriorModel(
        LdaPriorModelConfig(n_components=2, max_iter=5, random_state=42)
    ).fit_from_artifacts(
        corpus_path=corpus_path,
        dictionary_path=dictionary_path,
        metadata_path=metadata_path,
        models_dir=tmp_path / "models" / "lda",
        reports_dir=tmp_path / "reports" / "chapter4",
    )

    rows = _read_csv(result.topic_word_path)
    weights_by_topic: dict[str, float] = {}
    for row in rows:
        topic_id = row["topic_id"]
        weights_by_topic[topic_id] = weights_by_topic.get(topic_id, 0.0) + float(
            row["weight"]
        )

    assert set(weights_by_topic) == {"0", "1"}
    for weight_sum in weights_by_topic.values():
        assert weight_sum == pytest.approx(1.0, abs=1e-9)


def test_lda_prior_metadata_contains_training_parameters(tmp_path: Path) -> None:
    """Метаданные обучения должны фиксировать параметры LDA_prior."""

    corpus_path, dictionary_path, metadata_path = _build_test_corpus(tmp_path)
    result = LdaPriorModel(
        LdaPriorModelConfig(
            n_components=2,
            doc_topic_prior=0.25,
            topic_word_prior=0.5,
            max_iter=5,
            random_state=77,
        )
    ).fit_from_artifacts(
        corpus_path=corpus_path,
        dictionary_path=dictionary_path,
        metadata_path=metadata_path,
        models_dir=tmp_path / "models" / "lda",
        reports_dir=tmp_path / "reports" / "chapter4",
    )

    metadata = _read_json(result.metadata_path)

    assert metadata["model_name"] == "LDA_prior"
    assert metadata["n_components"] == 2
    assert metadata["doc_topic_prior"] == 0.25
    assert metadata["topic_word_prior"] == 0.5
    assert metadata["learning_method"] == "batch"
    assert metadata["max_iter"] == 5
    assert metadata["random_state"] == 77
    assert metadata["document_count"] == 6
    assert metadata["corpus_hash"] == result.corpus_hash
    assert metadata["model_hash"] == result.model_hash


def test_lda_prior_is_reproducible_with_fixed_random_state(tmp_path: Path) -> None:
    """Одинаковый random_state должен давать одинаковые theta_prior."""

    corpus_path, dictionary_path, metadata_path = _build_test_corpus(tmp_path)
    config = LdaPriorModelConfig(n_components=2, max_iter=5, random_state=123)

    result_1 = LdaPriorModel(config).fit_from_artifacts(
        corpus_path=corpus_path,
        dictionary_path=dictionary_path,
        metadata_path=metadata_path,
        models_dir=tmp_path / "models_1" / "lda",
        reports_dir=tmp_path / "reports_1" / "chapter4",
    )
    result_2 = LdaPriorModel(config).fit_from_artifacts(
        corpus_path=corpus_path,
        dictionary_path=dictionary_path,
        metadata_path=metadata_path,
        models_dir=tmp_path / "models_2" / "lda",
        reports_dir=tmp_path / "reports_2" / "chapter4",
    )

    theta_1 = _read_csv(result_1.theta_prior_path)
    theta_2 = _read_csv(result_2.theta_prior_path)

    assert theta_1 == theta_2


def test_lda_prior_rejects_too_large_number_of_topics(tmp_path: Path) -> None:
    """Число тем не должно превышать размер словаря токенов."""

    corpus_path, dictionary_path, metadata_path = _build_test_corpus(tmp_path)
    dictionary = _read_json(dictionary_path)
    token_count = int(dictionary["token_count"])

    model = LdaPriorModel(
        LdaPriorModelConfig(
            n_components=token_count + 1,
            max_iter=2,
            random_state=42,
        )
    )

    with pytest.raises(ValueError, match="n_components"):
        model.fit_from_artifacts(
            corpus_path=corpus_path,
            dictionary_path=dictionary_path,
            metadata_path=metadata_path,
            models_dir=tmp_path / "models" / "lda",
            reports_dir=tmp_path / "reports" / "chapter4",
        )


def test_lda_prior_config_rejects_invalid_parameters() -> None:
    """Конфигурация LDA_prior должна отклонять некорректные параметры."""

    with pytest.raises(ValueError, match="n_components"):
        LdaPriorModelConfig(n_components=1).validate()

    with pytest.raises(ValueError, match="learning_method"):
        LdaPriorModelConfig(n_components=2, learning_method="invalid").validate()

    with pytest.raises(ValueError, match="doc_topic_prior"):
        LdaPriorModelConfig(n_components=2, doc_topic_prior=0).validate()


def test_lda_prior_api_does_not_accept_fact_or_quality_paths() -> None:
    """API обучения LDA_prior не должен иметь аргументы для фактических данных."""

    from inspect import signature

    parameters = signature(LdaPriorModel.fit_from_artifacts).parameters

    assert "fact_features_path" not in parameters
    assert "quality_targets_path" not in parameters
    assert "all_features_path" not in parameters
