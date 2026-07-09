"""Тесты расчета метрик основной модели LDA_prior."""

import csv
import json
from pathlib import Path

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
from manual_coding_sim.lda.topic_metrics import (
    LdaTopicMetricsConfig,
    LdaTopicMetricsEvaluator,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Записать тестовый CSV с априорными признаками."""

    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Прочитать CSV-артефакт."""

    with path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def _read_json(path: Path) -> dict[str, object]:
    """Прочитать JSON-артефакт."""

    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _prior_rows() -> list[dict[str, object]]:
    """Вернуть корпус с двумя группами априорных признаков."""

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
            "has_control": 1,
            "message_length": 14,
            "procedure_type": "simple",
            "operator_skill": "medium",
        },
        {
            "run_id": "r004",
            "protocol_id": "p004",
            "scenario_id": "s004",
            "has_control": 0,
            "message_length": 40,
            "procedure_type": "complex",
            "operator_skill": "low",
        },
        {
            "run_id": "r005",
            "protocol_id": "p005",
            "scenario_id": "s005",
            "has_control": 0,
            "message_length": 45,
            "procedure_type": "complex",
            "operator_skill": "low",
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


def _build_and_train_prior(tmp_path: Path):
    """Построить тестовый корпус и обучить LDA_prior."""

    prior_path = tmp_path / "prior_features.csv"
    data_dir = tmp_path / "data" / "processed" / "lda"
    models_dir = tmp_path / "models" / "lda"
    reports_dir = tmp_path / "reports" / "chapter4"
    _write_csv(prior_path, _prior_rows())

    corpus_result = LdaCorpusBuilder(
        config=LdaCorpusBuilderConfig(
            tokenization=LdaTokenizationConfig(df_min=1, df_max_ratio=1.0),
        )
    ).build_from_csv(prior_path, data_dir)

    training_result = LdaPriorModel(
        LdaPriorModelConfig(n_components=2, max_iter=5, random_state=42),
    ).fit_from_artifacts(
        corpus_path=corpus_result.corpus_path,
        dictionary_path=corpus_result.dictionary_path,
        metadata_path=corpus_result.metadata_path,
        models_dir=models_dir,
        reports_dir=reports_dir,
    )

    return corpus_result, training_result, reports_dir


def test_topic_metrics_create_required_reports(tmp_path: Path) -> None:
    """Расчет метрик должен создать CSV- и JSON-отчеты."""

    corpus_result, training_result, reports_dir = _build_and_train_prior(tmp_path)

    result = LdaTopicMetricsEvaluator(
        LdaTopicMetricsConfig(top_n=3),
    ).evaluate_from_artifacts(
        model_path=training_result.model_path,
        corpus_path=corpus_result.corpus_path,
        dictionary_path=corpus_result.dictionary_path,
        metadata_path=corpus_result.metadata_path,
        reports_dir=reports_dir,
    )

    assert result.metrics_csv_path.exists()
    assert result.metrics_json_path.exists()
    assert result.document_count == 6
    assert result.token_count > 0
    assert result.n_components == 2
    assert result.top_n == 3
    assert len(result.model_hash) == 64


def test_topic_metrics_json_contains_expected_values(tmp_path: Path) -> None:
    """JSON-отчет должен содержать основные метрики и top-токены."""

    corpus_result, training_result, reports_dir = _build_and_train_prior(tmp_path)

    result = LdaTopicMetricsEvaluator(
        LdaTopicMetricsConfig(top_n=4),
    ).evaluate_from_artifacts(
        model_path=training_result.model_path,
        corpus_path=corpus_result.corpus_path,
        dictionary_path=corpus_result.dictionary_path,
        metadata_path=corpus_result.metadata_path,
        reports_dir=reports_dir,
    )
    payload = _read_json(result.metrics_json_path)

    assert payload["model_name"] == "LDA_prior"
    assert payload["n_components"] == 2
    assert payload["top_n"] == 4
    assert payload["document_count"] == 6
    assert payload["corpus_hash"] == corpus_result.corpus_hash
    assert payload["model_hash"] == result.model_hash
    assert payload["perplexity"] == pytest.approx(result.perplexity)
    assert payload["mean_coherence"] == pytest.approx(result.mean_coherence)
    assert payload["topic_diversity"] == pytest.approx(result.topic_diversity)
    assert len(payload["topics"]) == 2
    for topic in payload["topics"]:
        assert len(topic["top_tokens"]) == 4
        assert "coherence" in topic


def test_topic_metrics_csv_contains_compact_summary(tmp_path: Path) -> None:
    """CSV-отчет должен содержать компактную сводку метрик."""

    corpus_result, training_result, reports_dir = _build_and_train_prior(tmp_path)

    result = LdaTopicMetricsEvaluator(
        LdaTopicMetricsConfig(top_n=3),
    ).evaluate_from_artifacts(
        model_path=training_result.model_path,
        corpus_path=corpus_result.corpus_path,
        dictionary_path=corpus_result.dictionary_path,
        metadata_path=corpus_result.metadata_path,
        reports_dir=reports_dir,
    )
    rows = _read_csv(result.metrics_csv_path)
    metrics = {row["metric"]: row["value"] for row in rows}

    assert metrics["model_name"] == "LDA_prior"
    assert metrics["n_components"] == "2"
    assert metrics["top_n"] == "3"
    assert float(metrics["perplexity"]) > 0
    assert "mean_coherence" in metrics
    assert 0 < float(metrics["topic_diversity"]) <= 1
    assert len(metrics["model_hash"]) == 64


def test_topic_diversity_is_bounded(tmp_path: Path) -> None:
    """Topic diversity должна находиться в диапазоне от 0 до 1."""

    corpus_result, training_result, reports_dir = _build_and_train_prior(tmp_path)

    result = LdaTopicMetricsEvaluator(
        LdaTopicMetricsConfig(top_n=5),
    ).evaluate_from_artifacts(
        model_path=training_result.model_path,
        corpus_path=corpus_result.corpus_path,
        dictionary_path=corpus_result.dictionary_path,
        metadata_path=corpus_result.metadata_path,
        reports_dir=reports_dir,
    )

    assert 0 < result.topic_diversity <= 1
    assert result.perplexity > 0


def test_topic_metrics_are_reproducible_for_same_model(tmp_path: Path) -> None:
    """Повторный расчет по тем же артефактам должен давать те же метрики."""

    corpus_result, training_result, reports_dir = _build_and_train_prior(tmp_path)
    evaluator = LdaTopicMetricsEvaluator(LdaTopicMetricsConfig(top_n=3))

    result_1 = evaluator.evaluate_from_artifacts(
        model_path=training_result.model_path,
        corpus_path=corpus_result.corpus_path,
        dictionary_path=corpus_result.dictionary_path,
        metadata_path=corpus_result.metadata_path,
        reports_dir=reports_dir,
    )
    result_2 = evaluator.evaluate_from_artifacts(
        model_path=training_result.model_path,
        corpus_path=corpus_result.corpus_path,
        dictionary_path=corpus_result.dictionary_path,
        metadata_path=corpus_result.metadata_path,
        reports_dir=reports_dir,
    )

    assert result_1.perplexity == pytest.approx(result_2.perplexity)
    assert result_1.mean_coherence == pytest.approx(result_2.mean_coherence)
    assert result_1.topic_diversity == pytest.approx(result_2.topic_diversity)


def test_topic_metrics_config_rejects_invalid_top_n() -> None:
    """Конфигурация метрик должна отклонять слишком малый top_n."""

    with pytest.raises(ValueError, match="top_n"):
        LdaTopicMetricsConfig(top_n=1).validate()


def test_topic_metrics_reject_missing_model_file(tmp_path: Path) -> None:
    """Расчет метрик должен явно сообщать об отсутствующей модели."""

    corpus_result, _training_result, reports_dir = _build_and_train_prior(tmp_path)

    with pytest.raises(FileNotFoundError, match="LDA-модели"):
        LdaTopicMetricsEvaluator().evaluate_from_artifacts(
            model_path=tmp_path / "models" / "lda" / "missing.joblib",
            corpus_path=corpus_result.corpus_path,
            dictionary_path=corpus_result.dictionary_path,
            metadata_path=corpus_result.metadata_path,
            reports_dir=reports_dir,
        )


def test_topic_metrics_can_protect_existing_reports(tmp_path: Path) -> None:
    """При overwrite=False существующие отчеты не должны перезаписываться."""

    corpus_result, training_result, reports_dir = _build_and_train_prior(tmp_path)
    evaluator = LdaTopicMetricsEvaluator(LdaTopicMetricsConfig(top_n=3))
    evaluator.evaluate_from_artifacts(
        model_path=training_result.model_path,
        corpus_path=corpus_result.corpus_path,
        dictionary_path=corpus_result.dictionary_path,
        metadata_path=corpus_result.metadata_path,
        reports_dir=reports_dir,
    )

    protected_evaluator = LdaTopicMetricsEvaluator(
        LdaTopicMetricsConfig(top_n=3, overwrite=False),
    )
    with pytest.raises(FileExistsError, match="перезапись"):
        protected_evaluator.evaluate_from_artifacts(
            model_path=training_result.model_path,
            corpus_path=corpus_result.corpus_path,
            dictionary_path=corpus_result.dictionary_path,
            metadata_path=corpus_result.metadata_path,
            reports_dir=reports_dir,
        )
