"""Тесты подбора числа латентных факторов K для LDA_prior."""

import csv
import json
from pathlib import Path

import pytest

from manual_coding_sim.lda import LdaKSelectionConfig, LdaKSelector
from manual_coding_sim.lda.config import LdaTokenizationConfig
from manual_coding_sim.lda.corpus_builder import (
    LdaCorpusBuilder,
    LdaCorpusBuilderConfig,
)
from manual_coding_sim.lda.k_selection import LdaKSelectionCandidate


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Записать тестовый CSV с априорными признаками."""

    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Прочитать CSV-отчет в список словарей."""

    with path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def _read_json(path: Path) -> dict[str, object]:
    """Прочитать JSON-отчет."""

    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _prior_rows() -> list[dict[str, object]]:
    """Вернуть корпус с несколькими группами априорных признаков."""

    return [
        {
            "run_id": "r001",
            "protocol_id": "p001",
            "scenario_id": "s001",
            "has_control": 1,
            "message_length": 10,
            "procedure_type": "simple",
            "operator_skill": "high",
            "noise_level": "low",
        },
        {
            "run_id": "r002",
            "protocol_id": "p002",
            "scenario_id": "s002",
            "has_control": 1,
            "message_length": 12,
            "procedure_type": "simple",
            "operator_skill": "high",
            "noise_level": "low",
        },
        {
            "run_id": "r003",
            "protocol_id": "p003",
            "scenario_id": "s003",
            "has_control": 1,
            "message_length": 14,
            "procedure_type": "simple",
            "operator_skill": "medium",
            "noise_level": "medium",
        },
        {
            "run_id": "r004",
            "protocol_id": "p004",
            "scenario_id": "s004",
            "has_control": 0,
            "message_length": 40,
            "procedure_type": "complex",
            "operator_skill": "low",
            "noise_level": "high",
        },
        {
            "run_id": "r005",
            "protocol_id": "p005",
            "scenario_id": "s005",
            "has_control": 0,
            "message_length": 45,
            "procedure_type": "complex",
            "operator_skill": "low",
            "noise_level": "high",
        },
        {
            "run_id": "r006",
            "protocol_id": "p006",
            "scenario_id": "s006",
            "has_control": 0,
            "message_length": 50,
            "procedure_type": "complex",
            "operator_skill": "medium",
            "noise_level": "medium",
        },
    ]


def _build_test_corpus(tmp_path: Path):
    """Построить тестовый LDA-корпус для подбора K."""

    prior_path = tmp_path / "prior_features.csv"
    data_dir = tmp_path / "data" / "processed" / "lda"
    _write_csv(prior_path, _prior_rows())

    return LdaCorpusBuilder(
        config=LdaCorpusBuilderConfig(
            tokenization=LdaTokenizationConfig(df_min=1, df_max_ratio=1.0),
        )
    ).build_from_csv(prior_path, data_dir)


def _run_k_selection(tmp_path: Path):
    """Выполнить быстрый подбор K на тестовом корпусе."""

    corpus_result = _build_test_corpus(tmp_path)
    selector = LdaKSelector(
        LdaKSelectionConfig(
            k_values=(2, 3),
            max_iter=3,
            random_state=42,
            top_n=3,
        )
    )
    result = selector.select_from_artifacts(
        corpus_path=corpus_result.corpus_path,
        dictionary_path=corpus_result.dictionary_path,
        metadata_path=corpus_result.metadata_path,
        models_dir=tmp_path / "models" / "lda",
        reports_dir=tmp_path / "reports" / "chapter4",
    )
    return corpus_result, result


def test_k_selection_config_rejects_invalid_values() -> None:
    """Конфигурация должна отвергать некорректный набор K."""

    with pytest.raises(ValueError, match="не должен быть пустым"):
        LdaKSelectionConfig(k_values=()).validate()
    with pytest.raises(ValueError, match="повторяющиеся"):
        LdaKSelectionConfig(k_values=(2, 2)).validate()
    with pytest.raises(ValueError, match="не меньше 2"):
        LdaKSelectionConfig(k_values=(1, 2)).validate()
    with pytest.raises(ValueError, match="top_n"):
        LdaKSelectionConfig(k_values=(2, 3), top_n=1).validate()


def test_k_selector_creates_required_reports(tmp_path: Path) -> None:
    """Подбор K должен создать CSV-, JSON- и Markdown-отчеты."""

    _, result = _run_k_selection(tmp_path)

    assert result.report_csv_path.exists()
    assert result.report_json_path.exists()
    assert result.report_md_path.exists()
    assert result.recommended_k in {2, 3}
    assert len(result.candidates) == 2


def test_k_selection_json_contains_candidates_and_recommendation(
    tmp_path: Path,
) -> None:
    """JSON-отчет должен содержать кандидатов и выбранное значение K."""

    _, result = _run_k_selection(tmp_path)
    payload = _read_json(result.report_json_path)

    assert payload["model_name"] == "LDA_prior"
    assert payload["selection_method"] == "weighted_min_max_score"
    assert payload["recommended_k"] == result.recommended_k
    assert payload["random_state"] == 42
    assert payload["max_iter"] == 3
    assert len(payload["candidates"]) == 2

    for candidate in payload["candidates"]:
        assert candidate["k"] in {2, 3}
        assert candidate["perplexity"] > 0
        assert 0 <= candidate["normalized_coherence"] <= 1
        assert 0 <= candidate["normalized_inverse_perplexity"] <= 1
        assert 0 <= candidate["normalized_topic_diversity"] <= 1
        assert 0 <= candidate["selection_score"] <= 1
        assert Path(candidate["model_path"]).exists()
        assert Path(candidate["metrics_json_path"]).exists()
        assert len(candidate["model_hash"]) == 64


def test_k_selection_csv_marks_single_recommended_row(tmp_path: Path) -> None:
    """CSV-отчет должен пометить ровно одну рекомендованную строку."""

    _, result = _run_k_selection(tmp_path)
    rows = _read_csv(result.report_csv_path)
    recommended_rows = [row for row in rows if row["is_recommended"] == "true"]

    assert len(rows) == 2
    assert len(recommended_rows) == 1
    assert int(recommended_rows[0]["k"]) == result.recommended_k
    for row in rows:
        assert int(row["k"]) in {2, 3}
        assert float(row["perplexity"]) > 0
        assert 0 <= float(row["selection_score"]) <= 1


def test_k_selection_markdown_contains_summary(tmp_path: Path) -> None:
    """Markdown-отчет должен содержать краткую сводку выбора K."""

    _, result = _run_k_selection(tmp_path)
    content = result.report_md_path.read_text(encoding="utf-8")

    assert "# Отчет подбора числа латентных факторов K" in content
    assert f"K = {result.recommended_k}" in content
    assert "Perplexity" in content
    assert "Topic diversity" in content


def test_k_selector_recommends_candidate_with_max_score(tmp_path: Path) -> None:
    """Рекомендованное K должно соответствовать максимальной оценке."""

    _, result = _run_k_selection(tmp_path)
    best_candidate = max(
        result.candidates,
        key=lambda candidate: (candidate.selection_score, -candidate.k),
    )

    assert result.recommended_k == best_candidate.k


def test_k_selector_blocks_report_overwrite_when_requested(tmp_path: Path) -> None:
    """При overwrite=False нельзя перезаписывать итоговые отчеты."""

    corpus_result, _ = _run_k_selection(tmp_path)
    selector = LdaKSelector(
        LdaKSelectionConfig(
            k_values=(2, 3),
            max_iter=3,
            random_state=42,
            top_n=3,
            overwrite=False,
        )
    )

    with pytest.raises(FileExistsError, match="перезапись"):
        selector.select_from_artifacts(
            corpus_path=corpus_result.corpus_path,
            dictionary_path=corpus_result.dictionary_path,
            metadata_path=corpus_result.metadata_path,
            models_dir=tmp_path / "models" / "lda",
            reports_dir=tmp_path / "reports" / "chapter4",
        )


def test_k_selection_candidate_dataclass_accepts_paths(tmp_path: Path) -> None:
    """Dataclass кандидата должен хранить пути к модели и метрикам."""

    candidate = LdaKSelectionCandidate(
        k=2,
        perplexity=10.0,
        mean_coherence=-1.0,
        topic_diversity=0.5,
        normalized_coherence=1.0,
        normalized_inverse_perplexity=1.0,
        normalized_topic_diversity=0.5,
        selection_score=0.833333333333,
        model_path=tmp_path / "lda_prior.joblib",
        metrics_json_path=tmp_path / "topic_metrics.json",
        model_hash="a" * 64,
    )

    assert candidate.k == 2
    assert candidate.model_path.name == "lda_prior.joblib"
    assert len(candidate.model_hash) == 64
