"""Тесты анализа устойчивости латентных факторов LDA."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from manual_coding_sim.lda import (
    LdaCorpusBuilder,
    LdaCorpusBuilderConfig,
    LdaTokenizationConfig,
    LdaTopicStabilityAnalyzer,
    LdaTopicStabilityConfig,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Сохранить тестовые строки в CSV-файл."""

    path.parent.mkdir(parents=True, exist_ok=True)
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
    """Вернуть небольшой корпус с двумя устойчивыми группами признаков."""

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
    """Построить тестовый LDA-корпус для анализа устойчивости."""

    prior_path = tmp_path / "prior_features.csv"
    data_dir = tmp_path / "data" / "processed" / "lda"
    _write_csv(prior_path, _prior_rows())

    return LdaCorpusBuilder(
        config=LdaCorpusBuilderConfig(
            tokenization=LdaTokenizationConfig(df_min=1, df_max_ratio=1.0),
        )
    ).build_from_csv(prior_path, data_dir)


def _run_stability(tmp_path: Path):
    """Выполнить быстрый анализ устойчивости на тестовом корпусе."""

    corpus_result = _build_test_corpus(tmp_path)
    analyzer = LdaTopicStabilityAnalyzer(
        LdaTopicStabilityConfig(
            n_components=2,
            random_states=(11, 42, 77),
            max_iter=3,
        )
    )
    result = analyzer.analyze_from_artifacts(
        corpus_path=corpus_result.corpus_path,
        dictionary_path=corpus_result.dictionary_path,
        metadata_path=corpus_result.metadata_path,
        models_dir=tmp_path / "models" / "lda",
        reports_dir=tmp_path / "reports" / "chapter4",
    )
    return corpus_result, result


def test_topic_stability_config_rejects_invalid_values() -> None:
    """Конфигурация должна отвергать некорректные параметры."""

    with pytest.raises(ValueError, match="не меньше 2"):
        LdaTopicStabilityConfig(n_components=1, random_states=(1, 2)).validate()
    with pytest.raises(ValueError, match="не менее двух"):
        LdaTopicStabilityConfig(n_components=2, random_states=(1,)).validate()
    with pytest.raises(ValueError, match="повторяющиеся"):
        LdaTopicStabilityConfig(n_components=2, random_states=(1, 1)).validate()
    with pytest.raises(ValueError, match="max_iter"):
        LdaTopicStabilityConfig(
            n_components=2,
            random_states=(1, 2),
            max_iter=0,
        ).validate()


def test_topic_stability_creates_required_reports(tmp_path: Path) -> None:
    """Анализ устойчивости должен создать CSV-, JSON- и Markdown-отчеты."""

    _, result = _run_stability(tmp_path)

    assert result.report_csv_path.exists()
    assert result.report_json_path.exists()
    assert result.report_md_path.exists()
    assert result.n_components == 2
    assert result.random_states == (11, 42, 77)
    assert result.reference_random_state == 11
    assert len(result.runs) == 3


def test_topic_stability_json_contains_summary_and_runs(tmp_path: Path) -> None:
    """JSON-отчет должен содержать сводку, запуски и устойчивость тем."""

    _, result = _run_stability(tmp_path)
    payload = _read_json(result.report_json_path)

    assert payload["model_name"] == "LDA_prior"
    assert payload["analysis_method"] == "topic_word_cosine_similarity"
    assert payload["n_components"] == 2
    assert payload["random_states"] == [11, 42, 77]
    assert payload["reference_random_state"] == 11
    assert 0 <= payload["mean_stability"] <= 1
    assert 0 <= payload["min_stability"] <= 1
    assert len(payload["runs"]) == 3
    assert len(payload["per_topic_stability"]) == 2
    assert len(payload["pairwise_run_stability"]) == 3

    for run in payload["runs"]:
        assert Path(run["model_path"]).exists()
        assert Path(run["theta_prior_path"]).exists()
        assert Path(run["topic_word_path"]).exists()
        assert Path(run["metadata_path"]).exists()
        assert len(run["model_hash"]) == 64


def test_topic_stability_per_topic_matches_are_valid(tmp_path: Path) -> None:
    """Для каждой опорной темы должны быть указаны ближайшие темы запусков."""

    _, result = _run_stability(tmp_path)
    payload = _read_json(result.report_json_path)

    for topic_item in payload["per_topic_stability"]:
        assert topic_item["topic_id"] in {0, 1}
        assert 0 <= topic_item["mean_similarity"] <= 1
        assert 0 <= topic_item["min_similarity"] <= 1
        assert len(topic_item["matches"]) == 2
        for match in topic_item["matches"]:
            assert match["random_state"] in {42, 77}
            assert match["matched_topic_id"] in {0, 1}
            assert 0 <= match["similarity"] <= 1


def test_topic_stability_csv_contains_summary_topic_and_pairwise_rows(
    tmp_path: Path,
) -> None:
    """CSV-отчет должен содержать сводную, тематические и попарные строки."""

    _, result = _run_stability(tmp_path)
    rows = _read_csv(result.report_csv_path)
    row_types = [row["row_type"] for row in rows]

    assert row_types.count("summary") == 1
    assert row_types.count("topic") == 2
    assert row_types.count("pairwise_run") == 3
    for row in rows:
        assert 0 <= float(row["mean_similarity"]) <= 1
        assert 0 <= float(row["min_similarity"]) <= 1


def test_topic_stability_markdown_contains_summary(tmp_path: Path) -> None:
    """Markdown-отчет должен содержать краткую сводку устойчивости."""

    _, result = _run_stability(tmp_path)
    content = result.report_md_path.read_text(encoding="utf-8")

    assert "# Отчет устойчивости латентных факторов LDA_prior" in content
    assert "Число факторов K: 2" in content
    assert "Средняя устойчивость" in content
    assert "Попарная устойчивость запусков" in content


def test_topic_stability_blocks_report_overwrite_when_requested(
    tmp_path: Path,
) -> None:
    """При overwrite=False нельзя перезаписывать итоговые отчеты."""

    corpus_result, _ = _run_stability(tmp_path)
    analyzer = LdaTopicStabilityAnalyzer(
        LdaTopicStabilityConfig(
            n_components=2,
            random_states=(11, 42, 77),
            max_iter=3,
            overwrite=False,
        )
    )

    with pytest.raises(FileExistsError, match="перезапись"):
        analyzer.analyze_from_artifacts(
            corpus_path=corpus_result.corpus_path,
            dictionary_path=corpus_result.dictionary_path,
            metadata_path=corpus_result.metadata_path,
            models_dir=tmp_path / "models" / "lda",
            reports_dir=tmp_path / "reports" / "chapter4",
        )


def test_topic_stability_is_reproducible_for_same_inputs(tmp_path: Path) -> None:
    """Повторный расчет с теми же параметрами должен давать те же метрики."""

    _, first_result = _run_stability(tmp_path / "first")
    _, second_result = _run_stability(tmp_path / "second")

    assert first_result.mean_stability == pytest.approx(
        second_result.mean_stability,
    )
    assert first_result.min_stability == pytest.approx(
        second_result.min_stability,
    )
