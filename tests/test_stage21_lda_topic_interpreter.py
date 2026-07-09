"""Тесты этапа 21: интерпретация латентных факторов качества."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from manual_coding_sim.lda import (
    LdaTopicInterpreter,
    LdaTopicInterpreterConfig,
)


def _write_topic_word(path: Path) -> None:
    """Создать небольшой topic_word.csv для тестов интерпретатора."""

    rows = [
        {
            "topic_id": 0,
            "token_id": 0,
            "token": "operator_skill__level_low",
            "document_frequency": 4,
            "weight": "0.55",
        },
        {
            "topic_id": 0,
            "token_id": 1,
            "token": "procedure_complexity__level_high",
            "document_frequency": 5,
            "weight": "0.35",
        },
        {
            "topic_id": 0,
            "token_id": 2,
            "token": "control_available__present",
            "document_frequency": 3,
            "weight": "0.10",
        },
        {
            "topic_id": 1,
            "token_id": 0,
            "token": "message_length__level_high",
            "document_frequency": 4,
            "weight": "0.20",
        },
        {
            "topic_id": 1,
            "token_id": 1,
            "token": "noise_level__level_high",
            "document_frequency": 5,
            "weight": "0.70",
        },
        {
            "topic_id": 1,
            "token_id": 2,
            "token": "procedure_type__value_manual",
            "document_frequency": 2,
            "weight": "0.10",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "topic_id",
                "token_id",
                "token",
                "document_frequency",
                "weight",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Прочитать CSV-отчет интерпретатора."""

    with path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def test_interpreter_config_rejects_invalid_top_n() -> None:
    """Конфигурация должна запрещать неположительное число top-токенов."""

    with pytest.raises(ValueError, match="top_n"):
        LdaTopicInterpreterConfig(top_n=0).validate()


def test_interpreter_creates_all_reports(tmp_path: Path) -> None:
    """Интерпретатор должен создавать CSV, JSON и Markdown-отчеты."""

    topic_word_path = tmp_path / "topic_word.csv"
    reports_dir = tmp_path / "reports"
    _write_topic_word(topic_word_path)

    result = LdaTopicInterpreter(
        LdaTopicInterpreterConfig(top_n=2)
    ).interpret_from_topic_word(
        topic_word_path=topic_word_path,
        reports_dir=reports_dir,
    )

    assert result.interpretation_csv_path.exists()
    assert result.interpretation_json_path.exists()
    assert result.interpretation_md_path.exists()
    assert result.topic_count == 2
    assert result.top_n == 2


def test_interpreter_orders_tokens_by_weight(tmp_path: Path) -> None:
    """Top-токены должны сортироваться по убыванию веса внутри темы."""

    topic_word_path = tmp_path / "topic_word.csv"
    _write_topic_word(topic_word_path)

    result = LdaTopicInterpreter(
        LdaTopicInterpreterConfig(top_n=2)
    ).interpret_from_topic_word(topic_word_path, tmp_path / "reports")

    rows = _read_csv(result.interpretation_csv_path)
    first_topic = rows[0]
    assert first_topic["topic_id"] == "0"
    assert first_topic["dominant_token"] == "operator_skill__level_low"
    assert "operator_skill__level_low" in first_topic["top_tokens"]
    assert "procedure_complexity__level_high" in first_topic["top_tokens"]
    assert "control_available__present" not in first_topic["top_tokens"]


def test_interpreter_extracts_source_features(tmp_path: Path) -> None:
    """Отчет должен содержать исходные признаки, восстановленные из токенов."""

    topic_word_path = tmp_path / "topic_word.csv"
    _write_topic_word(topic_word_path)

    result = LdaTopicInterpreter(
        LdaTopicInterpreterConfig(top_n=3)
    ).interpret_from_topic_word(topic_word_path, tmp_path / "reports")

    rows = _read_csv(result.interpretation_csv_path)
    assert "operator_skill" in rows[0]["source_features"]
    assert "procedure_complexity" in rows[0]["source_features"]
    assert "noise_level" in rows[1]["source_features"]


def test_interpreter_json_contains_topics_and_comments(tmp_path: Path) -> None:
    """JSON-отчет должен содержать темы, названия и комментарии."""

    topic_word_path = tmp_path / "topic_word.csv"
    _write_topic_word(topic_word_path)

    result = LdaTopicInterpreter(
        LdaTopicInterpreterConfig(top_n=2)
    ).interpret_from_topic_word(topic_word_path, tmp_path / "reports")

    payload = json.loads(result.interpretation_json_path.read_text(encoding="utf-8"))
    assert payload["model_name"] == "LDA_prior"
    assert payload["topic_count"] == 2
    assert payload["allowed_for_apriori_forecast"] is True
    assert len(payload["topics"]) == 2
    assert payload["topics"][0]["suggested_factor_name"].startswith(
        "Латентный фактор качества 0"
    )
    assert "Доминирующий токен" in payload["topics"][0]["interpretation_comment"]


def test_interpreter_markdown_contains_topic_sections(tmp_path: Path) -> None:
    """Markdown-отчет должен быть пригоден для вставки в текст главы 4."""

    topic_word_path = tmp_path / "topic_word.csv"
    _write_topic_word(topic_word_path)

    result = LdaTopicInterpreter().interpret_from_topic_word(
        topic_word_path,
        tmp_path / "reports",
    )

    content = result.interpretation_md_path.read_text(encoding="utf-8")
    assert "# Интерпретация латентных факторов качества" in content
    assert "## Тема 0" in content
    assert "## Тема 1" in content
    assert "operator_skill__level_low" in content


def test_interpreter_rejects_missing_required_columns(tmp_path: Path) -> None:
    """Интерпретатор должен отклонять неполный topic_word.csv."""

    topic_word_path = tmp_path / "topic_word.csv"
    topic_word_path.write_text("topic_id,token,weight\n0,a__present,1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="обязательные колонки"):
        LdaTopicInterpreter().interpret_from_topic_word(
            topic_word_path,
            tmp_path / "reports",
        )


def test_interpreter_respects_overwrite_flag(tmp_path: Path) -> None:
    """При overwrite=False существующие отчеты не должны перезаписываться."""

    topic_word_path = tmp_path / "topic_word.csv"
    reports_dir = tmp_path / "reports"
    _write_topic_word(topic_word_path)
    LdaTopicInterpreter().interpret_from_topic_word(topic_word_path, reports_dir)

    with pytest.raises(FileExistsError, match="уже существуют"):
        LdaTopicInterpreter(
            LdaTopicInterpreterConfig(overwrite=False)
        ).interpret_from_topic_word(topic_word_path, reports_dir)


def test_interpreter_removes_diagnostic_prefix_from_source_feature(
    tmp_path: Path,
) -> None:
    """Префиксы diag_ и fact_ не должны засорять имя признака-источника."""

    topic_word_path = tmp_path / "topic_word.csv"
    with topic_word_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "topic_id",
                "token_id",
                "token",
                "document_frequency",
                "weight",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "topic_id": 0,
                    "token_id": 0,
                    "token": "diag_control_density__level_high",
                    "document_frequency": 2,
                    "weight": "0.6",
                },
                {
                    "topic_id": 0,
                    "token_id": 1,
                    "token": "fact_error_rate__level_low",
                    "document_frequency": 2,
                    "weight": "0.4",
                },
            ]
        )

    result = LdaTopicInterpreter(
        LdaTopicInterpreterConfig(model_name="LDA_full")
    ).interpret_from_topic_word(topic_word_path, tmp_path / "reports")

    rows = _read_csv(result.interpretation_csv_path)
    assert "control_density" in rows[0]["source_features"]
    assert "error_rate" in rows[0]["source_features"]
    assert "diag_control_density" not in rows[0]["source_features"]
