"""Локальные тесты генератора рисунка 4.5."""

from __future__ import annotations

import csv
import json
import struct
from pathlib import Path

import pytest

from manual_coding_sim.dissertation_figures.figure_4_5_top_tokens_by_topic import (
    FILE_STEM,
    TopicToken,
    generate,
    load_topic_interpretations,
    load_topic_tokens,
    select_top_tokens,
    token_to_russian_label,
    validate_topic_tokens,
)


def _write_reference_inputs(root: Path) -> tuple[Path, Path]:
    """Создать минимальные согласованные входные отчёты трёх тем."""

    report_dir = root / "reports" / "chapter4"
    report_dir.mkdir(parents=True, exist_ok=True)
    topic_word = report_dir / "topic_word.csv"
    with topic_word.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "topic_id",
                "token_id",
                "token",
                "document_frequency",
                "weight",
            ],
        )
        writer.writeheader()
        tokens = [
            "prior_total_nominal_time__level_high",
            "prior_attention_deficit__level_high",
            "prior_condition_time_pressure__level_low",
        ]
        for topic_id in range(3):
            weights = [0.5, 0.3, 0.2]
            for token_id, (token, weight) in enumerate(zip(tokens, weights, strict=True)):
                writer.writerow(
                    {
                        "topic_id": topic_id,
                        "token_id": token_id,
                        "token": token,
                        "document_frequency": 50 + token_id,
                        "weight": weight,
                    }
                )

    interpretation = report_dir / "topic_interpretation.json"
    interpretation.write_text(
        json.dumps(
            {
                "topics": [
                    {
                        "topic_id": topic_id,
                        "suggested_factor_name": f"Латентный фактор {topic_id}",
                        "dominant_token": "prior_total_nominal_time__level_high",
                        "top_weight_sum": 1.0,
                    }
                    for topic_id in range(3)
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return topic_word, interpretation


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG из заголовка IHDR."""

    with path.open("rb") as stream:
        stream.read(16)
        return struct.unpack(">II", stream.read(8))


def test_load_topic_tokens_reads_three_topics(tmp_path: Path) -> None:
    """Загрузчик должен читать три темы на общем словаре."""

    topic_word, _ = _write_reference_inputs(tmp_path)
    records = load_topic_tokens(topic_word)
    assert len(records) == 9
    assert sorted({record.topic_id for record in records}) == [0, 1, 2]


def test_validate_rejects_incomplete_topic_sequence() -> None:
    """Пропуск темы должен отклоняться."""

    records = (
        TopicToken(0, 0, "a", 10, 1.0),
        TopicToken(2, 0, "a", 10, 1.0),
    )
    with pytest.raises(ValueError, match="последовательность"):
        validate_topic_tokens(records)


def test_validate_rejects_different_vocabularies() -> None:
    """Темы на разных словарях должны отклоняться."""

    records = (
        TopicToken(0, 0, "a", 10, 1.0),
        TopicToken(1, 0, "b", 10, 1.0),
        TopicToken(2, 0, "a", 10, 1.0),
    )
    with pytest.raises(ValueError, match="одном словаре"):
        validate_topic_tokens(records)


def test_validate_rejects_non_normalized_weights() -> None:
    """Сумма весов каждой темы должна быть равна единице."""

    records = tuple(
        TopicToken(topic_id, 0, "a", 10, 0.8) for topic_id in range(3)
    )
    with pytest.raises(ValueError, match="равна единице"):
        validate_topic_tokens(records)


def test_select_top_tokens_orders_by_weight(tmp_path: Path) -> None:
    """Отбор должен возвращать токены по убыванию веса."""

    topic_word, _ = _write_reference_inputs(tmp_path)
    selected = select_top_tokens(load_topic_tokens(topic_word), top_n=3)
    assert [item.weight for item in selected[0]] == [0.5, 0.3, 0.2]


def test_token_translation_is_russian() -> None:
    """Технический токен должен преобразовываться в русскую подпись."""

    label = token_to_russian_label("prior_attention_deficit__level_high")
    assert label == "Дефицит внимания — высокий уровень"


def test_load_topic_interpretations_reads_all_topics(tmp_path: Path) -> None:
    """Интерпретации должны загружаться для тем 0–2."""

    _, interpretation = _write_reference_inputs(tmp_path)
    result = load_topic_interpretations(interpretation)
    assert [item.topic_id for item in result] == [0, 1, 2]


def test_generate_creates_expected_paths(tmp_path: Path) -> None:
    """Генератор должен создать PNG и SVG с установленным именем."""

    _write_reference_inputs(tmp_path)
    result = generate(project_root=tmp_path, top_n=3, dpi=300)
    assert result.png_path.name == f"{FILE_STEM}.png"
    assert result.svg_path.name == f"{FILE_STEM}.svg"
    assert result.png_path.is_file()
    assert result.svg_path.is_file()


def test_png_and_svg_are_dissertation_ready(tmp_path: Path) -> None:
    """PNG должен быть крупным, а SVG — содержать редактируемые подписи."""

    _write_reference_inputs(tmp_path)
    result = generate(project_root=tmp_path, top_n=3, dpi=300)
    width, height = _read_png_dimensions(result.png_path)
    svg = result.svg_path.read_text(encoding="utf-8")
    assert width >= 4500
    assert height >= 2300
    for text in (
        "Наиболее весомые токены",
        "Тема 0 — процедурная трудоёмкость",
        "Тема 1 — операционный риск",
        "Тема 2 — благоприятные условия",
        "Вес токена в распределении темы",
        "априорные латентные факторы",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg


def test_cli_main_generates_files(tmp_path: Path) -> None:
    """CLI должен завершаться успешно и создавать оба формата."""

    _write_reference_inputs(tmp_path)
    from manual_coding_sim.dissertation_figures.figure_4_5_top_tokens_by_topic import main

    exit_code = main(["--project-root", str(tmp_path), "--top-n", "3", "--dpi", "300"])
    assert exit_code == 0
    output_dir = tmp_path / "reports" / "chapter4" / "figures"
    assert (output_dir / f"{FILE_STEM}.png").is_file()
    assert (output_dir / f"{FILE_STEM}.svg").is_file()
