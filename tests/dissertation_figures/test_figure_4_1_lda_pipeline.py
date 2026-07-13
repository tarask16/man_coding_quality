"""Тесты генератора рисунка 4.1."""

from __future__ import annotations

import struct
from dataclasses import replace
from pathlib import Path

import pytest

from manual_coding_sim.dissertation_figures.figure_4_1_lda_pipeline import (
    CONNECTIONS,
    EXPECTED_STAGE_CODES,
    FILE_STEM,
    FORBIDDEN_INPUTS,
    MODEL_FACTS,
    OUTPUT_ARTIFACTS,
    OUTPUT_DIR,
    STAGES,
    PipelineConnection,
    generate,
    main,
    validate_lda_pipeline,
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать ширину и высоту PNG по заголовку файла."""

    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def test_constants_describe_complete_lda_pipeline() -> None:
    """Константы должны описывать полный шестишаговый LDA-конвейер."""

    assert tuple(stage.code for stage in STAGES) == EXPECTED_STAGE_CODES
    assert tuple(stage.number for stage in STAGES) == (1, 2, 3, 4, 5, 6)
    assert "theta_prior.csv" in OUTPUT_ARTIFACTS
    assert "topic_word.csv" in OUTPUT_ARTIFACTS
    assert "fact_features.csv" in FORBIDDEN_INPUTS
    assert "selected_k = 3" in MODEL_FACTS


def test_validator_accepts_reference_pipeline() -> None:
    """Эталонный LDA-конвейер должен проходить структурную проверку."""

    validate_lda_pipeline()


def test_validator_rejects_missing_main_connection() -> None:
    """Разрыв основного пути должен приводить к контролируемой ошибке."""

    broken = tuple(
        item
        for item in CONNECTIONS
        if not (item.source == "corpus_prior" and item.target == "dictionary")
    )
    with pytest.raises(ValueError, match="corpus_prior → dictionary"):
        validate_lda_pipeline(connections=broken)


def test_validator_requires_both_lda_outputs() -> None:
    """Модель должна формировать оба распределения: theta_prior и topic_word."""

    broken = tuple(item for item in CONNECTIONS if item.target != "topic_word")
    with pytest.raises(ValueError, match="theta_prior и topic_word"):
        validate_lda_pipeline(connections=broken)


def test_validator_rejects_fact_data_connection() -> None:
    """Фактические данные не должны поступать в основной LDA-конвейер."""

    contaminated = (*CONNECTIONS, PipelineConnection("fact_features", "lda_prior", "утечка"))
    with pytest.raises(ValueError, match="запрещены"):
        validate_lda_pipeline(connections=contaminated)


def test_validator_rejects_invalid_numbering() -> None:
    """Нарушение последовательной нумерации должно обнаруживаться."""

    malformed = list(STAGES)
    malformed[3] = replace(malformed[3], number=9)
    with pytest.raises(ValueError, match="Нумерация"):
        validate_lda_pipeline(stages=tuple(malformed))


def test_connections_preserve_expected_artifact_flow() -> None:
    """Связи должны вести от prior_features к модели и двум выходам."""

    pairs = {(item.source, item.target) for item in CONNECTIONS}
    assert ("prior_features", "discretization") in pairs
    assert ("discretization", "token_map") in pairs
    assert ("token_map", "corpus_prior") in pairs
    assert ("corpus_prior", "dictionary") in pairs
    assert ("dictionary", "lda_prior") in pairs
    assert ("lda_prior", "theta_prior") in pairs
    assert ("lda_prior", "topic_word") in pairs


def test_generate_creates_png_and_svg(tmp_path: Path) -> None:
    """Генератор должен создать оба формата в каталоге главы 4."""

    result = generate(project_root=tmp_path, dpi=300)
    expected_dir = tmp_path / OUTPUT_DIR

    assert result.png_path == expected_dir / f"{FILE_STEM}.png"
    assert result.svg_path == expected_dir / f"{FILE_STEM}.svg"
    assert result.png_path.is_file()
    assert result.svg_path.is_file()
    assert result.png_path.stat().st_size > 140_000
    assert result.svg_path.stat().st_size > 30_000


def test_png_and_svg_are_dissertation_ready(tmp_path: Path) -> None:
    """PNG должен быть крупным, а SVG — содержать редактируемый текст."""

    result = generate(project_root=tmp_path, dpi=300)
    width, height = _read_png_dimensions(result.png_path)
    svg = result.svg_path.read_text(encoding="utf-8")

    assert width >= 4200
    assert height >= 2200
    for text in (
        "Априорные признаки",
        "Дискретизация",
        "Токенизация",
        "Априорный корпус",
        "Словарь и матрица",
        "Обучение LDA_prior",
        "theta_prior.csv",
        "topic_word.csv",
        "LeakageGuard",
        "quality_targets.csv",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg


def test_cli_reports_both_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI должен сообщить об успешной генерации и вывести оба пути."""

    exit_code = main(["--project-root", str(tmp_path), "--dpi", "300"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Рисунок 4.1 успешно сформирован" in output
    assert ".png" in output
    assert ".svg" in output
