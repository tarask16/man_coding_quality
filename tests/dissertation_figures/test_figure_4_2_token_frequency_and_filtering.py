"""Тесты генератора рисунка 4.2."""

from __future__ import annotations

import json
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from manual_coding_sim.dissertation_figures.figure_4_2_token_frequency_and_filtering import (
    DEFAULT_DICTIONARY_PATH,
    DEFAULT_METADATA_PATH,
    FILE_STEM,
    OUTPUT_DIR,
    TokenFrequency,
    generate,
    load_filtering_parameters,
    load_token_frequencies,
    main,
    summarize_token_frequencies,
    validate_token_frequency_data,
)


REFERENCE_RECORDS: tuple[TokenFrequency, ...] = tuple(
    TokenFrequency(
        token_id=index,
        token=f"prior_feature_{index:03d}__level_mid",
        document_frequency=23 + index % 68,
    )
    for index in range(96)
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать ширину и высоту PNG по заголовку файла."""

    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def _write_reference_inputs(project_root: Path) -> tuple[Path, Path]:
    """Записать компактные входные JSON-файлы для тестов генератора."""

    dictionary_path = project_root / DEFAULT_DICTIONARY_PATH
    metadata_path = project_root / DEFAULT_METADATA_PATH
    dictionary_path.parent.mkdir(parents=True, exist_ok=True)

    dictionary_path.write_text(
        json.dumps(
            {
                "token_count": len(REFERENCE_RECORDS),
                "tokens": [
                    {
                        "token_id": record.token_id,
                        "token": record.token,
                        "document_frequency": record.document_frequency,
                    }
                    for record in REFERENCE_RECORDS
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "document_count": 150,
                "df_min": 2,
                "df_max_ratio": 0.95,
                "dictionary_token_count": 96,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return dictionary_path, metadata_path


def test_constants_point_to_chapter4_artifacts() -> None:
    """Константы должны задавать правильные входы и выходы главы 4."""

    assert OUTPUT_DIR == Path("reports/chapter4/figures")
    assert FILE_STEM == "figure_4_2_token_frequency_and_filtering"
    assert DEFAULT_DICTIONARY_PATH.name == "dictionary.json"
    assert DEFAULT_METADATA_PATH.name == "corpus_metadata.json"


def test_loaders_read_dictionary_and_filtering_parameters(tmp_path: Path) -> None:
    """Загрузчики должны корректно прочитать словарь и параметры фильтра."""

    dictionary_path, metadata_path = _write_reference_inputs(tmp_path)
    records = load_token_frequencies(dictionary_path)
    document_count, df_min, df_max_ratio = load_filtering_parameters(metadata_path)

    assert len(records) == 96
    assert records[0].token_id == 0
    assert records[-1].document_frequency == REFERENCE_RECORDS[-1].document_frequency
    assert (document_count, df_min, df_max_ratio) == (150, 2, 0.95)


def test_summary_describes_reference_dictionary() -> None:
    """Сводка должна отражать размер и диапазон документных частот."""

    summary = summarize_token_frequencies(
        REFERENCE_RECORDS,
        document_count=150,
        df_min=2,
        df_max_ratio=0.95,
    )

    assert summary.dictionary_token_count == 96
    assert summary.min_document_frequency == 23
    assert summary.max_document_frequency == 90
    assert summary.upper_document_frequency_limit == pytest.approx(142.5)
    assert 23.0 <= summary.mean_document_frequency <= 90.0


def test_validator_rejects_duplicate_token_id() -> None:
    """Повторяющийся token_id должен приводить к контролируемой ошибке."""

    malformed = list(REFERENCE_RECORDS)
    malformed[1] = replace(malformed[1], token_id=malformed[0].token_id)
    with pytest.raises(ValueError, match="token_id"):
        validate_token_frequency_data(
            malformed,
            document_count=150,
            df_min=2,
            df_max_ratio=0.95,
        )


def test_validator_rejects_frequency_below_df_min() -> None:
    """Итоговый токен ниже df_min должен быть отклонён."""

    malformed = list(REFERENCE_RECORDS)
    malformed[0] = replace(malformed[0], document_frequency=1)
    with pytest.raises(ValueError, match="ниже границы df_min"):
        validate_token_frequency_data(
            malformed,
            document_count=150,
            df_min=2,
            df_max_ratio=0.95,
        )


def test_validator_rejects_frequency_above_df_max_ratio() -> None:
    """Итоговый токен выше верхней границы должен быть отклонён."""

    malformed = list(REFERENCE_RECORDS)
    malformed[0] = replace(malformed[0], document_frequency=143)
    with pytest.raises(ValueError, match="выше границы df_max_ratio"):
        validate_token_frequency_data(
            malformed,
            document_count=150,
            df_min=2,
            df_max_ratio=0.95,
        )


def test_loader_rejects_declared_token_count_mismatch(tmp_path: Path) -> None:
    """Несовпадение token_count и списка tokens должно обнаруживаться."""

    dictionary_path, _ = _write_reference_inputs(tmp_path)
    payload = json.loads(dictionary_path.read_text(encoding="utf-8"))
    payload["token_count"] = 95
    dictionary_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="token_count"):
        load_token_frequencies(dictionary_path)


def test_generate_creates_png_and_svg(tmp_path: Path) -> None:
    """Генератор должен создать оба формата в каталоге главы 4."""

    _write_reference_inputs(tmp_path)
    result = generate(project_root=tmp_path, dpi=300)
    expected_dir = tmp_path / OUTPUT_DIR

    assert result.png_path == expected_dir / f"{FILE_STEM}.png"
    assert result.svg_path == expected_dir / f"{FILE_STEM}.svg"
    assert result.png_path.is_file()
    assert result.svg_path.is_file()
    assert result.png_path.stat().st_size > 120_000
    assert result.svg_path.stat().st_size > 35_000


def test_png_and_svg_are_dissertation_ready(tmp_path: Path) -> None:
    """PNG должен быть крупным, а SVG — содержать редактируемые подписи."""

    _write_reference_inputs(tmp_path)
    result = generate(project_root=tmp_path, dpi=300)
    width, height = _read_png_dimensions(result.png_path)
    svg = result.svg_path.read_text(encoding="utf-8")

    assert width >= 3800
    assert height >= 2100
    for text in (
        "Документная частота токенов",
        "df_min = 2",
        "df_max_ratio = 0.95",
        "V = 96",
        "Ранжированное распределение",
        "Итог частотной фильтрации",
        "отброшенные",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg


def test_cli_reports_both_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI должен сообщить об успешной генерации и вывести оба пути."""

    _write_reference_inputs(tmp_path)
    exit_code = main(["--project-root", str(tmp_path), "--dpi", "300"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Рисунок 4.2 успешно сформирован" in output
    assert ".png" in output
    assert ".svg" in output
