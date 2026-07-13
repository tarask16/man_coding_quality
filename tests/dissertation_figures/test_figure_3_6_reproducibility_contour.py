"""Тесты генератора рисунка 3.6."""

from __future__ import annotations

import struct
from dataclasses import replace
from pathlib import Path

import pytest

from manual_coding_sim.dissertation_figures.figure_3_6_reproducibility_contour import (
    ACCEPTANCE_CRITERIA,
    CONNECTIONS,
    EXPECTED_STAGE_CODES,
    FILE_STEM,
    OUTPUT_DIR,
    REQUIRED_ARTIFACTS,
    STAGES,
    StageConnection,
    generate,
    main,
    validate_reproducibility_contour,
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать ширину и высоту PNG по заголовку файла."""

    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def test_constants_describe_six_stage_contour() -> None:
    """Константы должны описывать полный шестишаговый контур."""

    assert tuple(stage.code for stage in STAGES) == EXPECTED_STAGE_CODES
    assert tuple(stage.number for stage in STAGES) == (1, 2, 3, 4, 5, 6)
    assert len(ACCEPTANCE_CRITERIA) == 4
    assert "checksums.json" in REQUIRED_ARTIFACTS
    assert "reproducibility_report.json" in REQUIRED_ARTIFACTS


def test_validator_accepts_reference_contour() -> None:
    """Эталонная схема должна проходить структурную проверку."""

    validate_reproducibility_contour()


def test_validator_rejects_missing_main_connection() -> None:
    """Разрыв основного пути должен приводить к контролируемой ошибке."""

    broken = tuple(
        item
        for item in CONNECTIONS
        if not (item.source == "artifacts" and item.target == "checksums")
    )
    with pytest.raises(ValueError, match="artifacts → checksums"):
        validate_reproducibility_contour(connections=broken)


def test_validator_requires_independent_rerun() -> None:
    """Контур без независимого повтора не является полным."""

    without_rerun = tuple(item for item in CONNECTIONS if item.kind != "rerun")
    with pytest.raises(ValueError, match="повторный запуск"):
        validate_reproducibility_contour(connections=without_rerun)


def test_validator_rejects_invalid_numbering() -> None:
    """Нарушение последовательной нумерации должно обнаруживаться."""

    malformed = list(STAGES)
    malformed[2] = replace(malformed[2], number=8)
    with pytest.raises(ValueError, match="Нумерация"):
        validate_reproducibility_contour(stages=tuple(malformed))


def test_stage_connections_include_checksums_and_tests() -> None:
    """Основной путь должен проходить через хэши и тесты к отчёту."""

    pairs = {(item.source, item.target) for item in CONNECTIONS}
    assert ("artifacts", "checksums") in pairs
    assert ("checksums", "tests") in pairs
    assert ("tests", "report") in pairs
    assert ("rerun", "checksums") in pairs


def test_generate_creates_png_and_svg(tmp_path: Path) -> None:
    """Генератор должен создать оба формата в каталоге главы 3."""

    result = generate(project_root=tmp_path, dpi=300)
    expected_dir = tmp_path / OUTPUT_DIR

    assert result.png_path == expected_dir / f"{FILE_STEM}.png"
    assert result.svg_path == expected_dir / f"{FILE_STEM}.svg"
    assert result.png_path.is_file()
    assert result.svg_path.is_file()
    assert result.png_path.stat().st_size > 150_000
    assert result.svg_path.stat().st_size > 30_000


def test_png_and_svg_are_dissertation_ready(tmp_path: Path) -> None:
    """PNG должен быть крупным, а SVG — содержать редактируемый текст."""

    result = generate(project_root=tmp_path, dpi=300)
    width, height = _read_png_dimensions(result.png_path)
    svg = result.svg_path.read_text(encoding="utf-8")

    assert width >= 4300
    assert height >= 2300
    for text in (
        "Конфигурация и seed",
        "Управляемый запуск",
        "CSV / JSON-артефакты",
        "Контрольные суммы",
        "Автоматические тесты",
        "Отчёт воспроизводимости",
        "Независимый повторный запуск",
        "SHA-256",
        "внешней валидности",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg


def test_cli_reports_both_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI должен сообщить об успешной генерации и вывести оба пути."""

    exit_code = main(["--project-root", str(tmp_path), "--dpi", "300"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Рисунок 3.6 успешно сформирован" in output
    assert ".png" in output
    assert ".svg" in output
