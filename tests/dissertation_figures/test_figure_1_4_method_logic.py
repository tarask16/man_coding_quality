"""Тесты генератора рисунка 1.4."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from manual_coding_sim.dissertation_figures.figure_1_4_method_logic import (
    FILE_STEM,
    OUTPUT_DIR,
    PIPELINE_STEPS,
    STEP_DETAILS,
    VALIDATION_INPUTS,
    generate,
    main,
    validate_method_logic,
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать ширину и высоту PNG из заголовка без внешних библиотек."""

    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def test_method_logic_contains_six_ordered_steps() -> None:
    """Схема должна фиксировать шесть последовательных этапов метода."""

    validate_method_logic()

    assert len(PIPELINE_STEPS) == 6
    assert len(STEP_DETAILS) == len(PIPELINE_STEPS)
    assert "Формальная модель" in PIPELINE_STEPS[0]
    assert "Компьютерное" in PIPELINE_STEPS[1]
    assert "X_prior" in PIPELINE_STEPS[2]
    assert "LDA_prior" in PIPELINE_STEPS[3]
    assert "Q_pred" in PIPELINE_STEPS[4]
    assert "Q_fact" in PIPELINE_STEPS[5]


def test_validation_inputs_are_external_to_prediction_contour() -> None:
    """Фактические файлы должны быть обозначены только как проверочные входы."""

    assert VALIDATION_INPUTS == ("quality_targets.csv", "fact_features.csv")
    assert all(
        input_name not in " ".join(PIPELINE_STEPS[:-1])
        for input_name in VALIDATION_INPUTS
    )


def test_validation_rejects_broken_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Проверка должна отклонять схему без обязательного этапа LDA_prior."""

    import manual_coding_sim.dissertation_figures.figure_1_4_method_logic as module

    broken_steps = list(module.PIPELINE_STEPS)
    broken_steps[3] = "Латентный профиль"
    monkeypatch.setattr(module, "PIPELINE_STEPS", tuple(broken_steps))

    with pytest.raises(ValueError, match="LDA_prior"):
        module.validate_method_logic()


def test_generate_creates_png_and_svg(tmp_path: Path) -> None:
    """Генератор должен создать оба обязательных формата рисунка."""

    result = generate(project_root=tmp_path, dpi=300)

    expected_dir = tmp_path / OUTPUT_DIR
    assert result.png_path == expected_dir / f"{FILE_STEM}.png"
    assert result.svg_path == expected_dir / f"{FILE_STEM}.svg"
    assert result.png_path.is_file()
    assert result.svg_path.is_file()
    assert result.png_path.stat().st_size > 100_000
    assert result.svg_path.stat().st_size > 15_000


def test_png_has_dissertation_ready_resolution(tmp_path: Path) -> None:
    """PNG должен иметь разрешение, достаточное для вставки в A4-документ."""

    result = generate(project_root=tmp_path, dpi=300)
    width, height = _read_png_dimensions(result.png_path)

    assert width >= 3400
    assert height >= 1800


def test_svg_contains_pipeline_and_leakage_restriction(tmp_path: Path) -> None:
    """SVG должен сохранять этапы метода и запрет методической утечки."""

    result = generate(project_root=tmp_path, dpi=300)
    svg = result.svg_path.read_text(encoding="utf-8")

    assert "Формальная модель" in svg
    assert "Компьютерное" in svg
    assert "X_prior" in svg
    assert "LDA_prior" in svg
    assert "Q_pred" in svg
    assert "Q_fact" in svg
    assert "quality_targets.csv" in svg
    assert "fact_features.csv" in svg
    assert "Запрещено использовать фактические признаки" in svg
    assert "априорный индекс относительного качества" in svg
    assert "<text" in svg
    assert "<path" in svg or "<rect" in svg


def test_cli_reports_both_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI должен завершаться успешно и выводить пути PNG и SVG."""

    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--dpi",
            "300",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Рисунок 1.4 успешно сформирован" in output
    assert ".png" in output
    assert ".svg" in output
