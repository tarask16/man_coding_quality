"""Тесты генератора рисунка 2.5."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from manual_coding_sim.dissertation_figures.figure_2_5_information_sets_and_tokenization import (
    EXPECTED_SET_CODES,
    EXPECTED_STEP_CODES,
    FILE_STEM,
    FORBIDDEN_SOURCE_CODES,
    FORMULA_CORPUS,
    INFORMATION_SETS,
    LEAKAGE_RULE,
    OUTPUT_DIR,
    TRANSFORMATION_STEPS,
    generate,
    main,
    validate_information_flow,
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG из заголовка без внешних библиотек."""

    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def test_information_sets_are_complete_and_ordered() -> None:
    """Схема должна содержать X_prior, X_fact и Y_fact."""

    validate_information_flow()

    assert tuple(item.code for item in INFORMATION_SETS) == EXPECTED_SET_CODES
    assert tuple(item.symbol for item in INFORMATION_SETS) == (
        "X_prior",
        "X_fact",
        "Y_fact",
    )


def test_only_prior_features_are_allowed_for_document() -> None:
    """Только X_prior может быть входом априорного документа."""

    allowed = tuple(item.code for item in INFORMATION_SETS if item.allowed_for_prior)
    forbidden = tuple(
        item.code for item in INFORMATION_SETS if not item.allowed_for_prior
    )

    assert allowed == ("prior",)
    assert forbidden == FORBIDDEN_SOURCE_CODES


def test_transformation_pipeline_is_complete() -> None:
    """Конвейер должен включать дискретизацию, токенизацию и документ."""

    assert tuple(step.code for step in TRANSFORMATION_STEPS) == EXPECTED_STEP_CODES
    assert all(len(step.description) == 3 for step in TRANSFORMATION_STEPS)
    assert all(step.example.strip() for step in TRANSFORMATION_STEPS)


def test_validation_rejects_fact_features_as_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Проверка должна отклонять разрешение X_fact для LDA_prior."""

    import manual_coding_sim.dissertation_figures.figure_2_5_information_sets_and_tokenization as module

    modified = tuple(
        module.InformationSet(
            code=item.code,
            symbol=item.symbol,
            title=item.title,
            examples=item.examples,
            allowed_for_prior=(item.code in {"prior", "fact"}),
        )
        for item in module.INFORMATION_SETS
    )
    monkeypatch.setattr(module, "INFORMATION_SETS", modified)

    with pytest.raises(ValueError, match="только признаки X_prior"):
        module.validate_information_flow()


def test_formulas_fix_corpus_and_leakage_rule() -> None:
    """Формульные подписи должны фиксировать корпус и запрет утечки."""

    assert "X_prior" in FORMULA_CORPUS
    assert "Discretize" in FORMULA_CORPUS
    assert "Tokenize" in FORMULA_CORPUS
    assert "d_i" in FORMULA_CORPUS
    assert "X_fact" in LEAKAGE_RULE
    assert "Y_fact" in LEAKAGE_RULE
    assert "↛" in LEAKAGE_RULE


def test_generate_creates_png_and_svg(tmp_path: Path) -> None:
    """Генератор должен создать оба обязательных формата рисунка."""

    result = generate(project_root=tmp_path, dpi=300)

    expected_dir = tmp_path / OUTPUT_DIR
    assert result.png_path == expected_dir / f"{FILE_STEM}.png"
    assert result.svg_path == expected_dir / f"{FILE_STEM}.svg"
    assert result.png_path.is_file()
    assert result.svg_path.is_file()
    assert result.png_path.stat().st_size > 150_000
    assert result.svg_path.stat().st_size > 18_000


def test_png_has_dissertation_ready_resolution(tmp_path: Path) -> None:
    """PNG должен иметь разрешение, достаточное для печатного документа."""

    result = generate(project_root=tmp_path, dpi=300)
    width, height = _read_png_dimensions(result.png_path)

    assert width >= 4200
    assert height >= 2300


def test_svg_contains_sets_pipeline_and_forbidden_links(tmp_path: Path) -> None:
    """SVG должен сохранять множества, этапы и запреты методической утечки."""

    result = generate(project_root=tmp_path, dpi=300)
    svg = result.svg_path.read_text(encoding="utf-8")

    for text in (
        "X_prior",
        "X_fact",
        "Y_fact",
        "Априорные признаки",
        "Фактические признаки",
        "Фактические показатели",
        "Дискретизация",
        "Токенизация",
        "Документ сценария",
        "LeakageGuard",
        "запрещено",
        "D_prior",
        "scenario_id",
        "protocol_id",
        "внешняя проверка",
    ):
        assert text in svg
    assert svg.count("запрещено") >= 2
    assert "<text" in svg
    assert "<path" in svg or "<rect" in svg


def test_cli_reports_both_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI должен завершаться успешно и вывести пути PNG и SVG."""

    exit_code = main(["--project-root", str(tmp_path), "--dpi", "300"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Рисунок 2.5 успешно сформирован" in output
    assert ".png" in output
    assert ".svg" in output
