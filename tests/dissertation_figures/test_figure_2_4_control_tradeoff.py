"""Тесты генератора рисунка 2.4."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from manual_coding_sim.dissertation_figures.figure_2_4_control_tradeoff import (
    CONTROL_BRANCHES,
    EXPECTED_CODES,
    EXPECTED_DIRECTIONS,
    FILE_STEM,
    FORMULA_QUALITY,
    FORMULA_RESIDUAL,
    FORMULA_TIME,
    OUTPUT_DIR,
    generate,
    main,
    validate_control_tradeoff,
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG из заголовка без внешних библиотек."""

    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def test_tradeoff_contains_accuracy_time_and_effort_branches() -> None:
    """Схема должна содержать три обязательные ветви влияния контроля."""

    validate_control_tradeoff()

    assert tuple(branch.code for branch in CONTROL_BRANCHES) == EXPECTED_CODES
    assert len(CONTROL_BRANCHES) == 3


def test_tradeoff_has_one_benefit_and_two_costs() -> None:
    """Контроль должен давать один положительный и два затратных эффекта."""

    directions = tuple(branch.direction for branch in CONTROL_BRANCHES)

    assert directions == EXPECTED_DIRECTIONS
    assert directions.count("benefit") == 1
    assert directions.count("cost") == 2


def test_each_branch_has_three_causal_links_and_criterion() -> None:
    """Каждая ветвь должна иметь механизм и частный критерий качества."""

    assert all(len(branch.mechanism) == 3 for branch in CONTROL_BRANCHES)
    assert all(branch.criterion.strip() for branch in CONTROL_BRANCHES)

    criteria = " ".join(branch.criterion for branch in CONTROL_BRANCHES)
    assert "q_acc" in criteria
    assert "q_res" in criteria
    assert "q_time" in criteria
    assert "q_effort" in criteria


def test_validation_rejects_missing_effort_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Проверка должна отклонять схему без трудоёмкостной ветви."""

    import manual_coding_sim.dissertation_figures.figure_2_4_control_tradeoff as module

    monkeypatch.setattr(
        module,
        "CONTROL_BRANCHES",
        tuple(branch for branch in module.CONTROL_BRANCHES if branch.code != "effort"),
    )

    with pytest.raises(ValueError, match="трудоёмкости"):
        module.validate_control_tradeoff()


def test_formulas_fix_residual_error_time_and_integral_quality() -> None:
    """Формульные подписи должны отражать три ключевые зависимости."""

    assert "P(D_j)" in FORMULA_RESIDUAL
    assert "P(C_j)" in FORMULA_RESIDUAL
    assert "Σ t_ctrl,r" in FORMULA_TIME
    assert "Σ w_m" in FORMULA_QUALITY


def test_generate_creates_png_and_svg(tmp_path: Path) -> None:
    """Генератор должен создать оба обязательных формата рисунка."""

    result = generate(project_root=tmp_path, dpi=300)

    expected_dir = tmp_path / OUTPUT_DIR
    assert result.png_path == expected_dir / f"{FILE_STEM}.png"
    assert result.svg_path == expected_dir / f"{FILE_STEM}.svg"
    assert result.png_path.is_file()
    assert result.svg_path.is_file()
    assert result.png_path.stat().st_size > 150_000
    assert result.svg_path.stat().st_size > 20_000


def test_png_has_dissertation_ready_resolution(tmp_path: Path) -> None:
    """PNG должен иметь разрешение, достаточное для печатного документа."""

    result = generate(project_root=tmp_path, dpi=300)
    width, height = _read_png_dimensions(result.png_path)

    assert width >= 4000
    assert height >= 2200


def test_svg_contains_branches_formulas_and_methodical_conclusion(
    tmp_path: Path,
) -> None:
    """SVG должен сохранять ветви, формулы и итоговый методический вывод."""

    result = generate(project_root=tmp_path, dpi=300)
    svg = result.svg_path.read_text(encoding="utf-8")

    for text in (
        "Контрольные процедуры",
        "Снижение остаточной ошибки",
        "Дополнительное время",
        "Дополнительная трудоёмкость",
        "q_acc",
        "q_res",
        "q_time",
        "q_effort",
        "P(D_j)",
        "P(C_j)",
        "T_ctrl",
        "Q(A)",
        "Интегральный результат",
        "Методический вывод",
        "компромиссом",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg or "<rect" in svg


def test_cli_reports_both_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI должен завершаться успешно и вывести пути PNG и SVG."""

    exit_code = main(["--project-root", str(tmp_path), "--dpi", "300"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Рисунок 2.4 успешно сформирован" in output
    assert ".png" in output
    assert ".svg" in output
