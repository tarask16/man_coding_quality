"""Локальные тесты генератора рисунка 5.1."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from manual_coding_sim.dissertation_figures.figure_5_1_prediction_pipeline import (
    ACCEPTANCE_CHECK_COUNT,
    CONNECTIONS,
    EXPECTED_ROW_COUNT,
    EXPECTED_STAGE_CODES,
    FILE_STEM,
    FORBIDDEN_INPUTS,
    OUTPUT_ARTIFACTS,
    STAGES,
    PredictionConnection,
    generate,
    validate_prediction_pipeline,
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG из заголовка IHDR."""

    with path.open("rb") as stream:
        stream.read(16)
        return struct.unpack(">II", stream.read(8))


def test_stage_order_matches_roadmap() -> None:
    """Последовательность этапов должна соответствовать roadmap главы 5."""

    assert tuple(stage.code for stage in STAGES) == EXPECTED_STAGE_CODES
    assert tuple(stage.number for stage in STAGES) == tuple(range(1, 9))


def test_pipeline_contains_required_output_artifacts() -> None:
    """В конвейере должны быть все основные выходные артефакты главы 5."""

    artifacts = {stage.artifact for stage in STAGES}
    assert set(OUTPUT_ARTIFACTS).issubset(artifacts)


def test_pipeline_uses_expected_acceptance_facts() -> None:
    """Схема должна фиксировать число проверок и строк итоговой приемки."""

    assert ACCEPTANCE_CHECK_COUNT == 18
    assert EXPECTED_ROW_COUNT == 150
    acceptance = next(stage for stage in STAGES if stage.code == "acceptance")
    assert "18 обязательных проверок" in acceptance.details
    assert "все выходы: 150 строк" in acceptance.details


def test_forbidden_inputs_are_explicitly_listed() -> None:
    """Фактические и диагностические таблицы должны быть явно запрещены."""

    assert "fact_features.csv" in FORBIDDEN_INPUTS
    assert "quality_targets.csv" in FORBIDDEN_INPUTS
    assert "theta_diag.csv" in FORBIDDEN_INPUTS
    assert "theta_full.csv" in FORBIDDEN_INPUTS


def test_validate_accepts_reference_pipeline() -> None:
    """Эталонный конвейер должен проходить структурную проверку."""

    validate_prediction_pipeline(STAGES, CONNECTIONS)


def test_validate_rejects_missing_stage() -> None:
    """Конвейер без одного этапа должен отклоняться."""

    with pytest.raises(ValueError, match="восемь этапов"):
        validate_prediction_pipeline(STAGES[:-1], CONNECTIONS)


def test_validate_rejects_broken_connection() -> None:
    """Разрыв последовательного пути должен отклоняться."""

    broken = tuple(
        connection
        for connection in CONNECTIONS
        if not (
            connection.source == "integral_quality"
            and connection.target == "uncertainty"
        )
    )
    with pytest.raises(ValueError, match="Отсутствуют обязательные связи"):
        validate_prediction_pipeline(STAGES, broken)


def test_validate_rejects_forbidden_data_connection() -> None:
    """Попытка включить фактические данные в расчетный контур должна отклоняться."""

    invalid = CONNECTIONS + (
        PredictionConnection("quality_targets", "integral_quality", "запрещённая связь"),
    )
    with pytest.raises(ValueError, match="запрещены"):
        validate_prediction_pipeline(STAGES, invalid)


def test_generate_creates_dissertation_ready_png_and_svg(tmp_path: Path) -> None:
    """PNG должен быть крупным, а SVG — содержать редактируемые подписи."""

    result = generate(project_root=tmp_path, dpi=300)
    width, height = _read_png_dimensions(result.png_path)
    svg = result.svg_path.read_text(encoding="utf-8")

    assert result.png_path.name == f"{FILE_STEM}.png"
    assert result.svg_path.name == f"{FILE_STEM}.svg"
    assert width >= 4000
    assert height >= 2200
    for text in (
        "Конвейер построения априорной интегральной оценки качества",
        "Загрузка априорных входов",
        "LeakageGuard",
        "Направленная нормировка",
        "Латентная компонента",
        "Частные критерии",
        "Интегральный индекс",
        "Диагностика неопределённости",
        "Финальная приёмка",
        "fact_features.csv",
        "quality_targets.csv",
        "Q_pred — априорный сравнительный индекс",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg


def test_cli_main_generates_files(tmp_path: Path) -> None:
    """CLI должен завершаться успешно и создавать оба формата."""

    from manual_coding_sim.dissertation_figures.figure_5_1_prediction_pipeline import main

    exit_code = main(["--project-root", str(tmp_path), "--dpi", "300"])
    assert exit_code == 0
    output_dir = tmp_path / "reports" / "chapter5" / "figures"
    assert (output_dir / f"{FILE_STEM}.png").is_file()
    assert (output_dir / f"{FILE_STEM}.svg").is_file()
