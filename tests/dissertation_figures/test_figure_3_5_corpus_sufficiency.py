"""Тесты генератора рисунка 3.5."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from manual_coding_sim.dissertation_figures.figure_3_5_corpus_sufficiency import (
    DEFAULT_INPUT_PATH,
    DEFAULT_MINIMUMS,
    DEFAULT_TOKEN_COUNT,
    EXPECTED_SCENARIO_COUNT,
    FILE_STEM,
    OUTPUT_DIR,
    REQUIRED_COLUMNS,
    build_sufficiency_metrics,
    generate,
    load_prior_features,
    main,
)


def _write_test_data(path: Path, *, rows: int = EXPECTED_SCENARIO_COUNT) -> Path:
    """Создать воспроизводимую таблицу априорных признаков."""

    index = np.arange(rows)
    frame = pd.DataFrame(
        {
            "scenario_id": [f"scn_{value:04d}" for value in index],
            "protocol_id": [f"prt_{value:04d}_00" for value in index],
            "prior_mean_complexity": (index % 5) + 1,
            "prior_mean_message_criticality": ((index // 2) % 5) + 1,
            "prior_operator_total_estimated_time": 20.0 + index * 0.51,
            "prior_condition_total_adjusted_time": 23.0 + index * 0.61,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG из заголовка без Pillow."""

    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def test_constants_describe_expected_gate() -> None:
    """Пороговая конфигурация должна соответствовать gate расширенного корпуса."""

    assert DEFAULT_INPUT_PATH == Path("data/processed/prior_features.csv")
    assert DEFAULT_TOKEN_COUNT == 96
    assert DEFAULT_MINIMUMS["documents"] == 100
    assert DEFAULT_MINIMUMS["tokens"] == 30
    assert len(REQUIRED_COLUMNS) == 6


def test_loader_reads_valid_prior_features(tmp_path: Path) -> None:
    """Загрузчик должен прочитать корректную таблицу априорных признаков."""

    path = _write_test_data(tmp_path / "prior_features.csv")
    frame = load_prior_features(path)

    assert frame.shape == (EXPECTED_SCENARIO_COUNT, len(REQUIRED_COLUMNS))
    assert frame["scenario_id"].nunique() == EXPECTED_SCENARIO_COUNT
    assert frame["prior_mean_complexity"].nunique() == 5


def test_loader_rejects_missing_column(tmp_path: Path) -> None:
    """Загрузчик должен отклонить таблицу без обязательного признака."""

    path = _write_test_data(tmp_path / "prior_features.csv")
    frame = pd.read_csv(path).drop(columns=["protocol_id"])
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="protocol_id"):
        load_prior_features(path)


def test_loader_rejects_non_numeric_feature(tmp_path: Path) -> None:
    """Нечисловой ключевой признак должен приводить к контролируемой ошибке."""

    path = _write_test_data(tmp_path / "prior_features.csv")
    frame = pd.read_csv(path)
    frame["prior_mean_complexity"] = frame["prior_mean_complexity"].astype(object)
    frame.loc[0, "prior_mean_complexity"] = "нет данных"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="prior_mean_complexity"):
        load_prior_features(path)


def test_metrics_match_extended_corpus(tmp_path: Path) -> None:
    """Показатели должны отражать фактические значения расширенного корпуса."""

    path = _write_test_data(tmp_path / "prior_features.csv")
    frame = load_prior_features(path)
    metrics = build_sufficiency_metrics(frame, token_count=96)
    by_key = {metric.key: metric for metric in metrics}

    assert len(metrics) == 8
    assert by_key["documents"].actual == 150
    assert by_key["scenarios"].actual == 150
    assert by_key["protocols"].actual == 150
    assert by_key["tokens"].actual == 96
    assert by_key["complexity_levels"].actual == 5
    assert by_key["criticality_levels"].actual == 5
    assert by_key["operator_time_levels"].actual == 150
    assert all(metric.passed for metric in metrics)


def test_metrics_allow_failed_gate_for_diagnostic_display(tmp_path: Path) -> None:
    """Недостаточный корпус должен отображаться как failed, а не скрываться."""

    path = _write_test_data(tmp_path / "prior_features.csv", rows=20)
    frame = load_prior_features(path)
    metrics = build_sufficiency_metrics(frame, token_count=10)
    by_key = {metric.key: metric for metric in metrics}

    assert not by_key["documents"].passed
    assert not by_key["tokens"].passed
    assert by_key["documents"].ratio == pytest.approx(0.2)


def test_metrics_reject_non_positive_token_count(tmp_path: Path) -> None:
    """Число токенов должно быть положительным."""

    path = _write_test_data(tmp_path / "prior_features.csv")
    frame = load_prior_features(path)

    with pytest.raises(ValueError, match="положительным"):
        build_sufficiency_metrics(frame, token_count=0)


def test_generate_creates_png_and_svg(tmp_path: Path) -> None:
    """Генератор должен создать оба формата в каталоге главы 3."""

    source = _write_test_data(tmp_path / DEFAULT_INPUT_PATH)
    result = generate(project_root=tmp_path, input_path=source, dpi=300)

    expected_dir = tmp_path / OUTPUT_DIR
    assert result.png_path == expected_dir / f"{FILE_STEM}.png"
    assert result.svg_path == expected_dir / f"{FILE_STEM}.svg"
    assert result.png_path.is_file()
    assert result.svg_path.is_file()
    assert result.png_path.stat().st_size > 100_000
    assert result.svg_path.stat().st_size > 30_000


def test_png_and_svg_are_dissertation_ready(tmp_path: Path) -> None:
    """PNG должен иметь высокое разрешение, а SVG — редактируемый текст."""

    source = _write_test_data(tmp_path / DEFAULT_INPUT_PATH)
    result = generate(project_root=tmp_path, input_path=source, dpi=300)
    width, height = _read_png_dimensions(result.png_path)
    svg = result.svg_path.read_text(encoding="utf-8")

    assert width >= 4000
    assert height >= 2500
    for text in (
        "Документы корпуса",
        "Уникальные сценарии",
        "Токены словаря LDA",
        "Уровни сложности сообщения",
        "CorpusSufficiencyAnalyzer",
        "passed",
        "не заменяет проверку внешней валидности",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg


def test_cli_reports_both_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI должен завершиться успешно и вывести пути к PNG и SVG."""

    source = _write_test_data(tmp_path / DEFAULT_INPUT_PATH)
    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--input",
            str(source),
            "--token-count",
            "96",
            "--dpi",
            "300",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Рисунок 3.5 успешно сформирован" in output
    assert ".png" in output
    assert ".svg" in output
