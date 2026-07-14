"""Тесты генератора рисунка 6.8 с ошибкой по доминирующему фактору."""

from __future__ import annotations

import csv
import struct
from pathlib import Path

import pytest

from manual_coding_sim.dissertation_figures.figure_6_8_error_by_dominant_topic import (
    DEFAULT_Q_FACT_PATH,
    DEFAULT_Q_PRED_PATH,
    DEFAULT_THETA_PATH,
    FILE_STEM,
    TOPIC_ORDER,
    TopicErrorPoint,
    generate,
    load_topic_error_points,
    main,
    summarise_errors_by_topic,
)


def _write_reference_inputs(project_root: Path) -> tuple[Path, Path, Path]:
    """Создать согласованные тестовые CSV для трёх доминирующих тем."""

    q_pred_path = project_root / DEFAULT_Q_PRED_PATH
    q_fact_path = project_root / DEFAULT_Q_FACT_PATH
    theta_path = project_root / DEFAULT_THETA_PATH
    q_pred_path.parent.mkdir(parents=True, exist_ok=True)
    q_fact_path.parent.mkdir(parents=True, exist_ok=True)
    theta_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = (0.25, 0.30, 0.38, 0.44, 0.72, 0.80)
    facts = (0.55, 0.58, 0.60, 0.70, 0.75, 0.82)
    theta = (
        (0.80, 0.10, 0.10),
        (0.70, 0.20, 0.10),
        (0.10, 0.80, 0.10),
        (0.20, 0.70, 0.10),
        (0.10, 0.10, 0.80),
        (0.15, 0.15, 0.70),
    )

    with q_pred_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["scenario_id", "protocol_id", "q_pred"],
        )
        writer.writeheader()
        for index, value in enumerate(predictions):
            writer.writerow(
                {
                    "scenario_id": f"scn_{index:04d}",
                    "protocol_id": f"prt_{index:04d}",
                    "q_pred": value,
                }
            )

    with q_fact_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["scenario_id", "protocol_id", "integral_quality"],
        )
        writer.writeheader()
        for index, value in enumerate(facts):
            writer.writerow(
                {
                    "scenario_id": f"scn_{index:04d}",
                    "protocol_id": f"prt_{index:04d}",
                    "integral_quality": value,
                }
            )

    with theta_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "scenario_id",
                "protocol_id",
                "theta_0",
                "theta_1",
                "theta_2",
            ],
        )
        writer.writeheader()
        for index, values in enumerate(theta):
            writer.writerow(
                {
                    "scenario_id": f"scn_{index:04d}",
                    "protocol_id": f"prt_{index:04d}",
                    "theta_0": values[0],
                    "theta_1": values[1],
                    "theta_2": values[2],
                }
            )
    return q_pred_path, q_fact_path, theta_path


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать ширину и высоту PNG без внешних библиотек."""

    with path.open("rb") as stream:
        signature = stream.read(24)
    if signature[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError("Файл не является PNG.")
    return struct.unpack(">II", signature[16:24])


def test_constants_fix_expected_output_contract() -> None:
    """Константы должны фиксировать входы и имя рисунка 6.8."""

    assert FILE_STEM == "error_by_dominant_topic"
    assert DEFAULT_Q_PRED_PATH == Path("reports/chapter5/q_pred.csv")
    assert DEFAULT_Q_FACT_PATH == Path("data/processed/quality_targets.csv")
    assert DEFAULT_THETA_PATH == Path("reports/chapter4/theta_prior.csv")
    assert TOPIC_ORDER == (0, 1, 2)


def test_load_topic_error_points_assigns_dominant_topics(tmp_path: Path) -> None:
    """Загрузчик должен вычислять dominant_topic по максимуму theta."""

    paths = _write_reference_inputs(tmp_path)
    points = load_topic_error_points(*paths)

    assert len(points) == 6
    assert tuple(point.dominant_topic for point in points) == (0, 0, 1, 1, 2, 2)
    assert points[0].absolute_error == pytest.approx(0.30)


def test_load_topic_error_points_rejects_mismatched_keys(tmp_path: Path) -> None:
    """Несовпадающие ключи трёх источников должны отклоняться."""

    q_pred_path, q_fact_path, theta_path = _write_reference_inputs(tmp_path)
    rows = list(csv.DictReader(theta_path.open("r", encoding="utf-8")))
    rows[-1]["scenario_id"] = "scn_other"
    with theta_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Ключи q_pred, quality_targets и theta_prior не совпадают"):
        load_topic_error_points(q_pred_path, q_fact_path, theta_path)


def test_load_topic_error_points_rejects_non_normalised_theta(tmp_path: Path) -> None:
    """Компоненты theta должны суммироваться в единицу."""

    q_pred_path, q_fact_path, theta_path = _write_reference_inputs(tmp_path)
    rows = list(csv.DictReader(theta_path.open("r", encoding="utf-8")))
    rows[0]["theta_2"] = "0.30"
    with theta_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="должна быть равна 1"):
        load_topic_error_points(q_pred_path, q_fact_path, theta_path)


def test_load_topic_error_points_requires_all_three_topics(tmp_path: Path) -> None:
    """Рисунок должен содержать наблюдения всех трёх доминирующих тем."""

    q_pred_path, q_fact_path, theta_path = _write_reference_inputs(tmp_path)
    rows = list(csv.DictReader(theta_path.open("r", encoding="utf-8")))
    for row in rows:
        row["theta_0"], row["theta_1"], row["theta_2"] = "0.8", "0.1", "0.1"
    with theta_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="отсутствуют сценарии с доминирующими темами"):
        load_topic_error_points(q_pred_path, q_fact_path, theta_path)


def test_summarise_errors_by_topic_calculates_group_statistics() -> None:
    """Сводка должна вычислять численность и средние ошибки групп."""

    points = (
        TopicErrorPoint("s0", "p0", 0.2, 0.5, 0.8, 0.1, 0.1, 0),
        TopicErrorPoint("s1", "p1", 0.3, 0.5, 0.7, 0.2, 0.1, 0),
        TopicErrorPoint("s2", "p2", 0.4, 0.6, 0.1, 0.8, 0.1, 1),
        TopicErrorPoint("s3", "p3", 0.5, 0.6, 0.2, 0.7, 0.1, 1),
        TopicErrorPoint("s4", "p4", 0.7, 0.75, 0.1, 0.1, 0.8, 2),
        TopicErrorPoint("s5", "p5", 0.8, 0.82, 0.15, 0.15, 0.7, 2),
    )
    summary = summarise_errors_by_topic(points)

    assert tuple(item.count for item in summary.topics) == (2, 2, 2)
    assert summary.topics[0].mean == pytest.approx(0.25)
    assert summary.topics[2].mean == pytest.approx(0.035)
    assert summary.best_observed_topic == 2
    assert summary.worst_observed_topic == 0


def test_summarise_errors_by_topic_rejects_missing_group() -> None:
    """Сводка без одной темы должна считаться неполной."""

    points = (
        TopicErrorPoint("s0", "p0", 0.2, 0.5, 0.8, 0.1, 0.1, 0),
        TopicErrorPoint("s1", "p1", 0.3, 0.5, 0.7, 0.2, 0.1, 0),
        TopicErrorPoint("s2", "p2", 0.4, 0.6, 0.1, 0.8, 0.1, 1),
    )

    with pytest.raises(ValueError, match="Отсутствует группа dominant_topic=2"):
        summarise_errors_by_topic(points)


def test_generate_creates_expected_output_names(tmp_path: Path) -> None:
    """Генератор должен создавать PNG и SVG с установленным именем."""

    _write_reference_inputs(tmp_path)
    result = generate(project_root=tmp_path, dpi=180)

    assert result.png_path.name == f"{FILE_STEM}.png"
    assert result.svg_path.name == f"{FILE_STEM}.svg"
    assert result.png_path.is_file()
    assert result.svg_path.is_file()


def test_png_and_svg_are_dissertation_ready(tmp_path: Path) -> None:
    """PNG должен быть крупным, а SVG — содержать редактируемые подписи."""

    _write_reference_inputs(tmp_path)
    result = generate(project_root=tmp_path, dpi=300)
    width, height = _read_png_dimensions(result.png_path)
    svg = result.svg_path.read_text(encoding="utf-8")

    assert width >= 4400
    assert height >= 2200
    for text in (
        "Абсолютная ошибка по доминирующему латентному фактору",
        "Распределение |Q_pred − Q_fact| внутри групп dominant_topic",
        "Процедурная трудоёмкость",
        "Операционный риск",
        "Благоприятные условия",
        "Описательная сводка групп",
        "ассоциативную интерпретацию",
        "не рассматривается как причина ошибки",
    ):
        assert text in svg


def test_cli_main_generates_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI должен формировать оба формата и печатать их пути."""

    q_pred_path, q_fact_path, theta_path = _write_reference_inputs(tmp_path)
    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--q-pred",
            str(q_pred_path),
            "--q-fact",
            str(q_fact_path),
            "--theta",
            str(theta_path),
            "--dpi",
            "180",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Рисунок 6.8 успешно сформирован." in output
    assert (tmp_path / "reports/chapter6/figures/error_by_dominant_topic.png").is_file()
    assert (tmp_path / "reports/chapter6/figures/error_by_dominant_topic.svg").is_file()
