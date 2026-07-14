"""Локальные тесты генератора рисунка 5.4."""

from __future__ import annotations

import csv
import struct
from pathlib import Path

import pytest

from manual_coding_sim.dissertation_figures.figure_5_4_q_pred_distribution import (
    DEFAULT_INPUT_PATH,
    FILE_STEM,
    HIGH_THRESHOLD,
    LOW_THRESHOLD,
    QUALITY_CLASSES,
    QPredRow,
    calculate_class_counts,
    calculate_summary,
    classify_q_pred,
    generate,
    load_q_pred,
    validate_q_pred_rows,
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG из заголовка IHDR."""

    with path.open("rb") as stream:
        stream.read(16)
        return struct.unpack(">II", stream.read(8))


def _reference_rows() -> tuple[QPredRow, ...]:
    """Сформировать тестовое распределение с тремя классами."""

    values = (0.10, 0.30, 0.449, 0.45, 0.55, 0.699, 0.70, 0.80, 0.90)
    return tuple(
        QPredRow(f"scn_{index:04d}", f"prt_{index:04d}", value)
        for index, value in enumerate(values, start=1)
    )


def _write_reference_input(project_root: Path) -> Path:
    """Записать тестовый q_pred.csv в стандартный каталог."""

    path = project_root / DEFAULT_INPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("scenario_id", "protocol_id", "q_pred"),
        )
        writer.writeheader()
        for row in _reference_rows():
            writer.writerow(
                {
                    "scenario_id": row.scenario_id,
                    "protocol_id": row.protocol_id,
                    "q_pred": row.q_pred,
                }
            )
    return path


def test_thresholds_and_class_order_match_chapter5() -> None:
    """Пороги и порядок классов должны совпадать с методикой главы 5."""

    assert LOW_THRESHOLD == pytest.approx(0.45)
    assert HIGH_THRESHOLD == pytest.approx(0.70)
    assert tuple(item.code for item in QUALITY_CLASSES) == ("low", "medium", "high")


def test_classification_obeys_half_open_intervals() -> None:
    """Граничные значения должны относиться к старшему классу."""

    assert classify_q_pred(0.449999) == "low"
    assert classify_q_pred(0.45) == "medium"
    assert classify_q_pred(0.699999) == "medium"
    assert classify_q_pred(0.70) == "high"


def test_class_counts_and_shares_are_consistent() -> None:
    """Численности классов должны суммироваться в размер выборки."""

    counts = calculate_class_counts(_reference_rows())
    assert tuple(item.count for item in counts) == (3, 3, 3)
    assert sum(item.count for item in counts) == 9
    assert sum(item.share for item in counts) == pytest.approx(1.0)


def test_summary_uses_population_standard_deviation() -> None:
    """Сводка должна рассчитывать статистику всего корпуса."""

    summary = calculate_summary(_reference_rows())
    assert summary.count == 9
    assert summary.minimum == pytest.approx(0.10)
    assert summary.maximum == pytest.approx(0.90)
    assert summary.median == pytest.approx(0.55)
    assert summary.standard_deviation > 0.0


def test_validate_rejects_value_outside_unit_interval() -> None:
    """Значение Q_pred вне [0; 1] должно отклоняться."""

    with pytest.raises(ValueError, match=r"диапазоне \[0; 1\]"):
        validate_q_pred_rows((QPredRow("scn", "prt", 1.2),))


def test_validate_rejects_duplicate_scenario() -> None:
    """Дублирование scenario_id должно отклоняться."""

    with pytest.raises(ValueError, match="сценариев должны быть уникальными"):
        validate_q_pred_rows(
            (
                QPredRow("scn", "prt_1", 0.4),
                QPredRow("scn", "prt_2", 0.6),
            )
        )


def test_load_q_pred_reads_reference_csv(tmp_path: Path) -> None:
    """Загрузчик должен прочитать корректный q_pred.csv."""

    source = _write_reference_input(tmp_path)
    rows = load_q_pred(source)
    assert len(rows) == 9
    assert rows[0].q_pred == pytest.approx(0.10)
    assert rows[-1].q_pred == pytest.approx(0.90)


def test_load_q_pred_rejects_missing_integral_column(tmp_path: Path) -> None:
    """CSV без q_pred должен отклоняться."""

    path = tmp_path / "q_pred.csv"
    path.write_text("scenario_id,protocol_id\nscn,prt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="отсутствует колонка q_pred"):
        load_q_pred(path)


def test_png_and_svg_are_dissertation_ready(tmp_path: Path) -> None:
    """PNG должен быть крупным, а SVG — содержать редактируемые подписи."""

    _write_reference_input(tmp_path)
    result = generate(project_root=tmp_path, dpi=300)
    width, height = _read_png_dimensions(result.png_path)
    svg = result.svg_path.read_text(encoding="utf-8")

    assert result.png_path.name == f"{FILE_STEM}.png"
    assert result.svg_path.name == f"{FILE_STEM}.svg"
    assert width >= 4200
    assert height >= 2200
    for text in (
        "Распределение интегрального прогнозного показателя Q_pred",
        "Гистограмма Q_pred и пороговые границы классов",
        "граница 0,45",
        "граница 0,70",
        "Численность классов качества",
        "Описательная статистика",
        "low",
        "medium",
        "high",
        "не подтверждает внешнюю точность",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg


def test_cli_main_generates_files(tmp_path: Path) -> None:
    """CLI должен завершаться успешно и создавать оба формата."""

    from manual_coding_sim.dissertation_figures.figure_5_4_q_pred_distribution import (
        main,
    )

    _write_reference_input(tmp_path)
    exit_code = main(["--project-root", str(tmp_path), "--dpi", "300"])
    assert exit_code == 0
    output_dir = tmp_path / "reports" / "chapter5" / "figures"
    assert (output_dir / f"{FILE_STEM}.png").is_file()
    assert (output_dir / f"{FILE_STEM}.svg").is_file()
