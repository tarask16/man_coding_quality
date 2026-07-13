"""Локальные тесты генератора рисунка 5.3."""

from __future__ import annotations

import csv
import struct
from pathlib import Path

import pytest

from manual_coding_sim.dissertation_figures.figure_5_3_partial_prediction_components import (
    CRITERIA,
    DEFAULT_INPUT_PATH,
    FILE_STEM,
    PartialPredictionRow,
    calculate_summaries,
    generate,
    load_partial_predictions,
    prediction_matrix,
    validate_partial_predictions,
)


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG из заголовка IHDR."""

    with path.open("rb") as stream:
        stream.read(16)
        return struct.unpack(">II", stream.read(8))


def _reference_rows() -> tuple[PartialPredictionRow, ...]:
    """Сформировать небольшой корректный набор частных прогнозов."""

    feature_weights = (0.65, 0.70, 0.70, 0.60, 0.60, 0.65)
    latent_weights = (0.35, 0.30, 0.30, 0.40, 0.40, 0.35)
    return (
        PartialPredictionRow(
            "scn_0001",
            "prt_0001",
            (0.20, 0.30, 0.40, 0.25, 0.35, 0.45),
            feature_weights,
            latent_weights,
        ),
        PartialPredictionRow(
            "scn_0002",
            "prt_0002",
            (0.40, 0.50, 0.60, 0.45, 0.55, 0.65),
            feature_weights,
            latent_weights,
        ),
        PartialPredictionRow(
            "scn_0003",
            "prt_0003",
            (0.60, 0.70, 0.80, 0.65, 0.75, 0.85),
            feature_weights,
            latent_weights,
        ),
        PartialPredictionRow(
            "scn_0004",
            "prt_0004",
            (0.80, 0.90, 0.55, 0.85, 0.95, 0.75),
            feature_weights,
            latent_weights,
        ),
        PartialPredictionRow(
            "scn_0005",
            "prt_0005",
            (0.35, 0.45, 0.50, 0.40, 0.50, 0.60),
            feature_weights,
            latent_weights,
        ),
        PartialPredictionRow(
            "scn_0006",
            "prt_0006",
            (0.55, 0.65, 0.70, 0.60, 0.70, 0.80),
            feature_weights,
            latent_weights,
        ),
    )


def _write_reference_input(project_root: Path) -> Path:
    """Записать тестовый q_pred_components.csv в стандартный каталог."""

    path = project_root / DEFAULT_INPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scenario_id", "protocol_id"]
    for criterion in CRITERIA:
        fieldnames.extend(
            (
                criterion.prediction_column,
                criterion.feature_weight_column,
                criterion.latent_weight_column,
            )
        )

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in _reference_rows():
            data: dict[str, object] = {
                "scenario_id": row.scenario_id,
                "protocol_id": row.protocol_id,
            }
            for index, criterion in enumerate(CRITERIA):
                data[criterion.prediction_column] = row.values[index]
                data[criterion.feature_weight_column] = row.feature_weights[index]
                data[criterion.latent_weight_column] = row.latent_weights[index]
            writer.writerow(data)
    return path


def test_criteria_match_six_chapter5_components() -> None:
    """Порядок критериев должен совпадать с методикой главы 5."""

    assert tuple(criterion.code for criterion in CRITERIA) == (
        "q_acc",
        "q_time",
        "q_effort",
        "q_res",
        "q_rep",
        "q_fit",
    )
    assert tuple(criterion.prediction_column for criterion in CRITERIA) == (
        "q_acc_pred",
        "q_time_pred",
        "q_effort_pred",
        "q_res_pred",
        "q_rep_pred",
        "q_fit_pred",
    )


def test_prediction_matrix_has_expected_shape() -> None:
    """Матрица должна содержать одну строку на сценарий и шесть колонок."""

    matrix = prediction_matrix(_reference_rows())
    assert matrix.shape == (6, 6)
    assert matrix.min() >= 0.0
    assert matrix.max() <= 1.0


def test_summaries_preserve_weights_and_statistics() -> None:
    """Сводка должна содержать статистику и постоянные веса компонентов."""

    summaries = calculate_summaries(_reference_rows())
    assert len(summaries) == 6
    assert summaries[0].mean == pytest.approx(0.4833333333)
    assert summaries[0].feature_weight == pytest.approx(0.65)
    assert summaries[0].latent_weight == pytest.approx(0.35)
    assert summaries[-1].maximum == pytest.approx(0.85)


def test_validate_rejects_prediction_outside_unit_interval() -> None:
    """Значение вне [0; 1] должно отклоняться."""

    row = _reference_rows()[0]
    invalid = (
        PartialPredictionRow(
            row.scenario_id,
            row.protocol_id,
            (1.2, *row.values[1:]),
            row.feature_weights,
            row.latent_weights,
        ),
    )
    with pytest.raises(ValueError, match="лежать в \\[0; 1\\]"):
        validate_partial_predictions(invalid)


def test_validate_rejects_non_unit_weight_sum() -> None:
    """Веса признаковой и латентной компонент должны суммироваться в единицу."""

    row = _reference_rows()[0]
    invalid = (
        PartialPredictionRow(
            row.scenario_id,
            row.protocol_id,
            row.values,
            (0.80, *row.feature_weights[1:]),
            row.latent_weights,
        ),
    )
    with pytest.raises(ValueError, match="Сумма весов"):
        validate_partial_predictions(invalid)


def test_validate_rejects_duplicate_scenario() -> None:
    """Дублирование идентификатора сценария должно отклоняться."""

    rows = _reference_rows()
    duplicate = (rows[0], rows[0])
    with pytest.raises(ValueError, match="сценариев должны быть уникальными"):
        validate_partial_predictions(duplicate)


def test_load_partial_predictions_reads_reference_csv(tmp_path: Path) -> None:
    """Загрузчик должен прочитать корректный тестовый CSV."""

    source = _write_reference_input(tmp_path)
    rows = load_partial_predictions(source)
    assert len(rows) == 6
    assert rows[0].scenario_id == "scn_0001"
    assert rows[-1].values[-1] == pytest.approx(0.80)


def test_load_partial_predictions_rejects_missing_column(tmp_path: Path) -> None:
    """CSV без обязательной прогнозной колонки должен отклоняться."""

    path = tmp_path / "q_pred_components.csv"
    path.write_text("scenario_id,protocol_id,q_acc_pred\n", encoding="utf-8")
    with pytest.raises(ValueError, match="отсутствуют обязательные колонки"):
        load_partial_predictions(path)


def test_png_and_svg_are_dissertation_ready(tmp_path: Path) -> None:
    """PNG должен быть крупным, а SVG — содержать редактируемые подписи."""

    _write_reference_input(tmp_path)
    result = generate(project_root=tmp_path, dpi=300)
    width, height = _read_png_dimensions(result.png_path)
    svg = result.svg_path.read_text(encoding="utf-8")

    assert result.png_path.name == f"{FILE_STEM}.png"
    assert result.svg_path.name == f"{FILE_STEM}.svg"
    assert width >= 4400
    assert height >= 2200
    for text in (
        "Распределения частных прогнозных критериев",
        "Форма распределений и квартильная структура",
        "Сводные характеристики и веса компонентов",
        "q_acc_pred",
        "q_time_pred",
        "q_effort_pred",
        "q_res_pred",
        "q_rep_pred",
        "q_fit_pred",
        "априорные q_j,pred",
        "не являются независимыми",
    ):
        assert text in svg
    assert "<text" in svg
    assert "<path" in svg


def test_cli_main_generates_files(tmp_path: Path) -> None:
    """CLI должен завершаться успешно и создавать оба формата."""

    from manual_coding_sim.dissertation_figures.figure_5_3_partial_prediction_components import (
        main,
    )

    _write_reference_input(tmp_path)
    exit_code = main(["--project-root", str(tmp_path), "--dpi", "300"])
    assert exit_code == 0
    output_dir = tmp_path / "reports" / "chapter5" / "figures"
    assert (output_dir / f"{FILE_STEM}.png").is_file()
    assert (output_dir / f"{FILE_STEM}.svg").is_file()
