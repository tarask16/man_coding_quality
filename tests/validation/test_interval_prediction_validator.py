"""Тесты проверки интервального прогноза главы 6."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from manual_coding_sim.validation.chapter6_config import (
    Chapter6MergeConfig,
    Chapter6ValidationConfig,
)
from manual_coding_sim.validation.chapter6_runner import main
from manual_coding_sim.validation.interval_prediction_validator import (
    IntervalPredictionValidationError,
    IntervalPredictionValidator,
)


def test_known_interval_metrics(tmp_path: Path) -> None:
    """Основные метрики должны совпадать с ручным расчетом."""

    result = _make_validator(tmp_path).validate(_build_dataset())
    metrics = result.report["metrics"]

    assert result.passed is True
    assert metrics["coverage_rate"] == pytest.approx(5 / 8)
    assert metrics["covered_count"] == 5
    assert metrics["miss_count"] == 3
    assert metrics["mean_interval_width"] == pytest.approx(0.11)
    assert metrics["median_interval_width"] == pytest.approx(0.1)
    assert metrics["miss_lower_count"] == 1
    assert metrics["miss_upper_count"] == 2
    assert metrics["mean_distance_to_interval"] == pytest.approx(0.04)
    assert metrics["mean_miss_distance"] == pytest.approx(0.32 / 3)
    assert metrics["max_distance_to_interval"] == pytest.approx(0.15)


def test_interval_boundaries_are_inclusive(tmp_path: Path) -> None:
    """Фактические значения на обеих границах считаются покрытыми."""

    result = _make_validator(tmp_path).validate(_build_dataset())

    assert result.details.loc[1, "q_fact"] == pytest.approx(
        result.details.loc[1, "q_pred_upper"]
    )
    assert result.details.loc[2, "q_fact"] == pytest.approx(
        result.details.loc[2, "q_pred_upper"]
    )
    assert result.details.loc[3, "q_fact"] == pytest.approx(
        result.details.loc[3, "q_pred_lower"]
    )
    assert result.details.loc[[1, 2, 3], "is_covered"].all()


def test_miss_directions_and_distances(tmp_path: Path) -> None:
    """Направление промаха и расстояние считаются от ближайшей границы."""

    result = _make_validator(tmp_path).validate(_build_dataset())
    details = result.details.set_index("scenario_id")

    assert details.loc["S6", "miss_direction"] == "above_upper"
    assert details.loc["S6", "distance_to_interval"] == pytest.approx(0.02)
    assert details.loc["S7", "miss_direction"] == "below_lower"
    assert details.loc["S7", "distance_to_interval"] == pytest.approx(0.15)
    assert details.loc["S8", "miss_direction"] == "above_upper"
    assert details.loc["S8", "distance_to_interval"] == pytest.approx(0.15)
    assert details.loc["S1", "distance_to_interval"] == pytest.approx(0.0)


def test_slices_by_classes_and_dominant_factor(tmp_path: Path) -> None:
    """Срезы должны покрывать все сценарии и сохранять заданный порядок."""

    result = _make_validator(tmp_path).validate(_build_dataset())
    slices = result.report["slices"]

    factual = {row["group"]: row for row in slices["by_factual_class"]}
    predicted = {row["group"]: row for row in slices["by_predicted_class"]}
    dominant = {row["group"]: row for row in slices["by_dominant_factor"]}

    assert [row["group"] for row in slices["by_factual_class"]] == [
        "low",
        "medium",
        "high",
    ]
    assert factual["low"]["count"] == 3
    assert factual["medium"]["count"] == 2
    assert factual["high"]["count"] == 3
    assert predicted["low"]["count"] == 2
    assert predicted["medium"]["count"] == 4
    assert predicted["high"]["count"] == 2
    assert dominant["theta_0"]["count"] == 3
    assert dominant["theta_1"]["count"] == 3
    assert dominant["theta_2"]["count"] == 2


def test_uncertainty_quantiles_cover_all_rows(tmp_path: Path) -> None:
    """Четыре квартиля должны быть непересекающимися и равными по размеру."""

    result = _make_validator(tmp_path).validate(_build_dataset())
    details = result.details
    rows = result.report["slices"]["by_uncertainty_quantile"]

    assert set(details["uncertainty_quantile"]) == {"Q1", "Q2", "Q3", "Q4"}
    assert [row["count"] for row in rows] == [2, 2, 2, 2]
    assert sum(row["count"] for row in rows) == 8
    assert rows[0]["uncertainty_score_max"] < rows[-1]["uncertainty_score_min"]


def test_existing_annotations_are_checked(tmp_path: Path) -> None:
    """Сохраненные классы и доминирующий фактор должны быть согласованы."""

    dataset = _build_dataset()
    dataset["q_pred_class"] = dataset["q_pred"].map(_quality_class)
    dataset["q_fact_class"] = dataset["q_fact"].map(_quality_class)
    dataset["theta_dominant_topic"] = dataset[
        ["theta_0", "theta_1", "theta_2"]
    ].idxmax(axis=1)

    valid = _make_validator(tmp_path).validate(dataset)
    assert valid.passed is True

    dataset.loc[0, "theta_dominant_topic"] = "theta_2"
    with pytest.raises(IntervalPredictionValidationError, match="повторному"):
        _make_validator(tmp_path).validate(dataset)


def test_missing_required_column_is_rejected(tmp_path: Path) -> None:
    """Отсутствие границы интервала должно блокировать этап 8."""

    invalid = _build_dataset().drop(columns=["q_pred_upper"])

    with pytest.raises(IntervalPredictionValidationError, match="q_pred_upper"):
        _make_validator(tmp_path).validate(invalid)


def test_non_unique_key_is_rejected(tmp_path: Path) -> None:
    """Повтор составного ключа должен блокировать интервальную проверку."""

    invalid = _build_dataset()
    invalid.loc[1, ["scenario_id", "protocol_id"]] = ["S1", "P1"]

    with pytest.raises(IntervalPredictionValidationError, match="не является уникальным"):
        _make_validator(tmp_path).validate(invalid)


def test_non_finite_and_out_of_range_values_are_rejected(tmp_path: Path) -> None:
    """NaN, бесконечности и значения вне диапазона [0; 1] недопустимы."""

    non_finite = _build_dataset()
    non_finite.loc[0, "uncertainty_score"] = np.nan
    with pytest.raises(IntervalPredictionValidationError, match="NaN"):
        _make_validator(tmp_path).validate(non_finite)

    out_of_range = _build_dataset()
    out_of_range.loc[0, "q_pred_upper"] = 1.1
    with pytest.raises(IntervalPredictionValidationError, match="диапазоне"):
        _make_validator(tmp_path).validate(out_of_range)


def test_invalid_interval_order_is_rejected(tmp_path: Path) -> None:
    """Нижняя граница не может превышать верхнюю."""

    invalid = _build_dataset()
    invalid.loc[0, ["q_pred_lower", "q_pred_upper"]] = [0.3, 0.1]

    with pytest.raises(IntervalPredictionValidationError, match="превышает"):
        _make_validator(tmp_path).validate(invalid)


def test_point_prediction_outside_interval_is_rejected(tmp_path: Path) -> None:
    """Точечный прогноз должен находиться внутри собственного интервала."""

    invalid = _build_dataset()
    invalid.loc[0, "q_pred"] = 0.4

    with pytest.raises(IntervalPredictionValidationError, match="внутри"):
        _make_validator(tmp_path).validate(invalid)


def test_reports_are_saved_to_stage8_paths(tmp_path: Path) -> None:
    """Этап должен создавать три предусмотренных артефакта."""

    dataset_path = tmp_path / "reports/chapter6/validation_dataset.csv"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    _build_dataset().to_csv(dataset_path, index=False)

    result = _make_validator(tmp_path).validate_and_save()

    assert result.details_path == (
        tmp_path / "reports/chapter6/interval_coverage_details.csv"
    )
    assert result.json_path == (
        tmp_path / "reports/chapter6/interval_coverage_report.json"
    )
    assert result.markdown_path == (
        tmp_path / "reports/chapter6/interval_coverage_report.md"
    )
    assert result.details_path.exists()
    assert result.json_path.exists()
    assert result.markdown_path.exists()

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["stage"] == 8
    assert payload["passed"] is True
    assert payload["row_count"] == 8


def test_cli_validates_interval_prediction(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI этапа 8 должен создать отчеты и вывести основные метрики."""

    config_path = _write_config(tmp_path)
    dataset_path = tmp_path / "reports/chapter6/validation_dataset.csv"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    _build_dataset().to_csv(dataset_path, index=False)

    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--config",
            str(config_path.relative_to(tmp_path)),
            "--validate-interval-prediction",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Проверка интервального прогноза качества завершена" in captured.out
    assert "Coverage rate: 0.6250000000" in captured.out
    assert "Этап 8 выполнен" in captured.out
    assert (tmp_path / "reports/chapter6/interval_coverage_report.json").exists()


def test_cli_can_run_stages_7_and_8_together(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Флаги этапов 7 и 8 должны выполняться последовательно."""

    config_path = _write_config(tmp_path)
    dataset_path = tmp_path / "reports/chapter6/validation_dataset.csv"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    _build_dataset().to_csv(dataset_path, index=False)

    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--config",
            str(config_path.relative_to(tmp_path)),
            "--validate-classification",
            "--validate-interval-prediction",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Этап 7 выполнен" in captured.out
    assert "Этап 8 выполнен" in captured.out


def _make_validator(project_root: Path) -> IntervalPredictionValidator:
    """Создать валидатор для восьми тестовых сценариев."""

    config = Chapter6ValidationConfig(
        merge=Chapter6MergeConfig(expected_row_count=8),
    )
    return IntervalPredictionValidator(config=config, project_root=project_root)


def _build_dataset() -> pd.DataFrame:
    """Сформировать датасет с известным покрытием и тремя промахами."""

    return pd.DataFrame(
        {
            "scenario_id": [f"S{index}" for index in range(1, 9)],
            "protocol_id": [f"P{index}" for index in range(1, 9)],
            "q_fact": [0.2, 0.4, 0.5, 0.6, 0.72, 0.9, 0.3, 0.8],
            "q_pred": [0.2, 0.35, 0.45, 0.65, 0.75, 0.85, 0.5, 0.6],
            "q_pred_lower": [0.1, 0.3, 0.4, 0.6, 0.7, 0.8, 0.45, 0.55],
            "q_pred_upper": [0.3, 0.4, 0.5, 0.7, 0.8, 0.88, 0.55, 0.65],
            "uncertainty_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "theta_0": [0.7, 0.6, 0.2, 0.1, 0.1, 0.2, 0.8, 0.2],
            "theta_1": [0.2, 0.3, 0.7, 0.8, 0.2, 0.1, 0.1, 0.7],
            "theta_2": [0.1, 0.1, 0.1, 0.1, 0.7, 0.7, 0.1, 0.1],
        }
    )


def _quality_class(value: float) -> str:
    """Преобразовать тестовое значение в класс по порогам главы 6."""

    if value < 0.45:
        return "low"
    if value < 0.70:
        return "medium"
    return "high"


def _write_config(project_root: Path) -> Path:
    """Записать минимальную конфигурацию CLI для восьми сценариев."""

    config_path = project_root / "configs/chapter6.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        """
merge:
  key_columns: [scenario_id, protocol_id]
  validation: one_to_one
  expected_row_count: 8
decision_thresholds:
  low_max: 0.45
  high_min: 0.70
bootstrap:
  resamples: 100
  confidence_level: 0.95
  random_seed: 42
  sampling_unit: scenario_id
""".strip(),
        encoding="utf-8",
    )
    return config_path
