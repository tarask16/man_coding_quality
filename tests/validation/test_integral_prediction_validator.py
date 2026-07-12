"""Тесты этапа 5: метрики интегрального априорного прогноза."""

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
from manual_coding_sim.validation.integral_prediction_validator import (
    IntegralPredictionValidationError,
    IntegralPredictionValidator,
)


def test_errors_and_basic_metrics_are_calculated(tmp_path: Path) -> None:
    """Построчные ошибки и основные метрики должны соответствовать формулам."""

    result = _make_validator(tmp_path).validate(_build_dataset())

    assert result.errors["prediction_error"].tolist() == pytest.approx(
        [0.1, -0.1, -0.1, -0.1]
    )
    assert result.errors["absolute_error"].tolist() == pytest.approx([0.1] * 4)
    assert result.errors["squared_error"].tolist() == pytest.approx([0.01] * 4)
    assert result.report["metrics"]["mae"] == pytest.approx(0.1)
    assert result.report["metrics"]["rmse"] == pytest.approx(0.1)
    assert result.report["metrics"]["bias"] == pytest.approx(-0.05)
    assert result.report["metrics"]["median_absolute_error"] == pytest.approx(0.1)
    assert result.report["metrics"]["max_absolute_error"] == pytest.approx(0.1)


def test_rank_metrics_equal_one_for_identical_order(tmp_path: Path) -> None:
    """Одинаковый порядок сценариев должен давать единичную ранговую связь."""

    result = _make_validator(tmp_path).validate(_build_dataset())
    metrics = result.report["metrics"]

    assert metrics["pearson"] > 0.95
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["kendall"] == pytest.approx(1.0)


def test_perfect_prediction_has_zero_errors_and_r2_one(tmp_path: Path) -> None:
    """Точный прогноз должен иметь нулевую ошибку и R², равный единице."""

    dataset = _build_dataset()
    dataset["q_pred"] = dataset["q_fact"]
    result = _make_validator(tmp_path).validate(dataset)
    metrics = result.report["metrics"]

    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["bias"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_negative_r2_is_valid_external_result(tmp_path: Path) -> None:
    """Отрицательный R² не должен блокировать сохранение внешней проверки."""

    dataset = _build_dataset()
    dataset["q_pred"] = [0.9, 0.8, 0.2, 0.1]
    result = _make_validator(tmp_path).validate(dataset)

    assert result.passed is True
    assert result.report["metrics"]["r2"] < 0.0


def test_missing_required_column_is_rejected(tmp_path: Path) -> None:
    """Отсутствие q_fact должно блокировать расчет метрик."""

    invalid = _build_dataset().drop(columns=["q_fact"])

    with pytest.raises(IntegralPredictionValidationError, match="q_fact"):
        _make_validator(tmp_path).validate(invalid)


def test_non_unique_composite_key_is_rejected(tmp_path: Path) -> None:
    """Повтор составного ключа должен блокировать этап 5."""

    invalid = _build_dataset()
    invalid.loc[1, ["scenario_id", "protocol_id"]] = ["S1", "P1"]

    with pytest.raises(IntegralPredictionValidationError, match="не является уникальным"):
        _make_validator(tmp_path).validate(invalid)


def test_non_finite_and_out_of_range_values_are_rejected(tmp_path: Path) -> None:
    """NaN и значения вне шкалы качества должны блокировать расчет."""

    non_finite = _build_dataset()
    non_finite.loc[0, "q_pred"] = np.nan
    with pytest.raises(IntegralPredictionValidationError, match="NaN"):
        _make_validator(tmp_path).validate(non_finite)

    out_of_range = _build_dataset()
    out_of_range.loc[0, "q_fact"] = 1.1
    with pytest.raises(IntegralPredictionValidationError, match="диапазоне"):
        _make_validator(tmp_path).validate(out_of_range)


def test_constant_quality_vector_is_rejected(tmp_path: Path) -> None:
    """Постоянная выборка не позволяет рассчитывать корреляции и R²."""

    invalid = _build_dataset()
    invalid["q_fact"] = 0.5

    with pytest.raises(IntegralPredictionValidationError, match="q_fact"):
        _make_validator(tmp_path).validate(invalid)


def test_reports_are_saved_to_stage5_paths(tmp_path: Path) -> None:
    """Должны создаваться CSV-, JSON- и Markdown-артефакты этапа 5."""

    dataset_path = tmp_path / "reports/chapter6/validation_dataset.csv"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    _build_dataset().to_csv(dataset_path, index=False)

    result = _make_validator(tmp_path).validate_and_save()

    assert result.csv_path == (
        tmp_path / "reports/chapter6/integral_prediction_errors.csv"
    )
    assert result.json_path == tmp_path / "reports/chapter6/validation_metrics.json"
    assert result.markdown_path == tmp_path / "reports/chapter6/validation_metrics.md"
    assert result.csv_path.exists()
    assert result.json_path.exists()
    assert result.markdown_path.exists()

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["stage"] == 5
    assert payload["passed"] is True
    assert payload["error_definition"] == "prediction_error = q_pred - q_fact"


def test_cli_calculates_integral_metrics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI этапа 5 должен сформировать отчеты и вывести основные метрики."""

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
            "--calculate-integral-metrics",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Метрики интегрального априорного прогноза рассчитаны" in captured.out
    assert "MAE:" in captured.out
    assert "Этап 5 выполнен" in captured.out
    assert (tmp_path / "reports/chapter6/validation_metrics.json").exists()
    assert (tmp_path / "reports/chapter6/integral_prediction_errors.csv").exists()


def test_cli_can_run_stages_4_and_5_together(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Флаги этапов 4 и 5 должны работать последовательно за один запуск."""

    config_path = _write_config(tmp_path)
    dataset = _build_dataset()
    for column in ("q_acc", "q_time", "q_effort", "q_res", "q_rep", "q_fit"):
        dataset[column] = dataset["q_fact"]
    dataset["integral_quality"] = dataset["q_fact"]
    dataset_path = tmp_path / "reports/chapter6/validation_dataset.csv"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(dataset_path, index=False)

    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--config",
            str(config_path.relative_to(tmp_path)),
            "--validate-integral-quality",
            "--calculate-integral-metrics",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Этап 4 выполнен" in captured.out
    assert "Этап 5 выполнен" in captured.out


def _make_validator(project_root: Path) -> IntegralPredictionValidator:
    """Создать валидатор для четырех тестовых сценариев."""

    config = Chapter6ValidationConfig(
        merge=Chapter6MergeConfig(expected_row_count=4),
    )
    return IntegralPredictionValidator(config=config, project_root=project_root)


def _build_dataset() -> pd.DataFrame:
    """Сформировать минимальный корректный датасет этапа 5."""

    return pd.DataFrame(
        {
            "scenario_id": ["S1", "S2", "S3", "S4"],
            "protocol_id": ["P1", "P2", "P3", "P4"],
            "q_pred": [0.2, 0.4, 0.6, 0.8],
            "q_fact": [0.1, 0.5, 0.7, 0.9],
        }
    )


def _write_config(project_root: Path) -> Path:
    """Записать минимальную YAML-конфигурацию для CLI-теста."""

    config_path = project_root / "configs/chapter6.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        """
merge:
  key_columns: [scenario_id, protocol_id]
  validation: one_to_one
  expected_row_count: 4
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
