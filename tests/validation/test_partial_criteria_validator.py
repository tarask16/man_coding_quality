"""Тесты этапа 6: проверка частных прогнозных критериев."""

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
from manual_coding_sim.validation.partial_criteria_validator import (
    CRITERION_MAPPINGS,
    PartialCriteriaValidationError,
    PartialCriteriaValidator,
)


def test_all_six_criteria_are_calculated(tmp_path: Path) -> None:
    """Этап должен рассчитывать метрики для всех заданных пар критериев."""

    result = _make_validator(tmp_path).validate(_build_dataset())

    assert result.passed is True
    assert len(result.metrics_table) == 6
    assert result.metrics_table["criterion"].tolist() == [
        mapping[0] for mapping in CRITERION_MAPPINGS
    ]
    assert result.report["criterion_count"] == 6


def test_basic_metrics_match_formulas(tmp_path: Path) -> None:
    """MAE, RMSE и Bias должны соответствовать определению ошибки."""

    result = _make_validator(tmp_path).validate(_build_dataset())
    first = result.metrics_table.iloc[0]

    assert first["mae"] == pytest.approx(0.1)
    assert first["rmse"] == pytest.approx(0.1)
    assert first["bias"] == pytest.approx(0.1)
    assert first["max_absolute_error"] == pytest.approx(0.1)
    assert first["pearson"] == pytest.approx(1.0)
    assert first["spearman"] == pytest.approx(1.0)
    assert first["kendall"] == pytest.approx(1.0)
    assert first["r2"] == pytest.approx(0.8)


def test_perfect_partial_predictions_have_zero_error(tmp_path: Path) -> None:
    """Точные частные прогнозы должны иметь нулевую ошибку и R² = 1."""

    dataset = _build_dataset()
    for _, predicted_column, factual_column, _ in CRITERION_MAPPINGS:
        dataset[predicted_column] = dataset[factual_column]

    result = _make_validator(tmp_path).validate(dataset)

    assert result.metrics_table["mae"].tolist() == pytest.approx([0.0] * 6)
    assert result.metrics_table["rmse"].tolist() == pytest.approx([0.0] * 6)
    assert result.metrics_table["bias"].tolist() == pytest.approx([0.0] * 6)
    assert result.metrics_table["r2"].tolist() == pytest.approx([1.0] * 6)


def test_summary_identifies_best_and_worst_criteria(tmp_path: Path) -> None:
    """Сводка должна корректно определять критерии по MAE и Spearman."""

    dataset = _build_dataset()
    dataset["q_time_pred"] = dataset["q_time"]
    dataset["q_fit_pred"] = [0.8, 0.6, 0.4, 0.2]

    result = _make_validator(tmp_path).validate(dataset)
    summary = result.report["summary"]

    assert summary["best_mae_criterion"] == "q_time"
    assert summary["best_mae"] == pytest.approx(0.0)
    assert summary["worst_mae_criterion"] == "q_fit"
    assert summary["worst_spearman_criterion"] == "q_fit"
    assert summary["worst_spearman"] == pytest.approx(-1.0)


def test_missing_required_column_is_rejected(tmp_path: Path) -> None:
    """Отсутствие частного критерия должно блокировать этап 6."""

    invalid = _build_dataset().drop(columns=["q_rep_pred"])

    with pytest.raises(PartialCriteriaValidationError, match="q_rep_pred"):
        _make_validator(tmp_path).validate(invalid)


def test_non_unique_composite_key_is_rejected(tmp_path: Path) -> None:
    """Повтор составного ключа должен блокировать проверку критериев."""

    invalid = _build_dataset()
    invalid.loc[1, ["scenario_id", "protocol_id"]] = ["S1", "P1"]

    with pytest.raises(PartialCriteriaValidationError, match="не является уникальным"):
        _make_validator(tmp_path).validate(invalid)


def test_non_numeric_non_finite_and_out_of_range_values_are_rejected(
    tmp_path: Path,
) -> None:
    """Некорректные значения частных критериев должны блокировать расчет."""

    non_numeric = _build_dataset()
    non_numeric["q_acc_pred"] = non_numeric["q_acc_pred"].astype(object)
    non_numeric.loc[0, "q_acc_pred"] = "ошибка"
    with pytest.raises(PartialCriteriaValidationError, match="числовыми"):
        _make_validator(tmp_path).validate(non_numeric)

    non_finite = _build_dataset()
    non_finite.loc[0, "q_acc"] = np.nan
    with pytest.raises(PartialCriteriaValidationError, match="NaN"):
        _make_validator(tmp_path).validate(non_finite)

    out_of_range = _build_dataset()
    out_of_range.loc[0, "q_time_pred"] = 1.1
    with pytest.raises(PartialCriteriaValidationError, match="диапазоне"):
        _make_validator(tmp_path).validate(out_of_range)


def test_constant_criterion_is_rejected(tmp_path: Path) -> None:
    """Постоянный критерий не позволяет рассчитывать корреляции и R²."""

    invalid = _build_dataset()
    invalid["q_res"] = 0.5

    with pytest.raises(PartialCriteriaValidationError, match="q_res"):
        _make_validator(tmp_path).validate(invalid)


def test_reports_are_saved_to_stage6_paths(tmp_path: Path) -> None:
    """Должны создаваться CSV-, JSON- и Markdown-артефакты этапа 6."""

    dataset_path = tmp_path / "reports/chapter6/validation_dataset.csv"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    _build_dataset().to_csv(dataset_path, index=False)

    result = _make_validator(tmp_path).validate_and_save()

    assert result.csv_path == (
        tmp_path / "reports/chapter6/partial_criteria_validation.csv"
    )
    assert result.json_path == (
        tmp_path / "reports/chapter6/partial_criteria_validation_report.json"
    )
    assert result.markdown_path == (
        tmp_path / "reports/chapter6/partial_criteria_validation_report.md"
    )
    assert result.csv_path.exists()
    assert result.json_path.exists()
    assert result.markdown_path.exists()

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["stage"] == 6
    assert payload["passed"] is True
    assert payload["criterion_count"] == 6


def test_cli_validates_partial_criteria(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI этапа 6 должен сформировать отчеты и вывести сводные метрики."""

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
            "--validate-partial-criteria",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Проверка частных прогнозных критериев завершена" in captured.out
    assert "Проверено критериев: 6" in captured.out
    assert "Этап 6 выполнен" in captured.out
    assert (
        tmp_path / "reports/chapter6/partial_criteria_validation_report.json"
    ).exists()


def test_cli_can_run_stages_5_and_6_together(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Флаги этапов 5 и 6 должны выполняться последовательно."""

    config_path = _write_config(tmp_path)
    dataset = _build_dataset()
    dataset["q_pred"] = [0.2, 0.4, 0.6, 0.8]
    dataset["q_fact"] = [0.1, 0.3, 0.5, 0.7]
    dataset_path = tmp_path / "reports/chapter6/validation_dataset.csv"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(dataset_path, index=False)

    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--config",
            str(config_path.relative_to(tmp_path)),
            "--calculate-integral-metrics",
            "--validate-partial-criteria",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Этап 5 выполнен" in captured.out
    assert "Этап 6 выполнен" in captured.out


def _make_validator(project_root: Path) -> PartialCriteriaValidator:
    """Создать валидатор для четырех тестовых сценариев."""

    config = Chapter6ValidationConfig(
        merge=Chapter6MergeConfig(expected_row_count=4),
    )
    return PartialCriteriaValidator(config=config, project_root=project_root)


def _build_dataset() -> pd.DataFrame:
    """Сформировать минимальный корректный датасет этапа 6."""

    dataset = pd.DataFrame(
        {
            "scenario_id": ["S1", "S2", "S3", "S4"],
            "protocol_id": ["P1", "P2", "P3", "P4"],
        }
    )
    factual = [0.1, 0.3, 0.5, 0.7]
    predicted = [0.2, 0.4, 0.6, 0.8]
    for _, predicted_column, factual_column, _ in CRITERION_MAPPINGS:
        dataset[predicted_column] = predicted
        dataset[factual_column] = factual
    return dataset


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
