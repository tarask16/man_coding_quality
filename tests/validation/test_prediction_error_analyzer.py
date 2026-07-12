"""Тесты диагностического анализа ошибок прогноза этапа 11."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from manual_coding_sim.validation.chapter6_config import (
    Chapter6MergeConfig,
    Chapter6ValidationConfig,
)
from manual_coding_sim.validation.chapter6_runner import main
from manual_coding_sim.validation.prediction_error_analyzer import (
    GROUP_DIMENSION_ORDER,
    PredictionErrorAnalysisError,
    PredictionErrorAnalyzer,
)


def test_top_errors_are_sorted_by_absolute_error(tmp_path: Path) -> None:
    """Top ошибок должен быть отсортирован по убыванию модуля ошибки."""

    result = _make_analyzer(tmp_path).analyze(_build_dataset())

    assert len(result.top_errors) == 8
    assert result.top_errors["error_rank"].tolist() == list(range(1, 9))
    assert result.top_errors["absolute_error"].is_monotonic_decreasing


def test_prediction_error_formula_is_q_pred_minus_q_fact(tmp_path: Path) -> None:
    """Знак ошибки должен соответствовать определению этапа 5."""

    dataset = _build_dataset()
    result = _make_analyzer(tmp_path).analyze(dataset)
    first = result.top_errors.iloc[0]
    source = dataset.loc[dataset["scenario_id"] == first["scenario_id"]].iloc[0]

    assert first["prediction_error"] == pytest.approx(
        source["q_pred"] - source["q_fact"]
    )
    assert first["absolute_error"] == pytest.approx(
        abs(source["q_pred"] - source["q_fact"])
    )


def test_group_analysis_covers_every_scenario_in_each_slice(tmp_path: Path) -> None:
    """Каждый диагностический срез должен покрывать весь набор сценариев."""

    result = _make_analyzer(tmp_path).analyze(_build_dataset())
    totals = result.group_analysis.groupby("analysis_dimension")["count"].sum()

    assert set(totals.index) == set(GROUP_DIMENSION_ORDER)
    assert (totals == 8).all()


def test_uncertainty_spearman_matches_direct_calculation(tmp_path: Path) -> None:
    """Связь неопределенности с абсолютной ошибкой должна считаться напрямую."""

    dataset = _build_dataset()
    result = _make_analyzer(tmp_path).analyze(dataset)
    expected = spearmanr(
        dataset["uncertainty_score"],
        np.abs(dataset["q_pred"] - dataset["q_fact"]),
    ).statistic

    assert result.report["uncertainty_relation"][
        "spearman_absolute_error"
    ] == pytest.approx(expected)


def test_report_contains_all_diagnostic_correlations(tmp_path: Path) -> None:
    """Отчет должен содержать корреляции условий и фактических признаков."""

    result = _make_analyzer(tmp_path).analyze(_build_dataset())
    variables = {
        row["variable"] for row in result.report["diagnostic_correlations"]
    }

    assert variables == {
        "uncertainty_score",
        "fact_duration_sec",
        "fact_error_count",
        "fact_recheck_count",
        "fact_reject_count",
        "fact_success",
        "prior_condition_noise_level_norm",
        "prior_condition_time_pressure_norm",
    }


def test_report_contains_top_error_diagnostics(tmp_path: Path) -> None:
    """Top ошибок должен включать фактор, условия и фактические признаки."""

    result = _make_analyzer(tmp_path).analyze(_build_dataset())
    required = {
        "dominant_factor",
        "uncertainty_score",
        "noise_level",
        "time_pressure",
        "fact_duration_sec",
        "fact_error_count",
        "fact_recheck_count",
        "fact_reject_count",
        "fact_success",
    }

    assert required.issubset(result.top_errors.columns)


def test_report_records_methodological_restrictions(tmp_path: Path) -> None:
    """Отчет должен фиксировать отсутствие подгонки модели главы 5."""

    result = _make_analyzer(tmp_path).analyze(_build_dataset())
    checks = result.report["methodological_checks"]

    assert result.report["stage"] == 11
    assert result.report["passed"] is True
    assert checks["chapter5_prediction_modified"] is False
    assert checks["quality_thresholds_modified"] is False
    assert checks["factual_values_used_only_for_external_validation"] is True
    assert checks["analysis_is_diagnostic_not_calibrating"] is True


def test_input_dataset_is_not_modified(tmp_path: Path) -> None:
    """Анализатор не должен изменять проверочный датасет этапа 3."""

    dataset = _build_dataset()
    original = dataset.copy(deep=True)

    _make_analyzer(tmp_path).analyze(dataset)

    pd.testing.assert_frame_equal(dataset, original)


def test_missing_diagnostic_column_is_rejected(tmp_path: Path) -> None:
    """Отсутствие фактического диагностического признака блокирует этап."""

    invalid = _build_dataset().drop(columns=["fact_duration_sec"])

    with pytest.raises(PredictionErrorAnalysisError, match="fact_duration_sec"):
        _make_analyzer(tmp_path).analyze(invalid)


def test_duplicate_composite_key_is_rejected(tmp_path: Path) -> None:
    """Повтор составного ключа должен блокировать анализ ошибок."""

    invalid = _build_dataset()
    invalid.loc[1, ["scenario_id", "protocol_id"]] = ["S1", "P1"]

    with pytest.raises(PredictionErrorAnalysisError, match="не является уникальным"):
        _make_analyzer(tmp_path).analyze(invalid)


def test_nan_is_rejected(tmp_path: Path) -> None:
    """NaN в прогнозе не должен попадать в диагностический отчет."""

    invalid = _build_dataset()
    invalid.loc[0, "q_pred"] = np.nan

    with pytest.raises(PredictionErrorAnalysisError, match="NaN"):
        _make_analyzer(tmp_path).analyze(invalid)


def test_value_outside_unit_interval_is_rejected(tmp_path: Path) -> None:
    """Прогноз вне диапазона [0; 1] должен блокировать этап."""

    invalid = _build_dataset()
    invalid.loc[0, "uncertainty_score"] = 1.2

    with pytest.raises(PredictionErrorAnalysisError, match=r"\[0; 1\]"):
        _make_analyzer(tmp_path).analyze(invalid)


def test_negative_diagnostic_counter_is_rejected(tmp_path: Path) -> None:
    """Отрицательный фактический счетчик должен блокировать этап."""

    invalid = _build_dataset()
    invalid.loc[0, "fact_error_count"] = -1

    with pytest.raises(PredictionErrorAnalysisError, match="отрицательные"):
        _make_analyzer(tmp_path).analyze(invalid)


def test_non_integer_diagnostic_counter_is_rejected(tmp_path: Path) -> None:
    """Дробное число ошибок не должно приниматься как счетчик."""

    invalid = _build_dataset()
    invalid["fact_error_count"] = invalid["fact_error_count"].astype(float)
    invalid.loc[0, "fact_error_count"] = 1.5

    with pytest.raises(PredictionErrorAnalysisError, match="целые счетчики"):
        _make_analyzer(tmp_path).analyze(invalid)


def test_inconsistent_quality_class_is_rejected(tmp_path: Path) -> None:
    """Сохраненный класс должен совпадать с классом по порогам."""

    invalid = _build_dataset()
    invalid.loc[0, "q_pred_class"] = "high"

    with pytest.raises(PredictionErrorAnalysisError, match="q_pred_class"):
        _make_analyzer(tmp_path).analyze(invalid)


def test_inconsistent_dominant_topic_is_rejected(tmp_path: Path) -> None:
    """Доминирующая тема должна повторно проверяться по theta-профилю."""

    invalid = _build_dataset()
    invalid.loc[0, "theta_dominant_topic"] = "theta_2"

    with pytest.raises(PredictionErrorAnalysisError, match="доминирующая тема"):
        _make_analyzer(tmp_path).analyze(invalid)


def test_reports_are_saved_to_stage11_paths(tmp_path: Path) -> None:
    """Должны создаваться два CSV и два отчета этапа 11."""

    dataset_path = tmp_path / "reports/chapter6/validation_dataset.csv"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    _build_dataset().to_csv(dataset_path, index=False)

    result = _make_analyzer(tmp_path).analyze_and_save()
    expected_paths = (
        tmp_path / "reports/chapter6/top_prediction_errors.csv",
        tmp_path / "reports/chapter6/error_group_analysis.csv",
        tmp_path / "reports/chapter6/prediction_error_analysis.json",
        tmp_path / "reports/chapter6/prediction_error_analysis.md",
    )
    actual_paths = (
        result.top_errors_path,
        result.group_analysis_path,
        result.json_path,
        result.markdown_path,
    )

    assert actual_paths == expected_paths
    assert all(path.exists() for path in expected_paths)
    payload = json.loads(expected_paths[2].read_text(encoding="utf-8"))
    assert payload["stage"] == 11
    assert payload["passed"] is True


def test_cli_runs_prediction_error_analysis(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI этапа 11 должен сформировать все диагностические артефакты."""

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
            "--analyze-prediction-errors",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Анализ ошибок априорного прогноза завершен" in captured.out
    assert "Сценариев в top-10: 8" in captured.out
    assert "Этап 11 выполнен" in captured.out
    assert (tmp_path / "reports/chapter6/top_prediction_errors.csv").exists()
    assert (tmp_path / "reports/chapter6/prediction_error_analysis.json").exists()


def _make_analyzer(project_root: Path) -> PredictionErrorAnalyzer:
    """Создать анализатор для восьми тестовых сценариев."""

    config = Chapter6ValidationConfig(
        merge=Chapter6MergeConfig(expected_row_count=8),
    )
    return PredictionErrorAnalyzer(config=config, project_root=project_root)


def _build_dataset() -> pd.DataFrame:
    """Сформировать корректный проверочный датасет этапа 11."""

    q_pred = np.array([0.25, 0.30, 0.42, 0.55, 0.68, 0.78, 0.82, 0.70])
    q_fact = np.array([0.20, 0.35, 0.50, 0.60, 0.72, 0.80, 0.90, 0.65])
    theta = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.60, 0.30, 0.10],
            [0.20, 0.70, 0.10],
            [0.10, 0.75, 0.15],
            [0.10, 0.20, 0.70],
            [0.15, 0.10, 0.75],
            [0.20, 0.15, 0.65],
            [0.55, 0.25, 0.20],
        ]
    )
    frame = pd.DataFrame(
        {
            "scenario_id": [f"S{i}" for i in range(1, 9)],
            "protocol_id": [f"P{i}" for i in range(1, 9)],
            "q_pred": q_pred,
            "q_fact": q_fact,
            "uncertainty_score": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
            "theta_0": theta[:, 0],
            "theta_1": theta[:, 1],
            "theta_2": theta[:, 2],
            "prior_condition_noise_level_norm": [0.0, 1 / 3, 2 / 3, 1.0] * 2,
            "prior_condition_time_pressure_norm": [1.0, 2 / 3, 1 / 3, 0.0] * 2,
            "fact_duration_sec": [20, 30, 40, 50, 60, 70, 80, 90],
            "fact_error_count": [0, 1, 2, 3, 4, 5, 6, 8],
            "fact_recheck_count": [0, 1, 2, 0, 1, 2, 0, 1],
            "fact_reject_count": [0, 0, 0, 1, 0, 1, 1, 1],
            "fact_success": [1, 1, 1, 0, 1, 0, 0, 0],
        }
    )
    frame["q_pred_class"] = frame["q_pred"].map(_quality_class)
    frame["q_fact_class"] = frame["q_fact"].map(_quality_class)
    frame["theta_dominant_topic"] = frame[
        ["theta_0", "theta_1", "theta_2"]
    ].idxmax(axis=1)
    return frame


def _quality_class(value: float) -> str:
    """Классифицировать тестовое значение по порогам главы 6."""

    if value < 0.45:
        return "low"
    if value < 0.70:
        return "medium"
    return "high"


def _write_config(project_root: Path) -> Path:
    """Записать минимальную конфигурацию CLI этапа 11."""

    path = project_root / "configs/chapter6.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
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
    return path
