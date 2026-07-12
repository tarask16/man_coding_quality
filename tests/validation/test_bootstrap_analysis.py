"""Тесты этапа 10: bootstrap-анализ статистической устойчивости."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from manual_coding_sim.validation.bootstrap_analysis import (
    BASELINE_MODELS,
    METRIC_ORDER,
    MODEL_ORDER,
    BootstrapAnalysisError,
    BootstrapAnalysisValidator,
)
from manual_coding_sim.validation.chapter6_config import (
    Chapter6BootstrapConfig,
    Chapter6MergeConfig,
    Chapter6ValidationConfig,
)
from manual_coding_sim.validation.chapter6_runner import main


def test_bootstrap_contains_all_models_and_metrics(tmp_path: Path) -> None:
    """Таблица интервалов должна содержать 4 × 6 строк."""

    result = _make_validator(tmp_path).validate(_build_predictions())

    assert len(result.confidence_intervals) == len(MODEL_ORDER) * len(METRIC_ORDER)
    assert set(result.confidence_intervals["model"]) == set(MODEL_ORDER)
    assert set(result.confidence_intervals["metric"]) == set(METRIC_ORDER)
    assert result.passed is True


def test_bootstrap_differences_cover_all_baselines(tmp_path: Path) -> None:
    """Парные разности должны быть рассчитаны для каждого baseline и метрики."""

    result = _make_validator(tmp_path).validate(_build_predictions())

    assert len(result.model_differences) == len(BASELINE_MODELS) * len(METRIC_ORDER)
    assert set(result.model_differences["baseline"]) == set(BASELINE_MODELS)
    assert set(result.model_differences["metric"]) == set(METRIC_ORDER)


def test_bootstrap_is_deterministic_for_fixed_seed(tmp_path: Path) -> None:
    """Одинаковый random_seed должен давать идентичные интервалы."""

    first = _make_validator(tmp_path).validate(_build_predictions())
    second = _make_validator(tmp_path).validate(_build_predictions())

    pd.testing.assert_frame_equal(
        first.confidence_intervals,
        second.confidence_intervals,
    )
    pd.testing.assert_frame_equal(
        first.model_differences,
        second.model_differences,
    )


def test_point_mae_matches_direct_calculation(tmp_path: Path) -> None:
    """Точечная MAE модели главы 5 должна совпадать с прямым расчетом."""

    predictions = _build_predictions()
    result = _make_validator(tmp_path).validate(predictions)
    row = result.confidence_intervals.set_index(["model", "metric"]).loc[
        ("chapter5_model", "mae")
    ]
    expected = np.mean(
        np.abs(predictions["chapter5_model"] - predictions["q_fact"])
    )

    assert row["point_estimate"] == pytest.approx(expected)


def test_point_delta_uses_chapter5_minus_baseline(tmp_path: Path) -> None:
    """Точечная разность должна иметь направление chapter5 minus baseline."""

    predictions = _build_predictions()
    result = _make_validator(tmp_path).validate(predictions)
    row = result.model_differences.set_index(["baseline", "metric"]).loc[
        ("prior_only_baseline", "mae")
    ]
    chapter5_mae = np.mean(
        np.abs(predictions["chapter5_model"] - predictions["q_fact"])
    )
    prior_mae = np.mean(
        np.abs(predictions["prior_only_baseline"] - predictions["q_fact"])
    )

    assert row["point_delta"] == pytest.approx(chapter5_mae - prior_mae)
    assert row["delta_definition"] == "metric_chapter5_model - metric_baseline"


def test_confidence_intervals_are_ordered_and_complete(tmp_path: Path) -> None:
    """Все интервалы должны быть упорядочены и содержать 100 повторов."""

    result = _make_validator(tmp_path).validate(_build_predictions())

    assert (result.confidence_intervals["ci_lower"] <= result.confidence_intervals["ci_upper"]).all()
    assert (result.model_differences["ci_lower"] <= result.model_differences["ci_upper"]).all()
    assert (result.confidence_intervals["valid_resamples"] == 100).all()
    assert (result.model_differences["valid_resamples"] == 100).all()


def test_report_records_paired_bootstrap_contract(tmp_path: Path) -> None:
    """Отчет должен фиксировать выборку по сценариям и отсутствие обучения."""

    result = _make_validator(tmp_path).validate(_build_predictions())
    sampling = result.report["sampling"]
    checks = result.report["methodological_checks"]

    assert result.report["stage"] == 10
    assert sampling["method"] == "paired_cluster_percentile_bootstrap"
    assert sampling["sampling_unit"] == "scenario_id"
    assert sampling["sampling_unit_count"] == 6
    assert sampling["resamples"] == 100
    assert sampling["random_seed"] == 42
    assert checks["paired_resamples_used_for_all_models"] is True
    assert checks["models_refitted_inside_bootstrap"] is False
    assert checks["chapter5_prediction_modified"] is False


def test_input_predictions_are_not_modified(tmp_path: Path) -> None:
    """Bootstrap-анализ не должен изменять входную таблицу этапа 9."""

    predictions = _build_predictions()
    original = predictions.copy(deep=True)

    _make_validator(tmp_path).validate(predictions)

    pd.testing.assert_frame_equal(predictions, original)


def test_missing_model_column_is_rejected(tmp_path: Path) -> None:
    """Отсутствие прогноза baseline должно блокировать этап."""

    invalid = _build_predictions().drop(columns=["theta_only_baseline"])

    with pytest.raises(BootstrapAnalysisError, match="theta_only_baseline"):
        _make_validator(tmp_path).validate(invalid)


def test_duplicate_composite_key_is_rejected(tmp_path: Path) -> None:
    """Повтор составного ключа должен блокировать bootstrap-анализ."""

    invalid = _build_predictions()
    invalid.loc[1, ["scenario_id", "protocol_id"]] = ["S1", "P1"]

    with pytest.raises(BootstrapAnalysisError, match="не является уникальным"):
        _make_validator(tmp_path).validate(invalid)


def test_non_finite_prediction_is_rejected(tmp_path: Path) -> None:
    """NaN в прогнозах не должен попадать в bootstrap-распределение."""

    invalid = _build_predictions()
    invalid.loc[0, "chapter5_model"] = np.nan

    with pytest.raises(BootstrapAnalysisError, match="NaN"):
        _make_validator(tmp_path).validate(invalid)


def test_out_of_range_prediction_is_rejected(tmp_path: Path) -> None:
    """Значения вне [0; 1] должны блокировать этап."""

    invalid = _build_predictions()
    invalid.loc[0, "mean_baseline"] = 1.1

    with pytest.raises(BootstrapAnalysisError, match=r"\[0; 1\]"):
        _make_validator(tmp_path).validate(invalid)


def test_reports_are_saved_to_stage10_paths(tmp_path: Path) -> None:
    """Должны создаваться два CSV и два отчета этапа 10."""

    predictions_path = tmp_path / "reports/chapter6/baseline_predictions.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    _build_predictions().to_csv(predictions_path, index=False)

    result = _make_validator(tmp_path).validate_and_save()

    expected_paths = (
        tmp_path / "reports/chapter6/bootstrap_confidence_intervals.csv",
        tmp_path / "reports/chapter6/bootstrap_model_differences.csv",
        tmp_path / "reports/chapter6/bootstrap_report.json",
        tmp_path / "reports/chapter6/bootstrap_report.md",
    )
    actual_paths = (
        result.confidence_intervals_path,
        result.model_differences_path,
        result.json_path,
        result.markdown_path,
    )
    assert actual_paths == expected_paths
    assert all(path.exists() for path in expected_paths)

    payload = json.loads(expected_paths[2].read_text(encoding="utf-8"))
    assert payload["stage"] == 10
    assert payload["passed"] is True
    assert payload["sampling"]["resamples"] == 100


def test_cli_runs_bootstrap_analysis(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI этапа 10 должен сформировать bootstrap-артефакты."""

    config_path = _write_config(tmp_path)
    predictions_path = tmp_path / "reports/chapter6/baseline_predictions.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    _build_predictions().to_csv(predictions_path, index=False)

    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--config",
            str(config_path.relative_to(tmp_path)),
            "--bootstrap-analysis",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Bootstrap-анализ статистической устойчивости завершен" in captured.out
    assert "Bootstrap-повторов: 100" in captured.out
    assert "Этап 10 выполнен" in captured.out
    assert (tmp_path / "reports/chapter6/bootstrap_report.json").exists()


def test_cli_can_run_stages_9_and_10_together(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Этапы 9 и 10 должны выполняться последовательно за один запуск."""

    config_path = _write_config(tmp_path)
    dataset_path = tmp_path / "reports/chapter6/validation_dataset.csv"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    _build_validation_dataset().to_csv(dataset_path, index=False)

    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--config",
            str(config_path.relative_to(tmp_path)),
            "--compare-baselines",
            "--bootstrap-analysis",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Этап 9 выполнен" in captured.out
    assert "Этап 10 выполнен" in captured.out
    assert (tmp_path / "reports/chapter6/baseline_predictions.csv").exists()
    assert (tmp_path / "reports/chapter6/bootstrap_confidence_intervals.csv").exists()


def _make_validator(project_root: Path) -> BootstrapAnalysisValidator:
    """Создать валидатор со 100 повторными выборками."""

    config = Chapter6ValidationConfig(
        merge=Chapter6MergeConfig(expected_row_count=6),
        bootstrap=Chapter6BootstrapConfig(
            resamples=100,
            confidence_level=0.95,
            random_seed=42,
            sampling_unit="scenario_id",
        ),
    )
    return BootstrapAnalysisValidator(config=config, project_root=project_root)


def _build_predictions() -> pd.DataFrame:
    """Сформировать корректные прогнозы четырех моделей."""

    return pd.DataFrame(
        {
            "scenario_id": ["S1", "S2", "S3", "S4", "S5", "S6"],
            "protocol_id": ["P1", "P2", "P3", "P4", "P5", "P6"],
            "q_fact": [0.20, 0.40, 0.55, 0.68, 0.80, 0.90],
            "mean_baseline": [0.58, 0.58, 0.58, 0.60, 0.60, 0.60],
            "prior_only_baseline": [0.25, 0.42, 0.52, 0.66, 0.78, 0.82],
            "theta_only_baseline": [0.15, 0.45, 0.50, 0.70, 0.95, 0.75],
            "chapter5_model": [0.22, 0.38, 0.56, 0.69, 0.81, 0.88],
        }
    )


def _build_validation_dataset() -> pd.DataFrame:
    """Сформировать вход этапа 9 для совместного CLI-теста."""

    predictions = _build_predictions()
    return pd.DataFrame(
        {
            "scenario_id": predictions["scenario_id"],
            "protocol_id": predictions["protocol_id"],
            "q_fact": predictions["q_fact"],
            "q_pred": predictions["chapter5_model"],
            "q_latent": predictions["theta_only_baseline"],
            "q_acc_feature_component": [0.20, 0.35, 0.50, 0.65, 0.80, 0.85],
            "q_time_feature_component": [0.25, 0.40, 0.55, 0.60, 0.75, 0.80],
            "q_effort_feature_component": [0.30, 0.45, 0.50, 0.70, 0.85, 0.75],
            "q_res_feature_component": [0.20, 0.40, 0.60, 0.65, 0.80, 0.90],
            "q_rep_feature_component": [0.25, 0.35, 0.55, 0.75, 0.90, 0.85],
            "q_fit_feature_component": [0.30, 0.45, 0.50, 0.65, 0.75, 0.80],
        }
    )


def _write_config(project_root: Path) -> Path:
    """Записать минимальную конфигурацию CLI этапа 10."""

    path = project_root / "configs/chapter6.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
merge:
  key_columns: [scenario_id, protocol_id]
  validation: one_to_one
  expected_row_count: 6
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
