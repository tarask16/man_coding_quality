"""Тесты этапа 9: сравнение с базовыми моделями без LDA."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from manual_coding_sim.validation.baseline_models import (
    BaselineComparisonError,
    BaselineModelsValidator,
    MODEL_ORDER,
)
from manual_coding_sim.validation.chapter6_config import (
    Chapter6MergeConfig,
    Chapter6ValidationConfig,
)
from manual_coding_sim.validation.chapter6_runner import main


def test_prior_theta_and_chapter5_predictions_follow_contract(tmp_path: Path) -> None:
    """Три априорные схемы должны использовать строго заданные компоненты."""

    dataset = _build_dataset()
    result = _make_validator(tmp_path).validate(dataset)

    expected_prior = dataset[
        [
            "q_acc_feature_component",
            "q_time_feature_component",
            "q_effort_feature_component",
            "q_res_feature_component",
            "q_rep_feature_component",
            "q_fit_feature_component",
        ]
    ].mean(axis=1)
    assert result.predictions["prior_only_baseline"].tolist() == pytest.approx(
        expected_prior.tolist(),
        abs=1e-9,
    )
    assert result.predictions["theta_only_baseline"].tolist() == pytest.approx(
        dataset["q_latent"].tolist()
    )
    assert result.predictions["chapter5_model"].tolist() == pytest.approx(
        dataset["q_pred"].tolist()
    )


def test_mean_baseline_excludes_current_fold_targets(tmp_path: Path) -> None:
    """Mean baseline должен использовать среднее только по обучающим fold."""

    dataset = _build_dataset()
    result = _make_validator(tmp_path, fold_count=3).validate(dataset)
    predictions = result.predictions

    for fold_id in sorted(predictions["oof_fold"].unique()):
        test_mask = predictions["oof_fold"] == fold_id
        expected_mean = dataset.loc[~test_mask, "q_fact"].mean()
        assert predictions.loc[test_mask, "mean_baseline"].nunique() == 1
        assert predictions.loc[test_mask, "mean_baseline"].iloc[0] == pytest.approx(
            expected_mean
        )
        assert predictions.loc[
            test_mask, "mean_baseline_training_count"
        ].iloc[0] == int((~test_mask).sum())


def test_oof_split_is_deterministic_for_fixed_seed(tmp_path: Path) -> None:
    """Одинаковый random_seed должен давать идентичное разбиение fold."""

    dataset = _build_dataset()
    first = _make_validator(tmp_path, fold_count=3).validate(dataset)
    second = _make_validator(tmp_path, fold_count=3).validate(dataset)

    assert first.predictions["oof_fold"].tolist() == second.predictions[
        "oof_fold"
    ].tolist()
    assert first.predictions["mean_baseline"].tolist() == pytest.approx(
        second.predictions["mean_baseline"].tolist()
    )


def test_comparison_contains_four_models_and_finite_metrics(tmp_path: Path) -> None:
    """Таблица сравнения должна содержать четыре модели и конечные метрики."""

    result = _make_validator(tmp_path).validate(_build_dataset())

    assert result.comparison["model"].tolist() == list(MODEL_ORDER)
    numeric = result.comparison.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()
    assert result.passed is True


def test_chapter5_mae_matches_direct_calculation(tmp_path: Path) -> None:
    """MAE модели главы 5 должен совпадать с прямым расчетом по q_pred."""

    dataset = _build_dataset()
    result = _make_validator(tmp_path).validate(dataset)
    row = result.comparison.set_index("model").loc["chapter5_model"]
    expected_mae = np.mean(np.abs(dataset["q_pred"] - dataset["q_fact"]))

    assert row["mae"] == pytest.approx(expected_mae)


def test_quality_classes_use_fixed_thresholds(tmp_path: Path) -> None:
    """Классы baseline-прогнозов должны использовать пороги 0,45 и 0,70."""

    result = _make_validator(tmp_path).validate(_build_dataset())

    expected = ["low", "medium", "medium", "high", "high", "medium"]
    assert result.predictions["chapter5_model_class"].tolist() == expected


def test_report_records_leakage_protection(tmp_path: Path) -> None:
    """Отчет должен явно фиксировать методическую защиту от утечки."""

    result = _make_validator(tmp_path, fold_count=3).validate(_build_dataset())
    checks = result.report["leakage_checks"]

    assert checks["mean_baseline_is_out_of_fold"] is True
    assert checks["test_fold_targets_excluded_from_training_mean"] is True
    assert checks["prior_only_uses_q_fact"] is False
    assert checks["theta_only_uses_q_fact"] is False
    assert checks["chapter5_prediction_unchanged"] is True
    assert result.report["mean_baseline"]["actual_fold_count"] == 3


def test_missing_required_column_is_rejected(tmp_path: Path) -> None:
    """Отсутствие обязательной feature-компоненты должно блокировать этап."""

    invalid = _build_dataset().drop(columns=["q_fit_feature_component"])

    with pytest.raises(BaselineComparisonError, match="q_fit_feature_component"):
        _make_validator(tmp_path).validate(invalid)


def test_non_unique_composite_key_is_rejected(tmp_path: Path) -> None:
    """Повтор составного ключа должен блокировать baseline-сравнение."""

    invalid = _build_dataset()
    invalid.loc[1, ["scenario_id", "protocol_id"]] = ["S1", "P1"]

    with pytest.raises(BaselineComparisonError, match="не является уникальным"):
        _make_validator(tmp_path).validate(invalid)


def test_non_finite_value_is_rejected(tmp_path: Path) -> None:
    """NaN в baseline-компонентах должен блокировать расчет."""

    invalid = _build_dataset()
    invalid.loc[0, "q_latent"] = np.nan

    with pytest.raises(BaselineComparisonError, match="NaN"):
        _make_validator(tmp_path).validate(invalid)


def test_out_of_range_value_is_rejected(tmp_path: Path) -> None:
    """Значение вне шкалы [0; 1] должно блокировать расчет."""

    invalid = _build_dataset()
    invalid.loc[0, "q_acc_feature_component"] = 1.2

    with pytest.raises(BaselineComparisonError, match=r"\[0; 1\]"):
        _make_validator(tmp_path).validate(invalid)


def test_invalid_quality_weight_sum_is_rejected(tmp_path: Path) -> None:
    """Некорректная сумма весов главы 5 не должна использоваться молча."""

    weights_path = tmp_path / "configs/chapter5_quality_weights.yaml"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_text(
        """
quality_weights:
  q_acc: 0.10
  q_time: 0.10
  q_effort: 0.10
  q_res: 0.10
  q_rep: 0.10
  q_fit: 0.10
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(BaselineComparisonError, match="равна единице"):
        _make_validator(tmp_path).validate(_build_dataset())


def test_reports_are_saved_to_stage9_paths(tmp_path: Path) -> None:
    """Должны создаваться два CSV и два отчета этапа 9."""

    dataset_path = tmp_path / "reports/chapter6/validation_dataset.csv"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    _build_dataset().to_csv(dataset_path, index=False)

    result = _make_validator(tmp_path).validate_and_save()

    assert result.predictions_path == (
        tmp_path / "reports/chapter6/baseline_predictions.csv"
    )
    assert result.comparison_path == (
        tmp_path / "reports/chapter6/baseline_comparison.csv"
    )
    assert result.json_path == (
        tmp_path / "reports/chapter6/baseline_comparison_report.json"
    )
    assert result.markdown_path == (
        tmp_path / "reports/chapter6/baseline_comparison_report.md"
    )
    for path in (
        result.predictions_path,
        result.comparison_path,
        result.json_path,
        result.markdown_path,
    ):
        assert path is not None and path.exists()

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["stage"] == 9
    assert payload["passed"] is True
    assert payload["leakage_checks"]["chapter5_prediction_unchanged"] is True


def test_cli_compares_baselines(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI этапа 9 должен сформировать все baseline-артефакты."""

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
            "--compare-baselines",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Сравнение с базовыми моделями завершено" in captured.out
    assert "Mean baseline MAE" in captured.out
    assert "Этап 9 выполнен" in captured.out
    assert (tmp_path / "reports/chapter6/baseline_predictions.csv").exists()
    assert (tmp_path / "reports/chapter6/baseline_comparison_report.json").exists()


def test_cli_can_run_stages_5_and_9_together(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Флаги этапов 5 и 9 должны работать последовательно за один запуск."""

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
            "--compare-baselines",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Этап 5 выполнен" in captured.out
    assert "Этап 9 выполнен" in captured.out


def _make_validator(
    project_root: Path,
    fold_count: int = 5,
) -> BaselineModelsValidator:
    """Создать валидатор для шести тестовых сценариев."""

    config = Chapter6ValidationConfig(
        merge=Chapter6MergeConfig(expected_row_count=6),
    )
    return BaselineModelsValidator(
        config=config,
        project_root=project_root,
        fold_count=fold_count,
    )


def _build_dataset() -> pd.DataFrame:
    """Сформировать минимальный корректный датасет baseline-сравнения."""

    return pd.DataFrame(
        {
            "scenario_id": ["S1", "S2", "S3", "S4", "S5", "S6"],
            "protocol_id": ["P1", "P2", "P3", "P4", "P5", "P6"],
            "q_fact": [0.20, 0.40, 0.55, 0.68, 0.80, 0.90],
            "q_pred": [0.30, 0.50, 0.60, 0.75, 0.85, 0.65],
            "q_latent": [0.25, 0.45, 0.50, 0.70, 0.90, 0.80],
            "q_acc_feature_component": [0.20, 0.35, 0.50, 0.65, 0.80, 0.85],
            "q_time_feature_component": [0.25, 0.40, 0.55, 0.60, 0.75, 0.80],
            "q_effort_feature_component": [0.30, 0.45, 0.50, 0.70, 0.85, 0.75],
            "q_res_feature_component": [0.20, 0.40, 0.60, 0.65, 0.80, 0.90],
            "q_rep_feature_component": [0.25, 0.35, 0.55, 0.75, 0.90, 0.85],
            "q_fit_feature_component": [0.30, 0.45, 0.50, 0.65, 0.75, 0.80],
        }
    )


def _write_config(project_root: Path) -> Path:
    """Записать минимальную YAML-конфигурацию для CLI-тестов."""

    config_path = project_root / "configs/chapter6.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
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
    return config_path
