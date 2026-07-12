"""Тесты этапа 3: формирование проверочного датасета главы 6."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from manual_coding_sim.validation.chapter6_config import (
    load_chapter6_validation_config,
)
from manual_coding_sim.validation.chapter6_data_loader import (
    Chapter6DataLoader,
)
from manual_coding_sim.validation.chapter6_runner import main
from manual_coding_sim.validation.validation_dataset_builder import (
    ValidationDatasetBuildError,
    ValidationDatasetBuilder,
)


PREDICTED_COLUMNS = (
    "q_acc_pred",
    "q_time_pred",
    "q_effort_pred",
    "q_res_pred",
    "q_rep_pred",
    "q_fit_pred",
)
FACTUAL_COLUMNS = (
    "q_acc",
    "q_time",
    "q_effort",
    "q_res",
    "q_rep",
    "q_fit",
)


def test_validation_dataset_contains_required_columns(tmp_path: Path) -> None:
    """Итоговый CSV должен содержать прогноз, факт и латентный профиль."""

    config_path = _build_valid_project(tmp_path)
    result = _make_builder(tmp_path, config_path).build()

    required = {
        "scenario_id",
        "protocol_id",
        "q_pred",
        "q_fact",
        "integral_quality",
        "q_pred_class",
        "q_fact_class",
        "theta_0",
        "theta_1",
        "theta_2",
        "uncertainty_score",
        "q_pred_lower",
        "q_pred_upper",
        *PREDICTED_COLUMNS,
        *FACTUAL_COLUMNS,
    }
    assert len(result.dataset) == 3
    assert required.issubset(result.dataset.columns)
    assert result.dataset.columns.duplicated().sum() == 0


def test_quality_classes_use_fixed_boundaries(tmp_path: Path) -> None:
    """Границы 0,45 и 0,70 должны относиться к medium и high."""

    config_path = _build_valid_project(tmp_path)
    result = _make_builder(tmp_path, config_path).build()

    assert result.dataset["q_pred_class"].tolist() == [
        "low",
        "medium",
        "high",
    ]
    assert result.dataset["q_fact_class"].tolist() == [
        "low",
        "medium",
        "high",
    ]
    assert result.predicted_class_counts == {
        "low": 1,
        "medium": 1,
        "high": 1,
    }


def test_q_fact_is_exact_alias_of_integral_quality(tmp_path: Path) -> None:
    """Этап 3 не должен пересчитывать фактическое интегральное качество."""

    config_path = _build_valid_project(tmp_path)
    dataset = _make_builder(tmp_path, config_path).build().dataset

    pd.testing.assert_series_equal(
        dataset["q_fact"],
        dataset["integral_quality"],
        check_names=False,
    )


def test_baseline_components_and_normalized_features_are_preserved(
    tmp_path: Path,
) -> None:
    """Датасет должен сохранять данные для последующих baseline-анализов."""

    config_path = _build_valid_project(tmp_path)
    dataset = _make_builder(tmp_path, config_path).build().dataset

    assert "q_acc_feature_component" in dataset.columns
    assert "q_time_latent_component" in dataset.columns
    assert "prior_condition_time_pressure_norm" in dataset.columns
    assert "q_latent" in dataset.columns


def test_fact_diagnostic_columns_are_added(tmp_path: Path) -> None:
    """Фактические диагностические признаки должны входить в датасет."""

    config_path = _build_valid_project(tmp_path)
    dataset = _make_builder(tmp_path, config_path).build().dataset

    assert "fact_error_count" in dataset.columns
    assert "fact_duration_sec" in dataset.columns
    assert dataset["fact_error_count"].tolist() == [3, 1, 0]


def test_all_merge_steps_preserve_row_count(tmp_path: Path) -> None:
    """Каждое one_to_one-объединение должно сохранять три сценария."""

    config_path = _build_valid_project(tmp_path)
    result = _make_builder(tmp_path, config_path).build()

    assert len(result.merge_steps) == 7
    for step in result.merge_steps:
        assert step.row_count_before == 3
        assert step.row_count_after == 3


def test_dataset_is_saved_to_configured_stage3_path(tmp_path: Path) -> None:
    """Метод build_and_save должен создать validation_dataset.csv."""

    config_path = _build_valid_project(tmp_path)
    result = _make_builder(tmp_path, config_path).build_and_save()

    expected = tmp_path / "reports/chapter6/validation_dataset.csv"
    assert result.output_path == expected
    assert expected.exists()
    saved = pd.read_csv(expected)
    assert len(saved) == 3
    assert saved["scenario_id"].tolist() == ["S1", "S2", "S3"]


def test_missing_fact_diagnostics_are_rejected(tmp_path: Path) -> None:
    """Пустой набор диагностических fact_*-колонок должен блокировать этап."""

    config_path = _build_valid_project(tmp_path)
    config = _load_config(tmp_path, config_path)
    loaded = Chapter6DataLoader(config=config, project_root=tmp_path).load()
    invalid = replace(
        loaded,
        fact_features=loaded.fact_features[["scenario_id", "protocol_id"]],
    )

    with pytest.raises(
        ValidationDatasetBuildError,
        match="диагностические колонки",
    ):
        ValidationDatasetBuilder(config, tmp_path).build(invalid)


def test_cli_builds_validation_dataset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI этапа 3 должен сформировать CSV и вывести итоговую сводку."""

    config_path = _build_valid_project(tmp_path)
    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--config",
            str(config_path.relative_to(tmp_path)),
            "--build-validation-dataset",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Проверочный датасет главы 6 успешно сформирован" in captured.out
    assert "Строк: 3" in captured.out
    assert "Этап 3 выполнен" in captured.out
    assert (tmp_path / "reports/chapter6/validation_dataset.csv").exists()


def test_cli_can_validate_inputs_and_build_dataset_in_one_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Флаги этапов 2 и 3 должны корректно работать совместно."""

    config_path = _build_valid_project(tmp_path)
    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--config",
            str(config_path.relative_to(tmp_path)),
            "--validate-inputs",
            "--build-validation-dataset",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Проверка входных артефактов главы 6 успешно завершена" in captured.out
    assert "Проверочный датасет главы 6 успешно сформирован" in captured.out
    assert (
        tmp_path / "reports/chapter6/chapter6_input_validation_report.json"
    ).exists()


def _make_builder(
    project_root: Path,
    config_path: Path,
) -> ValidationDatasetBuilder:
    """Создать построитель на основе тестового проекта."""

    return ValidationDatasetBuilder(
        config=_load_config(project_root, config_path),
        project_root=project_root,
    )


def _load_config(project_root: Path, config_path: Path):
    """Загрузить конфигурацию тестового проекта."""

    return load_chapter6_validation_config(
        config_path=config_path.relative_to(project_root),
        project_root=project_root,
    )


def _build_valid_project(project_root: Path) -> Path:
    """Сформировать минимальный корректный комплект входов этапа 3."""

    chapter5 = project_root / "reports/chapter5"
    chapter4 = project_root / "reports/chapter4"
    processed = project_root / "data/processed"
    configs = project_root / "configs"
    for directory in (chapter5, chapter4, processed, configs):
        directory.mkdir(parents=True, exist_ok=True)

    keys = pd.DataFrame(
        {
            "scenario_id": ["S1", "S2", "S3"],
            "protocol_id": ["P1", "P2", "P3"],
            "run_id": ["R1", "R1", "R1"],
            "alternative_id": ["A1", "A2", "A3"],
        }
    )
    q_pred_values = [0.44, 0.45, 0.70]
    predicted = {
        column: q_pred_values for column in PREDICTED_COLUMNS
    }
    keys.assign(
        **predicted,
        q_latent=[0.1, 0.4, 0.8],
        q_pred=q_pred_values,
    ).to_csv(chapter5 / "q_pred.csv", index=False)

    component_values: dict[str, list[float]] = {
        "q_latent": [0.1, 0.4, 0.8],
    }
    for prefix in ("q_acc", "q_time", "q_effort", "q_res", "q_rep", "q_fit"):
        component_values[f"{prefix}_feature_component"] = [0.2, 0.5, 0.8]
        component_values[f"{prefix}_latent_component"] = [0.1, 0.4, 0.8]
        component_values[f"{prefix}_observed_weight"] = [0.6, 0.6, 0.6]
        component_values[f"{prefix}_latent_weight"] = [0.4, 0.4, 0.4]
        component_values[f"{prefix}_pred"] = q_pred_values
    keys.assign(**component_values).to_csv(
        chapter5 / "q_pred_components.csv",
        index=False,
    )

    keys.assign(
        q_pred=q_pred_values,
        theta_0=[0.7, 0.2, 0.1],
        theta_1=[0.2, 0.5, 0.1],
        theta_2=[0.1, 0.3, 0.8],
        theta_entropy=[0.5, 0.8, 0.4],
        lda_instability=[0.1, 0.1, 0.1],
        input_missing_share=[0.0, 0.0, 0.0],
        uncertainty_score=[0.1, 0.2, 0.3],
        interval_radius=[0.05, 0.10, 0.10],
        q_pred_lower=[0.39, 0.35, 0.60],
        q_pred_upper=[0.49, 0.55, 0.80],
    ).to_csv(chapter5 / "prediction_uncertainty.csv", index=False)

    keys.assign(
        prior_condition_time_pressure_norm=[0.8, 0.5, 0.2],
        prior_operator_attention_norm=[0.2, 0.5, 0.9],
    ).to_csv(chapter5 / "normalized_prior_features.csv", index=False)

    keys.assign(
        theta_0=[0.7, 0.2, 0.1],
        theta_1=[0.2, 0.5, 0.1],
        theta_2=[0.1, 0.3, 0.8],
        q_latent=[0.1, 0.4, 0.8],
        latent_direction_score=[-0.8, -0.4, 0.6],
        theta_dominant_topic=["theta_0", "theta_1", "theta_2"],
        theta_dominant_value=[0.7, 0.5, 0.8],
    ).to_csv(chapter5 / "latent_quality_component.csv", index=False)

    keys.assign(
        theta_0=[0.7, 0.2, 0.1],
        theta_1=[0.2, 0.5, 0.1],
        theta_2=[0.1, 0.3, 0.8],
        document_index=[0, 1, 2],
        selected_k=[3, 3, 3],
        random_state=[42, 42, 42],
    ).to_csv(chapter4 / "theta_prior.csv", index=False)

    factual_values = [0.20, 0.45, 0.70]
    factual = {
        column: factual_values for column in FACTUAL_COLUMNS
    }
    pd.DataFrame(
        {
            "scenario_id": ["S1", "S2", "S3"],
            **factual,
            "integral_quality": factual_values,
        }
    ).to_csv(processed / "quality_targets.csv", index=False)
    pd.DataFrame(
        {
            "scenario_id": ["S1", "S2", "S3"],
            "fact_error_count": [3, 1, 0],
            "fact_duration_sec": [90.0, 60.0, 30.0],
            "fact_success": [0, 1, 1],
        }
    ).to_csv(processed / "fact_features.csv", index=False)

    prediction_report = {
        "stage": 11,
        "row_count": 3,
        "method_safety": {
            "apriori_only": True,
            "leakage_check_passed": True,
            "forbidden_column_count": 0,
            "full_pipeline_completed": True,
        },
    }
    acceptance_report = {
        "stage": 12,
        "accepted": True,
        "checks": {
            "artifact_existence": True,
            "all_output_row_counts": True,
            "leakage_check_passed": True,
        },
        "row_counts": {
            "q_pred": 3,
            "q_pred_components": 3,
            "prediction_uncertainty": 3,
        },
        "method_safety": {
            "apriori_only": True,
            "leakage_check_passed": True,
            "forbidden_column_count": 0,
            "full_pipeline_completed": True,
        },
    }
    (chapter5 / "chapter5_prediction_report.json").write_text(
        json.dumps(prediction_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (chapter5 / "chapter5_acceptance_report.json").write_text(
        json.dumps(acceptance_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    config_path = configs / "chapter6.yaml"
    config_path.write_text(
        """
inputs:
  q_pred_path: reports/chapter5/q_pred.csv
  q_pred_components_path: reports/chapter5/q_pred_components.csv
  prediction_uncertainty_path: reports/chapter5/prediction_uncertainty.csv
  chapter5_prediction_report_path: reports/chapter5/chapter5_prediction_report.json
  chapter5_acceptance_report_path: reports/chapter5/chapter5_acceptance_report.json
  normalized_prior_features_path: reports/chapter5/normalized_prior_features.csv
  latent_quality_component_path: reports/chapter5/latent_quality_component.csv
  theta_prior_path: reports/chapter4/theta_prior.csv
  quality_targets_path: data/processed/quality_targets.csv
  fact_features_path: data/processed/fact_features.csv
outputs:
  reports_dir: reports/chapter6
  input_validation_report_json_path: reports/chapter6/chapter6_input_validation_report.json
  input_validation_report_md_path: reports/chapter6/chapter6_input_validation_report.md
  validation_dataset_path: reports/chapter6/validation_dataset.csv
merge:
  key_columns: [scenario_id, protocol_id]
  validation: one_to_one
  expected_row_count: 3
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
