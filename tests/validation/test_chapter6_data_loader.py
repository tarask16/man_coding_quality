"""Тесты этапа 2: загрузка и проверка входных артефактов главы 6."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from manual_coding_sim.validation.chapter6_config import (
    load_chapter6_validation_config,
)
from manual_coding_sim.validation.chapter6_data_loader import (
    Chapter6DataLoadError,
    Chapter6DataLoader,
)
from manual_coding_sim.validation.chapter6_runner import main


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


def test_valid_inputs_are_loaded_and_checked(tmp_path: Path) -> None:
    """Корректный комплект входов должен пройти все проверки."""

    config_path = _build_valid_project(tmp_path)
    loaded = _make_loader(tmp_path, config_path).load()

    assert loaded.validation_report.passed is True
    assert loaded.validation_report.checked_csv_count == 8
    assert loaded.validation_report.checked_json_count == 2
    assert len(loaded.q_pred) == 3
    assert "protocol_id" in loaded.quality_targets.columns
    assert "protocol_id" in loaded.fact_features.columns
    assert loaded.validation_report.artifact_checks[
        "quality_targets"
    ].reconstructed_key_columns == ("protocol_id",)


def test_source_csv_is_not_changed_during_key_reconstruction(
    tmp_path: Path,
) -> None:
    """Восстановление protocol_id должно выполняться только в памяти."""

    config_path = _build_valid_project(tmp_path)
    source_path = tmp_path / "data/processed/quality_targets.csv"
    original_text = source_path.read_text(encoding="utf-8")

    _make_loader(tmp_path, config_path).load()

    assert source_path.read_text(encoding="utf-8") == original_text
    assert "protocol_id" not in pd.read_csv(source_path).columns


def test_reports_are_created_using_stage1_output_fields(tmp_path: Path) -> None:
    """Отчеты должны сохраняться по полям путей из этапа 1."""

    config_path = _build_valid_project(tmp_path)
    loaded = _make_loader(tmp_path, config_path).load_and_save_report()

    expected_json = (
        tmp_path / "reports/chapter6/chapter6_input_validation_report.json"
    )
    expected_md = (
        tmp_path / "reports/chapter6/chapter6_input_validation_report.md"
    )
    assert loaded.report_json_path == expected_json
    assert loaded.report_markdown_path == expected_md
    assert expected_json.exists()
    assert expected_md.exists()
    payload = json.loads(expected_json.read_text(encoding="utf-8"))
    assert payload["stage"] == 2
    assert payload["passed"] is True


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    """Отсутствующий обязательный файл должен блокировать этап."""

    config_path = _build_valid_project(tmp_path)
    (tmp_path / "reports/chapter5/q_pred.csv").unlink()

    with pytest.raises(FileNotFoundError, match="не найден"):
        _make_loader(tmp_path, config_path).load()


def test_missing_required_column_is_rejected(tmp_path: Path) -> None:
    """Отсутствие обязательной колонки должно фиксироваться явно."""

    config_path = _build_valid_project(tmp_path)
    path = tmp_path / "reports/chapter5/q_pred.csv"
    pd.read_csv(path).drop(columns=["q_pred"]).to_csv(path, index=False)

    with pytest.raises(Chapter6DataLoadError, match="q_pred"):
        _make_loader(tmp_path, config_path).load()


def test_wrong_row_count_is_rejected(tmp_path: Path) -> None:
    """Все основные таблицы должны содержать ожидаемое число сценариев."""

    config_path = _build_valid_project(tmp_path)
    path = tmp_path / "reports/chapter5/q_pred.csv"
    pd.read_csv(path).iloc[:2].to_csv(path, index=False)

    with pytest.raises(Chapter6DataLoadError, match="Неверное число строк"):
        _make_loader(tmp_path, config_path).load()


def test_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    """Дубли составного ключа должны блокировать загрузку."""

    config_path = _build_valid_project(tmp_path)
    path = tmp_path / "reports/chapter5/q_pred.csv"
    table = pd.read_csv(path)
    table.loc[2, ["scenario_id", "protocol_id"]] = table.loc[
        1, ["scenario_id", "protocol_id"]
    ]
    table.to_csv(path, index=False)

    with pytest.raises(Chapter6DataLoadError, match="не уникален"):
        _make_loader(tmp_path, config_path).load()


def test_non_finite_values_are_rejected(tmp_path: Path) -> None:
    """Бесконечные значения не должны попадать в проверочный контур."""

    config_path = _build_valid_project(tmp_path)
    path = tmp_path / "reports/chapter5/q_pred.csv"
    table = pd.read_csv(path)
    table.loc[0, "q_pred"] = np.inf
    table.to_csv(path, index=False)

    with pytest.raises(Chapter6DataLoadError, match="inf или -inf"):
        _make_loader(tmp_path, config_path).load()


def test_values_outside_unit_interval_are_rejected(tmp_path: Path) -> None:
    """Нормированные показатели должны находиться в диапазоне [0; 1]."""

    config_path = _build_valid_project(tmp_path)
    path = tmp_path / "data/processed/quality_targets.csv"
    table = pd.read_csv(path)
    table.loc[0, "integral_quality"] = 1.2
    table.to_csv(path, index=False)

    with pytest.raises(Chapter6DataLoadError, match="диапазон"):
        _make_loader(tmp_path, config_path).load()


def test_inconsistent_q_pred_is_rejected(tmp_path: Path) -> None:
    """Q_pred должен совпадать между точечным и интервальным прогнозами."""

    config_path = _build_valid_project(tmp_path)
    path = tmp_path / "reports/chapter5/prediction_uncertainty.csv"
    table = pd.read_csv(path)
    table.loc[0, "q_pred"] += 0.1
    table.to_csv(path, index=False)

    with pytest.raises(Chapter6DataLoadError, match="Q_pred не совпадает"):
        _make_loader(tmp_path, config_path).load()


def test_failed_chapter5_acceptance_is_rejected(tmp_path: Path) -> None:
    """Непринятая версия главы 5 не может использоваться в главе 6."""

    config_path = _build_valid_project(tmp_path)
    path = tmp_path / "reports/chapter5/chapter5_acceptance_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["accepted"] = False
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(Chapter6DataLoadError, match="приемка"):
        _make_loader(tmp_path, config_path).load()


def test_unknown_scenario_prevents_protocol_reconstruction(
    tmp_path: Path,
) -> None:
    """Ключ нельзя восстанавливать для неизвестного scenario_id."""

    config_path = _build_valid_project(tmp_path)
    path = tmp_path / "data/processed/quality_targets.csv"
    table = pd.read_csv(path)
    table.loc[0, "scenario_id"] = "S_UNKNOWN"
    table.to_csv(path, index=False)

    with pytest.raises(Chapter6DataLoadError, match="невозможно восстановить"):
        _make_loader(tmp_path, config_path).load()


def test_control_scenario_error_contains_recovery_hint(tmp_path: Path) -> None:
    """Контрольный A_TEST-сценарий должен давать диагностическую подсказку."""

    config_path = _build_valid_project(tmp_path)
    path = tmp_path / "data/processed/quality_targets.csv"
    table = pd.read_csv(path)
    table.loc[0, "scenario_id"] = "A_TEST_FINAL"
    table.to_csv(path, index=False)

    with pytest.raises(Chapter6DataLoadError, match="перезаписал data/processed"):
        _make_loader(tmp_path, config_path).load()


def test_cli_generates_stage2_reports(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI с --validate-inputs должен выполнить этап и создать отчеты."""

    config_path = _build_valid_project(tmp_path)
    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--config",
            str(config_path.relative_to(tmp_path)),
            "--validate-inputs",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Проверка входных артефактов главы 6 успешно завершена" in captured.out
    assert "Проверено CSV-файлов: 8" in captured.out
    assert (
        tmp_path / "reports/chapter6/chapter6_input_validation_report.json"
    ).exists()


def test_cli_show_config_preserves_stage1_json_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Режим --show-config должен сохранять JSON-вывод этапа 1."""

    config_path = _build_valid_project(tmp_path)
    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--config",
            str(config_path.relative_to(tmp_path)),
            "--show-config",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"expected_row_count": 3' in captured.out
    assert "expected_row_count = 3" in captured.out


def _make_loader(
    project_root: Path,
    config_path: Path,
) -> Chapter6DataLoader:
    """Создать загрузчик на основе тестовой YAML-конфигурации."""

    config = load_chapter6_validation_config(
        config_path=config_path.relative_to(project_root),
        project_root=project_root,
    )
    return Chapter6DataLoader(config=config, project_root=project_root)


def _build_valid_project(project_root: Path) -> Path:
    """Сформировать минимальный корректный комплект из десяти входов."""

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
        }
    )
    predicted = {
        column: [0.25, 0.50, 0.75] for column in PREDICTED_COLUMNS
    }
    keys.assign(**predicted, q_pred=[0.25, 0.50, 0.75]).to_csv(
        chapter5 / "q_pred.csv", index=False
    )
    keys.assign(**predicted, q_latent=[0.1, 0.3, 0.8]).to_csv(
        chapter5 / "q_pred_components.csv", index=False
    )
    keys.assign(
        q_pred=[0.25, 0.50, 0.75],
        theta_0=[0.7, 0.2, 0.1],
        theta_1=[0.2, 0.5, 0.1],
        theta_2=[0.1, 0.3, 0.8],
        uncertainty_score=[0.1, 0.2, 0.3],
        q_pred_lower=[0.20, 0.40, 0.60],
        q_pred_upper=[0.30, 0.60, 0.90],
    ).to_csv(chapter5 / "prediction_uncertainty.csv", index=False)
    keys.assign(prior_feature_norm=[0.1, 0.5, 0.9]).to_csv(
        chapter5 / "normalized_prior_features.csv", index=False
    )
    latent = keys.assign(
        theta_0=[0.7, 0.2, 0.1],
        theta_1=[0.2, 0.5, 0.1],
        theta_2=[0.1, 0.3, 0.8],
        q_latent=[0.1, 0.3, 0.8],
    )
    latent.to_csv(chapter5 / "latent_quality_component.csv", index=False)
    latent.drop(columns=["q_latent"]).to_csv(
        chapter4 / "theta_prior.csv", index=False
    )

    factual = {column: [0.3, 0.6, 0.9] for column in FACTUAL_COLUMNS}
    pd.DataFrame(
        {
            "scenario_id": ["S1", "S2", "S3"],
            **factual,
            "integral_quality": [0.3, 0.6, 0.9],
        }
    ).to_csv(processed / "quality_targets.csv", index=False)
    pd.DataFrame(
        {
            "scenario_id": ["S1", "S2", "S3"],
            "fact_error_count": [2, 1, 0],
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
