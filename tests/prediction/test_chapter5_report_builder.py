"""Тесты итогового отчета главы 5."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from manual_coding_sim.prediction import (
    Chapter5PredictionConfig,
    Chapter5ReportBuildError,
    Chapter5ReportBuilder,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Сохранить тестовый JSON-артефакт."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _prepare_artifacts(project_root: Path) -> None:
    """Создать минимальный согласованный набор артефактов главы 5."""

    report_dir = project_root / "reports/chapter5"
    report_dir.mkdir(parents=True, exist_ok=True)

    q_pred = pd.DataFrame(
        {
            "scenario_id": ["S1", "S2", "S3"],
            "protocol_id": ["P1", "P2", "P3"],
            "q_pred": [0.3, 0.6, 0.8],
        }
    )
    q_pred.to_csv(report_dir / "q_pred.csv", index=False)

    components = pd.DataFrame(
        {
            "scenario_id": ["S1", "S2", "S3"],
            "protocol_id": ["P1", "P2", "P3"],
            "q_acc_pred": [0.2, 0.6, 0.9],
            "q_time_pred": [0.3, 0.6, 0.8],
            "q_effort_pred": [0.4, 0.6, 0.7],
            "q_res_pred": [0.5, 0.6, 0.7],
            "q_rep_pred": [0.2, 0.6, 0.9],
            "q_fit_pred": [0.3, 0.6, 0.8],
        }
    )
    components.to_csv(report_dir / "q_pred_components.csv", index=False)

    uncertainty = pd.DataFrame(
        {
            "scenario_id": ["S1", "S2", "S3"],
            "protocol_id": ["P1", "P2", "P3"],
            "q_pred": [0.3, 0.6, 0.8],
            "uncertainty_score": [0.1, 0.2, 0.3],
            "interval_radius": [0.01, 0.02, 0.03],
            "q_pred_lower": [0.29, 0.58, 0.77],
            "q_pred_upper": [0.31, 0.62, 0.83],
        }
    )
    uncertainty.to_csv(report_dir / "prediction_uncertainty.csv", index=False)
    pd.DataFrame({"scenario_id": ["S1", "S2", "S3"]}).to_csv(
        report_dir / "normalized_prior_features.csv",
        index=False,
    )
    pd.DataFrame({"scenario_id": ["S1", "S2", "S3"]}).to_csv(
        report_dir / "latent_quality_component.csv",
        index=False,
    )

    criteria = ["q_acc", "q_time", "q_effort", "q_res", "q_rep", "q_fit"]
    _write_json(
        report_dir / "chapter5_leakage_report.json",
        {"is_safe": True, "forbidden_columns": []},
    )
    _write_json(
        report_dir / "normalization_report.json",
        {
            "row_count": 3,
            "input_column_count": 10,
            "normalized_feature_count": 6,
            "skipped_feature_count": 1,
            "non_numeric_features": ["prior_kind"],
            "unknown_input_features": [],
            "missing_dictionary_features": [],
        },
    )
    _write_json(
        report_dir / "latent_quality_component_report.json",
        {
            "row_count": 3,
            "theta_columns": ["theta_0", "theta_1", "theta_2"],
            "factor_directions": {"theta_0": -1.0, "theta_1": -1.0, "theta_2": 1.0},
            "q_latent_min": 0.1,
            "q_latent_max": 0.9,
            "q_latent_mean": 0.5,
            "dominant_topic_counts": {"theta_0": 1, "theta_1": 1, "theta_2": 1},
        },
    )
    _write_json(
        report_dir / "q_pred_components_report.json",
        {"row_count": 3, "criteria": criteria},
    )
    _write_json(
        report_dir / "q_pred_report.json",
        {
            "row_count": 3,
            "criteria": criteria,
            "weights": {criterion: 1.0 / 6.0 for criterion in criteria},
            "weight_sum": 1.0,
            "q_pred_min": 0.3,
            "q_pred_max": 0.8,
            "q_pred_mean": 0.5666666667,
            "q_pred_std": 0.2516611478,
        },
    )
    _write_json(
        report_dir / "prediction_uncertainty_report.json",
        {
            "row_count": 3,
            "weight_sum": 1.0,
            "delta": 0.15,
            "mean_stability": 0.85,
            "input_missing_share": 0.0,
            "uncertainty_score_min": 0.1,
            "uncertainty_score_max": 0.3,
            "uncertainty_score_mean": 0.2,
            "interval_radius_min": 0.01,
            "interval_radius_max": 0.03,
            "interval_radius_mean": 0.02,
            "q_pred_lower_min": 0.29,
            "q_pred_upper_max": 0.83,
        },
    )
    _write_json(
        report_dir / "chapter5_pipeline_run_report.json",
        {"completed_steps": {"normalization": True, "q_pred": True, "uncertainty": True}},
    )


def test_chapter5_report_builder_builds_json_and_markdown(tmp_path: Path) -> None:
    """Построитель должен формировать итоговые JSON- и Markdown-отчеты."""

    _prepare_artifacts(tmp_path)
    builder = Chapter5ReportBuilder()
    config = Chapter5PredictionConfig()

    report = builder.build_report(config=config, project_root=tmp_path)
    json_path = tmp_path / "reports/chapter5/chapter5_prediction_report.json"
    markdown_path = tmp_path / "reports/chapter5/chapter5_prediction_report.md"
    builder.save_outputs(report, json_report_path=json_path, markdown_report_path=markdown_path)

    assert report.stage == 11
    assert report.row_count == 3
    assert report.method_safety["leakage_check_passed"] is True
    assert report.quality_class_counts == {"low": 1, "medium": 1, "high": 1}
    assert json_path.exists()
    assert markdown_path.exists()
    assert "Итоговый отчет главы 5" in markdown_path.read_text(encoding="utf-8")


def test_chapter5_report_builder_rejects_missing_artifact(tmp_path: Path) -> None:
    """Отсутствие обязательного артефакта должно блокировать отчет."""

    builder = Chapter5ReportBuilder()

    with pytest.raises(Chapter5ReportBuildError, match="Не найдены артефакты"):
        builder.build_report(config=Chapter5PredictionConfig(), project_root=tmp_path)
