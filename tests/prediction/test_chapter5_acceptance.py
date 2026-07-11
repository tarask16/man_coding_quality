"""Тесты финальной приемки артефактов главы 5."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from manual_coding_sim.prediction import (
    Chapter5AcceptanceError,
    Chapter5AcceptanceValidator,
    Chapter5PredictionConfig,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Сохранить тестовый JSON-файл."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _prepare_acceptance_artifacts(project_root: Path, row_count: int = 150) -> None:
    """Создать согласованный набор артефактов для приемочного теста."""

    data_dir = project_root / "data/processed"
    chapter4_dir = project_root / "reports/chapter4"
    chapter5_dir = project_root / "reports/chapter5"
    data_dir.mkdir(parents=True, exist_ok=True)
    chapter4_dir.mkdir(parents=True, exist_ok=True)
    chapter5_dir.mkdir(parents=True, exist_ok=True)

    scenario_ids = [f"S{i:03d}" for i in range(row_count)]
    protocol_ids = [f"P{i:03d}" for i in range(row_count)]
    base = pd.DataFrame({"scenario_id": scenario_ids, "protocol_id": protocol_ids})

    prior = base.copy()
    prior["prior_numeric"] = [i / row_count for i in range(row_count)]
    prior.to_csv(data_dir / "prior_features.csv", index=False)

    theta = base.copy()
    theta["theta_0"] = 0.2
    theta["theta_1"] = 0.3
    theta["theta_2"] = 0.5
    theta.to_csv(chapter4_dir / "theta_prior.csv", index=False)

    normalized = base.copy()
    normalized["prior_numeric_norm"] = prior["prior_numeric"]
    normalized.to_csv(chapter5_dir / "normalized_prior_features.csv", index=False)

    latent = base.copy()
    latent["q_latent"] = 0.5
    latent.to_csv(chapter5_dir / "latent_quality_component.csv", index=False)

    components = base.copy()
    for criterion in ["q_acc", "q_time", "q_effort", "q_res", "q_rep", "q_fit"]:
        components[f"{criterion}_pred"] = 0.5
    components.to_csv(chapter5_dir / "q_pred_components.csv", index=False)

    q_pred = base.copy()
    q_pred["q_pred"] = 0.5
    q_pred.to_csv(chapter5_dir / "q_pred.csv", index=False)

    uncertainty = base.copy()
    uncertainty["q_pred"] = 0.5
    uncertainty["uncertainty_score"] = 0.2
    uncertainty["interval_radius"] = 0.03
    uncertainty["q_pred_lower"] = 0.47
    uncertainty["q_pred_upper"] = 0.53
    uncertainty.to_csv(chapter5_dir / "prediction_uncertainty.csv", index=False)

    _write_json(chapter5_dir / "chapter5_leakage_report.json", {"is_safe": True, "forbidden_columns": []})
    _write_json(
        chapter5_dir / "chapter5_pipeline_run_report.json",
        {"completed_steps": {"normalization": True, "q_pred": True, "uncertainty": True}},
    )
    _write_json(
        chapter5_dir / "chapter5_prediction_report.json",
        {
            "stage": 11,
            "row_count": row_count,
            "method_safety": {"apriori_only": True},
        },
    )
    (chapter5_dir / "chapter5_prediction_report.md").write_text(
        "# Итоговый отчет главы 5\n",
        encoding="utf-8",
    )


def test_chapter5_acceptance_validator_accepts_consistent_artifacts(tmp_path: Path) -> None:
    """Приемка должна проходить для полного согласованного набора артефактов."""

    _prepare_acceptance_artifacts(tmp_path)
    validator = Chapter5AcceptanceValidator()

    report = validator.validate(config=Chapter5PredictionConfig(), project_root=tmp_path)
    json_path = tmp_path / "reports/chapter5/chapter5_acceptance_report.json"
    markdown_path = tmp_path / "reports/chapter5/chapter5_acceptance_report.md"
    validator.save_outputs(report, json_report_path=json_path, markdown_report_path=markdown_path)

    assert report.stage == 12
    assert report.accepted is True
    assert report.checks["theta_key_alignment"] is True
    assert report.row_counts["q_pred"] == 150
    assert report.quality_ranges["q_pred"] == {"min": 0.5, "max": 0.5}
    assert json_path.exists()
    assert markdown_path.exists()
    assert "Финальная приемка главы 5" in markdown_path.read_text(encoding="utf-8")


def test_chapter5_acceptance_validator_rejects_missing_artifacts(tmp_path: Path) -> None:
    """Отсутствие обязательных файлов должно блокировать финальную приемку."""

    validator = Chapter5AcceptanceValidator()

    with pytest.raises(Chapter5AcceptanceError, match="Не найдены артефакты"):
        validator.validate(config=Chapter5PredictionConfig(), project_root=tmp_path)
