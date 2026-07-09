"""Тесты единого runner-а главы 4."""

import csv
import json
from pathlib import Path

import pytest

from manual_coding_sim.lda.chapter4_runner import (
    Chapter4LdaRunner,
    Chapter4RunnerConfig,
)
from manual_coding_sim.lda.config import (
    Chapter4LdaConfig,
    LdaInputPaths,
    LdaModelConfig,
    LdaOutputPaths,
    LdaTokenizationConfig,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Записать тестовый CSV-файл."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, object]:
    """Прочитать JSON-файл."""

    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _prior_rows() -> list[dict[str, object]]:
    """Вернуть тестовые априорные признаки."""

    return [
        {
            "run_id": "r001",
            "protocol_id": "p001",
            "scenario_id": "s001",
            "has_control": 1,
            "message_length": 10,
            "procedure_type": "simple",
            "operator_skill": "high",
        },
        {
            "run_id": "r002",
            "protocol_id": "p002",
            "scenario_id": "s002",
            "has_control": 1,
            "message_length": 12,
            "procedure_type": "simple",
            "operator_skill": "high",
        },
        {
            "run_id": "r003",
            "protocol_id": "p003",
            "scenario_id": "s003",
            "has_control": 0,
            "message_length": 40,
            "procedure_type": "complex",
            "operator_skill": "low",
        },
        {
            "run_id": "r004",
            "protocol_id": "p004",
            "scenario_id": "s004",
            "has_control": 0,
            "message_length": 45,
            "procedure_type": "complex",
            "operator_skill": "low",
        },
        {
            "run_id": "r005",
            "protocol_id": "p005",
            "scenario_id": "s005",
            "has_control": 1,
            "message_length": 14,
            "procedure_type": "simple",
            "operator_skill": "medium",
        },
        {
            "run_id": "r006",
            "protocol_id": "p006",
            "scenario_id": "s006",
            "has_control": 0,
            "message_length": 50,
            "procedure_type": "complex",
            "operator_skill": "medium",
        },
    ]


def _diagnostic_rows() -> list[dict[str, object]]:
    """Вернуть тестовые диагностические признаки."""

    return [
        {
            "run_id": row["run_id"],
            "protocol_id": row["protocol_id"],
            "scenario_id": row["scenario_id"],
            "control_complexity": "low" if index < 2 else "high",
            "operator_stress": index / 10,
        }
        for index, row in enumerate(_prior_rows(), start=1)
    ]


def _fact_rows() -> list[dict[str, object]]:
    """Вернуть тестовые фактические признаки для диагностической LDA_full."""

    return [
        {
            "run_id": row["run_id"],
            "protocol_id": row["protocol_id"],
            "scenario_id": row["scenario_id"],
            "errors_total": index,
            "time_overrun": int(index >= 3),
        }
        for index, row in enumerate(_prior_rows())
    ]


def _prepare_project(tmp_path: Path, include_diagnostic: bool = True) -> None:
    """Создать минимальную структуру проекта для runner-а."""

    processed_dir = tmp_path / "data" / "processed"
    _write_csv(processed_dir / "prior_features.csv", _prior_rows())
    if include_diagnostic:
        _write_csv(processed_dir / "diagnostic_features.csv", _diagnostic_rows())
        _write_csv(processed_dir / "fact_features.csv", _fact_rows())


def _runner_config(build_diag: bool = True, build_full: bool = True) -> Chapter4RunnerConfig:
    """Вернуть быструю конфигурацию единого запуска главы 4."""

    return Chapter4RunnerConfig(
        chapter4=Chapter4LdaConfig(
            inputs=LdaInputPaths(),
            outputs=LdaOutputPaths(),
            tokenization=LdaTokenizationConfig(df_min=1, df_max_ratio=1.0),
            model=LdaModelConfig(
                k_values=(2, 3),
                selected_k=None,
                max_iter=2,
                random_seeds=(11, 42),
            ),
            build_lda_diag=build_diag,
            build_lda_full=build_full,
        ),
        top_n=3,
    )


def test_chapter4_runner_creates_complete_pipeline_artifacts(tmp_path: Path) -> None:
    """Runner должен создать полный набор артефактов главы 4."""

    _prepare_project(tmp_path)

    result = Chapter4LdaRunner(_runner_config()).run(project_root=tmp_path)

    reports_dir = tmp_path / "reports" / "chapter4"
    models_dir = tmp_path / "models" / "lda"
    data_dir = tmp_path / "data" / "processed" / "lda"
    assert result.selected_k in {2, 3}
    assert result.diag_result is not None
    assert result.full_result is not None
    assert (data_dir / "corpus_prior.csv").exists()
    assert (models_dir / "lda_prior.joblib").exists()
    assert (reports_dir / "theta_prior.csv").exists()
    assert (reports_dir / "topic_word.csv").exists()
    assert (reports_dir / "topic_metrics.json").exists()
    assert (reports_dir / "topic_stability_report.json").exists()
    assert (reports_dir / "topic_interpretation.md").exists()
    assert (reports_dir / "lda_diagnostic_metadata.json").exists()
    assert (reports_dir / "chapter4_lda_report.json").exists()

    payload = _read_json(reports_dir / "chapter4_lda_report.json")
    assert payload["status"] == "completed"
    assert payload["selected_k"] == result.selected_k
    assert payload["methodological_constraints"]["fact_features_forbidden_in_lda_prior"] is True


def test_chapter4_runner_supports_disabled_diagnostics(tmp_path: Path) -> None:
    """Runner должен уметь строить отчет без диагностических моделей."""

    _prepare_project(tmp_path, include_diagnostic=False)

    result = Chapter4LdaRunner(
        _runner_config(build_diag=False, build_full=False)
    ).run(project_root=tmp_path)

    payload = _read_json(result.report_result.report_json_path)
    assert result.diag_result is None
    assert result.full_result is None
    assert result.report_result.diagnostic_artifacts_included is False
    assert "lda_diagnostic_models" not in payload["completed_steps"]


def test_chapter4_runner_does_not_create_final_report_after_failure(
    tmp_path: Path,
) -> None:
    """При ошибке обязательного этапа runner не должен создавать итоговый отчет."""

    _prepare_project(tmp_path, include_diagnostic=False)

    with pytest.raises(FileNotFoundError):
        Chapter4LdaRunner(_runner_config()).run(project_root=tmp_path)

    assert not (tmp_path / "reports" / "chapter4" / "chapter4_lda_report.json").exists()


def test_chapter4_runner_requires_two_random_seeds() -> None:
    """Единый запуск должен требовать минимум два seed-а для устойчивости."""

    config = Chapter4RunnerConfig(
        chapter4=Chapter4LdaConfig(
            tokenization=LdaTokenizationConfig(df_min=1, df_max_ratio=1.0),
            model=LdaModelConfig(k_values=(2, 3), max_iter=2, random_seeds=(42,)),
        ),
        top_n=3,
    )

    with pytest.raises(ValueError, match="не менее двух random_seed"):
        Chapter4LdaRunner(config)
