"""Тесты финальной приемки программного контура главы 6 на этапе 14."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from manual_coding_sim.validation.chapter6_acceptance import (
    Chapter6AcceptanceBuilder,
    Chapter6AcceptanceError,
)
from manual_coding_sim.validation.chapter6_runner import main


ROW_COUNT = 6


def test_acceptance_creates_json_and_markdown(tmp_path: Path) -> None:
    """Приемка должна создавать машинный и текстовый акты."""

    _write_sources(tmp_path)
    result = _builder(tmp_path).build_and_save()

    assert result.accepted is True
    assert result.json_path is not None and result.json_path.exists()
    assert result.markdown_path is not None and result.markdown_path.exists()


def test_acceptance_report_has_required_contract(tmp_path: Path) -> None:
    """JSON должен соответствовать контракту этапа 14."""

    _write_sources(tmp_path)
    report = _builder(tmp_path).build().report

    assert report["stage"] == 14
    assert report["report_type"] == "chapter6_final_acceptance_report"
    assert report["accepted"] is True
    assert report["full_pipeline_completed"] is True
    assert report["prediction_model_frozen"] is True
    assert report["target_leakage_detected"] is False


def test_all_acceptance_checks_pass(tmp_path: Path) -> None:
    """Согласованный комплект должен проходить все проверки."""

    _write_sources(tmp_path)
    report = _builder(tmp_path).build().report

    assert report["check_count"] >= 25
    assert report["passed_check_count"] == report["check_count"]
    assert report["failed_check_count"] == 0
    assert report["failed_checks"] == []


def test_scientific_status_is_preserved(tmp_path: Path) -> None:
    """Техническая приемка не должна усиливать научный вывод."""

    _write_sources(tmp_path)
    report = _builder(tmp_path).build().report

    assert report["hypothesis_status"] == "hypothesis_partially_supported"
    assert report["scientific_status_is_not_overridden"] is True


def test_accepted_artifact_hashes_are_recorded(tmp_path: Path) -> None:
    """Акт должен фиксировать SHA-256 принятых источников."""

    _write_sources(tmp_path)
    report = _builder(tmp_path).build().report

    assert len(report["accepted_artifacts"]) == 13
    assert all(len(item["sha256"]) == 64 for item in report["accepted_artifacts"])


def test_missing_required_report_is_rejected(tmp_path: Path) -> None:
    """Отсутствие обязательного отчета должно блокировать приемку."""

    paths = _write_sources(tmp_path)
    paths["bootstrap"].unlink()

    with pytest.raises(FileNotFoundError, match="bootstrap"):
        _builder(tmp_path).build()


def test_wrong_report_stage_is_rejected(tmp_path: Path) -> None:
    """Неверный номер этапа должен блокировать приемку."""

    paths = _write_sources(tmp_path)
    payload = _read_json(paths["classification"])
    payload["stage"] = 70
    _write_json(paths["classification"], payload)

    with pytest.raises(Chapter6AcceptanceError, match="неверный номер этапа"):
        _builder(tmp_path).build()


def test_failed_source_report_is_rejected(tmp_path: Path) -> None:
    """Отрицательный passed исходного отчета должен блокировать приемку."""

    paths = _write_sources(tmp_path)
    payload = _read_json(paths["interval"])
    payload["passed"] = False
    _write_json(paths["interval"], payload)

    with pytest.raises(Chapter6AcceptanceError, match="положительного статуса"):
        _builder(tmp_path).build()


def test_duplicate_dataset_key_is_rejected(tmp_path: Path) -> None:
    """Повтор составного ключа должен приводить к отказу приемки."""

    paths = _write_sources(tmp_path)
    dataset = pd.read_csv(paths["dataset"])
    dataset.loc[1, ["scenario_id", "protocol_id"]] = dataset.loc[
        0, ["scenario_id", "protocol_id"]
    ]
    dataset.to_csv(paths["dataset"], index=False)
    _refresh_final_report_hashes(tmp_path)

    with pytest.raises(Chapter6AcceptanceError, match="composite_keys_unique"):
        _builder(tmp_path).build()


def test_out_of_range_quality_is_rejected(tmp_path: Path) -> None:
    """Выход показателя за [0; 1] должен блокировать приемку."""

    paths = _write_sources(tmp_path)
    dataset = pd.read_csv(paths["dataset"])
    dataset.loc[0, "q_pred"] = 1.2
    dataset.to_csv(paths["dataset"], index=False)
    _refresh_final_report_hashes(tmp_path)

    with pytest.raises(
        Chapter6AcceptanceError,
        match="quality_values_in_unit_interval",
    ):
        _builder(tmp_path).build()


def test_invalid_confusion_matrix_is_rejected(tmp_path: Path) -> None:
    """Матрица, не покрывающая все сценарии, должна блокировать приемку."""

    paths = _write_sources(tmp_path)
    matrix = pd.read_csv(paths["confusion"])
    matrix.loc[0, "low"] = 0
    matrix.to_csv(paths["confusion"], index=False)

    with pytest.raises(Chapter6AcceptanceError, match="confusion_matrix_created"):
        _builder(tmp_path).build()


def test_missing_baseline_model_is_rejected(tmp_path: Path) -> None:
    """Отсутствие одной baseline-модели должно блокировать приемку."""

    paths = _write_sources(tmp_path)
    frame = pd.read_csv(paths["baseline_table"])
    frame = frame[frame["model"] != "theta_only_baseline"]
    frame.to_csv(paths["baseline_table"], index=False)

    with pytest.raises(Chapter6AcceptanceError, match="baseline_models_built"):
        _builder(tmp_path).build()


def test_incomplete_bootstrap_differences_are_rejected(tmp_path: Path) -> None:
    """Неполный набор парных сравнений должен блокировать приемку."""

    paths = _write_sources(tmp_path)
    frame = pd.read_csv(paths["bootstrap_differences"]).iloc[:-1]
    frame.to_csv(paths["bootstrap_differences"], index=False)

    with pytest.raises(
        Chapter6AcceptanceError,
        match="paired_difference_intervals_calculated",
    ):
        _builder(tmp_path).build()


def test_incomplete_top10_is_rejected(tmp_path: Path) -> None:
    """Таблица менее чем из десяти ошибок должна блокировать приемку."""

    paths = _write_sources(tmp_path)
    frame = pd.read_csv(paths["top_errors"]).iloc[:-1]
    frame.to_csv(paths["top_errors"], index=False)

    with pytest.raises(Chapter6AcceptanceError, match="top10_errors_created"):
        _builder(tmp_path).build()


def test_missing_figure_is_rejected(tmp_path: Path) -> None:
    """Отсутствие рисунка из манифеста должно блокировать приемку."""

    paths = _write_sources(tmp_path)
    paths["figure_3"].unlink()

    with pytest.raises(Chapter6AcceptanceError, match="figures_created"):
        _builder(tmp_path).build()


def test_changed_chapter5_artifact_is_rejected(tmp_path: Path) -> None:
    """Изменение зафиксированного прогноза главы 5 должно блокировать приемку."""

    paths = _write_sources(tmp_path)
    paths["q_pred"].write_text("изменено", encoding="utf-8")

    with pytest.raises(
        Chapter6AcceptanceError,
        match="chapter5_artifacts_unchanged",
    ):
        _builder(tmp_path).build()


def test_incomplete_pipeline_is_rejected(tmp_path: Path) -> None:
    """Неполный CLI-контур должен блокировать финальную приемку."""

    paths = _write_sources(tmp_path)
    pipeline = _read_json(paths["pipeline"])
    pipeline["completed_steps"]["figures"] = False
    _write_json(paths["pipeline"], pipeline)

    with pytest.raises(
        Chapter6AcceptanceError,
        match="full_cli_pipeline_completed",
    ):
        _builder(tmp_path).build()


def test_synthetic_run_cannot_replace_main_dataset(tmp_path: Path) -> None:
    """Путь synthetic_runs не должен приниматься как основной эксперимент."""

    paths = _write_sources(tmp_path)
    synthetic = tmp_path / "synthetic_runs/main/validation_dataset.csv"
    synthetic.parent.mkdir(parents=True, exist_ok=True)
    synthetic.write_bytes(paths["dataset"].read_bytes())
    config = _config()
    config.outputs.validation_dataset_path = (
        "synthetic_runs/main/validation_dataset.csv"
    )

    with pytest.raises(
        Chapter6AcceptanceError,
        match="synthetic_holdout_excluded_from_main_acceptance",
    ):
        Chapter6AcceptanceBuilder(config=config, project_root=tmp_path).build()


def test_markdown_contains_acceptance_sections(tmp_path: Path) -> None:
    """Markdown должен содержать проверки, ограничения и решение."""

    _write_sources(tmp_path)
    result = _builder(tmp_path).build_and_save()
    text = result.markdown_path.read_text(encoding="utf-8")

    assert "Акт финальной приемки" in text
    assert "Приемочные проверки" in text
    assert "Методическая корректность" in text
    assert "Ограничения" in text
    assert "Программный контур главы 6 принят" in text


def test_cli_run_acceptance_creates_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI-флаг --run-acceptance должен создавать оба акта."""

    _write_sources(tmp_path)
    import manual_coding_sim.validation.chapter6_runner as runner

    monkeypatch.setattr(runner, "_load_config", lambda *_: _config())
    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--config",
            "configs/chapter6.yaml",
            "--run-acceptance",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Финальная приемка программного контура главы 6 завершена." in output
    assert "Этап 14 выполнен" in output
    assert (tmp_path / "reports/chapter6/chapter6_acceptance_report.json").exists()
    assert (tmp_path / "reports/chapter6/chapter6_acceptance_report.md").exists()


def _builder(project_root: Path) -> Chapter6AcceptanceBuilder:
    """Создать построитель приемки для тестового проекта."""

    return Chapter6AcceptanceBuilder(config=_config(), project_root=project_root)


def _config() -> SimpleNamespace:
    """Вернуть минимальную совместимую конфигурацию этапа 14."""

    outputs = SimpleNamespace(
        validation_dataset_path="reports/chapter6/validation_dataset.csv",
        chapter6_acceptance_report_json_path=(
            "reports/chapter6/chapter6_acceptance_report.json"
        ),
        chapter6_acceptance_report_md_path=(
            "reports/chapter6/chapter6_acceptance_report.md"
        ),
    )
    config = SimpleNamespace(
        outputs=outputs,
        merge=SimpleNamespace(
            key_columns=("scenario_id", "protocol_id"),
            validation="one_to_one",
            expected_row_count=ROW_COUNT,
        ),
        decision_thresholds=SimpleNamespace(low_max=0.45, high_min=0.70),
        bootstrap=SimpleNamespace(
            resamples=1000,
            confidence_level=0.95,
            random_seed=42,
            sampling_unit="scenario_id",
        ),
    )
    config.validate = lambda: None
    config.to_dict = lambda: {}
    return config


def _write_sources(project_root: Path) -> dict[str, Path]:
    """Создать полностью согласованный комплект приемочных источников."""

    reports = project_root / "reports/chapter6"
    figures_dir = reports / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    chapter5 = project_root / "reports/chapter5"
    chapter5.mkdir(parents=True, exist_ok=True)
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.DataFrame(
        {
            "scenario_id": [f"S{i}" for i in range(ROW_COUNT)],
            "protocol_id": [f"P{i}" for i in range(ROW_COUNT)],
            "q_pred": [0.20, 0.30, 0.45, 0.60, 0.75, 0.85],
            "q_fact": [0.25, 0.35, 0.50, 0.65, 0.80, 0.90],
            "integral_quality": [0.25, 0.35, 0.50, 0.65, 0.80, 0.90],
            "q_acc": [0.2, 0.3, 0.5, 0.6, 0.8, 0.9],
            "q_time": [0.3, 0.4, 0.5, 0.7, 0.8, 0.9],
            "q_effort": [0.2, 0.3, 0.4, 0.6, 0.7, 0.8],
            "q_res": [0.3, 0.4, 0.5, 0.6, 0.8, 0.9],
            "q_rep": [0.2, 0.4, 0.5, 0.7, 0.8, 0.9],
            "q_fit": [0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
            "q_acc_pred": [0.2, 0.3, 0.5, 0.6, 0.8, 0.9],
            "q_time_pred": [0.3, 0.4, 0.5, 0.7, 0.8, 0.9],
            "q_effort_pred": [0.2, 0.3, 0.4, 0.6, 0.7, 0.8],
            "q_res_pred": [0.3, 0.4, 0.5, 0.6, 0.8, 0.9],
            "q_rep_pred": [0.2, 0.4, 0.5, 0.7, 0.8, 0.9],
            "q_fit_pred": [0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
            "q_latent": [0.2, 0.3, 0.4, 0.6, 0.8, 0.9],
            "q_pred_lower": [0.1, 0.2, 0.3, 0.5, 0.7, 0.8],
            "q_pred_upper": [0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
            "uncertainty_score": [0.1, 0.2, 0.3, 0.2, 0.1, 0.1],
            "theta_0": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "theta_1": [0.3, 0.3, 0.3, 0.4, 0.3, 0.2],
            "theta_2": [0.1, 0.2, 0.3, 0.3, 0.5, 0.7],
            "prior_test_norm": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    dataset_path = reports / "validation_dataset.csv"
    dataset.to_csv(dataset_path, index=False)

    artifact_files: dict[str, Path] = {}
    for index, name in enumerate(
        (
            "q_pred",
            "q_pred_components",
            "prediction_uncertainty",
            "normalized_prior_features",
            "latent_quality_component",
            "theta_prior",
            "quality_targets",
            "fact_features",
        )
    ):
        path = project_root / f"data/input_{index}_{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "scenario_id": dataset["scenario_id"],
                "protocol_id": dataset["protocol_id"],
                "value": [0.5] * ROW_COUNT,
            }
        ).to_csv(path, index=False)
        artifact_files[name] = path

    input_report = {
        "stage": 2,
        "passed": True,
        "expected_row_count": ROW_COUNT,
        "checked_csv_count": 8,
        "checked_json_count": 2,
        "key_alignment_reference": "q_pred",
        "artifact_checks": {
            name: {
                "path": str(path.relative_to(project_root)),
                "row_count": ROW_COUNT,
                "required_columns_present": True,
                "unique_keys": True,
                "finite_values": True,
                "unit_interval_values": True,
                "key_set_aligned": True,
            }
            for name, path in artifact_files.items()
        },
        "q_pred_consistency": {"passed": True, "max_abs_difference": 0.0},
        "chapter5_acceptance": {"accepted": True},
    }
    integral_quality = {
        "stage": 4,
        "passed": True,
        "row_count": ROW_COUNT,
        "metrics": {
            "outside_tolerance_count": 0,
            "max_q_fact_alias_difference": 0.0,
        },
    }
    integral_prediction = {
        "stage": 5,
        "passed": True,
        "row_count": ROW_COUNT,
        "metrics": {
            "mae": 0.05,
            "rmse": 0.06,
            "bias": -0.01,
            "spearman": 0.90,
            "kendall": 0.75,
        },
    }
    partial = {
        "stage": 6,
        "passed": True,
        "row_count": ROW_COUNT,
        "criterion_count": 6,
        "metrics": [],
    }
    classification = {
        "stage": 7,
        "passed": True,
        "row_count": ROW_COUNT,
        "metrics": {"accuracy": 0.8, "balanced_accuracy": 0.8, "macro_f1": 0.8},
        "critical_errors": {"low_to_high": 0, "high_to_low": 0, "total": 0},
    }
    interval = {
        "stage": 8,
        "passed": True,
        "row_count": ROW_COUNT,
        "metrics": {"coverage_rate": 0.9},
    }
    leakage_checks = {
        "mean_baseline_is_out_of_fold": True,
        "test_fold_targets_excluded_from_training_mean": True,
        "prior_only_uses_q_fact": False,
        "theta_only_uses_q_fact": False,
        "chapter5_prediction_unchanged": True,
    }
    baseline = {
        "stage": 9,
        "passed": True,
        "row_count": ROW_COUNT,
        "metrics": [],
        "leakage_checks": leakage_checks,
    }
    bootstrap = {
        "stage": 10,
        "passed": True,
        "row_count": ROW_COUNT,
        "sampling": {
            "resamples": 1000,
            "confidence_level": 0.95,
            "random_seed": 42,
            "sampling_unit": "scenario_id",
        },
        "summary": {"comparison_count": 18},
    }
    error_analysis = {
        "stage": 11,
        "passed": True,
        "row_count": ROW_COUNT,
        "top_error_count": 10,
    }

    figure_items = []
    paths: dict[str, Path] = {"dataset": dataset_path}
    for index in range(8):
        figure_path = figures_dir / f"figure_{index}.png"
        figure_path.write_bytes(b"valid-png-placeholder")
        figure_items.append(
            {
                "filename": figure_path.name,
                "path": str(figure_path.relative_to(project_root)),
            }
        )
        paths[f"figure_{index}"] = figure_path
    figures = {
        "stage": 12,
        "passed": True,
        "row_count": ROW_COUNT,
        "figure_count": 8,
        "dpi": 300,
        "source_data_modified": False,
        "manual_data_substitution": False,
        "figures": figure_items,
    }

    report_payloads = {
        "input": ("chapter6_input_validation_report.json", input_report),
        "integral_quality": (
            "integral_quality_consistency_report.json",
            integral_quality,
        ),
        "integral": ("validation_metrics.json", integral_prediction),
        "partial": ("partial_criteria_validation_report.json", partial),
        "classification": ("classification_report.json", classification),
        "interval": ("interval_coverage_report.json", interval),
        "baselines": ("baseline_comparison_report.json", baseline),
        "bootstrap": ("bootstrap_report.json", bootstrap),
        "error": ("prediction_error_analysis.json", error_analysis),
        "figures": ("figures/figure_manifest.json", figures),
    }
    for name, (relative, payload) in report_payloads.items():
        path = reports / relative
        _write_json(path, payload)
        paths[name] = path

    confusion_path = reports / "confusion_matrix.csv"
    pd.DataFrame(
        {
            "factual_class": ["low", "medium", "high"],
            "low": [2, 0, 0],
            "medium": [0, 2, 0],
            "high": [0, 0, 2],
        }
    ).to_csv(confusion_path, index=False)
    paths["confusion"] = confusion_path

    baseline_table = reports / "baseline_comparison.csv"
    pd.DataFrame(
        {
            "model": [
                "mean_baseline",
                "prior_only_baseline",
                "theta_only_baseline",
                "chapter5_model",
            ],
            "mae": [0.2, 0.1, 0.3, 0.05],
        }
    ).to_csv(baseline_table, index=False)
    paths["baseline_table"] = baseline_table

    differences_path = reports / "bootstrap_model_differences.csv"
    rows = []
    for baseline_name in (
        "mean_baseline",
        "prior_only_baseline",
        "theta_only_baseline",
    ):
        for metric in ("mae", "rmse", "spearman", "kendall", "accuracy", "macro_f1"):
            rows.append(
                {
                    "baseline": baseline_name,
                    "metric": metric,
                    "ci_lower": -0.2,
                    "ci_upper": -0.1,
                    "conclusion": "chapter5_model_favored",
                }
            )
    pd.DataFrame(rows).to_csv(differences_path, index=False)
    paths["bootstrap_differences"] = differences_path

    top_errors_path = reports / "top_prediction_errors.csv"
    pd.DataFrame(
        {
            "scenario_id": [f"S{i % ROW_COUNT}" for i in range(10)],
            "protocol_id": [f"P{i % ROW_COUNT}" for i in range(10)],
            "absolute_error": [0.1 - i * 0.005 for i in range(10)],
        }
    ).to_csv(top_errors_path, index=False)
    paths["top_errors"] = top_errors_path

    final_md = reports / "chapter6_validation_report.md"
    final_md.write_text("# Итоговый отчет", encoding="utf-8")
    source_paths = [dataset_path] + [paths[name] for name in report_payloads]
    final_report = {
        "stage": 13,
        "passed": True,
        "row_count": ROW_COUNT,
        "hypothesis_status": "hypothesis_partially_supported",
        "technical_checks": {
            "all_source_reports_passed": True,
            "row_count_matches_expected": True,
            "composite_key_is_unique": True,
            "q_pred_consistency_confirmed": True,
            "chapter5_acceptance_confirmed": True,
            "baseline_leakage_checks_passed": True,
            "bootstrap_methodological_checks_passed": True,
            "prediction_model_not_modified": True,
            "quality_thresholds_not_modified": True,
            "figure_sources_not_modified": True,
            "manual_data_substitution_absent": True,
            "all_source_hashes_recorded": True,
        },
        "methodological_conclusion": {
            "recommended_claim": "Основная гипотеза подтверждена частично."
        },
        "limitations": ["Требуется независимая эмпирическая проверка."],
        "source_artifacts": [
            {
                "name": f"source_{index}",
                "path": str(path.relative_to(project_root)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for index, path in enumerate(source_paths)
        ],
    }
    final_report_path = reports / "chapter6_validation_report.json"
    _write_json(final_report_path, final_report)
    paths["final_report"] = final_report_path

    frozen_paths = {
        "reports/chapter5/q_pred.csv": chapter5 / "q_pred.csv",
        "reports/chapter5/q_pred_components.csv": chapter5 / "q_pred_components.csv",
        "reports/chapter5/prediction_uncertainty.csv": (
            chapter5 / "prediction_uncertainty.csv"
        ),
        "configs/chapter5.yaml": config_dir / "chapter5.yaml",
    }
    for relative, path in frozen_paths.items():
        path.write_text(f"frozen:{relative}", encoding="utf-8")
    pipeline = {
        "stage": 13,
        "passed": True,
        "full_pipeline_completed": True,
        "requested_optional_steps": {
            "build_figures": True,
            "build_report": True,
        },
        "completed_steps": {
            "input_validation": True,
            "validation_dataset": True,
            "integral_quality": True,
            "integral_metrics": True,
            "partial_criteria": True,
            "classification": True,
            "interval_prediction": True,
            "baseline_comparison": True,
            "bootstrap_analysis": True,
            "prediction_error_analysis": True,
            "figures": True,
            "final_report": True,
        },
        "methodological_checks": {
            "prediction_model_frozen": True,
            "chapter5_artifacts_unchanged": True,
            "target_leakage_detected": False,
            "factual_values_used_only_for_external_validation": True,
            "quality_thresholds_modified": False,
        },
        "frozen_artifact_hashes": {
            relative: _sha256(path) for relative, path in frozen_paths.items()
        },
        "error": None,
    }
    pipeline_path = reports / "chapter6_pipeline_run_report.json"
    _write_json(pipeline_path, pipeline)
    paths["pipeline"] = pipeline_path
    paths["q_pred"] = frozen_paths["reports/chapter5/q_pred.csv"]
    return paths


def _refresh_final_report_hashes(project_root: Path) -> None:
    """Обновить хэши источников после контролируемого изменения fixture."""

    path = project_root / "reports/chapter6/chapter6_validation_report.json"
    report = _read_json(path)
    for item in report["source_artifacts"]:
        source = project_root / item["path"]
        item["sha256"] = _sha256(source)
        item["size_bytes"] = source.stat().st_size
    _write_json(path, report)


def _write_json(path: Path, payload: object) -> None:
    """Записать JSON с русскоязычным содержимым."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    """Прочитать тестовый JSON-объект."""

    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    """Рассчитать SHA-256 тестового файла."""

    return hashlib.sha256(path.read_bytes()).hexdigest()
