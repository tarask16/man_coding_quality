"""Тесты итогового отчета и единого контура главы 6 на этапе 13."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from manual_coding_sim.validation.chapter6_config import (
    Chapter6MergeConfig,
    Chapter6ValidationConfig,
)
from manual_coding_sim.validation.chapter6_pipeline import (
    Chapter6Pipeline,
    Chapter6PipelineError,
)
from manual_coding_sim.validation.chapter6_report_builder import (
    HYPOTHESIS_NOT_SUPPORTED,
    HYPOTHESIS_PARTIALLY_SUPPORTED,
    HYPOTHESIS_SUPPORTED,
    Chapter6ReportBuildError,
    Chapter6ReportBuilder,
)
from manual_coding_sim.validation.chapter6_runner import main


ROW_COUNT = 6


def test_builder_creates_json_and_markdown(tmp_path: Path) -> None:
    """Построитель должен создавать оба итоговых отчета этапа 13."""

    _write_sources(tmp_path, mode="partial")
    result = _make_builder(tmp_path).build_and_save()

    assert result.passed is True
    assert result.json_path is not None and result.json_path.exists()
    assert result.markdown_path is not None and result.markdown_path.exists()


def test_final_report_has_stage_and_expected_row_count(tmp_path: Path) -> None:
    """Итоговый JSON должен фиксировать этап 13 и число сценариев."""

    _write_sources(tmp_path, mode="partial")
    result = _make_builder(tmp_path).build()

    assert result.report["stage"] == 13
    assert result.report["report_type"] == "chapter6_validation_final_report"
    assert result.report["row_count"] == ROW_COUNT
    assert result.report["expected_row_count"] == ROW_COUNT


def test_partial_evidence_produces_partial_status(tmp_path: Path) -> None:
    """Высокое ранжирование без улучшения MAE должно давать частичный статус."""

    _write_sources(tmp_path, mode="partial")
    result = _make_builder(tmp_path).build()

    assert result.hypothesis_status == HYPOTHESIS_PARTIALLY_SUPPORTED
    criteria = result.report["hypothesis"]["criteria"]
    assert criteria["ranking_point_advantage"] is True
    assert criteria["absolute_accuracy_stably_contradicted"] is True


def test_full_stable_advantage_produces_supported_status(tmp_path: Path) -> None:
    """Устойчивое преимущество во всех первичных измерениях подтверждает гипотезу."""

    _write_sources(tmp_path, mode="supported")
    result = _make_builder(tmp_path).build()

    assert result.hypothesis_status == HYPOTHESIS_SUPPORTED
    assert result.report["hypothesis"]["criteria"][
        "absolute_accuracy_stably_supported"
    ] is True
    assert result.report["hypothesis"]["criteria"][
        "ranking_stably_supported"
    ] is True


def test_absence_of_useful_advantage_produces_not_supported_status(
    tmp_path: Path,
) -> None:
    """Отсутствие преимущества должно приводить к отрицательному статусу."""

    _write_sources(tmp_path, mode="not_supported")
    result = _make_builder(tmp_path).build()

    assert result.hypothesis_status == HYPOTHESIS_NOT_SUPPORTED


def test_source_hashes_are_recorded(tmp_path: Path) -> None:
    """Для каждого источника должен быть записан SHA-256."""

    _write_sources(tmp_path, mode="partial")
    result = _make_builder(tmp_path).build()

    sources = result.report["source_artifacts"]
    assert len(sources) == 11
    assert all(len(item["sha256"]) == 64 for item in sources)
    assert result.report["technical_checks"]["all_source_hashes_recorded"] is True


def test_missing_required_report_is_rejected(tmp_path: Path) -> None:
    """Отсутствие отчета одного этапа должно блокировать итоговый отчет."""

    paths = _write_sources(tmp_path, mode="partial")
    paths["bootstrap"].unlink()

    with pytest.raises(FileNotFoundError, match="bootstrap"):
        _make_builder(tmp_path).build()


def test_wrong_stage_number_is_rejected(tmp_path: Path) -> None:
    """Неверный номер этапа в источнике должен блокировать сборку."""

    paths = _write_sources(tmp_path, mode="partial")
    payload = json.loads(paths["classification"].read_text(encoding="utf-8"))
    payload["stage"] = 70
    paths["classification"].write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(Chapter6ReportBuildError, match="неверный номер этапа"):
        _make_builder(tmp_path).build()


def test_failed_source_report_is_rejected(tmp_path: Path) -> None:
    """Отрицательный passed в исходном отчете должен блокировать этап 13."""

    paths = _write_sources(tmp_path, mode="partial")
    payload = json.loads(paths["interval"].read_text(encoding="utf-8"))
    payload["passed"] = False
    paths["interval"].write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(Chapter6ReportBuildError, match="положительного статуса"):
        _make_builder(tmp_path).build()


def test_cross_report_row_count_mismatch_is_rejected(tmp_path: Path) -> None:
    """Несогласованное число сценариев должно блокировать итоговый отчет."""

    paths = _write_sources(tmp_path, mode="partial")
    payload = json.loads(paths["integral"].read_text(encoding="utf-8"))
    payload["row_count"] = ROW_COUNT - 1
    paths["integral"].write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(Chapter6ReportBuildError, match="сценариев вместо"):
        _make_builder(tmp_path).build()


def test_duplicate_dataset_key_is_rejected(tmp_path: Path) -> None:
    """Повтор составного ключа должен блокировать итоговый отчет."""

    paths = _write_sources(tmp_path, mode="partial")
    dataset = pd.read_csv(paths["dataset"])
    dataset.loc[1, ["scenario_id", "protocol_id"]] = dataset.loc[
        0, ["scenario_id", "protocol_id"]
    ]
    dataset.to_csv(paths["dataset"], index=False)

    with pytest.raises(Chapter6ReportBuildError, match="не является уникальным"):
        _make_builder(tmp_path).build()


def test_missing_figure_from_manifest_is_rejected(tmp_path: Path) -> None:
    """Все восемь рисунков из манифеста должны существовать."""

    paths = _write_sources(tmp_path, mode="partial")
    (tmp_path / "reports/chapter6/figures/figure_3.png").unlink()

    with pytest.raises(FileNotFoundError, match="рисунок"):
        _make_builder(tmp_path).build()


def test_markdown_contains_scientific_sections(tmp_path: Path) -> None:
    """Markdown должен содержать гипотезу, метрики, ограничения и итоговый вывод."""

    _write_sources(tmp_path, mode="partial")
    result = _make_builder(tmp_path).build_and_save()
    text = result.markdown_path.read_text(encoding="utf-8")

    assert "Основная гипотеза" in text
    assert "Интегральный прогноз" in text
    assert "Сравнение с baseline-моделями" in text
    assert "Bootstrap-анализ" in text
    assert "Ограничения" in text
    assert HYPOTHESIS_PARTIALLY_SUPPORTED in text


def test_cli_build_report_creates_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI-флаг --build-report должен формировать итоговые артефакты."""

    _write_sources(tmp_path, mode="partial")
    config_path = _write_config(tmp_path)

    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--config",
            str(config_path),
            "--build-report",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Итоговый отчет экспериментальной проверки главы 6 сформирован." in output
    assert HYPOTHESIS_PARTIALLY_SUPPORTED in output
    assert (tmp_path / "reports/chapter6/chapter6_validation_report.json").exists()
    assert (tmp_path / "reports/chapter6/chapter6_validation_report.md").exists()


def test_pipeline_runs_required_steps_with_patched_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Оркестратор должен последовательно закрывать обязательные шаги 2--11."""

    import manual_coding_sim.validation.chapter6_pipeline as module

    calls: list[str] = []
    dataset = pd.DataFrame(
        {
            "scenario_id": [f"S{i}" for i in range(ROW_COUNT)],
            "protocol_id": [f"P{i}" for i in range(ROW_COUNT)],
            "q_pred": [0.2, 0.3, 0.4, 0.6, 0.8, 0.9],
            "q_fact": [0.25, 0.35, 0.45, 0.65, 0.75, 0.85],
        }
    )

    class FakeLoader:
        def __init__(self, **_: object) -> None:
            pass

        def load_and_save_report(self) -> SimpleNamespace:
            calls.append("input_validation")
            return SimpleNamespace(validation_report=SimpleNamespace(passed=True))

    class FakeDatasetBuilder:
        def __init__(self, **_: object) -> None:
            pass

        def build_and_save(self, **_: object) -> SimpleNamespace:
            calls.append("validation_dataset")
            return SimpleNamespace(dataset=dataset, output_path=None)

    class FakeStage:
        stage_name = ""

        def __init__(self, **_: object) -> None:
            pass

        def validate_and_save(self, **_: object) -> SimpleNamespace:
            calls.append(self.stage_name)
            result = SimpleNamespace(passed=True, report={"passed": True})
            if self.stage_name == "baseline_comparison":
                result.predictions = dataset.assign(
                    mean_baseline=0.5,
                    prior_only_baseline=0.5,
                    theta_only_baseline=0.5,
                    chapter5_model=dataset["q_pred"],
                )
            return result

        def analyze_and_save(self, **_: object) -> SimpleNamespace:
            calls.append(self.stage_name)
            return SimpleNamespace(passed=True, report={"passed": True})

    stage_mapping = {
        "IntegralQualityValidator": "integral_quality",
        "IntegralPredictionValidator": "integral_metrics",
        "PartialCriteriaValidator": "partial_criteria",
        "ClassificationValidator": "classification",
        "IntervalPredictionValidator": "interval_prediction",
        "BaselineModelsValidator": "baseline_comparison",
        "BootstrapAnalysisValidator": "bootstrap_analysis",
        "PredictionErrorAnalyzer": "prediction_error_analysis",
    }
    monkeypatch.setattr(module, "Chapter6DataLoader", FakeLoader)
    monkeypatch.setattr(module, "ValidationDatasetBuilder", FakeDatasetBuilder)
    for class_name, stage_name in stage_mapping.items():
        fake_class = type(
            f"Fake{class_name}",
            (FakeStage,),
            {"stage_name": stage_name},
        )
        monkeypatch.setattr(module, class_name, fake_class)

    result = Chapter6Pipeline(
        config=_config(),
        project_root=tmp_path,
    ).run()

    assert result.passed is True
    assert calls == [
        "input_validation",
        "validation_dataset",
        "integral_quality",
        "integral_metrics",
        "partial_criteria",
        "classification",
        "interval_prediction",
        "baseline_comparison",
        "bootstrap_analysis",
        "prediction_error_analysis",
    ]
    assert result.report_path.exists()


def test_pipeline_saves_failure_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """При ошибке шага оркестратор должен сохранить отрицательный отчет запуска."""

    import manual_coding_sim.validation.chapter6_pipeline as module

    class BrokenLoader:
        def __init__(self, **_: object) -> None:
            pass

        def load_and_save_report(self) -> None:
            raise ValueError("контрольная ошибка")

    monkeypatch.setattr(module, "Chapter6DataLoader", BrokenLoader)

    pipeline = Chapter6Pipeline(config=_config(), project_root=tmp_path)
    with pytest.raises(Chapter6PipelineError, match="контрольная ошибка"):
        pipeline.run()

    report_path = tmp_path / "reports/chapter6/chapter6_pipeline_run_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["full_pipeline_completed"] is False
    assert payload["step_results"]["input_validation"]["passed"] is False
    assert payload["error"]["type"] == "ValueError"


def _make_builder(project_root: Path) -> Chapter6ReportBuilder:
    """Создать построитель итогового отчета для шести сценариев."""

    return Chapter6ReportBuilder(config=_config(), project_root=project_root)


def _config() -> Chapter6ValidationConfig:
    """Вернуть минимальную конфигурацию этапа 13."""

    return Chapter6ValidationConfig(
        merge=Chapter6MergeConfig(expected_row_count=ROW_COUNT),
    )


def _write_config(project_root: Path) -> Path:
    """Записать минимальную YAML-конфигурацию CLI."""

    path = project_root / "configs/chapter6.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
outputs:
  reports_dir: reports/chapter6
  validation_dataset_path: reports/chapter6/validation_dataset.csv
  figures_dir: reports/chapter6/figures
  chapter6_validation_report_json_path: reports/chapter6/chapter6_validation_report.json
  chapter6_validation_report_md_path: reports/chapter6/chapter6_validation_report.md
  chapter6_pipeline_run_report_path: reports/chapter6/chapter6_pipeline_run_report.json
merge:
  key_columns: [scenario_id, protocol_id]
  validation: one_to_one
  expected_row_count: {ROW_COUNT}
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


def _write_sources(project_root: Path, *, mode: str) -> dict[str, Path]:
    """Создать согласованный комплект отчетов этапов 2--12."""

    reports = project_root / "reports/chapter6"
    figures_dir = reports / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    q_pred = [0.18, 0.32, 0.48, 0.62, 0.78, 0.90]
    q_fact = [0.22, 0.36, 0.52, 0.66, 0.82, 0.94]
    dataset = pd.DataFrame(
        {
            "scenario_id": [f"S{i}" for i in range(ROW_COUNT)],
            "protocol_id": [f"P{i}" for i in range(ROW_COUNT)],
            "q_pred": q_pred,
            "q_fact": q_fact,
        }
    )
    dataset_path = reports / "validation_dataset.csv"
    dataset.to_csv(dataset_path, index=False)

    input_report = {
        "stage": 2,
        "report_type": "chapter6_input_validation_report",
        "passed": True,
        "expected_row_count": ROW_COUNT,
        "checked_csv_count": 8,
        "checked_json_count": 2,
        "q_pred_consistency": {"passed": True, "max_abs_difference": 0.0},
        "chapter5_acceptance": {"accepted": True, "all_checks_passed": True},
    }
    integral_quality = {
        "stage": 4,
        "passed": True,
        "row_count": ROW_COUNT,
        "consistency_tolerance": 0.05,
        "metrics": {
            "mean_absolute_difference": 0.01,
            "max_absolute_difference": 0.03,
            "outside_tolerance_count": 0,
            "max_q_fact_alias_difference": 0.0,
        },
    }

    if mode == "not_supported":
        integral_metrics = {
            "mae": 0.20,
            "rmse": 0.24,
            "bias": -0.10,
            "median_absolute_error": 0.19,
            "max_absolute_error": 0.35,
            "pearson": 0.30,
            "spearman": 0.25,
            "kendall": 0.18,
            "r2": -1.2,
            "q_pred_mean": 0.50,
            "q_fact_mean": 0.60,
            "q_pred_std": 0.18,
            "q_fact_std": 0.12,
        }
    else:
        integral_metrics = {
            "mae": 0.08 if mode == "supported" else 0.16,
            "rmse": 0.10 if mode == "supported" else 0.19,
            "bias": -0.01 if mode == "supported" else -0.14,
            "median_absolute_error": 0.07,
            "max_absolute_error": 0.22,
            "pearson": 0.91,
            "spearman": 0.88,
            "kendall": 0.70,
            "r2": 0.72 if mode == "supported" else -1.9,
            "q_pred_mean": 0.55,
            "q_fact_mean": 0.56,
            "q_pred_std": 0.20,
            "q_fact_std": 0.19,
        }
    integral = {
        "stage": 5,
        "passed": True,
        "row_count": ROW_COUNT,
        "error_definition": "prediction_error = q_pred - q_fact",
        "metrics": integral_metrics,
    }
    partial = {
        "stage": 6,
        "passed": True,
        "row_count": ROW_COUNT,
        "criterion_count": 6,
        "summary": {
            "mean_mae": 0.15,
            "mean_rmse": 0.18,
            "mean_spearman": 0.70,
            "mean_kendall": 0.55,
            "best_mae_criterion": "q_effort",
            "worst_mae_criterion": "q_fit",
        },
        "metrics": [
            {"criterion": name, "mae": 0.10 + index * 0.01}
            for index, name in enumerate(
                ["q_acc", "q_time", "q_effort", "q_res", "q_rep", "q_fit"]
            )
        ],
    }
    classification = {
        "stage": 7,
        "passed": True,
        "row_count": ROW_COUNT,
        "metrics": {
            "accuracy": 0.78 if mode == "supported" else 0.45,
            "balanced_accuracy": 0.82 if mode == "supported" else 0.68,
            "macro_f1": 0.80 if mode == "supported" else 0.43,
            "weighted_f1": 0.79 if mode == "supported" else 0.56,
        },
        "critical_errors": {"low_to_high": 0, "high_to_low": 0, "total": 0},
        "per_class_metrics": [
            {"class_label": "low", "support": 1},
            {"class_label": "medium", "support": 3},
            {"class_label": "high", "support": 2},
        ],
    }
    interval = {
        "stage": 8,
        "passed": True,
        "row_count": ROW_COUNT,
        "coverage_condition": "q_pred_lower <= q_fact <= q_pred_upper",
        "metrics": {
            "coverage_rate": 0.90 if mode == "supported" else 0.19,
            "covered_count": 5 if mode == "supported" else 1,
            "miss_count": 1 if mode == "supported" else 5,
            "mean_interval_width": 0.10,
            "median_interval_width": 0.09,
            "miss_lower_count": 0,
            "miss_upper_count": 1 if mode == "supported" else 5,
            "mean_distance_to_interval": 0.01 if mode == "supported" else 0.12,
            "mean_miss_distance": 0.05,
            "max_distance_to_interval": 0.20,
        },
        "slices": {},
    }

    if mode == "supported":
        prior = {
            "mae": 0.14,
            "rmse": 0.17,
            "spearman": 0.80,
            "kendall": 0.61,
            "accuracy": 0.65,
            "balanced_accuracy": 0.69,
            "macro_f1": 0.66,
        }
        chapter5 = {
            "mae": 0.08,
            "rmse": 0.10,
            "spearman": 0.92,
            "kendall": 0.76,
            "accuracy": 0.80,
            "balanced_accuracy": 0.82,
            "macro_f1": 0.80,
        }
        prior_conclusions = {metric: "chapter5_model_favored" for metric in (
            "mae", "rmse", "spearman", "kendall", "accuracy", "macro_f1"
        )}
    elif mode == "not_supported":
        prior = {
            "mae": 0.09,
            "rmse": 0.11,
            "spearman": 0.75,
            "kendall": 0.60,
            "accuracy": 0.70,
            "balanced_accuracy": 0.72,
            "macro_f1": 0.70,
        }
        chapter5 = {
            "mae": 0.20,
            "rmse": 0.24,
            "spearman": 0.25,
            "kendall": 0.18,
            "accuracy": 0.40,
            "balanced_accuracy": 0.42,
            "macro_f1": 0.39,
        }
        prior_conclusions = {metric: "baseline_favored" for metric in (
            "mae", "rmse", "spearman", "kendall", "accuracy", "macro_f1"
        )}
    else:
        prior = {
            "mae": 0.11,
            "rmse": 0.13,
            "spearman": 0.86,
            "kendall": 0.67,
            "accuracy": 0.51,
            "balanced_accuracy": 0.65,
            "macro_f1": 0.39,
        }
        chapter5 = {
            "mae": 0.16,
            "rmse": 0.19,
            "spearman": 0.88,
            "kendall": 0.70,
            "accuracy": 0.45,
            "balanced_accuracy": 0.67,
            "macro_f1": 0.42,
        }
        prior_conclusions = {
            "mae": "baseline_favored",
            "rmse": "baseline_favored",
            "spearman": "no_stable_difference",
            "kendall": "no_stable_difference",
            "accuracy": "no_stable_difference",
            "macro_f1": "no_stable_difference",
        }

    def model_row(model: str, values: dict[str, float]) -> dict[str, float | str]:
        return {
            "model": model,
            "bias": 0.0,
            **values,
        }

    baseline = {
        "stage": 9,
        "passed": True,
        "row_count": ROW_COUNT,
        "metrics": [
            model_row(
                "mean_baseline",
                {
                    "mae": 0.18,
                    "rmse": 0.21,
                    "spearman": 0.02,
                    "kendall": 0.01,
                    "accuracy": 0.50,
                    "balanced_accuracy": 0.33,
                    "macro_f1": 0.26,
                },
            ),
            model_row("prior_only_baseline", prior),
            model_row(
                "theta_only_baseline",
                {
                    "mae": 0.25,
                    "rmse": 0.30,
                    "spearman": 0.79,
                    "kendall": 0.60,
                    "accuracy": 0.40,
                    "balanced_accuracy": 0.55,
                    "macro_f1": 0.35,
                },
            ),
            model_row("chapter5_model", chapter5),
        ],
        "best_models": {},
        "chapter5_differences": {},
        "leakage_checks": {
            "mean_baseline_is_out_of_fold": True,
            "test_fold_targets_excluded_from_training_mean": True,
            "prior_only_uses_q_fact": False,
            "theta_only_uses_q_fact": False,
            "chapter5_prediction_unchanged": True,
        },
    }

    differences = []
    all_baselines = ["mean_baseline", "prior_only_baseline", "theta_only_baseline"]
    metrics_order = ["mae", "rmse", "spearman", "kendall", "accuracy", "macro_f1"]
    for baseline_name in all_baselines:
        for metric in metrics_order:
            if baseline_name == "prior_only_baseline":
                conclusion = prior_conclusions[metric]
            elif mode == "not_supported":
                conclusion = "baseline_favored"
            else:
                conclusion = "chapter5_model_favored"
            differences.append(
                {
                    "baseline": baseline_name,
                    "metric": metric,
                    "point_delta": 0.1,
                    "ci_lower": 0.05,
                    "ci_upper": 0.15,
                    "ci_includes_zero": False,
                    "conclusion": conclusion,
                }
            )
    stable_chapter5 = sum(
        row["conclusion"] == "chapter5_model_favored" for row in differences
    )
    stable_baseline = sum(
        row["conclusion"] == "baseline_favored" for row in differences
    )
    bootstrap = {
        "stage": 10,
        "passed": True,
        "row_count": ROW_COUNT,
        "sampling": {
            "method": "paired_cluster_percentile_bootstrap",
            "sampling_unit": "scenario_id",
            "sampling_unit_count": ROW_COUNT,
            "resamples": 1000,
            "confidence_level": 0.95,
            "random_seed": 42,
        },
        "chapter5_confidence_intervals": [
            {
                "model": "chapter5_model",
                "metric": "mae",
                "point_estimate": chapter5["mae"],
                "ci_lower": chapter5["mae"] - 0.01,
                "ci_upper": chapter5["mae"] + 0.01,
            }
        ],
        "model_differences": differences,
        "summary": {
            "stable_chapter5_wins": stable_chapter5,
            "stable_baseline_wins": stable_baseline,
            "no_stable_difference": 18 - stable_chapter5 - stable_baseline,
            "comparison_count": 18,
        },
        "methodological_checks": {
            "paired_resamples_used_for_all_models": True,
            "fixed_stage9_predictions_used": True,
            "models_refitted_inside_bootstrap": False,
            "chapter5_prediction_modified": False,
            "quality_thresholds_modified": False,
            "factual_values_used_only_for_external_validation": True,
        },
    }
    error_analysis = {
        "stage": 11,
        "passed": True,
        "row_count": ROW_COUNT,
        "top_error_count": ROW_COUNT,
        "summary": {
            "mae": integral_metrics["mae"],
            "rmse": integral_metrics["rmse"],
            "bias": integral_metrics["bias"],
            "max_absolute_error": integral_metrics["max_absolute_error"],
            "underestimation_count": 5,
            "overestimation_count": 1,
            "exact_count": 0,
            "worst_dominant_factor_by_mae": "theta_0",
            "worst_dominant_factor_mae": 0.20,
        },
        "uncertainty_relation": {
            "spearman_absolute_error": 0.16,
        },
        "group_highlights": {},
        "methodological_checks": {
            "chapter5_prediction_modified": False,
            "quality_thresholds_modified": False,
            "factual_values_used_only_for_external_validation": True,
        },
    }

    figure_records = []
    for index in range(8):
        path = figures_dir / f"figure_{index}.png"
        path.write_bytes(b"synthetic-png-content-" + bytes([index]))
        figure_records.append(
            {
                "filename": path.name,
                "path": str(path.relative_to(project_root)),
                "sha256": "a" * 64,
                "width_px": 1200,
                "height_px": 800,
                "size_bytes": path.stat().st_size,
            }
        )
    figure_manifest = {
        "stage": 12,
        "passed": True,
        "row_count": ROW_COUNT,
        "figure_count": 8,
        "dpi": 300,
        "source_data_modified": False,
        "manual_data_substitution": False,
        "figures": figure_records,
    }

    payloads = {
        "input": (reports / "chapter6_input_validation_report.json", input_report),
        "quality": (
            reports / "integral_quality_consistency_report.json",
            integral_quality,
        ),
        "integral": (reports / "validation_metrics.json", integral),
        "partial": (reports / "partial_criteria_validation_report.json", partial),
        "classification": (reports / "classification_report.json", classification),
        "interval": (reports / "interval_coverage_report.json", interval),
        "baseline": (reports / "baseline_comparison_report.json", baseline),
        "bootstrap": (reports / "bootstrap_report.json", bootstrap),
        "error": (reports / "prediction_error_analysis.json", error_analysis),
        "figures": (figures_dir / "figure_manifest.json", figure_manifest),
    }
    result_paths = {"dataset": dataset_path}
    for name, (path, payload) in payloads.items():
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result_paths[name] = path
    return result_paths
