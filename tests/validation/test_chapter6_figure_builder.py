"""Тесты воспроизводимого построения рисунков главы 6 на этапе 12."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from manual_coding_sim.validation.chapter6_config import (
    Chapter6MergeConfig,
    Chapter6ValidationConfig,
)
from manual_coding_sim.validation.chapter6_figure_builder import (
    FIGURE_FILENAMES,
    Chapter6FigureBuildError,
    Chapter6FigureBuilder,
)
from manual_coding_sim.validation.chapter6_runner import main


ROW_COUNT = 12


def test_builder_creates_exact_figure_set(tmp_path: Path) -> None:
    """Этап должен создавать все восемь рисунков утвержденного перечня."""

    _write_sources(tmp_path)
    result = _make_builder(tmp_path).build_and_save()

    assert tuple(result.figure_paths) == FIGURE_FILENAMES
    assert all(path.exists() for path in result.figure_paths.values())
    assert result.manifest["figure_count"] == 8
    assert result.passed is True


def test_created_files_are_valid_png_with_sufficient_size(tmp_path: Path) -> None:
    """Рисунки должны иметь PNG-сигнатуру и достаточное разрешение."""

    _write_sources(tmp_path)
    result = _make_builder(tmp_path, dpi=120).build_and_save()

    for path in result.figure_paths.values():
        width, height = _read_png_dimensions(path)
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert width >= 900
        assert height >= 550
        assert path.stat().st_size > 10_000


def test_json_manifest_records_reproducibility(tmp_path: Path) -> None:
    """JSON-манифест должен фиксировать источники, хэши и параметры рендера."""

    _write_sources(tmp_path)
    result = _make_builder(tmp_path).build_and_save()
    payload = json.loads(result.manifest_json_path.read_text(encoding="utf-8"))

    assert payload["stage"] == 12
    assert payload["passed"] is True
    assert payload["row_count"] == ROW_COUNT
    assert payload["figure_count"] == 8
    assert payload["source_data_modified"] is False
    assert payload["manual_data_substitution"] is False
    assert len(payload["source_files"]) == 7
    assert len(payload["figures"]) == 8
    assert all(len(item["sha256"]) == 64 for item in payload["figures"])


def test_markdown_manifest_is_russian_and_complete(tmp_path: Path) -> None:
    """Markdown-манифест должен содержать русские пояснения и все имена файлов."""

    _write_sources(tmp_path)
    result = _make_builder(tmp_path).build_and_save()
    text = result.manifest_markdown_path.read_text(encoding="utf-8")

    assert "Манифест графических материалов главы 6" in text
    assert "Ручная подмена данных" in text
    assert all(filename in text for filename in FIGURE_FILENAMES)


def test_source_artifacts_are_not_modified(tmp_path: Path) -> None:
    """Построитель не должен изменять расчетные CSV этапов 3--11."""

    paths = _write_sources(tmp_path)
    before = {name: _sha256(path) for name, path in paths.items()}

    _make_builder(tmp_path).build_and_save()

    after = {name: _sha256(path) for name, path in paths.items()}
    assert after == before


def test_missing_source_file_is_rejected(tmp_path: Path) -> None:
    """Отсутствие обязательной таблицы должно блокировать этап 12."""

    paths = _write_sources(tmp_path)
    paths["baseline_comparison"].unlink()

    with pytest.raises(FileNotFoundError, match="baseline_comparison"):
        _make_builder(tmp_path).build_and_save()


def test_duplicate_composite_key_is_rejected(tmp_path: Path) -> None:
    """Повтор составного ключа в датасете должен блокировать рисунки."""

    paths = _write_sources(tmp_path)
    dataset = pd.read_csv(paths["validation_dataset"])
    dataset.loc[1, ["scenario_id", "protocol_id"]] = dataset.loc[
        0, ["scenario_id", "protocol_id"]
    ]
    dataset.to_csv(paths["validation_dataset"], index=False)

    with pytest.raises(Chapter6FigureBuildError, match="не является уникальным"):
        _make_builder(tmp_path).build_and_save()


def test_nonfinite_value_is_rejected(tmp_path: Path) -> None:
    """NaN в расчетном источнике не должен попадать на рисунок."""

    paths = _write_sources(tmp_path)
    errors = pd.read_csv(paths["integral_prediction_errors"])
    errors.loc[0, "absolute_error"] = np.nan
    errors.to_csv(paths["integral_prediction_errors"], index=False)

    with pytest.raises(Chapter6FigureBuildError, match="NaN"):
        _make_builder(tmp_path).build_and_save()


def test_confusion_matrix_total_is_revalidated(tmp_path: Path) -> None:
    """Матрица ошибок должна содержать ровно ожидаемое число сценариев."""

    paths = _write_sources(tmp_path)
    confusion = pd.read_csv(paths["confusion_matrix"])
    confusion.loc[0, "low"] += 1
    confusion.to_csv(paths["confusion_matrix"], index=False)

    with pytest.raises(Chapter6FigureBuildError, match="не равна числу сценариев"):
        _make_builder(tmp_path).build_and_save()


def test_all_baseline_models_are_required(tmp_path: Path) -> None:
    """График baseline должен основываться на полном наборе четырех моделей."""

    paths = _write_sources(tmp_path)
    baseline = pd.read_csv(paths["baseline_comparison"])
    baseline = baseline.loc[baseline["model"] != "theta_only_baseline"]
    baseline.to_csv(paths["baseline_comparison"], index=False)

    with pytest.raises(Chapter6FigureBuildError, match="четыре обязательные модели"):
        _make_builder(tmp_path).build_and_save()


def test_all_partial_criteria_are_required(tmp_path: Path) -> None:
    """Сравнение частных критериев должно включать все шесть показателей."""

    paths = _write_sources(tmp_path)
    partial = pd.read_csv(paths["partial_criteria_validation"])
    partial = partial.loc[partial["criterion"] != "q_fit"]
    partial.to_csv(paths["partial_criteria_validation"], index=False)

    with pytest.raises(Chapter6FigureBuildError, match="все шесть критериев"):
        _make_builder(tmp_path).build_and_save()


def test_cli_builds_figures_and_prints_stage_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI-флаг этапа 12 должен сформировать комплект и вывести статус."""

    _write_sources(tmp_path)
    config_path = _write_config(tmp_path)

    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--config",
            str(config_path),
            "--build-figures",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Графические материалы главы 6 успешно сформированы." in output
    assert "Рисунков: 8" in output
    assert "Разрешение: 300 DPI" in output
    assert "Этап 12 выполнен" in output
    assert all(
        (tmp_path / "reports/chapter6/figures" / filename).exists()
        for filename in FIGURE_FILENAMES
    )


def _make_builder(
    project_root: Path,
    *,
    dpi: int = 120,
) -> Chapter6FigureBuilder:
    """Создать построитель для тестового набора из двенадцати сценариев."""

    config = Chapter6ValidationConfig(
        merge=Chapter6MergeConfig(expected_row_count=ROW_COUNT),
    )
    return Chapter6FigureBuilder(
        config=config,
        project_root=project_root,
        dpi=dpi,
    )


def _write_sources(project_root: Path) -> dict[str, Path]:
    """Записать согласованный набор расчетных артефактов этапов 3--11."""

    reports = project_root / "reports/chapter6"
    reports.mkdir(parents=True, exist_ok=True)

    q_fact = np.array([0.20, 0.31, 0.42, 0.48, 0.55, 0.62, 0.68, 0.72, 0.78, 0.83, 0.88, 0.94])
    q_pred = np.array([0.24, 0.29, 0.39, 0.51, 0.50, 0.59, 0.65, 0.70, 0.75, 0.80, 0.84, 0.90])
    scenario_ids = [f"S{i:02d}" for i in range(ROW_COUNT)]
    protocol_ids = [f"P{i:02d}" for i in range(ROW_COUNT)]

    validation = pd.DataFrame(
        {
            "scenario_id": scenario_ids,
            "protocol_id": protocol_ids,
            "q_pred": q_pred,
            "q_fact": q_fact,
        }
    )
    validation_path = reports / "validation_dataset.csv"
    validation.to_csv(validation_path, index=False)

    prediction_error = q_pred - q_fact
    errors = validation.copy()
    errors["prediction_error"] = prediction_error
    errors["absolute_error"] = np.abs(prediction_error)
    errors["squared_error"] = prediction_error**2
    errors_path = reports / "integral_prediction_errors.csv"
    errors.to_csv(errors_path, index=False)

    factual_class = [_quality_class(value) for value in q_fact]
    predicted_class = [_quality_class(value) for value in q_pred]
    matrix = pd.crosstab(
        pd.Categorical(factual_class, categories=["low", "medium", "high"]),
        pd.Categorical(predicted_class, categories=["low", "medium", "high"]),
        dropna=False,
    )
    matrix.index.name = "factual_class"
    confusion_path = reports / "confusion_matrix.csv"
    matrix.reset_index().to_csv(confusion_path, index=False)

    baseline = pd.DataFrame(
        {
            "model": [
                "mean_baseline",
                "prior_only_baseline",
                "theta_only_baseline",
                "chapter5_model",
            ],
            "mae": [0.16, 0.11, 0.22, 0.06],
            "rmse": [0.19, 0.14, 0.27, 0.08],
            "spearman": [0.02, 0.72, 0.68, 0.91],
            "kendall": [0.01, 0.55, 0.49, 0.75],
        }
    )
    baseline_path = reports / "baseline_comparison.csv"
    baseline.to_csv(baseline_path, index=False)

    radius = np.full(ROW_COUNT, 0.08)
    intervals = validation.copy()
    intervals["q_pred_lower"] = np.clip(q_pred - radius, 0.0, 1.0)
    intervals["q_pred_upper"] = np.clip(q_pred + radius, 0.0, 1.0)
    intervals["is_covered"] = (
        (q_fact >= intervals["q_pred_lower"])
        & (q_fact <= intervals["q_pred_upper"])
    )
    intervals_path = reports / "interval_coverage_details.csv"
    intervals.to_csv(intervals_path, index=False)

    groups = pd.DataFrame(
        {
            "analysis_dimension": ["dominant_factor"] * 3,
            "group": ["theta_0", "theta_1", "theta_2"],
            "count": [4, 4, 4],
            "mae": [0.10, 0.08, 0.04],
            "bias": [-0.09, -0.06, -0.01],
        }
    )
    groups_path = reports / "error_group_analysis.csv"
    groups.to_csv(groups_path, index=False)

    partial = pd.DataFrame(
        {
            "criterion": [
                "q_acc",
                "q_time",
                "q_effort",
                "q_res",
                "q_rep",
                "q_fit",
            ],
            "mae": [0.07, 0.10, 0.08, 0.09, 0.11, 0.13],
            "rmse": [0.09, 0.13, 0.10, 0.12, 0.14, 0.16],
            "spearman": [0.88, 0.72, 0.77, 0.75, 0.70, 0.83],
            "kendall": [0.70, 0.54, 0.58, 0.57, 0.51, 0.64],
        }
    )
    partial_path = reports / "partial_criteria_validation.csv"
    partial.to_csv(partial_path, index=False)

    return {
        "validation_dataset": validation_path,
        "integral_prediction_errors": errors_path,
        "confusion_matrix": confusion_path,
        "baseline_comparison": baseline_path,
        "interval_coverage_details": intervals_path,
        "error_group_analysis": groups_path,
        "partial_criteria_validation": partial_path,
    }


def _write_config(project_root: Path) -> Path:
    """Записать минимальную конфигурацию CLI этапа 12."""

    path = project_root / "configs/chapter6.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
outputs:
  figures_dir: reports/chapter6/figures
merge:
  key_columns: [scenario_id, protocol_id]
  validation: one_to_one
  expected_row_count: 12
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


def _quality_class(value: float) -> str:
    """Классифицировать качество по порогам главы 6."""

    if value < 0.45:
        return "low"
    if value < 0.70:
        return "medium"
    return "high"


def _sha256(path: Path) -> str:
    """Рассчитать SHA-256 тестового файла."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_png_dimensions(path: Path) -> tuple[int, int]:
    """Прочитать размеры PNG из блока IHDR."""

    with path.open("rb") as stream:
        assert stream.read(8) == b"\x89PNG\r\n\x1a\n"
        assert struct.unpack(">I", stream.read(4))[0] >= 8
        assert stream.read(4) == b"IHDR"
        return struct.unpack(">II", stream.read(8))
