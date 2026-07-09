"""Тесты этапа 11: формирование датасета главы 3."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from manual_coding_sim.dataset_builder import (
    DatasetBuildResult,
    DatasetBuilder,
    DatasetBuilderConfig,
    build_dataset,
)


def _make_config(tmp_path: Path, run_count: int = 3) -> DatasetBuilderConfig:
    """Создает тестовую конфигурацию с временными каталогами."""
    return DatasetBuilderConfig(
        run_count=run_count,
        output_dir=tmp_path / "data" / "processed",
        reports_dir=tmp_path / "reports" / "chapter3",
    )


def test_dataset_builder_imports(tmp_path: Path) -> None:
    """Проверяет импортируемость построителя датасета."""
    builder = DatasetBuilder(_make_config(tmp_path))

    assert isinstance(builder.config, DatasetBuilderConfig)


def test_build_returns_dataset_build_result(tmp_path: Path) -> None:
    """Проверяет формирование датасета в памяти."""
    result = DatasetBuilder(_make_config(tmp_path)).build()

    assert isinstance(result, DatasetBuildResult)
    assert result.run_count == 3
    assert result.saved_files == {}


def test_dataset_rows_have_requested_count(tmp_path: Path) -> None:
    """Проверяет согласованность числа строк во всех таблицах."""
    result = DatasetBuilder(_make_config(tmp_path, run_count=4)).build()

    assert len(result.protocol_rows) == 4
    assert len(result.prior_feature_rows) == 4
    assert len(result.fact_feature_rows) == 4
    assert len(result.diagnostic_feature_rows) == 4
    assert len(result.quality_rows) == 4
    assert len(result.all_feature_rows) == 4


def test_prior_rows_do_not_contain_fact_features(tmp_path: Path) -> None:
    """Проверяет отсутствие фактических признаков в X_prior."""
    result = DatasetBuilder(_make_config(tmp_path)).build()
    prior_keys = set(result.prior_feature_rows[0])

    assert not any(key.startswith("fact_") for key in prior_keys)
    assert "prior_step_count" in prior_keys


def test_quality_rows_contain_quality_vector(tmp_path: Path) -> None:
    """Проверяет наличие всех компонентов q(A) в целевой таблице."""
    result = DatasetBuilder(_make_config(tmp_path)).build()
    quality_keys = set(result.quality_rows[0])

    assert {"q_acc", "q_time", "q_effort", "q_res", "q_rep", "q_fit"} <= quality_keys
    assert "integral_quality" in quality_keys


def test_summary_contains_dataset_statistics(tmp_path: Path) -> None:
    """Проверяет отчетную сводку по датасету."""
    result = DatasetBuilder(_make_config(tmp_path, run_count=5)).build()

    assert result.summary["run_count"] == 5
    assert result.summary["prior_feature_count"] > 0
    assert result.summary["fact_feature_count"] > 0
    assert 0.0 <= result.summary["mean_integral_quality"] <= 1.0


def test_build_and_save_creates_all_files(tmp_path: Path) -> None:
    """Проверяет сохранение CSV и JSON-артефактов."""
    result = DatasetBuilder(_make_config(tmp_path)).build_and_save()

    expected_keys = {
        "protocols",
        "prior_features",
        "fact_features",
        "diagnostic_features",
        "quality_targets",
        "all_features",
        "summary",
    }
    assert set(result.saved_files) == expected_keys
    assert all(Path(path).exists() for path in result.saved_files.values())


def test_saved_prior_csv_has_no_fact_columns(tmp_path: Path) -> None:
    """Проверяет, что сохраненный prior_features.csv не содержит X_fact."""
    result = DatasetBuilder(_make_config(tmp_path)).build_and_save()
    path = Path(result.saved_files["prior_features"])

    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)

    assert "prior_step_count" in header
    assert not any(column.startswith("fact_") for column in header)


def test_saved_quality_csv_has_quality_columns(tmp_path: Path) -> None:
    """Проверяет столбцы CSV с фактическими показателями качества."""
    result = DatasetBuilder(_make_config(tmp_path)).build_and_save()
    path = Path(result.saved_files["quality_targets"])

    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)

    assert "q_acc" in header
    assert "q_fit" in header
    assert "integral_quality" in header


def test_saved_summary_json_contains_saved_files(tmp_path: Path) -> None:
    """Проверяет JSON-сводку по сформированному датасету."""
    result = DatasetBuilder(_make_config(tmp_path)).build_and_save()
    path = Path(result.saved_files["summary"])
    summary = json.loads(path.read_text(encoding="utf-8"))

    assert summary["run_count"] == 3
    assert "saved_files" in summary
    assert "quality_targets" in summary["saved_files"]


def test_build_dataset_helper_saves_dataset(tmp_path: Path) -> None:
    """Проверяет вспомогательную функцию build_dataset()."""
    result = build_dataset(_make_config(tmp_path, run_count=2))

    assert result.run_count == 2
    assert Path(result.saved_files["protocols"]).exists()


def test_invalid_config_is_rejected(tmp_path: Path) -> None:
    """Проверяет отклонение некорректной конфигурации."""
    with pytest.raises(ValueError):
        DatasetBuilderConfig(run_count=0, output_dir=tmp_path).validate()

    with pytest.raises(ValueError):
        DatasetBuilderConfig(run_count=1, round_digits=-1).validate()


def test_overwrite_false_rejects_existing_files(tmp_path: Path) -> None:
    """Проверяет защиту от перезаписи сформированного датасета."""
    config = DatasetBuilderConfig(
        run_count=2,
        output_dir=tmp_path / "data" / "processed",
        reports_dir=tmp_path / "reports" / "chapter3",
        overwrite=False,
    )
    DatasetBuilder(config).build_and_save()

    with pytest.raises(FileExistsError):
        DatasetBuilder(config).build_and_save()
