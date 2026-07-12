"""Тесты этапа 7: проверка классификации уровней качества."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from manual_coding_sim.validation.chapter6_config import (
    Chapter6MergeConfig,
    Chapter6ValidationConfig,
)
from manual_coding_sim.validation.chapter6_runner import main
from manual_coding_sim.validation.classification_validator import (
    CLASS_LABELS,
    ClassificationValidationError,
    ClassificationValidator,
)


def test_known_confusion_matrix_and_general_metrics(tmp_path: Path) -> None:
    """Матрица ошибок и общие метрики должны соответствовать ручному расчету."""

    result = _make_validator(tmp_path).validate(_build_dataset())

    assert result.passed is True
    assert result.confusion_matrix.loc["low"].tolist() == [1, 1, 0]
    assert result.confusion_matrix.loc["medium"].tolist() == [0, 1, 1]
    assert result.confusion_matrix.loc["high"].tolist() == [1, 0, 1]
    assert result.report["metrics"]["accuracy"] == pytest.approx(0.5)
    assert result.report["metrics"]["balanced_accuracy"] == pytest.approx(0.5)
    assert result.report["metrics"]["macro_f1"] == pytest.approx(0.5)
    assert result.report["metrics"]["weighted_f1"] == pytest.approx(0.5)


def test_per_class_metrics_and_critical_errors(tmp_path: Path) -> None:
    """Метрики по классам и критические ошибки должны считаться раздельно."""

    result = _make_validator(tmp_path).validate(_build_dataset())
    rows = {row["class_label"]: row for row in result.report["per_class_metrics"]}

    for label in CLASS_LABELS:
        assert rows[label]["precision"] == pytest.approx(0.5)
        assert rows[label]["recall"] == pytest.approx(0.5)
        assert rows[label]["f1"] == pytest.approx(0.5)
        assert rows[label]["support"] == 2
        assert rows[label]["predicted_count"] == 2

    assert result.report["critical_errors"] == {
        "low_to_high": 0,
        "high_to_low": 1,
        "total": 1,
    }


def test_perfect_classification_has_unit_metrics(tmp_path: Path) -> None:
    """При полном совпадении классов все основные метрики должны быть равны 1."""

    dataset = _build_dataset()
    dataset["q_pred"] = dataset["q_fact"]

    result = _make_validator(tmp_path).validate(dataset)

    metrics = result.report["metrics"]
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["weighted_f1"] == pytest.approx(1.0)
    assert result.report["critical_errors"]["total"] == 0


def test_threshold_boundaries_are_applied_exactly(tmp_path: Path) -> None:
    """Значения на границах 0.45 и 0.70 должны попадать в заданные классы."""

    dataset = _build_dataset()
    dataset.loc[0, "q_pred"] = 0.449999
    dataset.loc[1, "q_pred"] = 0.45
    dataset.loc[2, "q_pred"] = 0.699999
    dataset.loc[3, "q_pred"] = 0.70

    result = _make_validator(tmp_path).validate(dataset)

    assert result.predictions.loc[0, "q_pred_class"] == "low"
    assert result.predictions.loc[1, "q_pred_class"] == "medium"
    assert result.predictions.loc[2, "q_pred_class"] == "medium"
    assert result.predictions.loc[3, "q_pred_class"] == "high"


def test_missing_required_column_is_rejected(tmp_path: Path) -> None:
    """Отсутствие q_pred или q_fact должно блокировать этап 7."""

    invalid = _build_dataset().drop(columns=["q_pred"])

    with pytest.raises(ClassificationValidationError, match="q_pred"):
        _make_validator(tmp_path).validate(invalid)


def test_non_unique_key_is_rejected(tmp_path: Path) -> None:
    """Повтор составного ключа должен блокировать классификационную проверку."""

    invalid = _build_dataset()
    invalid.loc[1, ["scenario_id", "protocol_id"]] = ["S1", "P1"]

    with pytest.raises(ClassificationValidationError, match="не является уникальным"):
        _make_validator(tmp_path).validate(invalid)


def test_invalid_numeric_values_are_rejected(tmp_path: Path) -> None:
    """Нечисловые, бесконечные и выходящие за диапазон значения недопустимы."""

    non_numeric = _build_dataset()
    non_numeric["q_pred"] = non_numeric["q_pred"].astype(object)
    non_numeric.loc[0, "q_pred"] = "ошибка"
    with pytest.raises(ClassificationValidationError, match="числовые"):
        _make_validator(tmp_path).validate(non_numeric)

    non_finite = _build_dataset()
    non_finite.loc[0, "q_fact"] = np.nan
    with pytest.raises(ClassificationValidationError, match="NaN"):
        _make_validator(tmp_path).validate(non_finite)

    out_of_range = _build_dataset()
    out_of_range.loc[0, "q_pred"] = 1.1
    with pytest.raises(ClassificationValidationError, match="диапазоне"):
        _make_validator(tmp_path).validate(out_of_range)


def test_stale_saved_classes_are_rejected(tmp_path: Path) -> None:
    """Классы этапа 3 должны совпадать с повторным расчетом по порогам."""

    invalid = _build_dataset()
    invalid["q_pred_class"] = invalid["q_pred"].map(
        lambda value: "low" if value < 0.45 else "medium" if value < 0.70 else "high"
    )
    invalid["q_fact_class"] = invalid["q_fact"].map(
        lambda value: "low" if value < 0.45 else "medium" if value < 0.70 else "high"
    )
    invalid.loc[0, "q_pred_class"] = "high"

    with pytest.raises(ClassificationValidationError, match="не соответствует"):
        _make_validator(tmp_path).validate(invalid)


def test_missing_factual_class_is_rejected(tmp_path: Path) -> None:
    """Для Balanced Accuracy в проверочной выборке нужны все три класса."""

    invalid = _build_dataset()
    invalid["q_fact"] = [0.2, 0.3, 0.5, 0.6, 0.55, 0.65]

    with pytest.raises(ClassificationValidationError, match="не содержит"):
        _make_validator(tmp_path).validate(invalid)


def test_reports_are_saved_to_stage7_paths(tmp_path: Path) -> None:
    """Этап должен создавать четыре предусмотренных артефакта."""

    dataset_path = tmp_path / "reports/chapter6/validation_dataset.csv"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    _build_dataset().to_csv(dataset_path, index=False)

    result = _make_validator(tmp_path).validate_and_save()

    assert result.predictions_path == (
        tmp_path / "reports/chapter6/classification_predictions.csv"
    )
    assert result.confusion_matrix_path == (
        tmp_path / "reports/chapter6/confusion_matrix.csv"
    )
    assert result.json_path == (
        tmp_path / "reports/chapter6/classification_report.json"
    )
    assert result.markdown_path == (
        tmp_path / "reports/chapter6/classification_report.md"
    )
    assert result.predictions_path.exists()
    assert result.confusion_matrix_path.exists()
    assert result.json_path.exists()
    assert result.markdown_path.exists()

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["stage"] == 7
    assert payload["passed"] is True
    assert payload["row_count"] == 6


def test_cli_validates_classification(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI этапа 7 должен создать отчеты и вывести основные метрики."""

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
            "--validate-classification",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Проверка классификации уровней качества завершена" in captured.out
    assert "Accuracy: 0.5000000000" in captured.out
    assert "Этап 7 выполнен" in captured.out
    assert (tmp_path / "reports/chapter6/classification_report.json").exists()


def test_cli_can_run_stages_6_and_7_together(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Флаги этапов 6 и 7 должны выполняться последовательно."""

    config_path = _write_config(tmp_path)
    dataset = _build_dataset()
    factual = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
    predicted = [0.2, 0.4, 0.6, 0.8, 0.7, 0.95]
    for name in ("acc", "time", "effort", "res", "rep", "fit"):
        dataset[f"q_{name}_pred"] = predicted
        dataset[f"q_{name}"] = factual
    dataset_path = tmp_path / "reports/chapter6/validation_dataset.csv"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(dataset_path, index=False)

    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--config",
            str(config_path.relative_to(tmp_path)),
            "--validate-partial-criteria",
            "--validate-classification",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Этап 6 выполнен" in captured.out
    assert "Этап 7 выполнен" in captured.out


def _make_validator(project_root: Path) -> ClassificationValidator:
    """Создать валидатор для шести тестовых сценариев."""

    config = Chapter6ValidationConfig(
        merge=Chapter6MergeConfig(expected_row_count=6),
    )
    return ClassificationValidator(config=config, project_root=project_root)


def _build_dataset() -> pd.DataFrame:
    """Сформировать датасет с известной матрицей ошибок 3 x 3."""

    return pd.DataFrame(
        {
            "scenario_id": ["S1", "S2", "S3", "S4", "S5", "S6"],
            "protocol_id": ["P1", "P2", "P3", "P4", "P5", "P6"],
            "q_fact": [0.2, 0.3, 0.5, 0.6, 0.8, 0.9],
            "q_pred": [0.2, 0.5, 0.5, 0.8, 0.8, 0.3],
        }
    )


def _write_config(project_root: Path) -> Path:
    """Записать минимальную конфигурацию CLI для шести сценариев."""

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
