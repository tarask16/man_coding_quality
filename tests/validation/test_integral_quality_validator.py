"""Тесты этапа 4: проверка фактического интегрального качества."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from manual_coding_sim.validation.chapter6_config import (
    Chapter6MergeConfig,
    Chapter6ValidationConfig,
)
from manual_coding_sim.validation.chapter6_runner import main
from manual_coding_sim.validation.integral_quality_validator import (
    IntegralQualityValidationError,
    IntegralQualityValidator,
    QUALITY_WEIGHTS,
)


QUALITY_COLUMNS = tuple(QUALITY_WEIGHTS)


def test_control_aggregation_uses_fixed_chapter5_weights(tmp_path: Path) -> None:
    """Контрольная сумма должна использовать зафиксированные веса главы 5."""

    dataset = _build_dataset()
    result = _make_validator(tmp_path).validate(dataset)

    expected = sum(dataset.loc[0, name] * weight for name, weight in QUALITY_WEIGHTS.items())
    assert result.details.loc[0, "q_fact_control"] == pytest.approx(expected)
    assert result.report["weight_sum"] == pytest.approx(1.0)
    assert result.report["weights"] == QUALITY_WEIGHTS


def test_integral_quality_remains_primary_q_fact(tmp_path: Path) -> None:
    """Контрольная агрегация не должна подменять целевую переменную."""

    result = _make_validator(tmp_path).validate(_build_dataset())

    assert result.report["q_fact_source"] == "integral_quality"
    assert result.details["q_fact"].tolist() == result.details["integral_quality"].tolist()
    assert not result.details["q_fact_control"].equals(result.details["q_fact"])


def test_result_passes_when_all_rows_are_within_tolerance(tmp_path: Path) -> None:
    """Проверка должна пройти при расхождении не более заданного допуска."""

    result = _make_validator(tmp_path, tolerance=0.05).validate(_build_dataset())

    assert result.passed is True
    assert result.report["metrics"]["outside_tolerance_count"] == 0
    assert result.report["metrics"]["consistency_rate"] == pytest.approx(1.0)


def test_result_fails_when_any_row_exceeds_tolerance(tmp_path: Path) -> None:
    """Хотя бы одно превышение допуска должно давать отрицательный статус."""

    result = _make_validator(tmp_path, tolerance=0.005).validate(_build_dataset())

    assert result.passed is False
    assert result.report["metrics"]["outside_tolerance_count"] > 0


def test_missing_quality_column_is_rejected(tmp_path: Path) -> None:
    """Отсутствие частного критерия должно блокировать этап 4."""

    invalid = _build_dataset().drop(columns=["q_rep"])

    with pytest.raises(IntegralQualityValidationError, match="q_rep"):
        _make_validator(tmp_path).validate(invalid)


def test_non_unique_composite_key_is_rejected(tmp_path: Path) -> None:
    """Повтор составного ключа должен блокировать контроль качества."""

    invalid = _build_dataset()
    invalid.loc[1, ["scenario_id", "protocol_id"]] = ["S1", "P1"]

    with pytest.raises(IntegralQualityValidationError, match="не является уникальным"):
        _make_validator(tmp_path).validate(invalid)


def test_q_fact_alias_mismatch_is_rejected(tmp_path: Path) -> None:
    """Этап 4 должен подтвердить неизменность q_fact после этапа 3."""

    invalid = _build_dataset()
    invalid.loc[0, "q_fact"] = invalid.loc[0, "integral_quality"] + 0.01

    with pytest.raises(IntegralQualityValidationError, match="точной копией"):
        _make_validator(tmp_path).validate(invalid)


def test_reports_are_saved_to_stage4_paths(tmp_path: Path) -> None:
    """Должны создаваться CSV-, JSON- и Markdown-артефакты этапа 4."""

    dataset_path = tmp_path / "reports/chapter6/validation_dataset.csv"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    _build_dataset().to_csv(dataset_path, index=False)

    result = _make_validator(tmp_path).validate_and_save()

    assert result.csv_path == tmp_path / "reports/chapter6/integral_quality_consistency.csv"
    assert result.json_path == (
        tmp_path / "reports/chapter6/integral_quality_consistency_report.json"
    )
    assert result.markdown_path == (
        tmp_path / "reports/chapter6/integral_quality_consistency_report.md"
    )
    assert result.csv_path.exists()
    assert result.json_path.exists()
    assert result.markdown_path.exists()
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["stage"] == 4
    assert payload["passed"] is True


def test_cli_validates_integral_quality(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI этапа 4 должен сформировать отчеты и вернуть успешный код."""

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
            "--validate-integral-quality",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Проверка фактического интегрального качества завершена" in captured.out
    assert "Этап 4 выполнен" in captured.out
    assert (tmp_path / "reports/chapter6/integral_quality_consistency_report.json").exists()


def test_cli_can_build_dataset_and_validate_stage4_in_one_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Этап 4 должен принимать датасет, сформированный в том же запуске CLI."""

    config_path = _write_config(tmp_path)
    dataset = _build_dataset()

    class _FakeBuildResult:
        output_path = tmp_path / "reports/chapter6/validation_dataset.csv"
        merge_steps = tuple()
        predicted_class_counts = {"low": 1, "medium": 1, "high": 1}
        factual_class_counts = {"low": 1, "medium": 1, "high": 1}

        def __init__(self) -> None:
            self.dataset = dataset

    def _fake_build_and_save(self, loaded_inputs=None):  # noqa: ANN001, ANN202
        self._resolve_output_path().parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(self._resolve_output_path(), index=False)
        return _FakeBuildResult()

    monkeypatch.setattr(
        "manual_coding_sim.validation.chapter6_runner.ValidationDatasetBuilder.build_and_save",
        _fake_build_and_save,
    )

    exit_code = main(
        [
            "--project-root",
            str(tmp_path),
            "--config",
            str(config_path.relative_to(tmp_path)),
            "--build-validation-dataset",
            "--validate-integral-quality",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Проверочный датасет главы 6 успешно сформирован" in captured.out
    assert "Этап 4 выполнен" in captured.out


def _make_validator(
    project_root: Path,
    tolerance: float = 0.05,
) -> IntegralQualityValidator:
    """Создать валидатор для трех тестовых сценариев."""

    config = Chapter6ValidationConfig(
        merge=Chapter6MergeConfig(expected_row_count=3),
    )
    return IntegralQualityValidator(
        config=config,
        project_root=project_root,
        tolerance=tolerance,
    )


def _build_dataset() -> pd.DataFrame:
    """Сформировать минимальный корректный датасет этапа 4."""

    rows = [
        {
            "scenario_id": "S1",
            "protocol_id": "P1",
            "q_acc": 0.80,
            "q_time": 0.70,
            "q_effort": 0.60,
            "q_res": 0.50,
            "q_rep": 0.40,
            "q_fit": 0.30,
            "integral_quality": 0.54,
            "q_fact": 0.54,
        },
        {
            "scenario_id": "S2",
            "protocol_id": "P2",
            "q_acc": 0.90,
            "q_time": 0.80,
            "q_effort": 0.70,
            "q_res": 0.60,
            "q_rep": 0.50,
            "q_fit": 0.40,
            "integral_quality": 0.64,
            "q_fact": 0.64,
        },
        {
            "scenario_id": "S3",
            "protocol_id": "P3",
            "q_acc": 1.00,
            "q_time": 0.90,
            "q_effort": 0.80,
            "q_res": 0.70,
            "q_rep": 0.60,
            "q_fit": 0.50,
            "integral_quality": 0.74,
            "q_fact": 0.74,
        },
    ]
    return pd.DataFrame(rows)


def _write_config(project_root: Path) -> Path:
    """Записать минимальную YAML-конфигурацию для CLI-теста."""

    config_path = project_root / "configs/chapter6.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        """
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
