"""Тесты конфигурационного каркаса главы 6."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from manual_coding_sim.validation.chapter6_config import (
    Chapter6BootstrapConfig,
    Chapter6ConfigError,
    Chapter6DecisionThresholds,
    Chapter6MergeConfig,
    Chapter6ValidationConfig,
    load_chapter6_validation_config,
)
from manual_coding_sim.validation.chapter6_runner import main

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_default_config_is_valid() -> None:
    """Конфигурация по умолчанию должна соответствовать методике главы 6."""

    config = Chapter6ValidationConfig()

    config.validate()

    assert config.merge.key_columns == ("scenario_id", "protocol_id")
    assert config.merge.validation == "one_to_one"
    assert config.merge.expected_row_count == 150
    assert config.bootstrap.resamples == 1000
    assert config.bootstrap.random_seed == 42


def test_project_yaml_loads_expected_values() -> None:
    """Основной YAML-файл проекта должен загружаться без ручной корректировки."""

    config = load_chapter6_validation_config(
        config_path="configs/chapter6.yaml",
        project_root=PROJECT_ROOT,
    )

    assert config.inputs.q_pred_path == Path("reports/chapter5/q_pred.csv")
    assert config.outputs.reports_dir == Path("reports/chapter6")
    assert config.decision_thresholds.low_max == pytest.approx(0.45)
    assert config.decision_thresholds.high_min == pytest.approx(0.70)
    assert config.bootstrap.sampling_unit == "scenario_id"


def test_invalid_threshold_order_is_rejected() -> None:
    """Пересекающиеся диапазоны классов должны блокировать запуск."""

    thresholds = Chapter6DecisionThresholds(low_max=0.75, high_min=0.70)

    with pytest.raises(Chapter6ConfigError, match="Пороги"):
        thresholds.validate()


def test_unexpected_merge_keys_are_rejected() -> None:
    """Проверочный датасет должен объединяться только по двум заданным ключам."""

    merge = Chapter6MergeConfig(key_columns=("scenario_id",))

    with pytest.raises(Chapter6ConfigError, match="Ключи объединения"):
        merge.validate()


def test_invalid_bootstrap_parameters_are_rejected() -> None:
    """Недостаточное число повторов и неизвестная единица выборки недопустимы."""

    too_few = Chapter6BootstrapConfig(resamples=10)
    wrong_unit = Chapter6BootstrapConfig(sampling_unit="alternative_id")

    with pytest.raises(Chapter6ConfigError, match="bootstrap-повторов"):
        too_few.validate(("scenario_id", "protocol_id"))
    with pytest.raises(Chapter6ConfigError, match="Единица bootstrap-выборки"):
        wrong_unit.validate(("scenario_id", "protocol_id"))


def test_loader_rejects_invalid_yaml_values(tmp_path: Path) -> None:
    """Загрузчик должен сообщать об ошибке методически неверного YAML."""

    config_path = tmp_path / "chapter6_invalid.yaml"
    config_path.write_text(
        """
merge:
  key_columns: [scenario_id, protocol_id]
  validation: one_to_one
  expected_row_count: 150
decision_thresholds:
  low_max: 0.8
  high_min: 0.7
bootstrap:
  resamples: 1000
  confidence_level: 0.95
  random_seed: 42
  sampling_unit: scenario_id
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(Chapter6ConfigError, match="Пороги"):
        load_chapter6_validation_config(config_path=config_path)


def test_runner_displays_checked_config(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI должен проверять и выводить полную конфигурацию главы 6."""

    exit_code = main(
        [
            "--project-root",
            str(PROJECT_ROOT),
            "--config",
            "configs/chapter6.yaml",
            "--show-config",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Конфигурационный каркас главы 6 успешно загружен" in captured.out
    assert '"expected_row_count": 150' in captured.out
    assert '"sampling_unit": "scenario_id"' in captured.out


def test_config_can_be_serialized_to_json() -> None:
    """Проверенная конфигурация должна сохранять машинно-читаемый вид."""

    config = load_chapter6_validation_config(
        config_path="configs/chapter6.yaml",
        project_root=PROJECT_ROOT,
    )

    serialized = json.dumps(config.to_dict(), ensure_ascii=False)

    assert "reports/chapter5/q_pred.csv" in serialized
    assert "one_to_one" in serialized
