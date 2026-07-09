"""Тесты этапа 12: запуск воспроизводимого эксперимента."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from manual_coding_sim.experiment_runner import (
    ExperimentRunResult,
    ExperimentRunner,
    ExperimentRunnerConfig,
    hash_dataset_result,
    run_experiment,
    run_experiment_from_yaml,
)


def _write_config(tmp_path: Path, run_count: int = 3, seed: int = 42) -> Path:
    """Создает временную YAML-конфигурацию эксперимента."""
    config_path = tmp_path / "base_experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: test_chapter3_experiment",
                f"  random_seed: {seed}",
                f"  run_count: {run_count}",
                "  scenario_id: A_TEST",
                f"  output_dir: {tmp_path.as_posix()}/data/processed",
                f"  reports_dir: {tmp_path.as_posix()}/reports/chapter3",
                "  overwrite: true",
                "  check_reproducibility: true",
            ],
        ),
        encoding="utf-8",
    )
    return config_path


def _make_config(tmp_path: Path, run_count: int = 3) -> ExperimentRunnerConfig:
    """Создает тестовую конфигурацию запускателя эксперимента."""
    return ExperimentRunnerConfig(
        experiment_name="test_chapter3_experiment",
        random_seed=42,
        run_count=run_count,
        scenario_id="A_TEST",
        output_dir=tmp_path / "data" / "processed",
        reports_dir=tmp_path / "reports" / "chapter3",
    )


def test_experiment_runner_imports(tmp_path: Path) -> None:
    """Проверяет импортируемость запускателя эксперимента."""
    runner = ExperimentRunner(_make_config(tmp_path))

    assert isinstance(runner.config, ExperimentRunnerConfig)


def test_default_config_is_valid() -> None:
    """Проверяет корректность конфигурации по умолчанию."""
    config = ExperimentRunnerConfig()

    config.validate()
    assert config.run_count > 0
    assert config.random_seed >= 0


def test_config_loads_from_yaml(tmp_path: Path) -> None:
    """Проверяет загрузку параметров эксперимента из YAML."""
    config_path = _write_config(tmp_path, run_count=4, seed=123)
    config = ExperimentRunnerConfig.from_yaml(config_path)

    assert config.experiment_name == "test_chapter3_experiment"
    assert config.random_seed == 123
    assert config.run_count == 4
    assert config.scenario_id == "A_TEST"


def test_missing_yaml_is_rejected(tmp_path: Path) -> None:
    """Проверяет отклонение отсутствующего файла конфигурации."""
    with pytest.raises(FileNotFoundError):
        ExperimentRunnerConfig.from_yaml(tmp_path / "missing.yaml")


def test_invalid_config_values_are_rejected(tmp_path: Path) -> None:
    """Проверяет отклонение некорректных параметров запуска."""
    with pytest.raises(ValueError):
        ExperimentRunnerConfig(run_count=0).validate()

    with pytest.raises(ValueError):
        ExperimentRunnerConfig(random_seed=-1).validate()

    with pytest.raises(ValueError):
        ExperimentRunnerConfig(scenario_id="").validate()

    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text("experiment:\n  name: bad\n  random_seed: 1\n  run_count: 0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ExperimentRunnerConfig.from_yaml(bad_path)


def test_config_converts_to_dataset_builder_config(tmp_path: Path) -> None:
    """Проверяет передачу зерен генераторов в конфигурацию датасета."""
    config = _make_config(tmp_path, run_count=5)
    dataset_config = config.to_dataset_builder_config()

    assert dataset_config.run_count == 5
    assert dataset_config.simulator_config.scenario_id == "A_TEST"
    assert dataset_config.simulator_config.message_random_seed == 42
    assert dataset_config.simulator_config.error_random_seed == 1042
    assert dataset_config.simulator_config.control_random_seed == 2042


def test_runner_run_returns_experiment_result(tmp_path: Path) -> None:
    """Проверяет запуск эксперимента и тип результата."""
    result = ExperimentRunner(_make_config(tmp_path)).run()

    assert isinstance(result, ExperimentRunResult)
    assert result.run_count == 3
    assert result.experiment_name == "test_chapter3_experiment"


def test_runner_creates_dataset_files(tmp_path: Path) -> None:
    """Проверяет сохранение табличных артефактов эксперимента."""
    result = ExperimentRunner(_make_config(tmp_path)).run()

    assert Path(result.saved_files["prior_features"]).exists()
    assert Path(result.saved_files["fact_features"]).exists()
    assert Path(result.saved_files["quality_targets"]).exists()


def test_runner_creates_experiment_report(tmp_path: Path) -> None:
    """Проверяет сохранение итогового отчета эксперимента."""
    result = ExperimentRunner(_make_config(tmp_path)).run()
    report = json.loads(result.report_path.read_text(encoding="utf-8"))

    assert result.report_path.exists()
    assert report["experiment_name"] == "test_chapter3_experiment"
    assert report["scenario_id"] == "A_TEST"
    assert report["reproducibility_ok"] is True


def test_reproducibility_hash_is_stable(tmp_path: Path) -> None:
    """Проверяет устойчивость контрольного хеша датасета."""
    config = _make_config(tmp_path)
    result = ExperimentRunner(config).run()
    same_hash = hash_dataset_result(result.dataset_result)

    assert len(result.reproducibility_hash) == 64
    assert result.reproducibility_hash == same_hash
    assert result.reproducibility_ok is True


def test_run_experiment_helper(tmp_path: Path) -> None:
    """Проверяет вспомогательную функцию run_experiment()."""
    result = run_experiment(_make_config(tmp_path, run_count=2))

    assert result.run_count == 2
    assert result.summary["run_count"] == 2


def test_run_experiment_from_yaml(tmp_path: Path) -> None:
    """Проверяет запуск эксперимента по YAML-конфигурации."""
    config_path = _write_config(tmp_path, run_count=2, seed=77)
    result = run_experiment_from_yaml(config_path)

    assert result.run_count == 2
    assert result.random_seed == 77
    assert result.summary["random_seed"] == 77


def test_report_contains_dataset_summary_and_saved_files(tmp_path: Path) -> None:
    """Проверяет полноту итоговой сводки эксперимента."""
    result = ExperimentRunner(_make_config(tmp_path)).run()
    summary = result.summary

    assert summary["dataset_summary"]["run_count"] == 3
    assert "protocols" in summary["saved_files"]
    assert "quality_targets" in summary["saved_files"]
    assert summary["dataset_summary"]["prior_feature_count"] > 0
