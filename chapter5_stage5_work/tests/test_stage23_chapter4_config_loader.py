"""Тесты загрузчика YAML-конфигурации главы 4."""

from pathlib import Path

import pytest

from manual_coding_sim.lda.config_loader import (
    Chapter4ConfigLoader,
    Chapter4ConfigOverrides,
    load_chapter4_runner_config,
)


def _write_config(path: Path) -> None:
    """Записать минимальный YAML-конфиг для тестов."""

    path.write_text(
        """
input:
  prior_features: data/processed/prior_features.csv
  diagnostic_features: data/processed/diagnostic_features.csv
  fact_features: data/processed/fact_features.csv
output:
  data_dir: data/processed/lda
  models_dir: models/lda
  reports_dir: reports/chapter4
tokenization:
  df_min: 1
  df_max_ratio: 1.0
  numeric_strategy: quantile
  numeric_bins: 3
lda:
  k_values: [2, 3]
  selected_k: null
  learning_method: batch
  max_iter: 2
  random_seeds: [11, 42]
diagnostics:
  build_lda_diag: true
  build_lda_full: true
runner:
  top_n: 3
  overwrite: false
safety:
  forbid_fact_features_in_prior: true
  forbid_quality_targets_in_prior: true
  require_corpus_hash: true
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _touch(path: Path) -> None:
    """Создать пустой входной файл для проверки путей."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test\n", encoding="utf-8")


def test_config_loader_reads_yaml_sections(tmp_path: Path) -> None:
    """Загрузчик должен корректно читать основные секции YAML."""

    config_path = tmp_path / "chapter4_lda.yaml"
    _write_config(config_path)

    result = Chapter4ConfigLoader().load(config_path)
    config = result.runner_config

    assert result.config_path == config_path
    assert config.chapter4.inputs.prior_features_path == Path(
        "data/processed/prior_features.csv"
    )
    assert config.chapter4.outputs.reports_dir == Path("reports/chapter4")
    assert config.chapter4.tokenization.df_min == 1
    assert config.chapter4.model.k_values == (2, 3)
    assert config.chapter4.model.random_seeds == (11, 42)
    assert config.top_n == 3
    assert config.overwrite is False


def test_config_loader_applies_cli_overrides(tmp_path: Path) -> None:
    """CLI-переопределения должны иметь приоритет над YAML."""

    config_path = tmp_path / "chapter4_lda.yaml"
    _write_config(config_path)

    config = load_chapter4_runner_config(
        config_path,
        Chapter4ConfigOverrides(
            overwrite=True,
            skip_diagnostic=True,
            selected_k=3,
        ),
    )

    assert config.overwrite is True
    assert config.chapter4.build_lda_diag is False
    assert config.chapter4.build_lda_full is False
    assert config.chapter4.model.selected_k == 3


def test_config_loader_validates_required_inputs(tmp_path: Path) -> None:
    """Проверка входов должна учитывать включенные диагностические модели."""

    config_path = tmp_path / "chapter4_lda.yaml"
    _write_config(config_path)
    project_root = tmp_path / "project"
    _touch(project_root / "data" / "processed" / "prior_features.csv")
    _touch(project_root / "data" / "processed" / "diagnostic_features.csv")
    _touch(project_root / "data" / "processed" / "fact_features.csv")

    loader = Chapter4ConfigLoader()
    config = loader.load(config_path).runner_config

    loader.validate_required_inputs(config, project_root=project_root)


def test_config_loader_reports_missing_required_inputs(tmp_path: Path) -> None:
    """При отсутствии обязательных CSV должен возникать FileNotFoundError."""

    config_path = tmp_path / "chapter4_lda.yaml"
    _write_config(config_path)
    project_root = tmp_path / "project"
    _touch(project_root / "data" / "processed" / "prior_features.csv")

    loader = Chapter4ConfigLoader()
    config = loader.load(config_path).runner_config

    with pytest.raises(FileNotFoundError, match="diagnostic_features"):
        loader.validate_required_inputs(config, project_root=project_root)


def test_config_loader_skip_diagnostic_requires_only_prior_input(tmp_path: Path) -> None:
    """При отключенной диагностике обязательным должен быть только prior-файл."""

    config_path = tmp_path / "chapter4_lda.yaml"
    _write_config(config_path)
    project_root = tmp_path / "project"
    _touch(project_root / "data" / "processed" / "prior_features.csv")

    loader = Chapter4ConfigLoader()
    config = loader.load(
        config_path,
        Chapter4ConfigOverrides(skip_diagnostic=True),
    ).runner_config

    loader.validate_required_inputs(config, project_root=project_root)


def test_config_loader_rejects_invalid_root_yaml(tmp_path: Path) -> None:
    """Корневой YAML-элемент не должен быть списком."""

    config_path = tmp_path / "chapter4_lda.yaml"
    config_path.write_text("- invalid\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Корневой элемент"):
        Chapter4ConfigLoader().load(config_path)
