"""Тесты базовой структуры LDA-пакета главы 4."""

from pathlib import Path

from manual_coding_sim.lda import Chapter4LdaConfig, LeakageGuard
from manual_coding_sim.lda.paths import (
    DEFAULT_LDA_DATA_DIR,
    DEFAULT_LDA_MODELS_DIR,
    DEFAULT_LDA_REPORTS_DIR,
    DEFAULT_PRIOR_FEATURES_PATH,
    resolve_project_path,
)


def test_lda_package_exports_core_objects() -> None:
    """Пакет должен экспортировать конфигурацию и защиту от утечки."""

    config = Chapter4LdaConfig()
    guard = LeakageGuard()

    assert config.inputs.prior_features_path == DEFAULT_PRIOR_FEATURES_PATH
    assert guard is not None


def test_default_output_paths_match_chapter4_structure() -> None:
    """Выходные пути должны соответствовать структуре главы 4."""

    assert DEFAULT_LDA_DATA_DIR == Path("data/processed/lda")
    assert DEFAULT_LDA_MODELS_DIR == Path("models/lda")
    assert DEFAULT_LDA_REPORTS_DIR == Path("reports/chapter4")


def test_resolve_project_path_keeps_absolute_path(tmp_path: Path) -> None:
    """Абсолютный путь не должен изменяться при разрешении путей проекта."""

    absolute_path = tmp_path / "prior_features.csv"

    assert resolve_project_path(Path("/project"), absolute_path) == absolute_path


def test_resolve_project_path_uses_project_root() -> None:
    """Относительный путь должен разрешаться от корня проекта."""

    project_root = Path("/project")
    relative_path = Path("data/processed/prior_features.csv")

    assert resolve_project_path(project_root, relative_path) == project_root / relative_path


def test_chapter4_config_validation_accepts_defaults() -> None:
    """Конфигурация по умолчанию должна быть валидной."""

    config = Chapter4LdaConfig()

    config.validate()
