"""Программный контур экспериментальной проверки достоверности главы 6.

На этапе 1 пакет предоставляет конфигурационный слой, соглашения о путях и
CLI для проверки и отображения параметров будущего валидационного контура.
"""

from manual_coding_sim.validation.chapter6_config import (
    Chapter6BootstrapConfig,
    Chapter6ConfigError,
    Chapter6ConfigLoader,
    Chapter6DecisionThresholds,
    Chapter6InputPaths,
    Chapter6MergeConfig,
    Chapter6OutputPaths,
    Chapter6RunnerConfig,
    Chapter6ValidationConfig,
    load_chapter6_validation_config,
)
from manual_coding_sim.validation.paths import resolve_project_path

__all__ = [
    "Chapter6BootstrapConfig",
    "Chapter6ConfigError",
    "Chapter6ConfigLoader",
    "Chapter6DecisionThresholds",
    "Chapter6InputPaths",
    "Chapter6MergeConfig",
    "Chapter6OutputPaths",
    "Chapter6RunnerConfig",
    "Chapter6ValidationConfig",
    "load_chapter6_validation_config",
    "resolve_project_path",
]
