"""Конфигурационные структуры для LDA-модуля главы 4.

Конфигурации задают только воспроизводимые параметры подготовки корпуса,
обучения LDA и размещения артефактов. На данном этапе они не запускают
обучение модели и не выполняют токенизацию.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from manual_coding_sim.lda.paths import (
    DEFAULT_DIAGNOSTIC_FEATURES_PATH,
    DEFAULT_FACT_FEATURES_PATH,
    DEFAULT_LDA_DATA_DIR,
    DEFAULT_LDA_MODELS_DIR,
    DEFAULT_LDA_REPORTS_DIR,
    DEFAULT_PRIOR_FEATURES_PATH,
    DEFAULT_PROTOCOLS_PATH,
    DEFAULT_QUALITY_TARGETS_PATH,
)


@dataclass(frozen=True)
class LdaInputPaths:
    """Пути к входным артефактам главы 3.

    Основная модель ``LDA_prior`` должна использовать только
    ``prior_features_path`` и служебные идентификаторы из ``protocols_path``.
    Остальные файлы допустимы только для диагностических режимов и последующих
    проверок качества прогноза.
    """

    protocols_path: Path = DEFAULT_PROTOCOLS_PATH
    prior_features_path: Path = DEFAULT_PRIOR_FEATURES_PATH
    diagnostic_features_path: Path = DEFAULT_DIAGNOSTIC_FEATURES_PATH
    fact_features_path: Path = DEFAULT_FACT_FEATURES_PATH
    quality_targets_path: Path = DEFAULT_QUALITY_TARGETS_PATH


@dataclass(frozen=True)
class LdaOutputPaths:
    """Каталоги выходных артефактов главы 4."""

    data_dir: Path = DEFAULT_LDA_DATA_DIR
    models_dir: Path = DEFAULT_LDA_MODELS_DIR
    reports_dir: Path = DEFAULT_LDA_REPORTS_DIR


@dataclass(frozen=True)
class LdaTokenizationConfig:
    """Параметры подготовки словаря и токенизированного корпуса."""

    df_min: int = 2
    df_max_ratio: float = 0.95
    numeric_strategy: str = "quantile"
    numeric_bins: int = 3

    def validate(self) -> None:
        """Проверить корректность параметров токенизации."""

        if self.df_min < 1:
            msg = "df_min должен быть положительным целым числом."
            raise ValueError(msg)
        if not 0 < self.df_max_ratio <= 1:
            msg = "df_max_ratio должен находиться в диапазоне (0, 1]."
            raise ValueError(msg)
        if self.numeric_bins < 2:
            msg = "numeric_bins должен быть не меньше 2."
            raise ValueError(msg)
        allowed_strategies = {"quantile", "uniform"}
        if self.numeric_strategy not in allowed_strategies:
            msg = (
                "numeric_strategy должен иметь значение "
                f"из множества {sorted(allowed_strategies)}."
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class LdaModelConfig:
    """Параметры обучения LDA-модели.

    ``selected_k`` может быть ``None`` до этапа выбора числа латентных факторов.
    На последующих этапах оно должно быть зафиксировано в отчете главы 4.
    """

    k_values: tuple[int, ...] = (3, 4, 5, 6, 7, 8)
    selected_k: int | None = None
    doc_topic_prior: float | None = None
    topic_word_prior: float | None = None
    learning_method: str = "batch"
    max_iter: int = 100
    random_seeds: tuple[int, ...] = (11, 42, 77, 101)

    def validate(self) -> None:
        """Проверить корректность параметров LDA-модели."""

        if not self.k_values:
            msg = "Необходимо указать хотя бы одно значение K."
            raise ValueError(msg)
        if any(k < 2 for k in self.k_values):
            msg = "Все значения K должны быть не меньше 2."
            raise ValueError(msg)
        if self.selected_k is not None and self.selected_k not in self.k_values:
            msg = "selected_k должен входить в перечень k_values."
            raise ValueError(msg)
        allowed_methods = {"batch", "online"}
        if self.learning_method not in allowed_methods:
            msg = (
                "learning_method должен иметь значение "
                f"из множества {sorted(allowed_methods)}."
            )
            raise ValueError(msg)
        if self.max_iter < 1:
            msg = "max_iter должен быть положительным целым числом."
            raise ValueError(msg)
        if not self.random_seeds:
            msg = "Необходимо указать хотя бы один random_seed."
            raise ValueError(msg)


@dataclass(frozen=True)
class Chapter4LdaConfig:
    """Единая конфигурация программного блока главы 4."""

    inputs: LdaInputPaths = field(default_factory=LdaInputPaths)
    outputs: LdaOutputPaths = field(default_factory=LdaOutputPaths)
    tokenization: LdaTokenizationConfig = field(default_factory=LdaTokenizationConfig)
    model: LdaModelConfig = field(default_factory=LdaModelConfig)
    build_lda_diag: bool = True
    build_lda_full: bool = True
    forbid_fact_features_in_prior: bool = True
    forbid_quality_targets_in_prior: bool = True
    require_corpus_hash: bool = True

    def validate(self) -> None:
        """Проверить вложенные конфигурации главы 4."""

        self.tokenization.validate()
        self.model.validate()
