"""Конфигурационный каркас программного блока главы 5.

На первом этапе фиксируются только структуры путей и базовые флаги. Весовые
коэффициенты, правила нормировки и параметры неопределенности будут добавлены
на отдельном этапе, чтобы не смешивать каркас пакета с расчетной логикой.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from manual_coding_sim.prediction.paths import (
    DEFAULT_CHAPTER5_REPORTS_DIR,
    DEFAULT_PREDICTION_REPORT_JSON_PATH,
    DEFAULT_PREDICTION_REPORT_MD_PATH,
    DEFAULT_PRIOR_FEATURES_PATH,
    DEFAULT_Q_PRED_COMPONENTS_PATH,
    DEFAULT_Q_PRED_PATH,
    DEFAULT_THETA_PRIOR_PATH,
    DEFAULT_TOPIC_INTERPRETATION_PATH,
)


@dataclass(frozen=True)
class Chapter5InputPaths:
    """Пути к входным артефактам априорного прогноза главы 5."""

    prior_features_path: Path = DEFAULT_PRIOR_FEATURES_PATH
    theta_prior_path: Path = DEFAULT_THETA_PRIOR_PATH
    topic_interpretation_path: Path = DEFAULT_TOPIC_INTERPRETATION_PATH


@dataclass(frozen=True)
class Chapter5OutputPaths:
    """Пути к выходным артефактам априорного прогноза главы 5."""

    reports_dir: Path = DEFAULT_CHAPTER5_REPORTS_DIR
    q_pred_path: Path = DEFAULT_Q_PRED_PATH
    q_pred_components_path: Path = DEFAULT_Q_PRED_COMPONENTS_PATH
    report_json_path: Path = DEFAULT_PREDICTION_REPORT_JSON_PATH
    report_md_path: Path = DEFAULT_PREDICTION_REPORT_MD_PATH


@dataclass(frozen=True)
class Chapter5PredictionConfig:
    """Единая конфигурация каркаса главы 5.

    Поле ``expected_topic_count`` фиксирует результат главы 4: для текущего
    расширенного корпуса используется три латентных фактора качества.
    """

    inputs: Chapter5InputPaths = field(default_factory=Chapter5InputPaths)
    outputs: Chapter5OutputPaths = field(default_factory=Chapter5OutputPaths)
    expected_topic_count: int = 3
    forbid_fact_features: bool = True
    forbid_quality_targets: bool = True
    overwrite: bool = True

    def validate(self) -> None:
        """Проверить базовую корректность конфигурации главы 5."""

        if self.expected_topic_count < 2:
            msg = "Число латентных факторов главы 5 должно быть не меньше 2."
            raise ValueError(msg)
        if not str(self.outputs.reports_dir):
            msg = "Каталог отчетов главы 5 не должен быть пустым."
            raise ValueError(msg)
