"""
Исследовательский симулятор процессов ручного кодирования.

Пакет предназначен для программной реализации главы 3 диссертации:
компьютерного моделирования процессов ручного кодирования и декодирования
с последующим формированием априорных, фактических и диагностических
признаков для оценки качества ручных средств кодирования информации.
"""

from manual_coding_sim.config import ExperimentConfig, load_experiment_config
from manual_coding_sim.types import (
    FeatureGroup,
    GeneratedMessage,
    MessageElement,
    QualityVector,
    ScenarioParameters,
)

__all__ = [
    "ExperimentConfig",
    "FeatureGroup",
    "GeneratedMessage",
    "MessageElement",
    "QualityVector",
    "ScenarioParameters",
    "load_experiment_config",
]

__version__ = "0.1.0"
