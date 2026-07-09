"""
Базовый пакет исследовательского симулятора ручного кодирования.

Пакет предназначен для программной реализации главы 3 диссертации:
компьютерного моделирования процессов ручного кодирования и декодирования
при априорной оценке качества ручных средств кодирования информации.
"""

from manual_coding_sim.config import load_experiment_config
from manual_coding_sim.types import (
    FeatureGroup,
    GeneratedMessage,
    MessageElement,
    QualityVector,
    ScenarioParameters,
)

__version__ = "0.1.0"

__all__ = [
    "FeatureGroup",
    "GeneratedMessage",
    "MessageElement",
    "QualityVector",
    "ScenarioParameters",
    "load_experiment_config",
]
