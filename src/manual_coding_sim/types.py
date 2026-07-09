"""
Базовые типы данных для исследовательского симулятора.

В этом модуле задаются только структуры данных. Алгоритмы генерации
сообщений, моделирования ошибок, контроля и расчета показателей качества
будут реализованы в последующих задачах.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FeatureGroup(StrEnum):
    """Группы признаков, используемые в главах 3–6 диссертации."""

    PRIOR = "prior"
    FACT = "fact"
    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True)
class ScenarioParameters:
    """Параметры сценария A = {S, O, U, G, K}."""

    scenario_id: str
    coding_tool_id: str
    operator_id: str
    condition_id: str
    message_class_id: str
    control_procedure_id: str


@dataclass(frozen=True)
class MessageElement:
    """Элемент исходного сообщения M, подлежащий ручному кодированию."""

    value: str
    element_type: str
    position: int
    criticality: float


@dataclass(frozen=True)
class GeneratedMessage:
    """Сгенерированное исходное сообщение M."""

    message_id: str
    message_class_id: str
    elements: tuple[MessageElement, ...]


@dataclass(frozen=True)
class QualityVector:
    """Вектор частных показателей качества ручного кодирования."""

    q_acc: float
    q_time: float
    q_effort: float
    q_res: float
    q_rep: float
    q_fit: float
