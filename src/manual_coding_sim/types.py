"""Базовые типы данных для исследовательского симулятора."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScenarioParameters:
    """
    Параметры сценария применения A = {S, O, U, G, K}.

    Поля класса пока содержат строковые идентификаторы сущностей.
    На последующих этапах они будут связаны с моделями средства
    кодирования, оператора, условий применения, класса сообщений
    и контрольных процедур.
    """

    scenario_id: str
    coding_tool_id: str
    operator_id: str
    condition_id: str
    message_class_id: str
    control_procedure_id: str


@dataclass(frozen=True)
class MessageElement:
    """Элемент исходного сообщения M."""

    value: str
    element_type: str
    position: int
    criticality: float


@dataclass(frozen=True)
class GeneratedMessage:
    """Исходное сообщение M как последовательность элементов."""

    message_id: str
    elements: tuple[MessageElement, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QualityVector:
    """
    Вектор частных показателей качества ручного кодирования.

    Показатели соответствуют обозначению q(A) из главы 3:
    точность, время, трудоемкость, устойчивость, повторяемость
    и пригодность сценария применения.
    """

    q_acc: float
    q_time: float
    q_effort: float
    q_res: float
    q_rep: float
    q_fit: float


@dataclass(frozen=True)
class FeatureGroup:
    """
    Группа признаков, полученных при моделировании.

    Используется для раздельного хранения априорных, фактических
    и диагностических признаков, чтобы исключить утечку фактических
    результатов в процедуру априорной оценки качества.
    """

    scenario_id: str
    prior_features: dict[str, float]
    fact_features: dict[str, float]
    diagnostic_features: dict[str, float]
