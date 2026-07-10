"""Каркас расчета интегральной априорной оценки ``Q_pred``."""

from __future__ import annotations


class IntegralQualityPredictor:
    """Агрегирует частные прогнозные критерии в интегральную оценку качества."""

    def predict(self) -> None:
        """Сообщить, что интегральный расчет будет реализован позже."""

        msg = "Расчет интегральной оценки Q_pred будет реализован на этапе 8."
        raise NotImplementedError(msg)
