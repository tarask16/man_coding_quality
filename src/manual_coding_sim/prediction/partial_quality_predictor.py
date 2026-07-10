"""Каркас расчета частных прогнозных критериев главы 5."""

from __future__ import annotations


class PartialQualityPredictor:
    """Рассчитывает прогнозные критерии ``q_acc``, ``q_time`` и другие."""

    def predict(self) -> None:
        """Сообщить, что частные критерии будут реализованы позже."""

        msg = "Расчет частных прогнозных критериев будет реализован на этапе 7."
        raise NotImplementedError(msg)
