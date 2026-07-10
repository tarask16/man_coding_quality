"""Каркас расчета неопределенности прогноза главы 5."""

from __future__ import annotations


class PredictionUncertaintyEstimator:
    """Оценивает неопределенность и интервалы для ``Q_pred``."""

    def estimate(self) -> None:
        """Сообщить, что расчет неопределенности будет реализован позже."""

        msg = "Расчет неопределенности прогноза будет реализован на этапе 9."
        raise NotImplementedError(msg)
