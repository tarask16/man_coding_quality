"""Каркас нормировки априорных признаков главы 5."""

from __future__ import annotations


class PriorFeatureNormalizer:
    """Нормирует признаки ``X_prior`` к диапазону [0, 1]."""

    def normalize(self) -> None:
        """Сообщить, что нормировка будет реализована на отдельном этапе."""

        msg = "Нормировка априорных признаков будет реализована на этапе 5."
        raise NotImplementedError(msg)
