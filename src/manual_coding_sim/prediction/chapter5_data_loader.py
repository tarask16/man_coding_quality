"""Каркас загрузчика входных данных главы 5.

На этапе 1 загрузчик только фиксирует контракт входных файлов. Полная проверка
структуры ``theta_prior.csv``, ``topic_interpretation.json`` и
``prior_features.csv`` будет реализована на этапе загрузки и объединения данных.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from manual_coding_sim.prediction.chapter5_config import Chapter5InputPaths


@dataclass(frozen=True)
class Chapter5InputContract:
    """Описание обязательных входных файлов главы 5."""

    prior_features_path: Path
    theta_prior_path: Path
    topic_interpretation_path: Path


class Chapter5DataLoader:
    """Загрузчик входных артефактов для последующего расчета ``Q_pred``."""

    def __init__(self, paths: Chapter5InputPaths | None = None) -> None:
        """Создать загрузчик с путями по умолчанию или пользовательскими путями."""

        self.paths = paths or Chapter5InputPaths()

    def describe_expected_inputs(self) -> Chapter5InputContract:
        """Вернуть описание обязательных входных файлов без чтения данных."""

        return Chapter5InputContract(
            prior_features_path=self.paths.prior_features_path,
            theta_prior_path=self.paths.theta_prior_path,
            topic_interpretation_path=self.paths.topic_interpretation_path,
        )

    def load(self) -> None:
        """Сообщить, что фактическая загрузка будет реализована следующим этапом."""

        msg = "Загрузка и объединение входных данных будет реализована на этапе 3."
        raise NotImplementedError(msg)
