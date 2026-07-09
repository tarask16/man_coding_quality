"""
Загрузка и проверка конфигурации вычислительного эксперимента.

В терминах диссертации конфигурация задает параметры воспроизводимого
моделирования сценариев A = {S, O, U, G, K}, где S — средство ручного
кодирования, O — оператор, U — условия применения, G — класс сообщений,
K — контрольные процедуры.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """Минимальная конфигурация вычислительного эксперимента главы 3."""

    name: str
    random_seed: int


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    """
    Загрузить базовую конфигурацию эксперимента из YAML-файла.

    Параметр random_seed фиксирует начальное состояние генераторов
    случайных чисел и используется для воспроизводимого формирования
    протоколов ручного кодирования.
    """

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw_config: dict[str, Any] = yaml.safe_load(file) or {}

    experiment_section = raw_config.get("experiment")
    if not isinstance(experiment_section, dict):
        raise ValueError("В конфигурации отсутствует раздел 'experiment'.")

    name = experiment_section.get("name")
    random_seed = experiment_section.get("random_seed")

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Поле 'experiment.name' должно быть непустой строкой.")

    if not isinstance(random_seed, int):
        raise ValueError("Поле 'experiment.random_seed' должно быть целым числом.")

    return ExperimentConfig(name=name, random_seed=random_seed)
