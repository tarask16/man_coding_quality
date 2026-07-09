"""Загрузка и проверка конфигурации вычислительного эксперимента."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_experiment_config(config_path: str | Path) -> dict[str, Any]:
    """
    Загружает конфигурацию вычислительного эксперимента главы 3.

    Конфигурация фиксирует параметры воспроизводимого моделирования.
    На этапе 1 обязательны только имя эксперимента и зерно генератора
    случайных чисел, используемое далее при формировании сценариев
    A = {S, O, U, G, K}.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Конфигурация должна быть YAML-словарем.")

    experiment = config.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("В конфигурации отсутствует раздел experiment.")

    if not experiment.get("name"):
        raise ValueError("В разделе experiment отсутствует поле name.")

    random_seed = experiment.get("random_seed")
    if not isinstance(random_seed, int):
        raise ValueError("Поле experiment.random_seed должно быть целым числом.")

    return config
