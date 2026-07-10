"""Пути по умолчанию для программного блока главы 5.

Модуль не выполняет расчет качества. Он только фиксирует соглашения о
размещении входных и выходных артефактов, чтобы последующие этапы могли
использовать единый набор путей.
"""

from __future__ import annotations

from pathlib import Path


DEFAULT_CHAPTER5_REPORTS_DIR = Path("reports/chapter5")
DEFAULT_PRIOR_FEATURES_PATH = Path("data/processed/prior_features.csv")
DEFAULT_THETA_PRIOR_PATH = Path("reports/chapter4/theta_prior.csv")
DEFAULT_TOPIC_INTERPRETATION_PATH = Path("reports/chapter4/topic_interpretation.json")
DEFAULT_Q_PRED_PATH = DEFAULT_CHAPTER5_REPORTS_DIR / "q_pred.csv"
DEFAULT_Q_PRED_COMPONENTS_PATH = DEFAULT_CHAPTER5_REPORTS_DIR / "q_pred_components.csv"
DEFAULT_PREDICTION_REPORT_JSON_PATH = (
    DEFAULT_CHAPTER5_REPORTS_DIR / "chapter5_prediction_report.json"
)
DEFAULT_PREDICTION_REPORT_MD_PATH = (
    DEFAULT_CHAPTER5_REPORTS_DIR / "chapter5_prediction_report.md"
)


def resolve_project_path(project_root: Path, path: Path) -> Path:
    """Вернуть абсолютный путь относительно корня проекта.

    Абсолютные пути не изменяются. Относительные пути разрешаются от
    ``project_root``. Такая функция нужна для CLI-запуска из любого рабочего
    каталога.
    """

    if path.is_absolute():
        return path
    return project_root / path
