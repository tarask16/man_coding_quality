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
DEFAULT_Q_PRED_REPORT_PATH = DEFAULT_CHAPTER5_REPORTS_DIR / "q_pred_report.json"
DEFAULT_Q_PRED_COMPONENTS_PATH = DEFAULT_CHAPTER5_REPORTS_DIR / "q_pred_components.csv"
DEFAULT_Q_PRED_COMPONENTS_REPORT_PATH = (
    DEFAULT_CHAPTER5_REPORTS_DIR / "q_pred_components_report.json"
)
DEFAULT_NORMALIZED_PRIOR_FEATURES_PATH = (
    DEFAULT_CHAPTER5_REPORTS_DIR / "normalized_prior_features.csv"
)
DEFAULT_NORMALIZATION_REPORT_PATH = (
    DEFAULT_CHAPTER5_REPORTS_DIR / "normalization_report.json"
)
DEFAULT_LATENT_QUALITY_COMPONENT_PATH = (
    DEFAULT_CHAPTER5_REPORTS_DIR / "latent_quality_component.csv"
)
DEFAULT_LATENT_QUALITY_COMPONENT_REPORT_PATH = (
    DEFAULT_CHAPTER5_REPORTS_DIR / "latent_quality_component_report.json"
)
DEFAULT_PREDICTION_REPORT_JSON_PATH = (
    DEFAULT_CHAPTER5_REPORTS_DIR / "chapter5_prediction_report.json"
)
DEFAULT_PREDICTION_REPORT_MD_PATH = (
    DEFAULT_CHAPTER5_REPORTS_DIR / "chapter5_prediction_report.md"
)

DEFAULT_PIPELINE_RUN_REPORT_PATH = (
    DEFAULT_CHAPTER5_REPORTS_DIR / "chapter5_pipeline_run_report.json"
)

DEFAULT_PREDICTION_UNCERTAINTY_PATH = (
    DEFAULT_CHAPTER5_REPORTS_DIR / "prediction_uncertainty.csv"
)
DEFAULT_PREDICTION_UNCERTAINTY_REPORT_PATH = (
    DEFAULT_CHAPTER5_REPORTS_DIR / "prediction_uncertainty_report.json"
)

DEFAULT_ACCEPTANCE_REPORT_JSON_PATH = (
    DEFAULT_CHAPTER5_REPORTS_DIR / "chapter5_acceptance_report.json"
)
DEFAULT_ACCEPTANCE_REPORT_MD_PATH = (
    DEFAULT_CHAPTER5_REPORTS_DIR / "chapter5_acceptance_report.md"
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
