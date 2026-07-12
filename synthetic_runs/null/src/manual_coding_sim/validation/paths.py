"""Пути программного контура экспериментальной проверки главы 6.

Модуль фиксирует единое размещение входных артефактов глав 3–5 и будущих
выходных артефактов главы 6. Расчет метрик и чтение файлов здесь не
выполняются.
"""

from __future__ import annotations

from pathlib import Path


DEFAULT_CHAPTER6_REPORTS_DIR = Path("reports/chapter6")
DEFAULT_CHAPTER6_FIGURES_DIR = DEFAULT_CHAPTER6_REPORTS_DIR / "figures"

DEFAULT_Q_PRED_PATH = Path("reports/chapter5/q_pred.csv")
DEFAULT_Q_PRED_COMPONENTS_PATH = Path("reports/chapter5/q_pred_components.csv")
DEFAULT_PREDICTION_UNCERTAINTY_PATH = Path(
    "reports/chapter5/prediction_uncertainty.csv"
)
DEFAULT_CHAPTER5_PREDICTION_REPORT_PATH = Path(
    "reports/chapter5/chapter5_prediction_report.json"
)
DEFAULT_CHAPTER5_ACCEPTANCE_REPORT_PATH = Path(
    "reports/chapter5/chapter5_acceptance_report.json"
)
DEFAULT_NORMALIZED_PRIOR_FEATURES_PATH = Path(
    "reports/chapter5/normalized_prior_features.csv"
)
DEFAULT_LATENT_QUALITY_COMPONENT_PATH = Path(
    "reports/chapter5/latent_quality_component.csv"
)
DEFAULT_THETA_PRIOR_PATH = Path("reports/chapter4/theta_prior.csv")
DEFAULT_QUALITY_TARGETS_PATH = Path("data/processed/quality_targets.csv")
DEFAULT_FACT_FEATURES_PATH = Path("data/processed/fact_features.csv")

DEFAULT_INPUT_VALIDATION_REPORT_JSON_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "chapter6_input_validation_report.json"
)
DEFAULT_INPUT_VALIDATION_REPORT_MD_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "chapter6_input_validation_report.md"
)
DEFAULT_VALIDATION_DATASET_PATH = DEFAULT_CHAPTER6_REPORTS_DIR / "validation_dataset.csv"
DEFAULT_INTEGRAL_QUALITY_CONSISTENCY_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "integral_quality_consistency.csv"
)
DEFAULT_INTEGRAL_QUALITY_REPORT_JSON_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "integral_quality_consistency_report.json"
)
DEFAULT_INTEGRAL_QUALITY_REPORT_MD_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "integral_quality_consistency_report.md"
)
DEFAULT_VALIDATION_METRICS_JSON_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "validation_metrics.json"
)
DEFAULT_VALIDATION_METRICS_MD_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "validation_metrics.md"
)
DEFAULT_INTEGRAL_PREDICTION_ERRORS_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "integral_prediction_errors.csv"
)
DEFAULT_PARTIAL_CRITERIA_VALIDATION_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "partial_criteria_validation.csv"
)
DEFAULT_CLASSIFICATION_REPORT_JSON_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "classification_report.json"
)
DEFAULT_CLASSIFICATION_REPORT_MD_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "classification_report.md"
)
DEFAULT_CONFUSION_MATRIX_PATH = DEFAULT_CHAPTER6_REPORTS_DIR / "confusion_matrix.csv"
DEFAULT_INTERVAL_COVERAGE_DETAILS_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "interval_coverage_details.csv"
)
DEFAULT_INTERVAL_COVERAGE_REPORT_JSON_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "interval_coverage_report.json"
)
DEFAULT_INTERVAL_COVERAGE_REPORT_MD_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "interval_coverage_report.md"
)
DEFAULT_BASELINE_PREDICTIONS_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "baseline_predictions.csv"
)
DEFAULT_BASELINE_COMPARISON_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "baseline_comparison.csv"
)
DEFAULT_BOOTSTRAP_CONFIDENCE_INTERVALS_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "bootstrap_confidence_intervals.csv"
)
DEFAULT_BOOTSTRAP_MODEL_DIFFERENCES_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "bootstrap_model_differences.csv"
)
DEFAULT_TOP_PREDICTION_ERRORS_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "top_prediction_errors.csv"
)
DEFAULT_ERROR_GROUP_ANALYSIS_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "error_group_analysis.csv"
)
DEFAULT_FINAL_REPORT_JSON_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "chapter6_validation_report.json"
)
DEFAULT_FINAL_REPORT_MD_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "chapter6_validation_report.md"
)
DEFAULT_PIPELINE_RUN_REPORT_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "chapter6_pipeline_run_report.json"
)
DEFAULT_ACCEPTANCE_REPORT_JSON_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "chapter6_acceptance_report.json"
)
DEFAULT_ACCEPTANCE_REPORT_MD_PATH = (
    DEFAULT_CHAPTER6_REPORTS_DIR / "chapter6_acceptance_report.md"
)


def resolve_project_path(project_root: Path, path: Path) -> Path:
    """Вернуть абсолютный путь к артефакту относительно корня проекта."""

    if path.is_absolute():
        return path
    return project_root / path
