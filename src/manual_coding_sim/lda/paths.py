"""Стандартные пути программного модуля главы 4.

Пути согласованы со структурой артефактов главы 3: входные CSV-файлы берутся
из ``data/processed``, а результаты LDA-анализа сохраняются в отдельные
каталоги ``data/processed/lda``, ``models/lda`` и ``reports/chapter4``.
"""

from pathlib import Path

# Входные артефакты, сформированные программной реализацией главы 3.
DEFAULT_PROTOCOLS_PATH = Path("data/processed/protocols.csv")
DEFAULT_PRIOR_FEATURES_PATH = Path("data/processed/prior_features.csv")
DEFAULT_DIAGNOSTIC_FEATURES_PATH = Path("data/processed/diagnostic_features.csv")
DEFAULT_FACT_FEATURES_PATH = Path("data/processed/fact_features.csv")
DEFAULT_QUALITY_TARGETS_PATH = Path("data/processed/quality_targets.csv")

# Выходные каталоги для программной реализации главы 4.
DEFAULT_LDA_DATA_DIR = Path("data/processed/lda")
DEFAULT_LDA_MODELS_DIR = Path("models/lda")
DEFAULT_LDA_REPORTS_DIR = Path("reports/chapter4")


def resolve_project_path(project_root: Path, path: Path) -> Path:
    """Вернуть абсолютный путь относительно корня проекта.

    Если ``path`` уже абсолютный, он возвращается без изменения. Это удобно для
    тестов и для запуска модуля из разных рабочих каталогов.
    """

    if path.is_absolute():
        return path
    return project_root / path
