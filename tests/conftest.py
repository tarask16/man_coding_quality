"""Общие фикстуры тестового контура.

Некоторые ранние тесты главы 3 выполняют малый контрольный эксперимент и
перезаписывают ``data/processed/prior_features.csv`` в корне проекта. Для главы 5
этот файл является входным артефактом расширенного корпуса, поэтому после
каждого теста он восстанавливается из снимка, сделанного до запуска теста.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTECTED_RELATIVE_PATHS = (
    Path("data/processed/prior_features.csv"),
    Path("reports/chapter4/theta_prior.csv"),
    Path("reports/chapter4/topic_interpretation.json"),
)


@pytest.fixture(autouse=True)
def restore_chapter5_input_artifacts(tmp_path: Path):
    """Восстановить входные артефакты главы 5 после теста.

    Фикстура защищает расширенный ``prior_features.csv`` от побочного эффекта
    старых тестов главы 3, не изменяя поведение самих проверяемых функций.
    """

    backups: dict[Path, Path] = {}
    for relative_path in PROTECTED_RELATIVE_PATHS:
        source_path = PROJECT_ROOT / relative_path
        if source_path.exists():
            backup_path = tmp_path / "artifact_backup" / relative_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, backup_path)
            backups[source_path] = backup_path

    yield

    for source_path, backup_path in backups.items():
        source_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup_path, source_path)
