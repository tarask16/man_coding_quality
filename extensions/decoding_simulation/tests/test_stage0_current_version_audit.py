"""Тест этапа 0: аудит текущей версии программного комплекса."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_audit_module(project_root: Path):
    """Загрузить audit-скрипт без изменения PYTHONPATH."""
    path = project_root / "extensions/decoding_simulation/tools/audit_current_version.py"
    spec = importlib.util.spec_from_file_location("decoding_stage0_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Не удалось загрузить audit-скрипт этапа 0.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage0_current_version_is_ready_for_isolated_extension() -> None:
    """Проверить наличие контрактов, необходимых для отдельного расширения."""
    project_root = Path(__file__).resolve().parents[3]
    module = _load_audit_module(project_root)
    audit = module.build_audit(project_root)

    assert audit["status"] == "ready_for_isolated_extension"
    assert audit["missing_required_core_or_test_files"] == []
    assert audit["syntax_errors"] == []
    assert all(not item["missing_symbols"] for item in audit["symbol_checks"])
    assert audit["all_required_csv_have_150_rows"] is True
