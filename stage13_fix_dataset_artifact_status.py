"""
Исправление функции контроля артефактов датасета для этапа 13.

Скрипт устраняет ошибку, при которой пустое значение пути интерпретировалось
как текущий каталог проекта. Из-за этого функция collect_dataset_artifact_status({})
ошибочно возвращала True для всех артефактов.

После исправления артефакт считается существующим только в том случае, если:
1) ключ присутствует в saved_files;
2) значение пути является непустой строкой;
3) указанный файл реально существует на диске.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path.cwd()
MODULE_PATH = ROOT / "src" / "manual_coding_sim" / "chapter3_report.py"
REPORT_PATH = ROOT / "reports" / "chapter3" / "stage13_fix_dataset_artifact_status_report.json"

OLD_FUNCTION = '''def collect_dataset_artifact_status(
    experiment_summary: dict[str, Any],
) -> dict[str, bool]:
    """Проверяет наличие табличных артефактов эксперимента."""
    saved_files = experiment_summary.get("saved_files", {})
    if not isinstance(saved_files, dict):
        return {key: False for key in EXPECTED_DATASET_KEYS}

    return {
        key: Path(str(saved_files.get(key, ""))).exists()
        for key in EXPECTED_DATASET_KEYS
    }
'''

NEW_FUNCTION = '''def collect_dataset_artifact_status(
    experiment_summary: dict[str, Any],
) -> dict[str, bool]:
    """Проверяет наличие табличных артефактов эксперимента.

    Пустая строка не должна считаться существующим путем. В Python выражение
    Path("").exists() проверяет текущий каталог и обычно возвращает True, что
    недопустимо для контроля CSV/JSON-артефактов датасета главы 3.
    """
    saved_files = experiment_summary.get("saved_files", {})
    if not isinstance(saved_files, dict) or not saved_files:
        return {key: False for key in EXPECTED_DATASET_KEYS}

    artifact_status: dict[str, bool] = {}
    for key in EXPECTED_DATASET_KEYS:
        raw_path = saved_files.get(key)
        if not isinstance(raw_path, str) or not raw_path.strip():
            artifact_status[key] = False
            continue

        artifact_status[key] = Path(raw_path).exists()

    return artifact_status
'''


def replace_function(module_text: str) -> str:
    """Заменяет некорректную реализацию функции контроля артефактов."""
    if OLD_FUNCTION in module_text:
        return module_text.replace(OLD_FUNCTION, NEW_FUNCTION)

    start_marker = "def collect_dataset_artifact_status("
    next_marker = "\ndef render_report_markdown("
    start_index = module_text.find(start_marker)
    end_index = module_text.find(next_marker)

    if start_index == -1 or end_index == -1 or end_index <= start_index:
        raise RuntimeError(
            "Не удалось найти функцию collect_dataset_artifact_status для замены.",
        )

    return module_text[:start_index] + NEW_FUNCTION + module_text[end_index + 1 :]


def check_python_syntax(path: Path) -> dict[str, str]:
    """Проверяет синтаксис Python-файла после исправления."""
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as error:
        return {"status": "ERROR", "message": str(error)}

    return {"status": "OK", "message": "Синтаксис корректен"}


def main() -> None:
    """Исправляет модуль этапа 13 и формирует служебный отчет."""
    if not MODULE_PATH.exists():
        raise FileNotFoundError(f"Модуль итоговой отчетности не найден: {MODULE_PATH}")

    original_text = MODULE_PATH.read_text(encoding="utf-8")
    updated_text = replace_function(original_text)
    MODULE_PATH.write_text(updated_text, encoding="utf-8")

    syntax_status = check_python_syntax(MODULE_PATH)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "stage": 13,
                "fix": "collect_dataset_artifact_status_empty_summary",
                "module": str(MODULE_PATH.relative_to(ROOT)),
                "syntax_status": syntax_status,
                "scientific_meaning": (
                    "Исправлена проверка наличия артефактов датасета главы 3: "
                    "пустая сводка эксперимента больше не считается подтверждением "
                    "наличия protocols.csv, X_prior, X_fact, X_diag и q(A)."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("ЭТАП 13. ИСПРАВЛЕНИЕ КОНТРОЛЯ АРТЕФАКТОВ ДАТАСЕТА")
    print("=" * 70)
    print(f"[{syntax_status['status']}] {MODULE_PATH.relative_to(ROOT)}")
    print(f"[OK] Отчет: {REPORT_PATH}")
    print()
    print("Теперь выполните команду:")
    print("python -m pytest")


if __name__ == "__main__":
    main()
