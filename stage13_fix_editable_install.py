"""
Исправляет запуск пакета manual_coding_sim через команду python -m.

Скрипт добавляет в pyproject.toml минимальные настройки сборки для src-layout:
- build-system на базе setuptools;
- поиск пакетов внутри каталога src.

После выполнения этого файла необходимо установить проект в editable-режиме:
python -m pip install -e .

Это позволит запускать итоговый отчет главы 3 командой:
python -m manual_coding_sim.chapter3_report
"""

from __future__ import annotations

import json
from pathlib import Path


BUILD_SYSTEM_BLOCK = """
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
""".strip()

SETUPTOOLS_FIND_BLOCK = """
[tool.setuptools.packages.find]
where = ["src"]
""".strip()

REPORT_PATH = Path("reports/chapter3/stage13_editable_install_fix_report.json")


def _append_block_if_missing(text: str, header: str, block: str) -> tuple[str, bool]:
    """Добавляет TOML-блок, если в файле еще нет указанного заголовка."""
    if header in text:
        return text, False

    separator = "\n\n" if text.endswith("\n") else "\n\n"
    updated_text = f"{text.rstrip()}{separator}{block}\n"
    return updated_text, True


def update_pyproject(pyproject_path: Path) -> dict[str, object]:
    """Обновляет pyproject.toml для корректной установки пакета из каталога src."""
    if not pyproject_path.exists():
        raise FileNotFoundError(f"Не найден файл {pyproject_path}")

    original_text = pyproject_path.read_text(encoding="utf-8-sig")
    updated_text = original_text

    updated_text, build_system_added = _append_block_if_missing(
        updated_text,
        "[build-system]",
        BUILD_SYSTEM_BLOCK,
    )
    updated_text, setuptools_find_added = _append_block_if_missing(
        updated_text,
        "[tool.setuptools.packages.find]",
        SETUPTOOLS_FIND_BLOCK,
    )

    changed = updated_text != original_text
    if changed:
        pyproject_path.write_text(updated_text, encoding="utf-8")

    return {
        "pyproject_path": str(pyproject_path),
        "changed": changed,
        "build_system_added": build_system_added,
        "setuptools_find_added": setuptools_find_added,
        "next_commands": [
            "python -m pip install -e .",
            "python -m manual_coding_sim.chapter3_report",
        ],
    }


def main() -> None:
    """Точка входа исправления запуска пакета."""
    project_root = Path.cwd()
    pyproject_path = project_root / "pyproject.toml"

    result = update_pyproject(pyproject_path)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("ЭТАП 13. ИСПРАВЛЕНИЕ ЗАПУСКА ПАКЕТА")
    print("=" * 56)
    print(f"[OK] Проверен файл: {pyproject_path}")
    if result["changed"]:
        print("[OK] pyproject.toml обновлен для editable-установки src-пакета")
    else:
        print("[OK] pyproject.toml уже содержит необходимые настройки")
    print(f"[OK] Отчет: {REPORT_PATH}")
    print()
    print("Теперь выполните команды:")
    print("python -m pip install -e .")
    print("python -m manual_coding_sim.chapter3_report")


if __name__ == "__main__":
    main()
