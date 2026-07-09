"""
Диагностический скрипт настройки окружения VSCode для программной реализации
главы 3 диссертации.

Назначение:
    Проверить, готово ли локальное Python-окружение к разработке
    исследовательского симулятора процессов ручного кодирования и декодирования.

Скрипт не изменяет проект автоматически. Он только выполняет проверки и выводит
отчет, который можно приложить к результатам первой задачи.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckResult:
    """Результат одной диагностической проверки окружения."""

    name: str
    passed: bool
    details: str


REQUIRED_PACKAGES = (
    "numpy",
    "pandas",
    "sklearn",
    "yaml",
    "pytest",
)

RECOMMENDED_DIRECTORIES = (
    "src",
    "src/manual_coding_sim",
    "configs",
    "data/raw",
    "data/processed",
    "reports/chapter3",
    "tests",
    ".vscode",
)

RECOMMENDED_FILES = (
    "pyproject.toml",
    "configs/base_experiment.yaml",
    ".vscode/settings.json",
    ".vscode/launch.json",
)


def check_python_version() -> CheckResult:
    """Проверить, что используется поддерживаемая версия Python."""

    version = sys.version_info
    passed = version.major == 3 and version.minor >= 11
    details = f"Python {version.major}.{version.minor}.{version.micro}"
    return CheckResult("Версия Python не ниже 3.11", passed, details)


def check_virtual_environment() -> CheckResult:
    """Проверить, запущен ли интерпретатор из виртуального окружения."""

    in_venv = sys.prefix != sys.base_prefix
    details = f"sys.prefix={sys.prefix}; sys.base_prefix={sys.base_prefix}"
    return CheckResult("Активировано виртуальное окружение", in_venv, details)


def check_package(package_name: str) -> CheckResult:
    """Проверить доступность Python-пакета по его импортируемому имени."""

    spec = importlib.util.find_spec(package_name)
    passed = spec is not None
    details = "найден" if passed else "не найден"
    return CheckResult(f"Пакет {package_name}", passed, details)


def check_directory(project_root: Path, relative_path: str) -> CheckResult:
    """Проверить наличие каталога проекта."""

    path = project_root / relative_path
    return CheckResult(
        f"Каталог {relative_path}",
        path.is_dir(),
        str(path),
    )


def check_file(project_root: Path, relative_path: str) -> CheckResult:
    """Проверить наличие служебного файла проекта."""

    path = project_root / relative_path
    return CheckResult(
        f"Файл {relative_path}",
        path.is_file(),
        str(path),
    )


def build_report(project_root: Path) -> list[CheckResult]:
    """Сформировать перечень проверок для текущего состояния проекта."""

    results: list[CheckResult] = [
        check_python_version(),
        check_virtual_environment(),
    ]

    for package_name in REQUIRED_PACKAGES:
        results.append(check_package(package_name))

    for directory in RECOMMENDED_DIRECTORIES:
        results.append(check_directory(project_root, directory))

    for file_name in RECOMMENDED_FILES:
        results.append(check_file(project_root, file_name))

    return results


def print_human_report(results: list[CheckResult], project_root: Path) -> None:
    """Вывести отчет в удобном для чтения виде."""

    print("ОТЧЕТ О ПРОВЕРКЕ ОКРУЖЕНИЯ VSCode")
    print("=" * 48)
    print(f"Корень проекта: {project_root}")
    print(f"Операционная система: {platform.platform()}")
    print(f"Исполняемый файл Python: {sys.executable}")
    print()

    for result in results:
        status = "OK" if result.passed else "FAIL"
        print(f"[{status:4}] {result.name}: {result.details}")

    failed = [result for result in results if not result.passed]
    print()
    print("ИТОГ")
    print("=" * 48)

    if not failed:
        print("Окружение готово к программной реализации главы 3.")
        print(
            "Можно переходить к следующей задаче: созданию базовой структуры "
            "Python-пакета исследовательского симулятора."
        )
        return

    print("Окружение требует донастройки. Не закрывайте задачу, пока есть FAIL.")
    print("Неисправленные пункты:")
    for result in failed:
        print(f"- {result.name}")


def save_json_report(results: list[CheckResult], project_root: Path) -> Path:
    """Сохранить машинно-читаемый отчет в reports/chapter3."""

    report_dir = project_root / "reports" / "chapter3"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "environment_check.json"
    payload = {
        "project_root": str(project_root),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "checks": [
            {
                "name": result.name,
                "passed": result.passed,
                "details": result.details,
            }
            for result in results
        ],
    }

    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def main() -> int:
    """Точка входа диагностического скрипта."""

    project_root = Path.cwd()
    results = build_report(project_root)
    print_human_report(results, project_root)
    report_path = save_json_report(results, project_root)
    print()
    print(f"JSON-отчет сохранен: {report_path}")

    has_failed = any(not result.passed for result in results)
    return 1 if has_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
