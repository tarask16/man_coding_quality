"""Аудит текущей версии перед подключением расширения декодирования.

Скрипт не изменяет файлы проекта. Он проверяет наличие обязательных модулей,
ключевые публичные классы, схемы CSV и фиксирует SHA-256 базовой версии.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

REQUIRED_CORE = (
    "pyproject.toml",
    "src/manual_coding_sim/__init__.py",
    "src/manual_coding_sim/types.py",
    "src/manual_coding_sim/message_model.py",
    "src/manual_coding_sim/procedure_model.py",
    "src/manual_coding_sim/operator_model.py",
    "src/manual_coding_sim/condition_model.py",
    "src/manual_coding_sim/error_model.py",
    "src/manual_coding_sim/control_model.py",
    "src/manual_coding_sim/protocol_simulator.py",
    "src/manual_coding_sim/feature_extractor.py",
    "src/manual_coding_sim/quality_calculator.py",
    "src/manual_coding_sim/dataset_builder.py",
    "src/manual_coding_sim/experiment_runner.py",
    "src/manual_coding_sim/experiments/extended_corpus_plan.py",
    "src/manual_coding_sim/experiments/extended_corpus_runner.py",
)

REQUIRED_TESTS = (
    "tests/test_stage2_message_model.py",
    "tests/test_stage3_procedure_model.py",
    "tests/test_stage4_operator_model.py",
    "tests/test_stage5_condition_model.py",
    "tests/test_stage6_error_model.py",
    "tests/test_stage7_control_model.py",
    "tests/test_stage8_protocol_simulator.py",
    "tests/test_stage9_feature_extractor.py",
    "tests/test_stage10_quality_calculator.py",
    "tests/test_stage11_dataset_builder.py",
    "tests/test_stage12_experiment_runner.py",
    "tests/test_stage13_chapter3_report.py",
    "tests/test_stage25_extended_corpus_plan.py",
    "tests/test_stage25_extended_corpus_runner.py",
    "tests/test_stage25_extended_corpus_cli.py",
)

REQUIRED_DATA = (
    "data/processed/protocols.csv",
    "data/processed/prior_features.csv",
    "data/processed/diagnostic_features.csv",
    "data/processed/fact_features.csv",
    "data/processed/quality_targets.csv",
    "data/processed/all_features.csv",
)

EXPECTED_SYMBOLS = {
    "src/manual_coding_sim/types.py": (
        "ScenarioParameters", "MessageElement", "GeneratedMessage", "QualityVector", "FeatureGroup",
    ),
    "src/manual_coding_sim/procedure_model.py": (
        "CodingOperationRule", "ProcedureStep", "ProcedurePlan", "ProcedureModel",
    ),
    "src/manual_coding_sim/error_model.py": (
        "ErrorStepOutcome", "ErrorProtocol", "ErrorModel",
    ),
    "src/manual_coding_sim/control_model.py": (
        "ControlStepOutcome", "ControlProtocol", "ControlModel",
    ),
    "src/manual_coding_sim/protocol_simulator.py": (
        "ProtocolSimulatorConfig", "SimulationResult", "ProtocolSimulator",
    ),
}


def _sha256(path: Path) -> str:
    """Рассчитать SHA-256 файла."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _top_level_symbols(path: Path) -> set[str]:
    """Получить имена классов и функций верхнего уровня Python-модуля."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }


def _csv_summary(path: Path) -> dict[str, object]:
    """Получить число строк, столбцов и заголовок CSV."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        row_count = sum(1 for _ in reader)
    return {
        "rows": row_count,
        "columns": len(header),
        "header": header,
        "sha256": _sha256(path),
    }


def build_audit(project_root: Path) -> dict[str, object]:
    """Сформировать машинно-читаемый аудит проекта."""
    required_paths = REQUIRED_CORE + REQUIRED_TESTS + REQUIRED_DATA
    file_status = {}
    for relative in required_paths:
        path = project_root / relative
        file_status[relative] = {
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "sha256": _sha256(path) if path.exists() and path.is_file() else None,
        }

    symbol_checks = []
    syntax_errors = []
    for relative, expected in EXPECTED_SYMBOLS.items():
        path = project_root / relative
        if not path.exists():
            symbol_checks.append({"file": relative, "missing_symbols": list(expected)})
            continue
        try:
            actual = _top_level_symbols(path)
        except SyntaxError as exc:
            syntax_errors.append({"file": relative, "error": str(exc)})
            continue
        symbol_checks.append({
            "file": relative,
            "missing_symbols": [name for name in expected if name not in actual],
        })

    csv_artifacts = {
        relative: _csv_summary(project_root / relative)
        for relative in REQUIRED_DATA
        if (project_root / relative).exists()
    }

    missing = [
        relative for relative in REQUIRED_CORE + REQUIRED_TESTS
        if not file_status[relative]["exists"]
    ]
    all_csv_150 = bool(csv_artifacts) and all(
        item["rows"] == 150 for item in csv_artifacts.values()
    )
    return {
        "status": "ready_for_isolated_extension" if not missing and not syntax_errors else "baseline_incomplete",
        "missing_required_core_or_test_files": missing,
        "syntax_errors": syntax_errors,
        "symbol_checks": symbol_checks,
        "csv_artifacts": csv_artifacts,
        "all_required_csv_have_150_rows": all_csv_150,
        "baseline_core_hashes": {
            relative: file_status[relative]["sha256"]
            for relative in REQUIRED_CORE
            if file_status[relative]["exists"]
        },
    }


def _markdown(audit: dict[str, object]) -> str:
    """Сформировать краткий Markdown-отчет."""
    lines = [
        "# Аудит текущей версии",
        "",
        f"Статус: **{audit['status']}**.",
        "",
        f"Отсутствующие файлы: {audit['missing_required_core_or_test_files'] or 'нет'}.",
        f"Синтаксические ошибки: {audit['syntax_errors'] or 'нет'}.",
        f"Все обязательные CSV содержат 150 строк: {audit['all_required_csv_have_150_rows']}.",
        "",
        "## CSV",
        "",
        "| Файл | Строк | Столбцов |",
        "|---|---:|---:|",
    ]
    for relative, info in audit["csv_artifacts"].items():
        lines.append(f"| `{relative}` | {info['rows']} | {info['columns']} |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("extensions/decoding_simulation/reports/stage0"),
    )
    return parser.parse_args()


def main() -> None:
    """Выполнить аудит и сохранить отчеты."""
    args = parse_args()
    root = args.project_root.resolve()
    report_dir = (root / args.report_dir).resolve() if not args.report_dir.is_absolute() else args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    audit = build_audit(root)
    (report_dir / "current_version_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "current_version_audit.md").write_text(
        _markdown(audit),
        encoding="utf-8",
    )
    print(f"Статус аудита: {audit['status']}")
    print(f"JSON-отчет: {report_dir / 'current_version_audit.json'}")
    print(f"Markdown-отчет: {report_dir / 'current_version_audit.md'}")
    if audit["status"] != "ready_for_isolated_extension":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
