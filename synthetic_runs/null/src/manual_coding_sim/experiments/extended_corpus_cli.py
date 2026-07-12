"""CLI для генерации расширенного корпуса главы 3."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from manual_coding_sim.experiments.extended_corpus_runner import ExtendedCorpusRunner


def build_arg_parser() -> argparse.ArgumentParser:
    """Создать парсер аргументов генератора расширенного корпуса."""

    parser = argparse.ArgumentParser(
        description="Генерация расширенного корпуса главы 3 для последующего LDA-анализа.",
    )
    parser.add_argument(
        "--config",
        default="configs/chapter3_extended_corpus.yaml",
        help="Путь к YAML-конфигурации генерации расширенного корпуса.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Корень проекта, относительно которого разрешаются выходные пути.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить CLI-запуск генерации корпуса и вернуть код завершения."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        runner = ExtendedCorpusRunner.from_yaml(args.config)
        result = runner.run(project_root=Path(args.project_root))
    except Exception as exc:  # noqa: BLE001 - CLI должен преобразовать ошибку в код.
        print(f"Ошибка генерации расширенного корпуса: {exc}", file=sys.stderr)
        return 1

    print("Расширенный корпус главы 3 сформирован.")
    print(f"Документы: {result.document_count}")
    print(f"Уникальные сценарии: {result.unique_scenario_count}")
    print(f"Уникальные протоколы: {result.unique_protocol_count}")
    print(f"Отчет: {result.summary_md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - точка входа проверяется через main().
    raise SystemExit(main())
