"""CLI-каркас запуска программного блока главы 5.

На этапе 1 runner проверяет импортируемость пакета и выводит диагностическое
сообщение. Полный расчетный pipeline будет собран после реализации загрузки,
нормировки, частных критериев, интегральной оценки и неопределенности.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from manual_coding_sim.prediction.chapter5_config import Chapter5PredictionConfig


def build_arg_parser() -> argparse.ArgumentParser:
    """Создать CLI-парсер для каркаса главы 5."""

    parser = argparse.ArgumentParser(
        description="Каркас запуска априорного прогнозирования качества главы 5."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Корень проекта. По умолчанию используется текущий каталог.",
    )
    parser.add_argument(
        "--config",
        default="configs/chapter5.yaml",
        help="Путь к конфигурации главы 5. Будет использован на следующем этапе.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить CLI-каркас главы 5 и вернуть код завершения."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    project_root = Path(args.project_root)
    config = Chapter5PredictionConfig()
    config.validate()

    print("Каркас программного блока главы 5 успешно загружен.")
    print(f"Корень проекта: {project_root}")
    print(f"Конфигурация главы 5: {args.config}")
    print("Расчет Q_pred не выполнялся: это будет реализовано на следующих этапах.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
