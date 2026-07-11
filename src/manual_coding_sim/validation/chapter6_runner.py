"""CLI конфигурационного каркаса экспериментальной проверки главы 6.

На этапе 1 runner только загружает и проверяет YAML-конфигурацию, а также
выводит ее в человекочитаемом или JSON-представлении. Чтение расчетных
артефактов будет добавлено на этапе 2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from manual_coding_sim.validation.chapter6_config import (
    Chapter6ConfigError,
    Chapter6ValidationConfig,
    load_chapter6_validation_config,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Создать парсер аргументов командной строки главы 6."""

    parser = argparse.ArgumentParser(
        description="Конфигурационный каркас экспериментальной проверки главы 6."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Корень проекта. По умолчанию используется текущий каталог.",
    )
    parser.add_argument(
        "--config",
        default="configs/chapter6.yaml",
        help="Путь к YAML-конфигурации главы 6.",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Вывести полную проверенную конфигурацию в формате JSON.",
    )
    parser.add_argument(
        "--show-resolved-paths",
        action="store_true",
        help="Вывести абсолютные пути входных и выходных артефактов.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Загрузить конфигурацию главы 6, вывести сведения и вернуть код завершения."""

    args = build_arg_parser().parse_args(argv)
    project_root = Path(args.project_root)
    try:
        config = load_chapter6_validation_config(
            config_path=args.config,
            project_root=project_root,
        )
    except (FileNotFoundError, Chapter6ConfigError) as error:
        print(f"Конфигурация главы 6 не загружена: {error}")
        return 1

    _print_summary(config, project_root, args.config)
    if args.show_config:
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
    if args.show_resolved_paths:
        print(
            json.dumps(
                config.resolved_paths(project_root.resolve()),
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


def _print_summary(
    config: Chapter6ValidationConfig,
    project_root: Path,
    config_path: str,
) -> None:
    """Вывести краткую сводку проверенной конфигурации."""

    thresholds = config.decision_thresholds
    bootstrap = config.bootstrap
    print("Конфигурационный каркас главы 6 успешно загружен.")
    print("Конфигурация главы 6 успешно проверена.")
    print(f"Корень проекта: {project_root}")
    print(f"Файл конфигурации: {config_path}")
    print(f"Ключи объединения: {config.merge.key_columns}")
    print(f"Режим объединения: {config.merge.validation}")
    print(f"Ожидаемое число сценариев: {config.merge.expected_row_count}")
    print(f"Пороги классов: low < {thresholds.low_max}; high >= {thresholds.high_min}")
    print(
        "Bootstrap: "
        f"{bootstrap.resamples} повторов, доверительный уровень "
        f"{bootstrap.confidence_level}, random_seed={bootstrap.random_seed}."
    )


if __name__ == "__main__":
    raise SystemExit(main())
