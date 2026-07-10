"""CLI-каркас запуска программного блока главы 5.

На этапе 2 runner умеет загружать и валидировать YAML-конфигурации главы 5.
Полный расчетный pipeline будет собран после реализации загрузки данных,
нормировки, частных критериев, интегральной оценки и неопределенности.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from manual_coding_sim.prediction.chapter5_config import (
    Chapter5PredictionConfig,
    load_chapter5_prediction_config,
)
from manual_coding_sim.prediction.chapter5_data_loader import (
    Chapter5DataLoadError,
    Chapter5DataLoader,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Создать CLI-парсер для главы 5."""

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
        help="Путь к конфигурации главы 5.",
    )
    parser.add_argument(
        "--validate-inputs",
        action="store_true",
        help="Проверить и объединить входные данные главы 5 без расчета Q_pred.",
    )
    return parser


def _load_config_or_default(project_root: Path, config_path: str) -> Chapter5PredictionConfig:
    """Загрузить конфигурацию, если файл существует, иначе вернуть настройки по умолчанию."""

    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_absolute():
        resolved_config_path = project_root / resolved_config_path
    if resolved_config_path.exists():
        return load_chapter5_prediction_config(config_path=config_path, project_root=project_root)

    print("Файл конфигурации главы 5 не найден, используется конфигурация по умолчанию.")
    return Chapter5PredictionConfig()


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить CLI-каркас главы 5 и вернуть код завершения."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    project_root = Path(args.project_root)
    config = _load_config_or_default(project_root=project_root, config_path=args.config)
    config.validate()

    print("Каркас программного блока главы 5 успешно загружен.")
    print("Конфигурация главы 5 успешно проверена.")
    print(f"Корень проекта: {project_root}")
    print(f"Конфигурация главы 5: {args.config}")
    print(f"Веса частных критериев: {config.quality_weights.weights}")
    print(f"Направления латентных факторов: {config.factor_directions.directions}")
    if args.validate_inputs:
        loader = Chapter5DataLoader(
            paths=config.inputs,
            project_root=project_root,
            expected_topic_count=config.expected_topic_count,
        )
        try:
            loaded_inputs = loader.load()
        except (Chapter5DataLoadError, FileNotFoundError) as error:
            print(f"Проверка входных данных главы 5 не пройдена: {error}")
            return 1
        report = loaded_inputs.validation_report
        print(
            "Данные главы 5 успешно загружены: "
            f"{report.scenario_count} сценариев, {report.topic_count} латентных фактора."
        )
        print(f"Ключи объединения: {report.merge_key_columns}")
    print("Расчет Q_pred не выполнялся: это будет реализовано на следующих этапах.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
