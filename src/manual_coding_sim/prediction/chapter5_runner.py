"""CLI-каркас запуска программного блока главы 5.

Runner умеет проверять конфигурацию и входные данные главы 5. На этапе 5
добавлена нормировка числовых априорных признаков ``X_prior`` с сохранением
таблицы ``normalized_prior_features.csv`` и отчета ``normalization_report.json``.
Полный расчет ``Q_pred`` будет собран на последующих этапах.
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
    Chapter5LoadedInputs,
)
from manual_coding_sim.prediction.chapter5_leakage_guard import (
    Chapter5LeakageError,
    Chapter5LeakageGuard,
)
from manual_coding_sim.prediction.paths import resolve_project_path
from manual_coding_sim.prediction.prior_feature_normalizer import (
    PriorFeatureNormalizer,
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
    parser.add_argument(
        "--normalize-inputs",
        action="store_true",
        help="Нормировать априорные признаки главы 5 и сохранить отчет.",
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

    loaded_inputs: Chapter5LoadedInputs | None = None
    if args.validate_inputs or args.normalize_inputs:
        loaded_inputs = _load_and_report_inputs(project_root, config)
        if loaded_inputs is None:
            return 1

    if args.normalize_inputs:
        _normalize_and_save(project_root, config, loaded_inputs)

    print("Расчет Q_pred не выполнялся: это будет реализовано на следующих этапах.")
    return 0


def _load_and_report_inputs(
    project_root: Path,
    config: Chapter5PredictionConfig,
) -> Chapter5LoadedInputs | None:
    """Загрузить входы главы 5 и вывести краткий отчет."""

    loader = Chapter5DataLoader(
        paths=config.inputs,
        project_root=project_root,
        expected_topic_count=config.expected_topic_count,
    )
    try:
        loaded_inputs = loader.load()
    except (Chapter5DataLoadError, Chapter5LeakageError, FileNotFoundError) as error:
        print(f"Проверка входных данных главы 5 не пройдена: {error}")
        return None

    leakage_guard = Chapter5LeakageGuard()
    leakage_report = leakage_guard.check_dataframe(
        loaded_inputs.prior_features,
        source_name=str(resolve_project_path(project_root, config.inputs.prior_features_path)),
    )
    leakage_report_path = resolve_project_path(
        project_root,
        config.outputs.reports_dir / "chapter5_leakage_report.json",
    )
    leakage_guard.save_json_report(leakage_report_path, leakage_report)
    report = loaded_inputs.validation_report
    print(
        "Данные главы 5 успешно загружены: "
        f"{report.scenario_count} сценариев, {report.topic_count} латентных фактора."
    )
    print(f"Ключи объединения: {report.merge_key_columns}")
    print("Проверка методической утечки: пройдена.")
    print(f"Отчет проверки утечки сохранен: {leakage_report_path}")
    return loaded_inputs


def _normalize_and_save(
    project_root: Path,
    config: Chapter5PredictionConfig,
    loaded_inputs: Chapter5LoadedInputs | None,
) -> None:
    """Выполнить нормировку априорных признаков и сохранить артефакты."""

    if loaded_inputs is None:
        msg = "Внутренняя ошибка: нормировка вызвана без загруженных входных данных."
        raise RuntimeError(msg)

    normalizer = PriorFeatureNormalizer(config.prior_feature_dictionary)
    result = normalizer.normalize(loaded_inputs.prior_features)
    normalized_path = resolve_project_path(
        project_root,
        config.outputs.normalized_prior_features_path,
    )
    report_path = resolve_project_path(project_root, config.outputs.normalization_report_path)
    normalizer.save_outputs(
        result,
        normalized_features_path=normalized_path,
        report_path=report_path,
    )
    print("Нормировка априорных признаков: выполнена.")
    print(f"Нормированных признаков: {result.report.normalized_feature_count}")
    print(f"Пропущено нечисловых признаков: {len(result.report.non_numeric_features)}")
    print(f"Таблица нормированных признаков сохранена: {normalized_path}")
    print(f"Отчет нормировки сохранен: {report_path}")


if __name__ == "__main__":
    raise SystemExit(main())
