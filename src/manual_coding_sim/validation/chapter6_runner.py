"""CLI конфигурационного каркаса и этапов проверки главы 6."""

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
from manual_coding_sim.validation.chapter6_data_loader import (
    Chapter6DataLoadError,
    Chapter6DataLoader,
    Chapter6LoadedInputs,
)
from manual_coding_sim.validation.integral_prediction_validator import (
    IntegralPredictionValidationError,
    IntegralPredictionValidator,
)
from manual_coding_sim.validation.integral_quality_validator import (
    IntegralQualityValidationError,
    IntegralQualityValidator,
)
from manual_coding_sim.validation.validation_dataset_builder import (
    ValidationDatasetBuildError,
    ValidationDatasetBuilder,
    ValidationDatasetBuildResult,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Создать CLI-парсер программного контура главы 6."""

    parser = argparse.ArgumentParser(
        description="Экспериментальная проверка априорной оценки качества главы 6."
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
        help="Показать проверенную конфигурацию этапа 1.",
    )
    parser.add_argument(
        "--validate-inputs",
        action="store_true",
        help="Проверить входные CSV/JSON-артефакты и создать отчеты этапа 2.",
    )
    parser.add_argument(
        "--build-validation-dataset",
        action="store_true",
        help=(
            "Сформировать единый проверочный датасет этапа 3 после "
            "повторной проверки входных артефактов."
        ),
    )
    parser.add_argument(
        "--validate-integral-quality",
        action="store_true",
        help=(
            "Проверить согласованность integral_quality с контрольной "
            "агрегацией шести частных критериев на этапе 4."
        ),
    )
    parser.add_argument(
        "--calculate-integral-metrics",
        action="store_true",
        help=(
            "Рассчитать ошибки и метрики интегрального прогноза на этапе 5."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить CLI главы 6 и вернуть код завершения."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    project_root = Path(args.project_root)

    try:
        config = _load_config(project_root, args.config)
    except (FileNotFoundError, Chapter6ConfigError) as error:
        print(f"Ошибка конфигурации главы 6: {error}")
        return 1

    _print_config(config, project_root, args.config, args.show_config)

    if not any(
        (
            args.validate_inputs,
            args.build_validation_dataset,
            args.validate_integral_quality,
            args.calculate_integral_metrics,
        )
    ):
        print(
            "Расчетные этапы не выполнялись. Используйте флаг "
            "--validate-inputs, --build-validation-dataset, "
            "--validate-integral-quality или --calculate-integral-metrics."
        )
        return 0

    loaded_inputs: Chapter6LoadedInputs | None = None
    if args.validate_inputs:
        try:
            loaded_inputs = Chapter6DataLoader(
                config=config,
                project_root=project_root,
            ).load_and_save_report()
        except (FileNotFoundError, OSError, Chapter6DataLoadError) as error:
            print(f"Ошибка проверки входных артефактов главы 6: {error}")
            return 1
        _print_stage2_result(loaded_inputs)

    dataset_result: ValidationDatasetBuildResult | None = None
    if args.build_validation_dataset:
        try:
            dataset_result = ValidationDatasetBuilder(
                config=config,
                project_root=project_root,
            ).build_and_save(loaded_inputs=loaded_inputs)
        except (
            FileNotFoundError,
            OSError,
            Chapter6DataLoadError,
            ValidationDatasetBuildError,
        ) as error:
            print(f"Ошибка формирования проверочного датасета главы 6: {error}")
            return 1
        _print_stage3_result(dataset_result)

    active_dataset = dataset_result.dataset if dataset_result is not None else None

    if args.validate_integral_quality:
        try:
            validation_result = IntegralQualityValidator(
                config=config,
                project_root=project_root,
            ).validate_and_save(dataset=active_dataset)
        except (
            FileNotFoundError,
            OSError,
            IntegralQualityValidationError,
            TypeError,
            ValueError,
        ) as error:
            print(f"Ошибка проверки фактического интегрального качества: {error}")
            return 1

        metrics = validation_result.report["metrics"]
        print("Проверка фактического интегрального качества завершена.")
        print(f"Сценариев: {validation_result.report['row_count']}")
        print(
            "Среднее абсолютное расхождение: "
            f"{metrics['mean_absolute_difference']:.10f}"
        )
        print(
            "Максимальное абсолютное расхождение: "
            f"{metrics['max_absolute_difference']:.10f}"
        )
        print(
            "Сценариев вне допуска: "
            f"{metrics['outside_tolerance_count']}"
        )
        if validation_result.csv_path is not None:
            print(f"Таблица согласованности: {validation_result.csv_path}")
        if validation_result.json_path is not None:
            print(f"JSON-отчет: {validation_result.json_path}")
        if validation_result.markdown_path is not None:
            print(f"Markdown-отчет: {validation_result.markdown_path}")

        if not validation_result.passed:
            print("Этап 4 не пройден: контрольное расхождение превышает допуск.")
            return 1

        print(
            "Этап 4 выполнен. Переход к этапу 5 требует "
            "отдельного подтверждения."
        )

    if not args.calculate_integral_metrics:
        return 0

    try:
        prediction_result = IntegralPredictionValidator(
            config=config,
            project_root=project_root,
        ).validate_and_save(dataset=active_dataset)
    except (
        FileNotFoundError,
        OSError,
        IntegralPredictionValidationError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Ошибка расчета метрик интегрального прогноза: {error}")
        return 1

    prediction_metrics = prediction_result.report["metrics"]
    print("Метрики интегрального априорного прогноза рассчитаны.")
    print(f"Сценариев: {prediction_result.report['row_count']}")
    print(f"MAE: {prediction_metrics['mae']:.10f}")
    print(f"RMSE: {prediction_metrics['rmse']:.10f}")
    print(f"Bias: {prediction_metrics['bias']:.10f}")
    print(f"Spearman: {prediction_metrics['spearman']:.10f}")
    print(f"Kendall: {prediction_metrics['kendall']:.10f}")
    print(f"R²: {prediction_metrics['r2']:.10f}")
    if prediction_result.csv_path is not None:
        print(f"Таблица ошибок: {prediction_result.csv_path}")
    if prediction_result.json_path is not None:
        print(f"JSON-отчет: {prediction_result.json_path}")
    if prediction_result.markdown_path is not None:
        print(f"Markdown-отчет: {prediction_result.markdown_path}")

    if not prediction_result.passed:
        print("Этап 5 не пройден: рассчитаны некорректные метрики.")
        return 1

    print("Этап 5 выполнен. Переход к этапу 6 требует отдельного подтверждения.")
    return 0


def _print_stage3_result(result: ValidationDatasetBuildResult) -> None:
    """Вывести результат формирования датасета этапа 3."""

    print("Проверочный датасет главы 6 успешно сформирован.")
    print(f"Строк: {len(result.dataset)}")
    print(f"Колонок: {len(result.dataset.columns)}")
    print(f"Шагов объединения: {len(result.merge_steps)}")
    print(
        "Прогнозные классы low/medium/high: "
        f"{result.predicted_class_counts['low']}/"
        f"{result.predicted_class_counts['medium']}/"
        f"{result.predicted_class_counts['high']}"
    )
    print(
        "Фактические классы low/medium/high: "
        f"{result.factual_class_counts['low']}/"
        f"{result.factual_class_counts['medium']}/"
        f"{result.factual_class_counts['high']}"
    )
    if result.output_path is not None:
        print(f"Проверочный датасет: {result.output_path}")
    print("Этап 3 выполнен. Переход к этапу 4 требует отдельного подтверждения.")


def _print_stage2_result(loaded: Chapter6LoadedInputs) -> None:
    """Вывести результат проверки входов этапа 2."""

    report = loaded.validation_report
    print("Проверка входных артефактов главы 6 успешно завершена.")
    print(f"Проверено CSV-файлов: {report.checked_csv_count}")
    print(f"Проверено JSON-файлов: {report.checked_json_count}")
    print(f"Сценариев в каждом основном артефакте: {report.expected_row_count}")
    print(
        "Максимальное расхождение Q_pred между артефактами: "
        f"{report.q_pred_consistency['max_abs_difference']:.12g}"
    )
    if loaded.report_json_path is not None:
        print(f"JSON-отчет: {loaded.report_json_path}")
    if loaded.report_markdown_path is not None:
        print(f"Markdown-отчет: {loaded.report_markdown_path}")
    print("Этап 2 выполнен. Переход к этапу 3 требует отдельного подтверждения.")


def _print_config(
    config: Chapter6ValidationConfig,
    project_root: Path,
    config_path: str,
    show_details: bool,
) -> None:
    """Вывести проверенную конфигурацию в формате этапа 1."""

    print("Конфигурационный каркас главы 6 успешно загружен.")
    print("Конфигурация главы 6 успешно проверена.")
    print(f"Корень проекта: {project_root}")
    print(f"Конфигурация главы 6: {config_path}")
    print(f"Ключи объединения: {config.merge.key_columns}")
    print(f"Режим объединения: {config.merge.validation}")
    print(f"Ожидаемое число сценариев: {config.merge.expected_row_count}")
    print(
        "Пороги классов качества: "
        f"low < {config.decision_thresholds.low_max}; "
        f"medium < {config.decision_thresholds.high_min}; далее high."
    )
    print(
        "Параметры bootstrap: "
        f"повторов {config.bootstrap.resamples}, "
        f"уровень доверия {config.bootstrap.confidence_level}, "
        f"random_seed {config.bootstrap.random_seed}."
    )
    if show_details:
        # Сначала выводится полный машинно-читаемый снимок конфигурации.
        # Дополнительные строки сохраняют совместимость с интерфейсом этапа 1.
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
        print(f"expected_row_count = {config.merge.expected_row_count}")
        print(f"join_keys = {', '.join(config.merge.key_columns)}")
        print(f"merge_validate = {config.merge.validation}")
        print(f"sampling_unit = {config.bootstrap.sampling_unit}")


def _load_config(
    project_root: Path,
    config_path: str,
) -> Chapter6ValidationConfig:
    """Загрузить конфигурацию через публичный API этапа 1."""

    return load_chapter6_validation_config(
        config_path=config_path,
        project_root=project_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
