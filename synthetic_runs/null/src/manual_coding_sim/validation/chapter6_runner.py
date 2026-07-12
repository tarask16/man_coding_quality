"""CLI конфигурационного каркаса и этапов проверки главы 6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from manual_coding_sim.validation.baseline_models import (
    BaselineComparisonError,
    BaselineComparisonResult,
    BaselineModelsValidator,
)
from manual_coding_sim.validation.bootstrap_analysis import (
    BootstrapAnalysisError,
    BootstrapAnalysisValidator,
)
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
from manual_coding_sim.validation.classification_validator import (
    ClassificationValidationError,
    ClassificationValidator,
)
from manual_coding_sim.validation.integral_prediction_validator import (
    IntegralPredictionValidationError,
    IntegralPredictionValidator,
)
from manual_coding_sim.validation.interval_prediction_validator import (
    IntervalPredictionValidationError,
    IntervalPredictionValidator,
)
from manual_coding_sim.validation.integral_quality_validator import (
    IntegralQualityValidationError,
    IntegralQualityValidator,
)
from manual_coding_sim.validation.partial_criteria_validator import (
    PartialCriteriaValidationError,
    PartialCriteriaValidator,
)
from manual_coding_sim.validation.prediction_error_analyzer import (
    PredictionErrorAnalysisError,
    PredictionErrorAnalyzer,
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
    parser.add_argument(
        "--validate-partial-criteria",
        action="store_true",
        help=(
            "Проверить шесть частных прогнозных критериев на этапе 6."
        ),
    )
    parser.add_argument(
        "--validate-classification",
        action="store_true",
        help=(
            "Проверить классификацию уровней качества на этапе 7."
        ),
    )
    parser.add_argument(
        "--validate-interval-prediction",
        action="store_true",
        help=(
            "Проверить покрытие и ширину прогнозных интервалов на этапе 8."
        ),
    )
    parser.add_argument(
        "--compare-baselines",
        action="store_true",
        help=(
            "Сравнить модель главы 5 с mean, prior-only и theta-only "
            "baseline на этапе 9."
        ),
    )
    parser.add_argument(
        "--bootstrap-analysis",
        action="store_true",
        help=(
            "Рассчитать bootstrap-доверительные интервалы метрик и "
            "парных разностей моделей на этапе 10."
        ),
    )
    parser.add_argument(
        "--analyze-prediction-errors",
        action="store_true",
        help=(
            "Выполнить диагностический анализ ошибок прогноза на этапе 11."
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
            args.validate_partial_criteria,
            args.validate_classification,
            args.validate_interval_prediction,
            args.compare_baselines,
            args.bootstrap_analysis,
            args.analyze_prediction_errors,
        )
    ):
        print(
            "Расчетные этапы не выполнялись. Используйте флаг "
            "--validate-inputs, --build-validation-dataset, "
            "--validate-integral-quality, --calculate-integral-metrics, "
            "--validate-partial-criteria, --validate-classification, "
            "--validate-interval-prediction, --compare-baselines, "
            "--bootstrap-analysis или --analyze-prediction-errors."
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

    if args.calculate_integral_metrics:
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

        print(
            "Этап 5 выполнен. Переход к этапу 6 требует "
            "отдельного подтверждения."
        )

    if args.validate_partial_criteria:
        try:
            partial_result = PartialCriteriaValidator(
                config=config,
                project_root=project_root,
            ).validate_and_save(dataset=active_dataset)
        except (
            FileNotFoundError,
            OSError,
            PartialCriteriaValidationError,
            TypeError,
            ValueError,
        ) as error:
            print(f"Ошибка проверки частных прогнозных критериев: {error}")
            return 1

        summary = partial_result.report["summary"]
        print("Проверка частных прогнозных критериев завершена.")
        print(f"Сценариев: {partial_result.report['row_count']}")
        print(f"Проверено критериев: {partial_result.report['criterion_count']}")
        print(f"Среднее MAE: {summary['mean_mae']:.10f}")
        print(f"Средний Spearman: {summary['mean_spearman']:.10f}")
        print(
            "Наименьший MAE: "
            f"{summary['best_mae_criterion']} = {summary['best_mae']:.10f}"
        )
        print(
            "Наибольший MAE: "
            f"{summary['worst_mae_criterion']} = {summary['worst_mae']:.10f}"
        )
        if partial_result.csv_path is not None:
            print(f"Таблица метрик: {partial_result.csv_path}")
        if partial_result.json_path is not None:
            print(f"JSON-отчет: {partial_result.json_path}")
        if partial_result.markdown_path is not None:
            print(f"Markdown-отчет: {partial_result.markdown_path}")

        if not partial_result.passed:
            print("Этап 6 не пройден: рассчитаны некорректные метрики.")
            return 1

        print(
            "Этап 6 выполнен. Переход к этапу 7 требует "
            "отдельного подтверждения."
        )

    if args.validate_classification:
        try:
            classification_result = ClassificationValidator(
                config=config,
                project_root=project_root,
            ).validate_and_save(dataset=active_dataset)
        except (
            FileNotFoundError,
            OSError,
            ClassificationValidationError,
            TypeError,
            ValueError,
        ) as error:
            print(f"Ошибка проверки классификации уровней качества: {error}")
            return 1

        classification_metrics = classification_result.report["metrics"]
        critical_errors = classification_result.report["critical_errors"]
        print("Проверка классификации уровней качества завершена.")
        print(f"Сценариев: {classification_result.report['row_count']}")
        print(f"Accuracy: {classification_metrics['accuracy']:.10f}")
        print(
            "Balanced Accuracy: "
            f"{classification_metrics['balanced_accuracy']:.10f}"
        )
        print(f"Macro F1: {classification_metrics['macro_f1']:.10f}")
        print(f"Weighted F1: {classification_metrics['weighted_f1']:.10f}")
        print(
            "Критических ошибок low->high: "
            f"{critical_errors['low_to_high']}"
        )
        print(
            "Критических ошибок high->low: "
            f"{critical_errors['high_to_low']}"
        )
        if classification_result.predictions_path is not None:
            print(
                "Таблица классификации: "
                f"{classification_result.predictions_path}"
            )
        if classification_result.confusion_matrix_path is not None:
            print(
                "Матрица ошибок: "
                f"{classification_result.confusion_matrix_path}"
            )
        if classification_result.json_path is not None:
            print(f"JSON-отчет: {classification_result.json_path}")
        if classification_result.markdown_path is not None:
            print(f"Markdown-отчет: {classification_result.markdown_path}")

        if not classification_result.passed:
            print("Этап 7 не пройден: классификационные метрики некорректны.")
            return 1

        print(
            "Этап 7 выполнен. Переход к этапу 8 требует "
            "отдельного подтверждения."
        )

    if args.validate_interval_prediction:
        try:
            interval_result = IntervalPredictionValidator(
                config=config,
                project_root=project_root,
            ).validate_and_save(dataset=active_dataset)
        except (
            FileNotFoundError,
            OSError,
            IntervalPredictionValidationError,
            TypeError,
            ValueError,
        ) as error:
            print(f"Ошибка проверки интервального прогноза: {error}")
            return 1

        interval_metrics = interval_result.report["metrics"]
        print("Проверка интервального прогноза качества завершена.")
        print(f"Сценариев: {interval_result.report['row_count']}")
        print(f"Coverage rate: {interval_metrics['coverage_rate']:.10f}")
        print(
            "Средняя ширина интервала: "
            f"{interval_metrics['mean_interval_width']:.10f}"
        )
        print(
            "Медианная ширина интервала: "
            f"{interval_metrics['median_interval_width']:.10f}"
        )
        print(
            "Факт ниже нижней границы: "
            f"{interval_metrics['miss_lower_count']}"
        )
        print(
            "Факт выше верхней границы: "
            f"{interval_metrics['miss_upper_count']}"
        )
        print(
            "Среднее расстояние до интервала: "
            f"{interval_metrics['mean_distance_to_interval']:.10f}"
        )
        if interval_result.details_path is not None:
            print(f"Таблица покрытия: {interval_result.details_path}")
        if interval_result.json_path is not None:
            print(f"JSON-отчет: {interval_result.json_path}")
        if interval_result.markdown_path is not None:
            print(f"Markdown-отчет: {interval_result.markdown_path}")

        if not interval_result.passed:
            print("Этап 8 не пройден: метрики интервалов рассчитаны некорректно.")
            return 1

        print(
            "Этап 8 выполнен. Переход к этапу 9 требует "
            "отдельного подтверждения."
        )

    baseline_result: BaselineComparisonResult | None = None
    if args.compare_baselines:
        try:
            baseline_result = BaselineModelsValidator(
                config=config,
                project_root=project_root,
            ).validate_and_save(dataset=active_dataset)
        except (
            FileNotFoundError,
            OSError,
            BaselineComparisonError,
            TypeError,
            ValueError,
        ) as error:
            print(f"Ошибка сравнения с базовыми моделями: {error}")
            return 1

        baseline_metrics = {
            row["model"]: row for row in baseline_result.report["metrics"]
        }
        best_models = baseline_result.report["best_models"]
        print("Сравнение с базовыми моделями завершено.")
        print(f"Сценариев: {baseline_result.report['row_count']}")
        print(
            "Mean baseline MAE: "
            f"{baseline_metrics['mean_baseline']['mae']:.10f}"
        )
        print(
            "Prior-only baseline MAE: "
            f"{baseline_metrics['prior_only_baseline']['mae']:.10f}"
        )
        print(
            "Theta-only baseline MAE: "
            f"{baseline_metrics['theta_only_baseline']['mae']:.10f}"
        )
        print(
            "Chapter 5 model MAE: "
            f"{baseline_metrics['chapter5_model']['mae']:.10f}"
        )
        print(
            "Лучший MAE: "
            f"{best_models['mae']['model']} = "
            f"{best_models['mae']['value']:.10f}"
        )
        print(
            "Лучший Spearman: "
            f"{best_models['spearman']['model']} = "
            f"{best_models['spearman']['value']:.10f}"
        )
        if baseline_result.predictions_path is not None:
            print(f"Baseline-прогнозы: {baseline_result.predictions_path}")
        if baseline_result.comparison_path is not None:
            print(f"Таблица сравнения: {baseline_result.comparison_path}")
        if baseline_result.json_path is not None:
            print(f"JSON-отчет: {baseline_result.json_path}")
        if baseline_result.markdown_path is not None:
            print(f"Markdown-отчет: {baseline_result.markdown_path}")

        if not baseline_result.passed:
            print("Этап 9 не пройден: baseline-метрики рассчитаны некорректно.")
            return 1

        print(
            "Этап 9 выполнен. Переход к этапу 10 требует "
            "отдельного подтверждения."
        )

    if args.bootstrap_analysis:
        try:
            bootstrap_result = BootstrapAnalysisValidator(
                config=config,
                project_root=project_root,
            ).validate_and_save(
                predictions=(
                    baseline_result.predictions
                    if baseline_result is not None
                    else None
                )
            )
        except (
            FileNotFoundError,
            OSError,
            BootstrapAnalysisError,
            TypeError,
            ValueError,
        ) as error:
            print(f"Ошибка bootstrap-анализа статистической устойчивости: {error}")
            return 1

        chapter5_intervals = {
            row["metric"]: row
            for row in bootstrap_result.report["chapter5_confidence_intervals"]
        }
        summary = bootstrap_result.report["summary"]
        sampling = bootstrap_result.report["sampling"]
        print("Bootstrap-анализ статистической устойчивости завершен.")
        print(f"Сценариев: {bootstrap_result.report['row_count']}")
        print(f"Bootstrap-повторов: {sampling['resamples']}")
        print(f"Уровень доверия: {sampling['confidence_level']:.4f}")
        print(
            "MAE модели главы 5: "
            f"{chapter5_intervals['mae']['point_estimate']:.10f} "
            f"[{chapter5_intervals['mae']['ci_lower']:.10f}; "
            f"{chapter5_intervals['mae']['ci_upper']:.10f}]"
        )
        print(
            "Spearman модели главы 5: "
            f"{chapter5_intervals['spearman']['point_estimate']:.10f} "
            f"[{chapter5_intervals['spearman']['ci_lower']:.10f}; "
            f"{chapter5_intervals['spearman']['ci_upper']:.10f}]"
        )
        print(
            "Устойчивых преимуществ модели главы 5: "
            f"{summary['stable_chapter5_wins']}"
        )
        print(
            "Устойчивых преимуществ baseline: "
            f"{summary['stable_baseline_wins']}"
        )
        print(
            "Различий без статистически устойчивого вывода: "
            f"{summary['no_stable_difference']}"
        )
        if bootstrap_result.confidence_intervals_path is not None:
            print(
                "Доверительные интервалы: "
                f"{bootstrap_result.confidence_intervals_path}"
            )
        if bootstrap_result.model_differences_path is not None:
            print(
                "Парные разности моделей: "
                f"{bootstrap_result.model_differences_path}"
            )
        if bootstrap_result.json_path is not None:
            print(f"JSON-отчет: {bootstrap_result.json_path}")
        if bootstrap_result.markdown_path is not None:
            print(f"Markdown-отчет: {bootstrap_result.markdown_path}")

        if not bootstrap_result.passed:
            print("Этап 10 не пройден: bootstrap-результаты некорректны.")
            return 1

        print(
            "Этап 10 выполнен. Переход к этапу 11 требует "
            "отдельного подтверждения."
        )

    if args.analyze_prediction_errors:
        try:
            error_result = PredictionErrorAnalyzer(
                config=config,
                project_root=project_root,
            ).analyze_and_save(dataset=active_dataset)
        except (
            FileNotFoundError,
            OSError,
            PredictionErrorAnalysisError,
            TypeError,
            ValueError,
        ) as error:
            print(f"Ошибка диагностического анализа ошибок прогноза: {error}")
            return 1

        error_summary = error_result.report["summary"]
        uncertainty_relation = error_result.report["uncertainty_relation"]
        strongest_relation = error_summary[
            "strongest_diagnostic_absolute_error_relation"
        ]
        print("Анализ ошибок априорного прогноза завершен.")
        print(f"Сценариев: {error_result.report['row_count']}")
        print(f"Сценариев в top-10: {error_result.report['top_error_count']}")
        print(
            "Максимальная абсолютная ошибка: "
            f"{error_summary['max_absolute_error']:.10f}"
        )
        print(
            "Занижений / завышений: "
            f"{error_summary['underestimation_count']} / "
            f"{error_summary['overestimation_count']}"
        )
        print(
            "Spearman неопределенности с абсолютной ошибкой: "
            f"{uncertainty_relation['spearman_absolute_error']:.10f}"
        )
        print(
            "Наиболее сильная диагностическая связь: "
            f"{strongest_relation['variable']} = "
            f"{strongest_relation['spearman']:.10f}"
        )
        print(
            "Наибольшая MAE по доминирующему фактору: "
            f"{error_summary['worst_dominant_factor_by_mae']} = "
            f"{error_summary['worst_dominant_factor_mae']:.10f}"
        )
        if error_result.top_errors_path is not None:
            print(f"Top-10 ошибок: {error_result.top_errors_path}")
        if error_result.group_analysis_path is not None:
            print(f"Групповой анализ: {error_result.group_analysis_path}")
        if error_result.json_path is not None:
            print(f"JSON-отчет: {error_result.json_path}")
        if error_result.markdown_path is not None:
            print(f"Markdown-отчет: {error_result.markdown_path}")

        if not error_result.passed:
            print("Этап 11 не пройден: результаты анализа ошибок некорректны.")
            return 1

        print(
            "Этап 11 выполнен. Переход к этапу 12 требует "
            "отдельного подтверждения."
        )

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
