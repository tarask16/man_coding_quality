"""CLI-каркас запуска программного блока главы 5.

Runner умеет проверять конфигурацию и входные данные главы 5. На этапе 5
добавлена нормировка числовых априорных признаков ``X_prior`` с сохранением
таблицы ``normalized_prior_features.csv`` и отчета ``normalization_report.json``.
Этап 9 добавляет интервальную оценку неопределенности прогноза ``Q_pred``.
Этап 10 завершает CLI-контур единым флагом полного запуска.
Этап 11 добавляет формирование итогового JSON- и Markdown-отчета.
Этап 12 добавляет финальную приемку артефактов главы 5.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from manual_coding_sim.prediction.chapter5_acceptance import (
    Chapter5AcceptanceError,
    Chapter5AcceptanceReport,
    Chapter5AcceptanceValidator,
)
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
from manual_coding_sim.prediction.chapter5_pipeline import (
    Chapter5PipelineRunReport,
    Chapter5PipelineRunReporter,
)
from manual_coding_sim.prediction.chapter5_report_builder import (
    Chapter5FinalReport,
    Chapter5ReportBuilder,
    Chapter5ReportBuildError,
)
from manual_coding_sim.prediction.integral_quality_predictor import (
    IntegralQualityPredictionResult,
    IntegralQualityPredictor,
)
from manual_coding_sim.prediction.latent_quality_component import (
    LatentQualityComponentCalculator,
    LatentQualityComponentResult,
)
from manual_coding_sim.prediction.partial_quality_predictor import (
    PartialQualityPredictionResult,
    PartialQualityPredictor,
)
from manual_coding_sim.prediction.paths import resolve_project_path
from manual_coding_sim.prediction.prediction_uncertainty import (
    PredictionUncertaintyEstimator,
    PredictionUncertaintyResult,
)
from manual_coding_sim.prediction.prior_feature_normalizer import (
    PriorFeatureNormalizationResult,
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
    parser.add_argument(
        "--calculate-latent-component",
        action="store_true",
        help="Рассчитать латентную компоненту качества Q_lat и сохранить отчет.",
    )
    parser.add_argument(
        "--calculate-partial-criteria",
        action="store_true",
        help="Рассчитать частные прогнозные критерии и сохранить q_pred_components.csv.",
    )
    parser.add_argument(
        "--calculate-q-pred",
        action="store_true",
        help="Рассчитать интегральный прогнозный показатель Q_pred и сохранить q_pred.csv.",
    )
    parser.add_argument(
        "--estimate-uncertainty",
        action="store_true",
        help="Рассчитать неопределенность и интервалы прогноза Q_pred.",
    )
    parser.add_argument(
        "--run-full-pipeline",
        action="store_true",
        help=(
            "Выполнить полный контур главы 5: проверку входов, нормировку, "
            "Q_lat, частные критерии, Q_pred и интервальную оценку."
        ),
    )
    parser.add_argument(
        "--build-report",
        action="store_true",
        help="Сформировать итоговый JSON- и Markdown-отчет главы 5.",
    )
    parser.add_argument(
        "--run-acceptance",
        action="store_true",
        help="Выполнить финальную приемку артефактов главы 5.",
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




def _activate_full_pipeline_options(args: argparse.Namespace) -> None:
    """Включить все этапы расчета при запуске полного CLI-контура."""

    if not args.run_full_pipeline:
        return
    args.validate_inputs = True
    args.normalize_inputs = True
    args.calculate_latent_component = True
    args.calculate_partial_criteria = True
    args.calculate_q_pred = True
    args.estimate_uncertainty = True


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить CLI-каркас главы 5 и вернуть код завершения."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _activate_full_pipeline_options(args)
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
    if (
        args.validate_inputs
        or args.normalize_inputs
        or args.calculate_latent_component
        or args.calculate_partial_criteria
        or args.calculate_q_pred
        or args.estimate_uncertainty
    ):
        loaded_inputs = _load_and_report_inputs(project_root, config)
        if loaded_inputs is None:
            return 1

    normalization_result: PriorFeatureNormalizationResult | None = None
    if args.normalize_inputs:
        normalization_result = _normalize_and_save(project_root, config, loaded_inputs)

    latent_result: LatentQualityComponentResult | None = None
    if args.calculate_latent_component:
        latent_result = _calculate_latent_component_and_save(project_root, config, loaded_inputs)

    partial_result: PartialQualityPredictionResult | None = None
    if args.calculate_partial_criteria:
        partial_result = _calculate_partial_criteria_and_save(
            project_root,
            config,
            loaded_inputs,
            normalization_result,
            latent_result,
        )

    q_pred_result: IntegralQualityPredictionResult | None = None
    if args.calculate_q_pred:
        q_pred_result = _calculate_q_pred_and_save(
            project_root,
            config,
            loaded_inputs,
            normalization_result,
            latent_result,
            partial_result,
        )

    uncertainty_result: PredictionUncertaintyResult | None = None
    if args.estimate_uncertainty:
        uncertainty_result = _estimate_uncertainty_and_save(
            project_root,
            config,
            loaded_inputs,
            normalization_result,
            latent_result,
            partial_result,
            q_pred_result,
        )

    if args.run_full_pipeline:
        _save_pipeline_run_report(
            project_root,
            config,
            normalization_result,
            latent_result,
            partial_result,
            q_pred_result,
            uncertainty_result,
        )

    if args.build_report:
        try:
            _build_final_report(project_root, config)
        except Chapter5ReportBuildError as error:
            print(f"Итоговый отчет главы 5 не сформирован: {error}")
            return 1

    if args.run_acceptance:
        try:
            acceptance_report = _run_acceptance(project_root, config)
        except Chapter5AcceptanceError as error:
            print(f"Финальная приемка главы 5 не выполнена: {error}")
            return 1
        if not acceptance_report.accepted:
            print("Финальная приемка главы 5: не пройдена.")
            return 1

    if not (
        args.calculate_q_pred
        or args.run_full_pipeline
        or args.build_report
        or args.run_acceptance
    ):
        print(
            "Расчет Q_pred не выполнялся: интегральный показатель "
            "будет реализован на следующих этапах."
        )
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
) -> PriorFeatureNormalizationResult:
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
    return result


def _calculate_latent_component_and_save(
    project_root: Path,
    config: Chapter5PredictionConfig,
    loaded_inputs: Chapter5LoadedInputs | None,
) -> LatentQualityComponentResult:
    """Рассчитать латентную компоненту качества и сохранить артефакты."""

    if loaded_inputs is None:
        msg = "Внутренняя ошибка: расчет Q_lat вызван без загруженных входных данных."
        raise RuntimeError(msg)

    calculator = LatentQualityComponentCalculator(config.factor_directions)
    result = calculator.calculate(loaded_inputs.theta_prior)
    latent_path = resolve_project_path(
        project_root,
        config.outputs.latent_quality_component_path,
    )
    report_path = resolve_project_path(
        project_root,
        config.outputs.latent_quality_component_report_path,
    )
    calculator.save_outputs(
        result,
        latent_component_path=latent_path,
        report_path=report_path,
    )
    print("Латентная компонента качества: рассчитана.")
    print(f"Строк латентной компоненты: {result.report.row_count}")
    print(f"Минимальное q_latent: {result.report.q_latent_min:.6f}")
    print(f"Максимальное q_latent: {result.report.q_latent_max:.6f}")
    print(f"Таблица латентной компоненты сохранена: {latent_path}")
    print(f"Отчет латентной компоненты сохранен: {report_path}")
    return result


def _calculate_partial_criteria_and_save(
    project_root: Path,
    config: Chapter5PredictionConfig,
    loaded_inputs: Chapter5LoadedInputs | None,
    normalization_result: PriorFeatureNormalizationResult | None,
    latent_result: LatentQualityComponentResult | None,
) -> PartialQualityPredictionResult:
    """Рассчитать частные прогнозные критерии и сохранить компоненты."""

    if loaded_inputs is None:
        msg = "Внутренняя ошибка: расчет частных критериев вызван без входных данных."
        raise RuntimeError(msg)
    if normalization_result is None:
        normalizer = PriorFeatureNormalizer(config.prior_feature_dictionary)
        normalization_result = normalizer.normalize(loaded_inputs.prior_features)
    if latent_result is None:
        latent_calculator = LatentQualityComponentCalculator(config.factor_directions)
        latent_result = latent_calculator.calculate(loaded_inputs.theta_prior)

    predictor = PartialQualityPredictor(config.feature_weights)
    result = predictor.predict(
        normalization_result.normalized_features,
        latent_result.latent_quality,
    )
    components_path = resolve_project_path(project_root, config.outputs.q_pred_components_path)
    report_path = resolve_project_path(project_root, config.outputs.q_pred_components_report_path)
    predictor.save_outputs(
        result,
        components_path=components_path,
        report_path=report_path,
    )
    print("Частные прогнозные критерии: рассчитаны.")
    print(f"Строк компонентов качества: {result.report.row_count}")
    print(f"Критериев рассчитано: {len(result.report.criteria)}")
    print(f"Таблица компонентов качества сохранена: {components_path}")
    print(f"Отчет компонентов качества сохранен: {report_path}")
    return result


def _calculate_q_pred_and_save(
    project_root: Path,
    config: Chapter5PredictionConfig,
    loaded_inputs: Chapter5LoadedInputs | None,
    normalization_result: PriorFeatureNormalizationResult | None,
    latent_result: LatentQualityComponentResult | None,
    partial_result: PartialQualityPredictionResult | None,
) -> IntegralQualityPredictionResult:
    """Рассчитать интегральный показатель Q_pred и сохранить артефакты."""

    if partial_result is None:
        partial_result = _calculate_partial_criteria_and_save(
            project_root,
            config,
            loaded_inputs,
            normalization_result,
            latent_result,
        )

    predictor = IntegralQualityPredictor(config.quality_weights)
    result = predictor.predict(partial_result.components)
    q_pred_path = resolve_project_path(project_root, config.outputs.q_pred_path)
    report_path = resolve_project_path(project_root, config.outputs.q_pred_report_path)
    predictor.save_outputs(
        result,
        q_pred_path=q_pred_path,
        report_path=report_path,
    )
    print("Интегральный прогнозный показатель Q_pred: рассчитан.")
    print(f"Строк интегрального прогноза: {result.report.row_count}")
    print(f"Минимальное Q_pred: {result.report.q_pred_min:.6f}")
    print(f"Максимальное Q_pred: {result.report.q_pred_max:.6f}")
    print(f"Таблица интегрального прогноза сохранена: {q_pred_path}")
    print(f"Отчет интегрального прогноза сохранен: {report_path}")
    return result


def _estimate_uncertainty_and_save(
    project_root: Path,
    config: Chapter5PredictionConfig,
    loaded_inputs: Chapter5LoadedInputs | None,
    normalization_result: PriorFeatureNormalizationResult | None,
    latent_result: LatentQualityComponentResult | None,
    partial_result: PartialQualityPredictionResult | None,
    q_pred_result: IntegralQualityPredictionResult | None,
) -> PredictionUncertaintyResult:
    """Рассчитать неопределенность прогноза и сохранить интервальные оценки."""

    if loaded_inputs is None:
        msg = "Внутренняя ошибка: расчет неопределенности вызван без входных данных."
        raise RuntimeError(msg)
    if normalization_result is None:
        normalizer = PriorFeatureNormalizer(config.prior_feature_dictionary)
        normalization_result = normalizer.normalize(loaded_inputs.prior_features)
    if q_pred_result is None:
        q_pred_result = _calculate_q_pred_and_save(
            project_root,
            config,
            loaded_inputs,
            normalization_result,
            latent_result,
            partial_result,
        )

    estimator = PredictionUncertaintyEstimator(config.uncertainty)
    result = estimator.estimate(
        q_pred_result.q_pred,
        loaded_inputs.theta_prior,
        normalization_result.normalized_features,
    )
    uncertainty_path = resolve_project_path(
        project_root,
        config.outputs.prediction_uncertainty_path,
    )
    report_path = resolve_project_path(
        project_root,
        config.outputs.prediction_uncertainty_report_path,
    )
    estimator.save_outputs(
        result,
        uncertainty_path=uncertainty_path,
        report_path=report_path,
    )
    print("Неопределенность прогноза Q_pred: рассчитана.")
    print(f"Строк интервального прогноза: {result.report.row_count}")
    print(f"Минимальное uncertainty_score: {result.report.uncertainty_score_min:.6f}")
    print(f"Максимальное uncertainty_score: {result.report.uncertainty_score_max:.6f}")
    print(f"Минимальный радиус интервала: {result.report.interval_radius_min:.6f}")
    print(f"Максимальный радиус интервала: {result.report.interval_radius_max:.6f}")
    print(f"Таблица неопределенности сохранена: {uncertainty_path}")
    print(f"Отчет неопределенности сохранен: {report_path}")
    return result


def _save_pipeline_run_report(
    project_root: Path,
    config: Chapter5PredictionConfig,
    normalization_result: PriorFeatureNormalizationResult | None,
    latent_result: LatentQualityComponentResult | None,
    partial_result: PartialQualityPredictionResult | None,
    q_pred_result: IntegralQualityPredictionResult | None,
    uncertainty_result: PredictionUncertaintyResult | None,
) -> Chapter5PipelineRunReport:
    """Сохранить сводный JSON-отчет полного CLI-контура главы 5."""

    reporter = Chapter5PipelineRunReporter()
    report = reporter.build_report(
        config=config,
        project_root=project_root,
        normalization_result=normalization_result,
        latent_result=latent_result,
        partial_result=partial_result,
        q_pred_result=q_pred_result,
        uncertainty_result=uncertainty_result,
    )
    report_path = resolve_project_path(project_root, config.outputs.pipeline_run_report_path)
    reporter.save_report(report, report_path)
    print("Полный контур главы 5: выполнен.")
    print(f"Отчет полного CLI-запуска сохранен: {report_path}")
    return report


def _build_final_report(
    project_root: Path,
    config: Chapter5PredictionConfig,
) -> Chapter5FinalReport:
    """Сформировать и сохранить итоговый отчет главы 5."""

    builder = Chapter5ReportBuilder()
    report = builder.build_report(config=config, project_root=project_root)
    json_path = resolve_project_path(project_root, config.outputs.report_json_path)
    markdown_path = resolve_project_path(project_root, config.outputs.report_md_path)
    builder.save_outputs(
        report,
        json_report_path=json_path,
        markdown_report_path=markdown_path,
    )
    print("Итоговый отчет главы 5: сформирован.")
    print(f"Строк итогового отчета: {report.row_count}")
    print(f"JSON-отчет главы 5 сохранен: {json_path}")
    print(f"Markdown-отчет главы 5 сохранен: {markdown_path}")
    return report


def _run_acceptance(
    project_root: Path,
    config: Chapter5PredictionConfig,
) -> Chapter5AcceptanceReport:
    """Выполнить финальную приемку артефактов главы 5."""

    validator = Chapter5AcceptanceValidator()
    report = validator.validate(config=config, project_root=project_root)
    json_path = resolve_project_path(project_root, config.outputs.acceptance_report_json_path)
    markdown_path = resolve_project_path(project_root, config.outputs.acceptance_report_md_path)
    validator.save_outputs(
        report,
        json_report_path=json_path,
        markdown_report_path=markdown_path,
    )
    print("Финальная приемка главы 5: выполнена.")
    print(f"Статус приемки: {'пройдена' if report.accepted else 'не пройдена'}.")
    print(f"Проверок выполнено: {len(report.checks)}")
    print(f"JSON-отчет приемки сохранен: {json_path}")
    print(f"Markdown-отчет приемки сохранен: {markdown_path}")
    return report


if __name__ == "__main__":
    raise SystemExit(main())
