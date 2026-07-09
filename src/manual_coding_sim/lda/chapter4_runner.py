"""Единый runner программного блока главы 4.

Runner последовательно запускает уже реализованные компоненты LDA-модуля:
построение корпуса, подбор числа факторов, обучение ``LDA_prior``, расчет
метрик, анализ устойчивости, интерпретацию тем, диагностические модели и
итоговый отчет. Если любой обязательный этап завершается ошибкой, запуск
прерывается и итоговый отчет не создается.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from manual_coding_sim.lda.chapter4_report import (
    Chapter4LdaReportBuilder,
    Chapter4LdaReportConfig,
    Chapter4LdaReportResult,
)
from manual_coding_sim.lda.config import Chapter4LdaConfig
from manual_coding_sim.lda.corpus_builder import (
    LdaCorpusBuilder,
    LdaCorpusBuilderConfig,
    LdaCorpusBuildResult,
)
from manual_coding_sim.lda.k_selection import (
    LdaKSelectionConfig,
    LdaKSelectionResult,
    LdaKSelector,
)
from manual_coding_sim.lda.lda_diagnostic_model import (
    LdaDiagnosticModel,
    LdaDiagnosticModelConfig,
    LdaDiagnosticTrainingResult,
)
from manual_coding_sim.lda.lda_prior_model import (
    LdaPriorModel,
    LdaPriorModelConfig,
    LdaPriorTrainingResult,
)
from manual_coding_sim.lda.paths import resolve_project_path
from manual_coding_sim.lda.topic_interpreter import (
    LdaTopicInterpretationResult,
    LdaTopicInterpreter,
    LdaTopicInterpreterConfig,
)
from manual_coding_sim.lda.topic_metrics import (
    LdaTopicMetricsConfig,
    LdaTopicMetricsEvaluator,
    LdaTopicMetricsResult,
)
from manual_coding_sim.lda.topic_stability import (
    LdaTopicStabilityAnalyzer,
    LdaTopicStabilityConfig,
    LdaTopicStabilityResult,
)


@dataclass(frozen=True)
class Chapter4RunnerConfig:
    """Параметры единого запуска программного блока главы 4."""

    chapter4: Chapter4LdaConfig = field(default_factory=Chapter4LdaConfig)
    top_n: int = 10
    overwrite: bool = True

    def validate(self) -> None:
        """Проверить корректность параметров runner-а."""

        self.chapter4.validate()
        if self.top_n < 2:
            msg = "top_n должен быть не меньше 2."
            raise ValueError(msg)
        if len(self.chapter4.model.random_seeds) < 2:
            msg = "Для единого запуска главы 4 нужно не менее двух random_seed."
            raise ValueError(msg)


@dataclass(frozen=True)
class Chapter4RunResult:
    """Результат единого запуска главы 4."""

    selected_k: int
    corpus_result: LdaCorpusBuildResult
    k_selection_result: LdaKSelectionResult
    prior_training_result: LdaPriorTrainingResult
    metrics_result: LdaTopicMetricsResult
    stability_result: LdaTopicStabilityResult
    interpretation_result: LdaTopicInterpretationResult
    report_result: Chapter4LdaReportResult
    diag_result: LdaDiagnosticTrainingResult | None = None
    full_result: LdaDiagnosticTrainingResult | None = None


class Chapter4LdaRunner:
    """Запускает полный воспроизводимый сценарий главы 4."""

    def __init__(self, config: Chapter4RunnerConfig | None = None) -> None:
        """Создать runner главы 4."""

        self.config = config or Chapter4RunnerConfig()
        self.config.validate()

    def run(self, project_root: str | Path = ".") -> Chapter4RunResult:
        """Выполнить полный запуск главы 4 относительно корня проекта."""

        root = Path(project_root)
        paths = self._resolve_paths(root)
        chapter_config = self.config.chapter4
        model_config = chapter_config.model
        random_state = model_config.random_seeds[0]

        corpus_result = LdaCorpusBuilder(
            LdaCorpusBuilderConfig(
                tokenization=chapter_config.tokenization,
                overwrite=self.config.overwrite,
            )
        ).build_from_csv(
            prior_features_path=paths["prior_features"],
            output_dir=paths["data_dir"],
        )

        k_selection_result = LdaKSelector(
            LdaKSelectionConfig(
                k_values=model_config.k_values,
                doc_topic_prior=model_config.doc_topic_prior,
                topic_word_prior=model_config.topic_word_prior,
                learning_method=model_config.learning_method,
                max_iter=model_config.max_iter,
                random_state=random_state,
                top_n=self.config.top_n,
                overwrite=self.config.overwrite,
            )
        ).select_from_artifacts(
            corpus_path=corpus_result.corpus_path,
            dictionary_path=corpus_result.dictionary_path,
            metadata_path=corpus_result.metadata_path,
            models_dir=paths["models_dir"],
            reports_dir=paths["reports_dir"],
        )

        selected_k = model_config.selected_k or k_selection_result.recommended_k
        prior_training_result = LdaPriorModel(
            LdaPriorModelConfig(
                n_components=selected_k,
                doc_topic_prior=model_config.doc_topic_prior,
                topic_word_prior=model_config.topic_word_prior,
                learning_method=model_config.learning_method,
                max_iter=model_config.max_iter,
                random_state=random_state,
                overwrite=self.config.overwrite,
            )
        ).fit_from_artifacts(
            corpus_path=corpus_result.corpus_path,
            dictionary_path=corpus_result.dictionary_path,
            metadata_path=corpus_result.metadata_path,
            models_dir=paths["models_dir"],
            reports_dir=paths["reports_dir"],
        )

        metrics_result = LdaTopicMetricsEvaluator(
            LdaTopicMetricsConfig(
                top_n=self.config.top_n,
                overwrite=self.config.overwrite,
            )
        ).evaluate_from_artifacts(
            model_path=prior_training_result.model_path,
            corpus_path=corpus_result.corpus_path,
            dictionary_path=corpus_result.dictionary_path,
            metadata_path=corpus_result.metadata_path,
            reports_dir=paths["reports_dir"],
        )

        stability_result = LdaTopicStabilityAnalyzer(
            LdaTopicStabilityConfig(
                n_components=selected_k,
                random_states=model_config.random_seeds,
                doc_topic_prior=model_config.doc_topic_prior,
                topic_word_prior=model_config.topic_word_prior,
                learning_method=model_config.learning_method,
                max_iter=model_config.max_iter,
                overwrite=self.config.overwrite,
            )
        ).analyze_from_artifacts(
            corpus_path=corpus_result.corpus_path,
            dictionary_path=corpus_result.dictionary_path,
            metadata_path=corpus_result.metadata_path,
            models_dir=paths["models_dir"],
            reports_dir=paths["reports_dir"],
        )

        interpretation_result = LdaTopicInterpreter(
            LdaTopicInterpreterConfig(
                top_n=self.config.top_n,
                model_name="LDA_prior",
                overwrite=self.config.overwrite,
            )
        ).interpret_from_topic_word(
            topic_word_path=prior_training_result.topic_word_path,
            reports_dir=paths["reports_dir"],
        )

        diag_result = None
        full_result = None
        if chapter_config.build_lda_diag:
            diag_result = self._fit_diagnostic_model(
                kind="diag",
                selected_k=selected_k,
                prior_features_path=paths["prior_features"],
                extension_features_path=paths["diagnostic_features"],
                data_dir=paths["data_dir"],
                models_dir=paths["models_dir"],
                reports_dir=paths["reports_dir"],
            )
        if chapter_config.build_lda_full:
            full_result = self._fit_diagnostic_model(
                kind="full",
                selected_k=selected_k,
                prior_features_path=paths["prior_features"],
                extension_features_path=paths["fact_features"],
                data_dir=paths["data_dir"],
                models_dir=paths["models_dir"],
                reports_dir=paths["reports_dir"],
            )

        report_result = Chapter4LdaReportBuilder(
            Chapter4LdaReportConfig(
                overwrite=self.config.overwrite,
                require_diagnostic_artifacts=(
                    chapter_config.build_lda_diag and chapter_config.build_lda_full
                ),
            )
        ).build_from_artifacts(
            data_dir=paths["data_dir"],
            models_dir=paths["models_dir"],
            reports_dir=paths["reports_dir"],
            selected_k=selected_k,
        )

        return Chapter4RunResult(
            selected_k=selected_k,
            corpus_result=corpus_result,
            k_selection_result=k_selection_result,
            prior_training_result=prior_training_result,
            metrics_result=metrics_result,
            stability_result=stability_result,
            interpretation_result=interpretation_result,
            diag_result=diag_result,
            full_result=full_result,
            report_result=report_result,
        )

    def _fit_diagnostic_model(
        self,
        kind: str,
        selected_k: int,
        prior_features_path: Path,
        extension_features_path: Path,
        data_dir: Path,
        models_dir: Path,
        reports_dir: Path,
    ) -> LdaDiagnosticTrainingResult:
        """Обучить одну диагностическую модель в изолированном режиме."""

        chapter_config = self.config.chapter4
        model_config = chapter_config.model
        return LdaDiagnosticModel(
            LdaDiagnosticModelConfig(
                diagnostic_kind=kind,
                n_components=selected_k,
                tokenization=chapter_config.tokenization,
                doc_topic_prior=model_config.doc_topic_prior,
                topic_word_prior=model_config.topic_word_prior,
                learning_method=model_config.learning_method,
                max_iter=model_config.max_iter,
                random_state=model_config.random_seeds[0],
                overwrite=self.config.overwrite,
            )
        ).fit_from_csv(
            prior_features_path=prior_features_path,
            extension_features_path=extension_features_path,
            data_dir=data_dir,
            models_dir=models_dir,
            reports_dir=reports_dir,
        )

    def _resolve_paths(self, project_root: Path) -> dict[str, Path]:
        """Разрешить относительные пути конфигурации относительно корня проекта."""

        chapter_config = self.config.chapter4
        return {
            "prior_features": resolve_project_path(
                project_root,
                chapter_config.inputs.prior_features_path,
            ),
            "diagnostic_features": resolve_project_path(
                project_root,
                chapter_config.inputs.diagnostic_features_path,
            ),
            "fact_features": resolve_project_path(
                project_root,
                chapter_config.inputs.fact_features_path,
            ),
            "data_dir": resolve_project_path(project_root, chapter_config.outputs.data_dir),
            "models_dir": resolve_project_path(project_root, chapter_config.outputs.models_dir),
            "reports_dir": resolve_project_path(project_root, chapter_config.outputs.reports_dir),
        }
