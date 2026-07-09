"""Тесты генерации расширенного корпуса главы 3."""

import csv
import json
from pathlib import Path

import pytest

from manual_coding_sim.experiments.extended_corpus_plan import ExtendedCorpusPlanConfig
from manual_coding_sim.experiments.extended_corpus_runner import (
    ExtendedCorpusOutputPaths,
    ExtendedCorpusRunner,
    ExtendedCorpusRunnerConfig,
)
from manual_coding_sim.lda.config import LdaTokenizationConfig
from manual_coding_sim.lda.corpus_builder import LdaCorpusBuilder, LdaCorpusBuilderConfig
from manual_coding_sim.lda.corpus_sufficiency import (
    CorpusCoverageRule,
    CorpusSufficiencyAnalyzer,
    CorpusSufficiencyConfig,
)
from manual_coding_sim.lda.leakage_guard import LeakageGuard


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Прочитать CSV-файл в список строк."""

    with path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def _outputs(tmp_path: Path) -> ExtendedCorpusOutputPaths:
    """Сформировать тестовые выходные пути."""

    return ExtendedCorpusOutputPaths(
        data_dir=Path("data/processed"),
        reports_dir=Path("reports/chapter3"),
        protocols_path=Path("data/processed/protocols.csv"),
        prior_features_path=Path("data/processed/prior_features.csv"),
        diagnostic_features_path=Path("data/processed/diagnostic_features.csv"),
        fact_features_path=Path("data/processed/fact_features.csv"),
        quality_targets_path=Path("data/processed/quality_targets.csv"),
        all_features_path=Path("data/processed/all_features.csv"),
        summary_json_path=Path("reports/chapter3/extended_corpus_summary.json"),
        summary_md_path=Path("reports/chapter3/extended_corpus_summary.md"),
    )


def _runner(tmp_path: Path, document_count: int = 120) -> ExtendedCorpusRunner:
    """Создать runner с временными выходными путями."""

    return ExtendedCorpusRunner(
        ExtendedCorpusRunnerConfig(
            plan=ExtendedCorpusPlanConfig(
                document_count=document_count,
                random_seed=20260709,
            ),
            outputs=_outputs(tmp_path),
            overwrite=True,
        )
    )


def test_extended_corpus_runner_creates_chapter3_artifacts(tmp_path: Path) -> None:
    """Runner должен создать стандартные CSV-артефакты главы 3."""

    result = _runner(tmp_path, 120).run(project_root=tmp_path)

    assert result.document_count == 120
    assert result.unique_scenario_count == 120
    assert result.unique_protocol_count == 120
    assert result.protocols_path.exists()
    assert result.prior_features_path.exists()
    assert result.diagnostic_features_path.exists()
    assert result.fact_features_path.exists()
    assert result.quality_targets_path.exists()
    assert result.all_features_path.exists()
    assert result.summary_json_path.exists()
    assert result.summary_md_path.exists()


def test_generated_prior_features_pass_leakage_guard(tmp_path: Path) -> None:
    """Априорный файл расширенного корпуса не должен содержать утечки результата."""

    result = _runner(tmp_path, 120).run(project_root=tmp_path)
    rows = _read_csv(result.prior_features_path)

    LeakageGuard().validate_prior_input(
        source_paths=[result.prior_features_path],
        columns=rows[0].keys(),
    )
    assert "q_acc" not in rows[0]
    assert "integral_quality" not in rows[0]
    assert not any(column.startswith("fact_") for column in rows[0])


def test_generated_corpus_passes_sufficiency_after_dictionary_build(tmp_path: Path) -> None:
    """Сгенерированный корпус должен проходить проверку достаточности главы 4."""

    result = _runner(tmp_path, 120).run(project_root=tmp_path)
    lda_data_dir = tmp_path / "data" / "processed" / "lda"
    corpus_result = LdaCorpusBuilder(
        LdaCorpusBuilderConfig(
            tokenization=LdaTokenizationConfig(
                df_min=2,
                df_max_ratio=0.95,
                numeric_bins=3,
            ),
            overwrite=True,
        )
    ).build_from_csv(result.prior_features_path, lda_data_dir)

    sufficiency = CorpusSufficiencyAnalyzer(
        CorpusSufficiencyConfig(
            min_documents=100,
            min_unique_scenarios=100,
            min_unique_protocols=100,
            min_vocabulary_tokens=30,
            coverage_rules=(
                CorpusCoverageRule("prior_mean_complexity", 3),
                CorpusCoverageRule("prior_mean_message_criticality", 3),
                CorpusCoverageRule("prior_operator_total_estimated_time", 3),
                CorpusCoverageRule("prior_condition_total_adjusted_time", 3),
            ),
        )
    ).analyze_from_files(
        result.prior_features_path,
        tmp_path / "reports" / "chapter4",
        corpus_result.dictionary_path,
    )

    assert sufficiency.passed is True
    assert sufficiency.document_count == 120
    assert sufficiency.vocabulary_token_count >= 30


def test_quality_targets_are_separated_from_prior_features(tmp_path: Path) -> None:
    """Целевые показатели должны храниться отдельно от априорных признаков."""

    result = _runner(tmp_path, 30).run(project_root=tmp_path)
    prior_rows = _read_csv(result.prior_features_path)
    target_rows = _read_csv(result.quality_targets_path)

    assert len(prior_rows) == len(target_rows) == 30
    assert "integral_quality" not in prior_rows[0]
    assert "integral_quality" in target_rows[0]
    assert "q_acc" in target_rows[0]


def test_extended_corpus_runner_respects_overwrite_false(tmp_path: Path) -> None:
    """Runner не должен перезаписывать файлы при overwrite=False."""

    _runner(tmp_path, 10).run(project_root=tmp_path)
    runner = ExtendedCorpusRunner(
        ExtendedCorpusRunnerConfig(
            plan=ExtendedCorpusPlanConfig(document_count=10),
            outputs=_outputs(tmp_path),
            overwrite=False,
        )
    )

    with pytest.raises(FileExistsError):
        runner.run(project_root=tmp_path)


def test_extended_corpus_runner_loads_yaml_config(tmp_path: Path) -> None:
    """Runner должен создаваться из YAML-конфигурации."""

    config_path = tmp_path / "chapter3_extended_corpus.yaml"
    config_path.write_text(
        """
extended_corpus:
  overwrite: true
  plan:
    document_count: 25
    random_seed: 123
    protocols_per_scenario: 1
  output:
    data_dir: data/processed
    reports_dir: reports/chapter3
""".strip(),
        encoding="utf-8",
    )

    runner = ExtendedCorpusRunner.from_yaml(config_path)
    result = runner.run(project_root=tmp_path)

    assert result.document_count == 25
    assert result.prior_features_path.exists()


def test_extended_corpus_summary_contains_methodological_note(tmp_path: Path) -> None:
    """Машинный отчет должен фиксировать способ формирования корпуса."""

    result = _runner(tmp_path, 20).run(project_root=tmp_path)
    payload = json.loads(result.summary_json_path.read_text(encoding="utf-8"))

    assert payload["document_count"] == 20
    assert "не являются копиями" in payload["methodological_note"]
    assert payload["coverage"]["message_complexity_values"]
