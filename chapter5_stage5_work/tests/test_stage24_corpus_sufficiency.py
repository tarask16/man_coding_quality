"""Тесты проверки достаточности расширенного корпуса."""

import csv
import json
from pathlib import Path

import pytest

from manual_coding_sim.lda.corpus_sufficiency import (
    CorpusCoverageRule,
    CorpusSufficiencyAnalyzer,
    CorpusSufficiencyConfig,
    CorpusSufficiencyConfigLoader,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Записать тестовый CSV-файл."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Записать тестовый JSON-файл."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def _rows(count: int = 12) -> list[dict[str, object]]:
    """Вернуть тестовые априорные признаки с управляемым покрытием."""

    complexity_values = ["low", "mid", "high"]
    condition_values = ["normal", "noise", "time_limit"]
    return [
        {
            "run_id": f"r{index:03d}",
            "protocol_id": f"p{index:03d}",
            "scenario_id": f"s{index:03d}",
            "prior_mean_complexity": complexity_values[index % 3],
            "prior_condition_mode": condition_values[index % 3],
            "prior_total_nominal_time": 10 + index,
        }
        for index in range(count)
    ]


def test_sufficiency_analyzer_passes_extended_corpus(tmp_path: Path) -> None:
    """Расширенный корпус должен проходить заданные пороги достаточности."""

    prior_path = tmp_path / "data" / "processed" / "prior_features.csv"
    dictionary_path = tmp_path / "data" / "processed" / "lda" / "dictionary.json"
    reports_dir = tmp_path / "reports" / "chapter4"
    _write_csv(prior_path, _rows(12))
    _write_json(
        dictionary_path,
        {f"token_{index}": {"token_id": index} for index in range(8)},
    )

    result = CorpusSufficiencyAnalyzer(
        CorpusSufficiencyConfig(
            min_documents=10,
            min_unique_scenarios=10,
            min_unique_protocols=10,
            min_vocabulary_tokens=8,
            coverage_rules=(
                CorpusCoverageRule("prior_mean_complexity", 3),
                CorpusCoverageRule("prior_condition_mode", 3),
            ),
        )
    ).analyze_from_files(prior_path, reports_dir, dictionary_path)

    assert result.passed is True
    assert result.document_count == 12
    assert result.vocabulary_token_count == 8
    assert result.report_json_path.exists()
    assert result.report_md_path.exists()


def test_sufficiency_analyzer_fails_small_corpus(tmp_path: Path) -> None:
    """Малый корпус должен давать рекомендации по расширению."""

    prior_path = tmp_path / "data" / "processed" / "prior_features.csv"
    reports_dir = tmp_path / "reports" / "chapter4"
    _write_csv(prior_path, _rows(5))

    result = CorpusSufficiencyAnalyzer(
        CorpusSufficiencyConfig(
            min_documents=10,
            min_unique_scenarios=10,
            min_unique_protocols=10,
            min_vocabulary_tokens=10,
        )
    ).analyze_from_files(prior_path, reports_dir)

    assert result.passed is False
    assert result.document_count == 5
    assert result.recommendations
    assert "Увеличить число документов" in result.recommendations[0]


def test_sufficiency_analyzer_checks_coverage_columns(tmp_path: Path) -> None:
    """Анализатор должен выявлять недостаточное покрытие признаков."""

    prior_path = tmp_path / "prior_features.csv"
    reports_dir = tmp_path / "reports"
    rows = _rows(12)
    for row in rows:
        row["prior_mean_complexity"] = "low"
    _write_csv(prior_path, rows)

    result = CorpusSufficiencyAnalyzer(
        CorpusSufficiencyConfig(
            min_documents=5,
            min_unique_scenarios=5,
            min_unique_protocols=5,
            min_vocabulary_tokens=2,
            coverage_rules=(CorpusCoverageRule("prior_mean_complexity", 3),),
        )
    ).analyze_from_files(prior_path, reports_dir)

    assert result.passed is False
    assert result.coverage_results[0].unique_count == 1
    assert result.coverage_results[0].passed is False


def test_sufficiency_analyzer_reports_missing_coverage_column(tmp_path: Path) -> None:
    """Отсутствующая колонка покрытия должна попадать в рекомендации."""

    prior_path = tmp_path / "prior_features.csv"
    reports_dir = tmp_path / "reports"
    _write_csv(prior_path, _rows(12))

    result = CorpusSufficiencyAnalyzer(
        CorpusSufficiencyConfig(
            min_documents=5,
            min_unique_scenarios=5,
            min_unique_protocols=5,
            min_vocabulary_tokens=2,
            coverage_rules=(CorpusCoverageRule("prior_missing_feature", 2),),
        )
    ).analyze_from_files(prior_path, reports_dir)

    assert result.passed is False
    assert result.coverage_results[0].exists is False
    assert any("prior_missing_feature" in item for item in result.recommendations)


def test_sufficiency_analyzer_respects_overwrite_false(tmp_path: Path) -> None:
    """При overwrite=False существующие отчеты нельзя перезаписывать."""

    prior_path = tmp_path / "prior_features.csv"
    reports_dir = tmp_path / "reports"
    _write_csv(prior_path, _rows(12))
    reports_dir.mkdir(parents=True)
    (reports_dir / "corpus_sufficiency_report.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        CorpusSufficiencyAnalyzer(
            CorpusSufficiencyConfig(
                min_documents=5,
                min_unique_scenarios=5,
                min_unique_protocols=5,
                min_vocabulary_tokens=2,
                overwrite=False,
            )
        ).analyze_from_files(prior_path, reports_dir)


def test_sufficiency_config_loader_reads_corpus_section(tmp_path: Path) -> None:
    """Загрузчик должен читать секцию corpus из YAML-конфига."""

    config_path = tmp_path / "chapter4_extended.yaml"
    config_path.write_text(
        """
corpus:
  min_documents: 120
  min_unique_scenarios: 100
  min_unique_protocols: 90
  min_vocabulary_tokens: 40
  overwrite: false
  coverage_rules:
    - column: prior_mean_complexity
      min_unique_values: 3
""".strip(),
        encoding="utf-8",
    )

    config = CorpusSufficiencyConfigLoader().load(config_path)

    assert config.min_documents == 120
    assert config.min_unique_scenarios == 100
    assert config.min_unique_protocols == 90
    assert config.min_vocabulary_tokens == 40
    assert config.overwrite is False
    assert config.coverage_rules[0].column == "prior_mean_complexity"


def test_sufficiency_config_loader_supports_overwrite_override(tmp_path: Path) -> None:
    """CLI-переопределение overwrite должно иметь приоритет над YAML."""

    config_path = tmp_path / "chapter4_extended.yaml"
    config_path.write_text("corpus:\n  overwrite: false\n", encoding="utf-8")

    config = CorpusSufficiencyConfigLoader().load(
        config_path,
        overwrite_override=True,
    )

    assert config.overwrite is True
