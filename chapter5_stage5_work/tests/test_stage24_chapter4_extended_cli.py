"""Тесты расширенного CLI-запуска главы 4."""

import csv
import json
from pathlib import Path

from manual_coding_sim.lda.chapter4_extended_cli import (
    build_extended_arg_parser,
    main,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Записать тестовый CSV-файл."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, object]:
    """Прочитать JSON-файл."""

    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _prior_rows(count: int = 8) -> list[dict[str, object]]:
    """Вернуть тестовые априорные признаки."""

    return [
        {
            "run_id": f"r{index:03d}",
            "protocol_id": f"p{index:03d}",
            "scenario_id": f"s{index:03d}",
            "has_control": int(index % 2 == 0),
            "message_length": 10 + index,
            "procedure_type": "simple" if index % 2 == 0 else "complex",
            "operator_skill": ["low", "medium", "high"][index % 3],
        }
        for index in range(count)
    ]


def _diagnostic_rows(count: int = 8) -> list[dict[str, object]]:
    """Вернуть тестовые диагностические признаки."""

    return [
        {
            "run_id": f"r{index:03d}",
            "protocol_id": f"p{index:03d}",
            "scenario_id": f"s{index:03d}",
            "control_complexity": "low" if index % 2 == 0 else "high",
            "operator_stress": index / 10,
        }
        for index in range(count)
    ]


def _fact_rows(count: int = 8) -> list[dict[str, object]]:
    """Вернуть тестовые фактические признаки."""

    return [
        {
            "run_id": f"r{index:03d}",
            "protocol_id": f"p{index:03d}",
            "scenario_id": f"s{index:03d}",
            "errors_total": index % 4,
            "time_overrun": int(index % 3 == 0),
        }
        for index in range(count)
    ]


def _write_config(path: Path, corpus_min_documents: int = 6) -> None:
    """Записать быстрый YAML-конфиг расширенного CLI."""

    path.write_text(
        f"""
input:
  prior_features: data/processed/prior_features.csv
  diagnostic_features: data/processed/diagnostic_features.csv
  fact_features: data/processed/fact_features.csv
output:
  data_dir: data/processed/lda
  models_dir: models/lda
  reports_dir: reports/chapter4
tokenization:
  df_min: 1
  df_max_ratio: 1.0
  numeric_strategy: quantile
  numeric_bins: 3
lda:
  k_values: [2, 3]
  selected_k: null
  learning_method: batch
  max_iter: 2
  random_seeds: [11, 42]
diagnostics:
  build_lda_diag: true
  build_lda_full: true
runner:
  top_n: 3
  overwrite: true
corpus:
  min_documents: {corpus_min_documents}
  min_unique_scenarios: {corpus_min_documents}
  min_unique_protocols: {corpus_min_documents}
  min_vocabulary_tokens: 3
  overwrite: true
  coverage_rules:
    - column: procedure_type
      min_unique_values: 2
    - column: operator_skill
      min_unique_values: 3
""".strip(),
        encoding="utf-8",
    )


def _prepare_project(tmp_path: Path, count: int = 8, corpus_min_documents: int = 6) -> Path:
    """Создать тестовый проект для расширенного CLI."""

    project_root = tmp_path / "project"
    processed_dir = project_root / "data" / "processed"
    _write_csv(processed_dir / "prior_features.csv", _prior_rows(count))
    _write_csv(processed_dir / "diagnostic_features.csv", _diagnostic_rows(count))
    _write_csv(processed_dir / "fact_features.csv", _fact_rows(count))
    config_path = tmp_path / "chapter4_extended.yaml"
    _write_config(config_path, corpus_min_documents=corpus_min_documents)
    return config_path


def test_extended_arg_parser_contains_corpus_flag() -> None:
    """Парсер должен поддерживать разрешение отладочного запуска на малом корпусе."""

    parser = build_extended_arg_parser()
    args = parser.parse_args(
        [
            "--config",
            "configs/chapter4_extended_corpus.yaml",
            "--allow-insufficient-corpus",
        ]
    )

    assert args.allow_insufficient_corpus is True


def test_extended_cli_runs_when_corpus_is_sufficient(tmp_path: Path) -> None:
    """Расширенный CLI должен запускать главу 4 при достаточном корпусе."""

    config_path = _prepare_project(tmp_path, count=8, corpus_min_documents=6)
    project_root = tmp_path / "project"

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--project-root",
            str(project_root),
            "--selected-k",
            "2",
        ]
    )

    report_path = project_root / "reports" / "chapter4" / "chapter4_lda_report.json"
    sufficiency_path = (
        project_root / "reports" / "chapter4" / "corpus_sufficiency_report.json"
    )
    assert exit_code == 0
    assert report_path.exists()
    assert sufficiency_path.exists()
    assert _read_json(sufficiency_path)["passed"] is True


def test_extended_cli_stops_small_corpus_before_lda_run(tmp_path: Path) -> None:
    """При малом корпусе CLI должен остановиться до запуска главы 4."""

    config_path = _prepare_project(tmp_path, count=5, corpus_min_documents=10)
    project_root = tmp_path / "project"

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--project-root",
            str(project_root),
        ]
    )

    assert exit_code == 2
    assert (
        project_root / "reports" / "chapter4" / "corpus_sufficiency_report.json"
    ).exists()
    assert not (
        project_root / "reports" / "chapter4" / "chapter4_lda_report.json"
    ).exists()


def test_extended_cli_can_allow_debug_run_on_small_corpus(tmp_path: Path) -> None:
    """Флаг allow-insufficient-corpus должен разрешать отладочный запуск."""

    config_path = _prepare_project(tmp_path, count=6, corpus_min_documents=20)
    project_root = tmp_path / "project"

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--project-root",
            str(project_root),
            "--selected-k",
            "2",
            "--allow-insufficient-corpus",
        ]
    )

    assert exit_code == 0
    assert (
        project_root / "reports" / "chapter4" / "chapter4_lda_report.json"
    ).exists()
    assert _read_json(
        project_root / "reports" / "chapter4" / "corpus_sufficiency_report.json"
    )["passed"] is False


def test_extended_cli_returns_error_for_missing_inputs(tmp_path: Path) -> None:
    """Отсутствующие входные CSV должны давать код ошибки."""

    config_path = tmp_path / "chapter4_extended.yaml"
    _write_config(config_path, corpus_min_documents=5)

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--project-root",
            str(tmp_path / "missing_project"),
        ]
    )

    assert exit_code == 1
