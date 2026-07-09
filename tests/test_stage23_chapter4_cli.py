"""Тесты CLI-запуска главы 4."""

import csv
import json
from pathlib import Path

from manual_coding_sim.lda.chapter4_cli import build_arg_parser, main


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


def _prior_rows() -> list[dict[str, object]]:
    """Вернуть тестовые априорные признаки."""

    return [
        {
            "run_id": "r001",
            "protocol_id": "p001",
            "scenario_id": "s001",
            "has_control": 1,
            "message_length": 10,
            "procedure_type": "simple",
            "operator_skill": "high",
        },
        {
            "run_id": "r002",
            "protocol_id": "p002",
            "scenario_id": "s002",
            "has_control": 1,
            "message_length": 12,
            "procedure_type": "simple",
            "operator_skill": "high",
        },
        {
            "run_id": "r003",
            "protocol_id": "p003",
            "scenario_id": "s003",
            "has_control": 0,
            "message_length": 40,
            "procedure_type": "complex",
            "operator_skill": "low",
        },
        {
            "run_id": "r004",
            "protocol_id": "p004",
            "scenario_id": "s004",
            "has_control": 0,
            "message_length": 45,
            "procedure_type": "complex",
            "operator_skill": "low",
        },
        {
            "run_id": "r005",
            "protocol_id": "p005",
            "scenario_id": "s005",
            "has_control": 1,
            "message_length": 14,
            "procedure_type": "simple",
            "operator_skill": "medium",
        },
        {
            "run_id": "r006",
            "protocol_id": "p006",
            "scenario_id": "s006",
            "has_control": 0,
            "message_length": 50,
            "procedure_type": "complex",
            "operator_skill": "medium",
        },
    ]


def _write_config(path: Path) -> None:
    """Записать быстрый YAML-конфиг для CLI-тестов."""

    path.write_text(
        """
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
  doc_topic_prior: null
  topic_word_prior: null
  learning_method: batch
  max_iter: 2
  random_seeds: [11, 42]
diagnostics:
  build_lda_diag: false
  build_lda_full: false
runner:
  top_n: 3
  overwrite: true
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _prepare_project(tmp_path: Path) -> Path:
    """Создать минимальный проект для CLI-запуска."""

    project_root = tmp_path / "project"
    _write_csv(project_root / "data" / "processed" / "prior_features.csv", _prior_rows())
    config_path = tmp_path / "chapter4_lda.yaml"
    _write_config(config_path)
    return config_path


def test_cli_arg_parser_contains_expected_arguments() -> None:
    """CLI-парсер должен поддерживать обязательные аргументы задачи."""

    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--config",
            "configs/chapter4_lda.yaml",
            "--project-root",
            ".",
            "--overwrite",
            "--skip-diagnostic",
            "--selected-k",
            "2",
        ]
    )

    assert args.config == "configs/chapter4_lda.yaml"
    assert args.project_root == "."
    assert args.overwrite is True
    assert args.skip_diagnostic is True
    assert args.selected_k == 2


def test_cli_runs_chapter4_pipeline_successfully(tmp_path: Path, capsys) -> None:
    """CLI должен запускать полный pipeline и возвращать код 0."""

    config_path = _prepare_project(tmp_path)
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

    captured = capsys.readouterr()
    report_path = project_root / "reports" / "chapter4" / "chapter4_lda_report.json"
    payload = _read_json(report_path)
    assert exit_code == 0
    assert "LDA-pipeline выполнен успешно" in captured.out
    assert payload["status"] == "completed"
    assert payload["selected_k"] == 2
    assert (project_root / "reports" / "chapter4" / "theta_prior.csv").exists()


def test_cli_returns_error_code_when_required_input_missing(
    tmp_path: Path,
    capsys,
) -> None:
    """CLI должен вернуть код 1 при отсутствии обязательного входного файла."""

    config_path = tmp_path / "chapter4_lda.yaml"
    _write_config(config_path)
    project_root = tmp_path / "project"

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--project-root",
            str(project_root),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Ошибка запуска главы 4" in captured.err
    assert not (project_root / "reports" / "chapter4" / "chapter4_lda_report.json").exists()


def test_cli_skip_diagnostic_overrides_yaml(tmp_path: Path) -> None:
    """Флаг skip-diagnostic должен отключать диагностические модели."""

    config_path = _prepare_project(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "build_lda_diag: false\n  build_lda_full: false",
            "build_lda_diag: true\n  build_lda_full: true",
        ),
        encoding="utf-8",
    )
    project_root = tmp_path / "project"

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--project-root",
            str(project_root),
            "--skip-diagnostic",
            "--selected-k",
            "2",
        ]
    )

    report_path = project_root / "reports" / "chapter4" / "chapter4_lda_report.json"
    payload = _read_json(report_path)
    assert exit_code == 0
    assert "lda_diagnostic_models" not in payload["completed_steps"]
