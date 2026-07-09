"""Тесты CLI генерации расширенного корпуса."""

from pathlib import Path

from manual_coding_sim.experiments.extended_corpus_cli import build_arg_parser, main


def test_extended_corpus_cli_parser_reads_arguments() -> None:
    """CLI-парсер должен читать путь к конфигу и корень проекта."""

    parser = build_arg_parser()
    args = parser.parse_args(["--config", "cfg.yaml", "--project-root", "project"])

    assert args.config == "cfg.yaml"
    assert args.project_root == "project"


def test_extended_corpus_cli_generates_artifacts(tmp_path: Path) -> None:
    """CLI должен создавать расширенный корпус по YAML-конфигурации."""

    config_path = tmp_path / "configs" / "chapter3_extended_corpus.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
extended_corpus:
  overwrite: true
  plan:
    document_count: 30
    random_seed: 42
    protocols_per_scenario: 1
  output:
    data_dir: data/processed
    reports_dir: reports/chapter3
""".strip(),
        encoding="utf-8",
    )

    exit_code = main(["--config", str(config_path), "--project-root", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "data" / "processed" / "prior_features.csv").exists()
    assert (tmp_path / "reports" / "chapter3" / "extended_corpus_summary.md").exists()


def test_extended_corpus_cli_returns_error_for_missing_config(tmp_path: Path) -> None:
    """При отсутствующем конфиге CLI должен вернуть код ошибки."""

    exit_code = main([
        "--config",
        str(tmp_path / "missing.yaml"),
        "--project-root",
        str(tmp_path),
    ])

    assert exit_code == 1
