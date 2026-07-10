"""Тесты CLI-каркаса главы 5."""

from pathlib import Path

from manual_coding_sim.prediction.chapter5_runner import build_arg_parser, main


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_chapter5_runner_arg_parser() -> None:
    """CLI-парсер должен принимать путь к проекту и конфигурации."""

    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--project-root",
            "/project",
            "--config",
            "configs/chapter5.yaml",
        ]
    )

    assert args.project_root == "/project"
    assert args.config == "configs/chapter5.yaml"


def test_chapter5_runner_outputs_russian_message(capsys) -> None:
    """Runner этапа 2 должен завершаться успешно и выводить русское сообщение."""

    exit_code = main(["--project-root", str(PROJECT_ROOT)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Каркас программного блока главы 5 успешно загружен" in captured.out
    assert "Конфигурация главы 5 успешно проверена" in captured.out
    assert "Расчет Q_pred не выполнялся" in captured.out


def test_chapter5_runner_normalizes_inputs(capsys) -> None:
    """Runner этапа 5 должен сохранять нормированные априорные признаки."""

    exit_code = main(
        [
            "--project-root",
            str(PROJECT_ROOT),
            "--config",
            "configs/chapter5.yaml",
            "--normalize-inputs",
        ]
    )

    captured = capsys.readouterr()
    normalized_path = PROJECT_ROOT / "reports/chapter5/normalized_prior_features.csv"
    report_path = PROJECT_ROOT / "reports/chapter5/normalization_report.json"
    assert exit_code == 0
    assert "Нормировка априорных признаков: выполнена" in captured.out
    assert normalized_path.exists()
    assert report_path.exists()
