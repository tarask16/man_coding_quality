"""Тесты CLI-каркаса главы 5."""

from manual_coding_sim.prediction.chapter5_runner import build_arg_parser, main


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
    """Runner этапа 1 должен завершаться успешно и выводить русское сообщение."""

    exit_code = main(["--project-root", "/project"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Каркас программного блока главы 5 успешно загружен" in captured.out
    assert "Расчет Q_pred не выполнялся" in captured.out
