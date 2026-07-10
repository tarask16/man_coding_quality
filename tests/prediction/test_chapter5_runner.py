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


def test_chapter5_runner_calculates_latent_component(capsys) -> None:
    """Runner этапа 6 должен сохранять латентную компоненту качества."""

    exit_code = main(
        [
            "--project-root",
            str(PROJECT_ROOT),
            "--config",
            "configs/chapter5.yaml",
            "--calculate-latent-component",
        ]
    )

    captured = capsys.readouterr()
    latent_path = PROJECT_ROOT / "reports/chapter5/latent_quality_component.csv"
    report_path = PROJECT_ROOT / "reports/chapter5/latent_quality_component_report.json"
    assert exit_code == 0
    assert "Латентная компонента качества: рассчитана" in captured.out
    assert latent_path.exists()
    assert report_path.exists()


def test_chapter5_runner_calculates_partial_criteria(capsys) -> None:
    """Runner этапа 7 должен сохранять частные прогнозные критерии."""

    exit_code = main(
        [
            "--project-root",
            str(PROJECT_ROOT),
            "--config",
            "configs/chapter5.yaml",
            "--calculate-partial-criteria",
        ]
    )

    captured = capsys.readouterr()
    components_path = PROJECT_ROOT / "reports/chapter5/q_pred_components.csv"
    report_path = PROJECT_ROOT / "reports/chapter5/q_pred_components_report.json"
    assert exit_code == 0
    assert "Частные прогнозные критерии: рассчитаны" in captured.out
    assert components_path.exists()
    assert report_path.exists()


def test_chapter5_runner_calculates_q_pred(capsys) -> None:
    """Runner этапа 8 должен сохранять интегральный прогноз Q_pred."""

    exit_code = main(
        [
            "--project-root",
            str(PROJECT_ROOT),
            "--config",
            "configs/chapter5.yaml",
            "--calculate-q-pred",
        ]
    )

    captured = capsys.readouterr()
    q_pred_path = PROJECT_ROOT / "reports/chapter5/q_pred.csv"
    report_path = PROJECT_ROOT / "reports/chapter5/q_pred_report.json"
    assert exit_code == 0
    assert "Интегральный прогнозный показатель Q_pred: рассчитан" in captured.out
    assert q_pred_path.exists()
    assert report_path.exists()
