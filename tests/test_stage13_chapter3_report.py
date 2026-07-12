"""Тесты этапа 13: итоговая отчетность реализации главы 3."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from manual_coding_sim.chapter3_report import (
    Chapter3ImplementationReport,
    Chapter3ImplementationReportConfig,
    EXPECTED_DATASET_KEYS,
    EXPECTED_SOURCE_FILES,
    EXPECTED_TEST_FILES,
    build_chapter3_implementation_report,
    collect_dataset_artifact_status,
    collect_file_status,
    render_report_markdown,
)


def _make_config(tmp_path: Path, run_experiment: bool = True) -> Chapter3ImplementationReportConfig:
    """Создать изолированную тестовую конфигурацию итогового отчета."""

    project_root = _prepare_sandbox_project(tmp_path)
    return Chapter3ImplementationReportConfig(
        project_root=project_root,
        output_json=Path("reports/chapter3/chapter3_implementation_report.json"),
        output_markdown=Path("reports/chapter3/chapter3_implementation_report.md"),
        run_experiment=run_experiment,
        experiment_run_count=2,
        random_seed=42,
        scenario_id="A_TEST_FINAL",
    )


def _prepare_sandbox_project(tmp_path: Path) -> Path:
    """Подготовить временный корень, не изменяющий рабочие артефакты проекта."""

    project_root = tmp_path / "sandbox_project"
    for relative_path in (*EXPECTED_SOURCE_FILES, *EXPECTED_TEST_FILES):
        target = project_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    return project_root


def test_report_config_is_valid(tmp_path: Path) -> None:
    """Проверяет корректность конфигурации итогового отчета."""
    config = _make_config(tmp_path)

    config.validate()
    assert config.experiment_run_count == 2
    assert config.scenario_id == "A_TEST_FINAL"


def test_invalid_report_config_is_rejected(tmp_path: Path) -> None:
    """Проверяет отклонение некорректной конфигурации отчета."""
    with pytest.raises(ValueError):
        Chapter3ImplementationReportConfig(
            project_root=Path.cwd(),
            output_json=tmp_path / "r.json",
            experiment_run_count=0,
        ).validate()

    with pytest.raises(ValueError):
        Chapter3ImplementationReportConfig(
            project_root=Path.cwd(),
            output_json=tmp_path / "r.json",
            random_seed=-1,
        ).validate()

    with pytest.raises(ValueError):
        Chapter3ImplementationReportConfig(
            project_root=Path.cwd(),
            output_json=tmp_path / "r.json",
            scenario_id="",
        ).validate()


def test_expected_source_file_list_contains_core_modules() -> None:
    """Проверяет наличие ключевых модулей в списке контроля."""
    assert "src/manual_coding_sim/message_model.py" in EXPECTED_SOURCE_FILES
    assert "src/manual_coding_sim/protocol_simulator.py" in EXPECTED_SOURCE_FILES
    assert "src/manual_coding_sim/experiment_runner.py" in EXPECTED_SOURCE_FILES
    assert "src/manual_coding_sim/chapter3_report.py" in EXPECTED_SOURCE_FILES


def test_expected_test_file_list_contains_all_stages() -> None:
    """Проверяет наличие тестовых файлов этапов 1-13."""
    assert len(EXPECTED_TEST_FILES) == 13
    assert "tests/test_stage1_package_structure.py" in EXPECTED_TEST_FILES
    assert "tests/test_stage13_chapter3_report.py" in EXPECTED_TEST_FILES


def test_collect_file_status_for_source_files() -> None:
    """Проверяет контроль наличия исходных файлов."""
    status = collect_file_status(Path.cwd(), EXPECTED_SOURCE_FILES)

    assert status["src/manual_coding_sim/message_model.py"] is True
    assert status["src/manual_coding_sim/chapter3_report.py"] is True
    assert all(isinstance(value, bool) for value in status.values())


def test_build_report_without_experiment(tmp_path: Path) -> None:
    """Проверяет формирование отчета без запуска эксперимента."""
    report = build_chapter3_implementation_report(
        _make_config(tmp_path, run_experiment=False),
    )

    assert isinstance(report, Chapter3ImplementationReport)
    assert report.json_path.exists()
    assert report.markdown_path.exists()
    assert report.all_required_files_present is True


def test_report_json_contains_scientific_chain(tmp_path: Path) -> None:
    """Проверяет наличие научной цепочки моделирования в JSON."""
    report = build_chapter3_implementation_report(
        _make_config(tmp_path, run_experiment=False),
    )
    payload = json.loads(report.json_path.read_text(encoding="utf-8"))

    assert payload["summary"]["scientific_chain"] == "S, O, U, G, K → M → E_h → C → D_h → M'"
    assert payload["summary"]["source_files_present"] == payload["summary"]["source_files_total"]
    assert payload["summary"]["test_files_present"] == payload["summary"]["test_files_total"]


def test_report_markdown_contains_dissertation_terms(tmp_path: Path) -> None:
    """Проверяет наличие терминов диссертации в Markdown-отчете."""
    report = build_chapter3_implementation_report(
        _make_config(tmp_path, run_experiment=False),
    )
    markdown = report.markdown_path.read_text(encoding="utf-8")

    assert "ручного кодирования" in markdown
    assert "X_prior" in markdown
    assert "q(A)" in markdown
    assert "S, O, U, G, K" in markdown


def test_build_report_with_control_experiment(tmp_path: Path) -> None:
    """Проверяет отчет с малым контрольным экспериментом."""
    report = build_chapter3_implementation_report(_make_config(tmp_path))

    assert report.experiment_summary["run_count"] == 2
    assert report.experiment_summary["scenario_id"] == "A_TEST_FINAL"
    assert report.experiment_summary["reproducibility_ok"] is True


def test_dataset_artifacts_are_present_after_experiment(tmp_path: Path) -> None:
    """Проверяет наличие CSV/JSON-артефактов после эксперимента."""
    report = build_chapter3_implementation_report(_make_config(tmp_path))

    assert set(report.dataset_artifact_status) == set(EXPECTED_DATASET_KEYS)
    assert report.all_dataset_artifacts_present is True


def test_collect_dataset_artifact_status_rejects_empty_summary() -> None:
    """Проверяет отсутствие артефактов при пустой сводке."""
    status = collect_dataset_artifact_status({})

    assert set(status) == set(EXPECTED_DATASET_KEYS)
    assert not any(status.values())


def test_render_report_markdown_uses_status_markers(tmp_path: Path) -> None:
    """Проверяет Markdown-представление статусов."""
    report = build_chapter3_implementation_report(
        _make_config(tmp_path, run_experiment=False),
    )
    payload = json.loads(report.json_path.read_text(encoding="utf-8"))
    markdown = render_report_markdown(payload)

    assert "[OK]" in markdown
    assert "## Состав научных компонентов" in markdown
    assert "## Ограничение области реализации" in markdown


def test_report_properties_reflect_statuses(tmp_path: Path) -> None:
    """Проверяет свойства итогового отчета."""
    report = build_chapter3_implementation_report(_make_config(tmp_path))

    assert report.all_required_files_present is True
    assert report.all_dataset_artifacts_present is True
    assert report.summary["dataset_artifacts_present"] == len(EXPECTED_DATASET_KEYS)
