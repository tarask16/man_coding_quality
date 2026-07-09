"""
Создание модуля итоговой отчетности программной реализации главы 3.

Скрипт относится к этапу 13 программной реализации главы 3 диссертации.
Он создает модуль, который фиксирует состав реализованных компонентов,
проверяет наличие исходных файлов, тестов и экспериментальных артефактов,
а также формирует итоговый JSON- и Markdown-отчет по исследовательскому
симулятору процессов ручного кодирования.

На этом этапе не выполняется LDA-моделирование и не строится прогнозная
априорная оценка качества. Задача этапа — зафиксировать завершенность
программной реализации главы 3 и получить отчетные материалы для текста
диссертации.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import textwrap


ROOT = Path.cwd()
SRC_DIR = ROOT / "src" / "manual_coding_sim"
TESTS_DIR = ROOT / "tests"
REPORTS_DIR = ROOT / "reports" / "chapter3"


def write_text_file(path: Path, content: str) -> None:
    """Записывает текстовый файл в кодировке UTF-8 без лишних отступов."""
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_content = textwrap.dedent(content).lstrip("\n")
    path.write_text(normalized_content, encoding="utf-8")


def append_text_once(path: Path, marker: str, content: str) -> None:
    """Добавляет текстовый блок в файл только один раз."""
    current_text = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in current_text:
        return
    normalized_content = textwrap.dedent(content).strip("\n")
    path.write_text(current_text.rstrip() + "\n\n" + normalized_content + "\n", encoding="utf-8")


def check_python_syntax(path: Path) -> dict[str, str]:
    """Проверяет синтаксическую корректность Python-файла."""
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        return {"status": "OK", "message": "Синтаксис корректен"}
    except SyntaxError as error:
        return {"status": "ERROR", "message": str(error)}


def main() -> None:
    """Создает модуль итоговой отчетности, тесты и отчет этапа 13."""
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    write_text_file(
        SRC_DIR / "chapter3_report.py",
        r'''
        """Итоговая отчетность программной реализации главы 3.

        Модуль фиксирует состав исследовательского симулятора процессов
        ручного кодирования, проверяет наличие исходных файлов, тестов,
        табличных артефактов и отчета воспроизводимого эксперимента.

        Отчет используется для подтверждения того, что глава 3 имеет
        программную реализацию, способную воспроизводимо формировать данные
        для адаптированной LDA-модели, метода априорной оценки качества и
        последующей проверки достоверности прогноза.
        """

        from __future__ import annotations

        import json
        from dataclasses import dataclass
        from pathlib import Path
        from typing import Any

        from manual_coding_sim.experiment_runner import (
            ExperimentRunner,
            ExperimentRunnerConfig,
        )


        EXPECTED_SOURCE_FILES: tuple[str, ...] = (
            "src/manual_coding_sim/__init__.py",
            "src/manual_coding_sim/config.py",
            "src/manual_coding_sim/types.py",
            "src/manual_coding_sim/message_model.py",
            "src/manual_coding_sim/procedure_model.py",
            "src/manual_coding_sim/operator_model.py",
            "src/manual_coding_sim/condition_model.py",
            "src/manual_coding_sim/error_model.py",
            "src/manual_coding_sim/control_model.py",
            "src/manual_coding_sim/protocol_simulator.py",
            "src/manual_coding_sim/feature_extractor.py",
            "src/manual_coding_sim/quality_calculator.py",
            "src/manual_coding_sim/dataset_builder.py",
            "src/manual_coding_sim/experiment_runner.py",
            "src/manual_coding_sim/chapter3_report.py",
        )

        EXPECTED_TEST_FILES: tuple[str, ...] = (
            "tests/test_stage1_package_structure.py",
            "tests/test_stage2_message_model.py",
            "tests/test_stage3_procedure_model.py",
            "tests/test_stage4_operator_model.py",
            "tests/test_stage5_condition_model.py",
            "tests/test_stage6_error_model.py",
            "tests/test_stage7_control_model.py",
            "tests/test_stage8_protocol_simulator.py",
            "tests/test_stage9_feature_extractor.py",
            "tests/test_stage10_quality_calculator.py",
            "tests/test_stage11_dataset_builder.py",
            "tests/test_stage12_experiment_runner.py",
            "tests/test_stage13_chapter3_report.py",
        )

        EXPECTED_DATASET_KEYS: tuple[str, ...] = (
            "protocols",
            "prior_features",
            "fact_features",
            "diagnostic_features",
            "quality_targets",
            "all_features",
            "summary",
        )

        SCIENTIFIC_COMPONENTS: tuple[dict[str, str], ...] = (
            {
                "component": "G",
                "module": "message_model.py",
                "meaning": "модель класса исходных сообщений M",
            },
            {
                "component": "S",
                "module": "procedure_model.py",
                "meaning": "модель средства ручного кодирования",
            },
            {
                "component": "O",
                "module": "operator_model.py",
                "meaning": "модель оператора ручного кодирования",
            },
            {
                "component": "U",
                "module": "condition_model.py",
                "meaning": "модель условий применения",
            },
            {
                "component": "ErrorModel",
                "module": "error_model.py",
                "meaning": "вероятностная модель ошибок",
            },
            {
                "component": "K",
                "module": "control_model.py",
                "meaning": "модель контрольных процедур",
            },
            {
                "component": "ProtocolSimulator",
                "module": "protocol_simulator.py",
                "meaning": "интегральный симулятор протоколов",
            },
            {
                "component": "FeatureExtractor",
                "module": "feature_extractor.py",
                "meaning": "разделение X_prior, X_fact и X_diag",
            },
            {
                "component": "QualityCalculator",
                "module": "quality_calculator.py",
                "meaning": "расчет q(A)",
            },
            {
                "component": "DatasetBuilder",
                "module": "dataset_builder.py",
                "meaning": "формирование табличного датасета",
            },
            {
                "component": "ExperimentRunner",
                "module": "experiment_runner.py",
                "meaning": "воспроизводимый запуск вычислительного эксперимента",
            },
        )


        @dataclass(frozen=True)
        class Chapter3ImplementationReportConfig:
            """Конфигурация формирования итогового отчета главы 3."""

            project_root: Path | str = Path.cwd()
            output_json: Path | str = Path("reports") / "chapter3" / "chapter3_implementation_report.json"
            output_markdown: Path | str = Path("reports") / "chapter3" / "chapter3_implementation_report.md"
            run_experiment: bool = True
            experiment_run_count: int = 5
            random_seed: int = 42
            scenario_id: str = "A_001"
            overwrite: bool = True

            def validate(self) -> None:
                """Проверяет корректность параметров итогового отчета."""
                root = Path(self.project_root)
                if not root.exists():
                    raise FileNotFoundError(f"Корень проекта не найден: {root}")

                if self.experiment_run_count <= 0:
                    raise ValueError("experiment_run_count должен быть положительным.")

                if self.random_seed < 0:
                    raise ValueError("random_seed не должен быть отрицательным.")

                if not self.scenario_id:
                    raise ValueError("scenario_id не должен быть пустым.")


        @dataclass(frozen=True)
        class Chapter3ImplementationReport:
            """Итоговый отчет о программной реализации главы 3."""

            summary: dict[str, Any]
            source_file_status: dict[str, bool]
            test_file_status: dict[str, bool]
            dataset_artifact_status: dict[str, bool]
            experiment_summary: dict[str, Any]
            json_path: Path
            markdown_path: Path

            @property
            def all_required_files_present(self) -> bool:
                """Возвращает признак наличия всех обязательных файлов."""
                return all(self.source_file_status.values()) and all(
                    self.test_file_status.values(),
                )

            @property
            def all_dataset_artifacts_present(self) -> bool:
                """Возвращает признак наличия всех артефактов датасета."""
                return all(self.dataset_artifact_status.values())


        def build_chapter3_implementation_report(
            config: Chapter3ImplementationReportConfig | None = None,
        ) -> Chapter3ImplementationReport:
            """Формирует итоговый отчет по программной реализации главы 3."""
            report_config = config or Chapter3ImplementationReportConfig()
            report_config.validate()
            project_root = Path(report_config.project_root)

            source_status = collect_file_status(project_root, EXPECTED_SOURCE_FILES)
            test_status = collect_file_status(project_root, EXPECTED_TEST_FILES)
            experiment_summary: dict[str, Any] = {}
            dataset_status: dict[str, bool] = {
                key: False for key in EXPECTED_DATASET_KEYS
            }

            if report_config.run_experiment:
                experiment_summary = _run_control_experiment(
                    project_root=project_root,
                    report_config=report_config,
                )
                dataset_status = collect_dataset_artifact_status(experiment_summary)

            summary = {
                "title": "Итоговый отчет программной реализации главы 3",
                "scientific_chain": "S, O, U, G, K → M → E_h → C → D_h → M'",
                "implemented_scope": (
                    "компьютерное моделирование процессов ручного кодирования, "
                    "разделение признаков и формирование датасета"
                ),
                "not_implemented_in_chapter3": [
                    "адаптированная LDA-модель латентных факторов качества",
                    "построение априорной прогнозной оценки качества",
                    "проверка достоверности прогноза по train/val/test-разбиению",
                ],
                "source_files_total": len(source_status),
                "source_files_present": sum(source_status.values()),
                "test_files_total": len(test_status),
                "test_files_present": sum(test_status.values()),
                "dataset_artifacts_total": len(dataset_status),
                "dataset_artifacts_present": sum(dataset_status.values()),
                "scientific_components": list(SCIENTIFIC_COMPONENTS),
            }

            json_path = project_root / Path(report_config.output_json)
            markdown_path = project_root / Path(report_config.output_markdown)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.parent.mkdir(parents=True, exist_ok=True)

            payload = {
                "summary": summary,
                "source_file_status": source_status,
                "test_file_status": test_status,
                "dataset_artifact_status": dataset_status,
                "experiment_summary": experiment_summary,
            }
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            markdown_path.write_text(
                render_report_markdown(payload),
                encoding="utf-8",
            )

            return Chapter3ImplementationReport(
                summary=summary,
                source_file_status=source_status,
                test_file_status=test_status,
                dataset_artifact_status=dataset_status,
                experiment_summary=experiment_summary,
                json_path=json_path,
                markdown_path=markdown_path,
            )


        def collect_file_status(
            project_root: Path,
            relative_paths: tuple[str, ...],
        ) -> dict[str, bool]:
            """Проверяет наличие файлов относительно корня проекта."""
            return {
                path: (project_root / path).exists()
                for path in relative_paths
            }


        def collect_dataset_artifact_status(
            experiment_summary: dict[str, Any],
        ) -> dict[str, bool]:
            """Проверяет наличие табличных артефактов эксперимента."""
            saved_files = experiment_summary.get("saved_files", {})
            if not isinstance(saved_files, dict):
                return {key: False for key in EXPECTED_DATASET_KEYS}

            return {
                key: Path(str(saved_files.get(key, ""))).exists()
                for key in EXPECTED_DATASET_KEYS
            }


        def render_report_markdown(payload: dict[str, Any]) -> str:
            """Формирует Markdown-представление итогового отчета."""
            summary = payload["summary"]
            experiment_summary = payload.get("experiment_summary", {})
            source_status = payload["source_file_status"]
            test_status = payload["test_file_status"]
            dataset_status = payload["dataset_artifact_status"]

            lines = [
                "# Итоговый отчет программной реализации главы 3",
                "",
                "## Назначение",
                "",
                (
                    "Программная реализация подтверждает возможность "
                    "воспроизводимого компьютерного моделирования процессов "
                    "ручного кодирования и формирования данных для дальнейшей "
                    "априорной оценки качества."
                ),
                "",
                "## Реализованная цепочка",
                "",
                f"`{summary['scientific_chain']}`",
                "",
                "## Состав научных компонентов",
                "",
                "| Компонент | Модуль | Смысл |",
                "|---|---|---|",
            ]

            for item in summary["scientific_components"]:
                lines.append(
                    f"| {item['component']} | {item['module']} | {item['meaning']} |",
                )

            lines.extend(
                [
                    "",
                    "## Контроль файлов",
                    "",
                    f"Исходные файлы: {summary['source_files_present']} "
                    f"из {summary['source_files_total']}.",
                    f"Тестовые файлы: {summary['test_files_present']} "
                    f"из {summary['test_files_total']}.",
                    f"Артефакты датасета: {summary['dataset_artifacts_present']} "
                    f"из {summary['dataset_artifacts_total']}.",
                    "",
                    "## Исходные файлы",
                    "",
                ],
            )
            lines.extend(_status_lines(source_status))
            lines.extend(["", "## Тестовые файлы", ""])
            lines.extend(_status_lines(test_status))
            lines.extend(["", "## Артефакты датасета", ""])
            lines.extend(_status_lines(dataset_status))

            if experiment_summary:
                lines.extend(
                    [
                        "",
                        "## Воспроизводимый эксперимент",
                        "",
                        f"Эксперимент: `{experiment_summary.get('experiment_name')}`",
                        f"Сценарий: `{experiment_summary.get('scenario_id')}`",
                        f"Число прогонов: `{experiment_summary.get('run_count')}`",
                        f"random_seed: `{experiment_summary.get('random_seed')}`",
                        (
                            "Контроль воспроизводимости: "
                            f"`{experiment_summary.get('reproducibility_ok')}`"
                        ),
                        (
                            "Контрольный хеш: "
                            f"`{experiment_summary.get('reproducibility_hash')}`"
                        ),
                    ],
                )

            lines.extend(
                [
                    "",
                    "## Ограничение области реализации",
                    "",
                ],
            )
            for item in summary["not_implemented_in_chapter3"]:
                lines.append(f"- {item}")

            return "\n".join(lines) + "\n"


        def _status_lines(status: dict[str, bool]) -> list[str]:
            """Преобразует словарь статусов в Markdown-строки."""
            return [
                f"- [{'OK' if exists else 'MISSING'}] `{path}`"
                for path, exists in status.items()
            ]


        def _run_control_experiment(
            project_root: Path,
            report_config: Chapter3ImplementationReportConfig,
        ) -> dict[str, Any]:
            """Запускает малый контрольный эксперимент для итогового отчета."""
            experiment_config = ExperimentRunnerConfig(
                experiment_name="chapter3_final_control_experiment",
                random_seed=report_config.random_seed,
                run_count=report_config.experiment_run_count,
                scenario_id=report_config.scenario_id,
                output_dir=project_root / "data" / "processed",
                reports_dir=project_root / "reports" / "chapter3",
                overwrite=report_config.overwrite,
                check_reproducibility=True,
            )
            result = ExperimentRunner(experiment_config).run()
            return result.summary


        def main() -> None:
            """CLI-точка входа для формирования итогового отчета главы 3."""
            report = build_chapter3_implementation_report()
            print("Итоговый отчет программной реализации главы 3 сформирован.")
            print(f"JSON: {report.json_path}")
            print(f"Markdown: {report.markdown_path}")
            print(f"Все обязательные файлы: {report.all_required_files_present}")
            print(f"Все артефакты датасета: {report.all_dataset_artifacts_present}")


        if __name__ == "__main__":
            main()
        ''',
    )

    append_text_once(
        SRC_DIR / "__init__.py",
        "from manual_coding_sim.chapter3_report import",
        r'''
        # Экспорт этапа 13: итоговая отчетность программной реализации главы 3.
        from manual_coding_sim.chapter3_report import (
            Chapter3ImplementationReport,
            Chapter3ImplementationReportConfig,
            build_chapter3_implementation_report,
        )
        ''',
    )

    write_text_file(
        TESTS_DIR / "test_stage13_chapter3_report.py",
        r'''
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
            """Создает тестовую конфигурацию итогового отчета."""
            return Chapter3ImplementationReportConfig(
                project_root=Path.cwd(),
                output_json=tmp_path / "chapter3_implementation_report.json",
                output_markdown=tmp_path / "chapter3_implementation_report.md",
                run_experiment=run_experiment,
                experiment_run_count=2,
                random_seed=42,
                scenario_id="A_TEST_FINAL",
            )


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
        ''',
    )

    created_files = [
        SRC_DIR / "chapter3_report.py",
        SRC_DIR / "__init__.py",
        TESTS_DIR / "test_stage13_chapter3_report.py",
    ]
    syntax_report = {
        str(path.relative_to(ROOT)): check_python_syntax(path)
        for path in created_files
    }

    report = {
        "stage": 13,
        "title": "Тесты, отчетность и фиксация программной реализации главы 3",
        "created_files": [str(path.relative_to(ROOT)) for path in created_files],
        "syntax_report": syntax_report,
        "scientific_scope": (
            "Итоговая фиксация программной реализации главы 3: проверка состава "
            "модулей, тестов, экспериментальных артефактов и формирование "
            "JSON/Markdown-отчета для диссертации."
        ),
        "not_implemented_in_chapter3": [
            "адаптированная LDA-модель латентных факторов качества",
            "построение априорной прогнозной оценки качества",
            "проверка достоверности прогноза на train/val/test-разбиении",
        ],
    }

    report_path = REPORTS_DIR / "stage13_chapter3_report_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("ЭТАП 13. ИТОГОВАЯ ОТЧЕТНОСТЬ ПРОГРАММНОЙ РЕАЛИЗАЦИИ ГЛАВЫ 3")
    print("=" * 80)
    for path in created_files:
        rel_path = path.relative_to(ROOT)
        status = syntax_report[str(rel_path)]["status"]
        print(f"[{status}] {rel_path}")
    print(f"[OK] Отчет: {report_path}")
    print()
    print("Теперь выполните команду:")
    print("python -m pytest")
    print()
    print("После успешных тестов можно сформировать итоговый отчет командой:")
    print("python -m manual_coding_sim.chapter3_report")


if __name__ == "__main__":
    main()
