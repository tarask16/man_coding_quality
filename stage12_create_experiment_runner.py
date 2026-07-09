"""
Создание модуля запуска воспроизводимого эксперимента ExperimentRunner.

Скрипт относится к этапу 12 программной реализации главы 3 диссертации.
Он создает единую точку запуска вычислительного эксперимента, которая
загружает YAML-конфигурацию, формирует датасет и сохраняет отчет о
воспроизводимости результатов.

На этом этапе не выполняется LDA-моделирование, не строится априорный
прогноз качества и не проводится проверка достоверности прогноза. Задача
этапа — обеспечить управляемый запуск уже реализованной цепочки
моделирования и формирования табличных артефактов.
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
    """Создает модуль ExperimentRunner, тесты и отчет этапа 12."""
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    write_text_file(
        SRC_DIR / "experiment_runner.py",
        r'''
        """Запуск воспроизводимого вычислительного эксперимента главы 3.

        Модуль является единой точкой запуска программной реализации главы 3.
        Он загружает YAML-конфигурацию, передает параметры в DatasetBuilder,
        сохраняет табличные артефакты и формирует отчет о воспроизводимости.

        На данном этапе модуль не выполняет LDA-моделирование и не строит
        априорную прогнозную оценку качества. Его задача — воспроизводимо
        получить данные, необходимые для следующих глав диссертации.
        """

        from __future__ import annotations

        import argparse
        import hashlib
        import json
        from dataclasses import dataclass
        from pathlib import Path
        from typing import Any

        import yaml

        from manual_coding_sim.dataset_builder import (
            DatasetBuildResult,
            DatasetBuilder,
            DatasetBuilderConfig,
        )
        from manual_coding_sim.protocol_simulator import ProtocolSimulatorConfig


        @dataclass(frozen=True)
        class ExperimentRunnerConfig:
            """Конфигурация запуска вычислительного эксперимента.

            Параметры задают имя эксперимента, число прогонов, зерно генератора
            случайных чисел, идентификатор сценария A = {S, O, U, G, K}, а также
            каталоги для табличных данных и отчетов.
            """

            experiment_name: str = "chapter3_reproducible_experiment"
            random_seed: int = 42
            run_count: int = 10
            scenario_id: str = "A_001"
            output_dir: Path | str = Path("data") / "processed"
            reports_dir: Path | str = Path("reports") / "chapter3"
            overwrite: bool = True
            round_digits: int = 6
            check_reproducibility: bool = True

            def validate(self) -> None:
                """Проверяет корректность параметров эксперимента."""
                if not self.experiment_name:
                    raise ValueError("Имя эксперимента не задано.")

                if self.random_seed < 0:
                    raise ValueError("random_seed не должен быть отрицательным.")

                if self.run_count <= 0:
                    raise ValueError("run_count должен быть положительным.")

                if not self.scenario_id:
                    raise ValueError("Идентификатор сценария A не задан.")

                if self.round_digits < 0:
                    raise ValueError("round_digits не должен быть отрицательным.")

                Path(self.output_dir)
                Path(self.reports_dir)

            @classmethod
            def from_yaml(cls, config_path: str | Path) -> "ExperimentRunnerConfig":
                """Загружает конфигурацию эксперимента из YAML-файла."""
                path = Path(config_path)
                if not path.exists():
                    raise FileNotFoundError(f"Файл конфигурации не найден: {path}")

                with path.open("r", encoding="utf-8") as file:
                    config = yaml.safe_load(file)

                if not isinstance(config, dict):
                    raise ValueError("Конфигурация должна быть YAML-словарем.")

                experiment = config.get("experiment")
                if not isinstance(experiment, dict):
                    raise ValueError("В конфигурации отсутствует раздел experiment.")

                runner_config = cls(
                    experiment_name=str(
                        experiment.get("name", cls.experiment_name),
                    ),
                    random_seed=int(
                        experiment.get("random_seed", cls.random_seed),
                    ),
                    run_count=int(experiment.get("run_count", cls.run_count)),
                    scenario_id=str(
                        experiment.get("scenario_id", cls.scenario_id),
                    ),
                    output_dir=experiment.get("output_dir", cls.output_dir),
                    reports_dir=experiment.get("reports_dir", cls.reports_dir),
                    overwrite=bool(experiment.get("overwrite", cls.overwrite)),
                    round_digits=int(
                        experiment.get("round_digits", cls.round_digits),
                    ),
                    check_reproducibility=bool(
                        experiment.get(
                            "check_reproducibility",
                            cls.check_reproducibility,
                        ),
                    ),
                )
                runner_config.validate()
                return runner_config

            def to_dataset_builder_config(self) -> DatasetBuilderConfig:
                """Преобразует конфигурацию запуска в конфигурацию DatasetBuilder."""
                self.validate()
                simulator_config = ProtocolSimulatorConfig(
                    scenario_id=self.scenario_id,
                    message_random_seed=self.random_seed,
                    error_random_seed=self.random_seed + 1000,
                    control_random_seed=self.random_seed + 2000,
                )
                return DatasetBuilderConfig(
                    run_count=self.run_count,
                    output_dir=self.output_dir,
                    reports_dir=self.reports_dir,
                    overwrite=self.overwrite,
                    round_digits=self.round_digits,
                    simulator_config=simulator_config,
                )


        @dataclass(frozen=True)
        class ExperimentRunResult:
            """Результат запуска воспроизводимого эксперимента главы 3."""

            experiment_name: str
            run_count: int
            random_seed: int
            dataset_result: DatasetBuildResult
            report_path: Path
            reproducibility_hash: str
            reproducibility_ok: bool
            summary: dict[str, Any]

            @property
            def saved_files(self) -> dict[str, str]:
                """Возвращает словарь сохраненных артефактов эксперимента."""
                return self.dataset_result.saved_files


        class ExperimentRunner:
            """Организатор воспроизводимого вычислительного эксперимента.

            ExperimentRunner не содержит новой научной модели. Он связывает
            уже реализованные компоненты в управляемый сценарий запуска и
            фиксирует отчетные признаки воспроизводимости.
            """

            def __init__(self, config: ExperimentRunnerConfig | None = None) -> None:
                """Инициализирует запускатель эксперимента."""
                self.config = config or ExperimentRunnerConfig()
                self.config.validate()

            def run(self) -> ExperimentRunResult:
                """Запускает эксперимент, сохраняет датасет и отчет."""
                dataset_config = self.config.to_dataset_builder_config()
                dataset_result = DatasetBuilder(dataset_config).build_and_save()
                reproducibility_hash = hash_dataset_result(dataset_result)
                reproducibility_ok = True

                if self.config.check_reproducibility:
                    repeat_config = DatasetBuilderConfig(
                        run_count=dataset_config.run_count,
                        output_dir=dataset_config.output_dir,
                        reports_dir=dataset_config.reports_dir,
                        overwrite=True,
                        round_digits=dataset_config.round_digits,
                        simulator_config=dataset_config.simulator_config,
                    )
                    repeat_result = DatasetBuilder(repeat_config).build()
                    reproducibility_ok = (
                        reproducibility_hash == hash_dataset_result(repeat_result)
                    )

                summary = self._build_experiment_summary(
                    dataset_result=dataset_result,
                    reproducibility_hash=reproducibility_hash,
                    reproducibility_ok=reproducibility_ok,
                )
                report_path = Path(self.config.reports_dir) / "experiment_run_report.json"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                return ExperimentRunResult(
                    experiment_name=self.config.experiment_name,
                    run_count=self.config.run_count,
                    random_seed=self.config.random_seed,
                    dataset_result=dataset_result,
                    report_path=report_path,
                    reproducibility_hash=reproducibility_hash,
                    reproducibility_ok=reproducibility_ok,
                    summary=summary,
                )

            def _build_experiment_summary(
                self,
                dataset_result: DatasetBuildResult,
                reproducibility_hash: str,
                reproducibility_ok: bool,
            ) -> dict[str, Any]:
                """Формирует итоговую сводку запуска эксперимента."""
                return {
                    "experiment_name": self.config.experiment_name,
                    "scenario_id": self.config.scenario_id,
                    "run_count": self.config.run_count,
                    "random_seed": self.config.random_seed,
                    "message_random_seed": self.config.random_seed,
                    "error_random_seed": self.config.random_seed + 1000,
                    "control_random_seed": self.config.random_seed + 2000,
                    "reproducibility_hash": reproducibility_hash,
                    "reproducibility_ok": reproducibility_ok,
                    "dataset_summary": dataset_result.summary,
                    "saved_files": dataset_result.saved_files,
                }


        def hash_dataset_result(result: DatasetBuildResult) -> str:
            """Вычисляет контрольный хеш содержательной части датасета."""
            payload = {
                "protocol_rows": result.protocol_rows,
                "prior_feature_rows": result.prior_feature_rows,
                "fact_feature_rows": result.fact_feature_rows,
                "diagnostic_feature_rows": result.diagnostic_feature_rows,
                "quality_rows": result.quality_rows,
                "all_feature_rows": result.all_feature_rows,
                "summary": _summary_without_saved_files(result.summary),
            }
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()


        def _summary_without_saved_files(summary: dict[str, Any]) -> dict[str, Any]:
            """Исключает пути файлов из сводки перед проверкой хеша."""
            return {
                key: value
                for key, value in summary.items()
                if key != "saved_files"
            }


        def run_experiment(
            config: ExperimentRunnerConfig | None = None,
        ) -> ExperimentRunResult:
            """Запускает воспроизводимый эксперимент главы 3."""
            return ExperimentRunner(config).run()


        def run_experiment_from_yaml(config_path: str | Path) -> ExperimentRunResult:
            """Загружает YAML-конфигурацию и запускает эксперимент."""
            config = ExperimentRunnerConfig.from_yaml(config_path)
            return run_experiment(config)


        def main() -> None:
            """CLI-точка входа для запуска эксперимента из командной строки."""
            parser = argparse.ArgumentParser(
                description="Запуск воспроизводимого эксперимента главы 3.",
            )
            parser.add_argument(
                "--config",
                default="configs/base_experiment.yaml",
                help="Путь к YAML-конфигурации эксперимента.",
            )
            args = parser.parse_args()

            result = run_experiment_from_yaml(args.config)
            print("Воспроизводимый эксперимент главы 3 выполнен.")
            print(f"Эксперимент: {result.experiment_name}")
            print(f"Число прогонов: {result.run_count}")
            print(f"random_seed: {result.random_seed}")
            print(f"Контроль воспроизводимости: {result.reproducibility_ok}")
            print(f"Отчет: {result.report_path}")


        if __name__ == "__main__":
            main()
        ''',
    )

    append_text_once(
        SRC_DIR / "__init__.py",
        "from manual_coding_sim.experiment_runner import",
        r'''
        # Экспорт этапа 12: запуск воспроизводимого эксперимента главы 3.
        from manual_coding_sim.experiment_runner import (
            ExperimentRunResult,
            ExperimentRunner,
            ExperimentRunnerConfig,
            hash_dataset_result,
            run_experiment,
            run_experiment_from_yaml,
        )
        ''',
    )

    write_text_file(
        TESTS_DIR / "test_stage12_experiment_runner.py",
        r'''
        """Тесты этапа 12: запуск воспроизводимого эксперимента."""

        from __future__ import annotations

        import json
        from pathlib import Path

        import pytest

        from manual_coding_sim.experiment_runner import (
            ExperimentRunResult,
            ExperimentRunner,
            ExperimentRunnerConfig,
            hash_dataset_result,
            run_experiment,
            run_experiment_from_yaml,
        )


        def _write_config(tmp_path: Path, run_count: int = 3, seed: int = 42) -> Path:
            """Создает временную YAML-конфигурацию эксперимента."""
            config_path = tmp_path / "base_experiment.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "experiment:",
                        "  name: test_chapter3_experiment",
                        f"  random_seed: {seed}",
                        f"  run_count: {run_count}",
                        "  scenario_id: A_TEST",
                        f"  output_dir: {tmp_path.as_posix()}/data/processed",
                        f"  reports_dir: {tmp_path.as_posix()}/reports/chapter3",
                        "  overwrite: true",
                        "  check_reproducibility: true",
                    ],
                ),
                encoding="utf-8",
            )
            return config_path


        def _make_config(tmp_path: Path, run_count: int = 3) -> ExperimentRunnerConfig:
            """Создает тестовую конфигурацию запускателя эксперимента."""
            return ExperimentRunnerConfig(
                experiment_name="test_chapter3_experiment",
                random_seed=42,
                run_count=run_count,
                scenario_id="A_TEST",
                output_dir=tmp_path / "data" / "processed",
                reports_dir=tmp_path / "reports" / "chapter3",
            )


        def test_experiment_runner_imports(tmp_path: Path) -> None:
            """Проверяет импортируемость запускателя эксперимента."""
            runner = ExperimentRunner(_make_config(tmp_path))

            assert isinstance(runner.config, ExperimentRunnerConfig)


        def test_default_config_is_valid() -> None:
            """Проверяет корректность конфигурации по умолчанию."""
            config = ExperimentRunnerConfig()

            config.validate()
            assert config.run_count > 0
            assert config.random_seed >= 0


        def test_config_loads_from_yaml(tmp_path: Path) -> None:
            """Проверяет загрузку параметров эксперимента из YAML."""
            config_path = _write_config(tmp_path, run_count=4, seed=123)
            config = ExperimentRunnerConfig.from_yaml(config_path)

            assert config.experiment_name == "test_chapter3_experiment"
            assert config.random_seed == 123
            assert config.run_count == 4
            assert config.scenario_id == "A_TEST"


        def test_missing_yaml_is_rejected(tmp_path: Path) -> None:
            """Проверяет отклонение отсутствующего файла конфигурации."""
            with pytest.raises(FileNotFoundError):
                ExperimentRunnerConfig.from_yaml(tmp_path / "missing.yaml")


        def test_invalid_config_values_are_rejected(tmp_path: Path) -> None:
            """Проверяет отклонение некорректных параметров запуска."""
            with pytest.raises(ValueError):
                ExperimentRunnerConfig(run_count=0).validate()

            with pytest.raises(ValueError):
                ExperimentRunnerConfig(random_seed=-1).validate()

            with pytest.raises(ValueError):
                ExperimentRunnerConfig(scenario_id="").validate()

            bad_path = tmp_path / "bad.yaml"
            bad_path.write_text("experiment:\n  name: bad\n  random_seed: 1\n  run_count: 0\n", encoding="utf-8")
            with pytest.raises(ValueError):
                ExperimentRunnerConfig.from_yaml(bad_path)


        def test_config_converts_to_dataset_builder_config(tmp_path: Path) -> None:
            """Проверяет передачу зерен генераторов в конфигурацию датасета."""
            config = _make_config(tmp_path, run_count=5)
            dataset_config = config.to_dataset_builder_config()

            assert dataset_config.run_count == 5
            assert dataset_config.simulator_config.scenario_id == "A_TEST"
            assert dataset_config.simulator_config.message_random_seed == 42
            assert dataset_config.simulator_config.error_random_seed == 1042
            assert dataset_config.simulator_config.control_random_seed == 2042


        def test_runner_run_returns_experiment_result(tmp_path: Path) -> None:
            """Проверяет запуск эксперимента и тип результата."""
            result = ExperimentRunner(_make_config(tmp_path)).run()

            assert isinstance(result, ExperimentRunResult)
            assert result.run_count == 3
            assert result.experiment_name == "test_chapter3_experiment"


        def test_runner_creates_dataset_files(tmp_path: Path) -> None:
            """Проверяет сохранение табличных артефактов эксперимента."""
            result = ExperimentRunner(_make_config(tmp_path)).run()

            assert Path(result.saved_files["prior_features"]).exists()
            assert Path(result.saved_files["fact_features"]).exists()
            assert Path(result.saved_files["quality_targets"]).exists()


        def test_runner_creates_experiment_report(tmp_path: Path) -> None:
            """Проверяет сохранение итогового отчета эксперимента."""
            result = ExperimentRunner(_make_config(tmp_path)).run()
            report = json.loads(result.report_path.read_text(encoding="utf-8"))

            assert result.report_path.exists()
            assert report["experiment_name"] == "test_chapter3_experiment"
            assert report["scenario_id"] == "A_TEST"
            assert report["reproducibility_ok"] is True


        def test_reproducibility_hash_is_stable(tmp_path: Path) -> None:
            """Проверяет устойчивость контрольного хеша датасета."""
            config = _make_config(tmp_path)
            result = ExperimentRunner(config).run()
            same_hash = hash_dataset_result(result.dataset_result)

            assert len(result.reproducibility_hash) == 64
            assert result.reproducibility_hash == same_hash
            assert result.reproducibility_ok is True


        def test_run_experiment_helper(tmp_path: Path) -> None:
            """Проверяет вспомогательную функцию run_experiment()."""
            result = run_experiment(_make_config(tmp_path, run_count=2))

            assert result.run_count == 2
            assert result.summary["run_count"] == 2


        def test_run_experiment_from_yaml(tmp_path: Path) -> None:
            """Проверяет запуск эксперимента по YAML-конфигурации."""
            config_path = _write_config(tmp_path, run_count=2, seed=77)
            result = run_experiment_from_yaml(config_path)

            assert result.run_count == 2
            assert result.random_seed == 77
            assert result.summary["random_seed"] == 77


        def test_report_contains_dataset_summary_and_saved_files(tmp_path: Path) -> None:
            """Проверяет полноту итоговой сводки эксперимента."""
            result = ExperimentRunner(_make_config(tmp_path)).run()
            summary = result.summary

            assert summary["dataset_summary"]["run_count"] == 3
            assert "protocols" in summary["saved_files"]
            assert "quality_targets" in summary["saved_files"]
            assert summary["dataset_summary"]["prior_feature_count"] > 0
        ''',
    )

    created_files = [
        SRC_DIR / "experiment_runner.py",
        SRC_DIR / "__init__.py",
        TESTS_DIR / "test_stage12_experiment_runner.py",
    ]
    syntax_report = {
        str(path.relative_to(ROOT)): check_python_syntax(path)
        for path in created_files
    }

    report = {
        "stage": 12,
        "title": "Запуск воспроизводимого эксперимента ExperimentRunner",
        "created_files": [str(path.relative_to(ROOT)) for path in created_files],
        "syntax_report": syntax_report,
        "scientific_scope": (
            "Единая точка запуска вычислительного эксперимента главы 3: "
            "загрузка YAML-конфигурации, формирование датасета, сохранение "
            "табличных артефактов и проверка воспроизводимости по контрольному хешу."
        ),
        "not_implemented_yet": [
            "токенизация признаков для LDA",
            "обучение LDA-модели",
            "построение априорной прогнозной оценки качества",
            "сравнение прогнозной и фактической оценки качества",
        ],
    }

    report_path = REPORTS_DIR / "stage12_experiment_runner_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("ЭТАП 12. ЗАПУСК ВОСПРОИЗВОДИМОГО ЭКСПЕРИМЕНТА EXPERIMENTRUNNER")
    print("=" * 80)
    for path in created_files:
        rel_path = path.relative_to(ROOT)
        status = syntax_report[str(rel_path)]["status"]
        print(f"[{status}] {rel_path}")
    print(f"[OK] Отчет: {report_path}")
    print()
    print("Теперь выполните команду:")
    print("python -m pytest")


if __name__ == "__main__":
    main()
