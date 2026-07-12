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
