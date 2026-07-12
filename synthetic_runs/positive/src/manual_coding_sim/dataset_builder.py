"""Формирование датасета по результатам моделирования ручного кодирования.

Модуль относится к главе 3 диссертации и переводит результаты
воспроизводимого моделирования в табличные артефакты:

* protocols.csv — контрольная таблица протоколов моделирования;
* prior_features.csv — априорные признаки X_prior;
* fact_features.csv — фактические признаки X_fact;
* diagnostic_features.csv — диагностические признаки X_diag;
* quality_targets.csv — фактические частные показатели качества q(A);
* all_features.csv — объединенная таблица признаков для анализа.

Модуль не выполняет тематическое моделирование LDA. Его задача —
подготовить воспроизводимый датасет для следующих этапов: адаптации
LDA-модели, построения метода априорной оценки качества и проверки
достоверности прогноза.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from manual_coding_sim.feature_extractor import (
    FeatureExtractor,
    feature_group_to_flat_row,
)
from manual_coding_sim.protocol_simulator import (
    ProtocolSimulator,
    ProtocolSimulatorConfig,
    SimulationResult,
    simulation_results_to_rows,
)
from manual_coding_sim.quality_calculator import (
    QualityAssessment,
    QualityCalculator,
    quality_assessments_to_rows,
)
from manual_coding_sim.types import FeatureGroup


TableRow = dict[str, int | float | str]


@dataclass(frozen=True)
class DatasetBuilderConfig:
    """Конфигурация формирования датасета главы 3.

    run_count задает число прогонов моделирования. output_dir хранит
    табличные артефакты, а reports_dir — сводный отчет. Параметр
    overwrite разрешает перезапись ранее сформированных таблиц.
    """

    run_count: int = 10
    output_dir: Path | str = Path("data") / "processed"
    reports_dir: Path | str = Path("reports") / "chapter3"
    overwrite: bool = True
    round_digits: int = 6
    simulator_config: ProtocolSimulatorConfig = field(
        default_factory=ProtocolSimulatorConfig,
    )

    def validate(self) -> None:
        """Проверяет корректность конфигурации построения датасета."""
        if self.run_count <= 0:
            raise ValueError("run_count должен быть положительным.")

        if self.round_digits < 0:
            raise ValueError("round_digits не должен быть отрицательным.")

        Path(self.output_dir)
        Path(self.reports_dir)
        self.simulator_config.validate()


@dataclass(frozen=True)
class DatasetBuildResult:
    """Результат формирования датасета по протоколам моделирования."""

    run_count: int
    protocol_rows: list[TableRow]
    prior_feature_rows: list[TableRow]
    fact_feature_rows: list[TableRow]
    diagnostic_feature_rows: list[TableRow]
    quality_rows: list[TableRow]
    all_feature_rows: list[TableRow]
    saved_files: dict[str, str]
    summary: dict[str, int | float | str]


class DatasetBuilder:
    """Построитель воспроизводимого датасета главы 3.

    DatasetBuilder объединяет ранее реализованные компоненты:
    ProtocolSimulator, FeatureExtractor и QualityCalculator. Он не
    меняет научную логику моделей, а только формирует согласованные
    таблицы для последующего анализа.
    """

    def __init__(self, config: DatasetBuilderConfig | None = None) -> None:
        """Инициализирует построитель датасета."""
        self.config = config or DatasetBuilderConfig()
        self.config.validate()

    def build(self) -> DatasetBuildResult:
        """Выполняет моделирование и формирует датасет в памяти."""
        simulator = ProtocolSimulator(self.config.simulator_config)
        results = simulator.simulate_batch(self.config.run_count)

        extractor = FeatureExtractor()
        feature_groups = extractor.extract_batch(results)

        calculator = QualityCalculator()
        assessments = calculator.calculate_batch(feature_groups)

        protocol_rows = self._add_run_ids(
            simulation_results_to_rows(results),
            results,
        )
        prior_rows = self._feature_rows(feature_groups, "prior")
        fact_rows = self._feature_rows(feature_groups, "fact")
        diagnostic_rows = self._feature_rows(feature_groups, "diagnostic")
        all_feature_rows = self._all_feature_rows(feature_groups, results)
        quality_rows = self._quality_rows(assessments, results)
        summary = self._build_summary(
            protocol_rows=protocol_rows,
            prior_rows=prior_rows,
            fact_rows=fact_rows,
            diagnostic_rows=diagnostic_rows,
            quality_rows=quality_rows,
        )

        return DatasetBuildResult(
            run_count=self.config.run_count,
            protocol_rows=protocol_rows,
            prior_feature_rows=prior_rows,
            fact_feature_rows=fact_rows,
            diagnostic_feature_rows=diagnostic_rows,
            quality_rows=quality_rows,
            all_feature_rows=all_feature_rows,
            saved_files={},
            summary=summary,
        )

    def build_and_save(self) -> DatasetBuildResult:
        """Формирует датасет и сохраняет CSV/JSON-артефакты."""
        result = self.build()
        output_dir = Path(self.config.output_dir)
        reports_dir = Path(self.config.reports_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {
            "protocols": str(output_dir / "protocols.csv"),
            "prior_features": str(output_dir / "prior_features.csv"),
            "fact_features": str(output_dir / "fact_features.csv"),
            "diagnostic_features": str(output_dir / "diagnostic_features.csv"),
            "quality_targets": str(output_dir / "quality_targets.csv"),
            "all_features": str(output_dir / "all_features.csv"),
            "summary": str(reports_dir / "dataset_summary.json"),
        }

        self._write_csv(Path(saved_files["protocols"]), result.protocol_rows)
        self._write_csv(Path(saved_files["prior_features"]), result.prior_feature_rows)
        self._write_csv(Path(saved_files["fact_features"]), result.fact_feature_rows)
        self._write_csv(
            Path(saved_files["diagnostic_features"]),
            result.diagnostic_feature_rows,
        )
        self._write_csv(Path(saved_files["quality_targets"]), result.quality_rows)
        self._write_csv(Path(saved_files["all_features"]), result.all_feature_rows)

        summary = dict(result.summary)
        summary["saved_files"] = saved_files
        Path(saved_files["summary"]).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return DatasetBuildResult(
            run_count=result.run_count,
            protocol_rows=result.protocol_rows,
            prior_feature_rows=result.prior_feature_rows,
            fact_feature_rows=result.fact_feature_rows,
            diagnostic_feature_rows=result.diagnostic_feature_rows,
            quality_rows=result.quality_rows,
            all_feature_rows=result.all_feature_rows,
            saved_files=saved_files,
            summary=summary,
        )

    def _add_run_ids(
        self,
        rows: list[TableRow],
        results: Iterable[SimulationResult],
    ) -> list[TableRow]:
        """Добавляет идентификаторы прогонов к строкам протоколов."""
        enriched_rows: list[TableRow] = []
        for run_id, (row, result) in enumerate(zip(rows, results), start=1):
            enriched_row: TableRow = {
                "run_id": run_id,
                "message_id": result.message.message_id,
            }
            enriched_row.update(row)
            enriched_rows.append(enriched_row)
        return enriched_rows

    def _feature_rows(
        self,
        feature_groups: Iterable[FeatureGroup],
        group_name: str,
    ) -> list[TableRow]:
        """Формирует строки для одной группы признаков."""
        rows: list[TableRow] = []
        for run_id, feature_group in enumerate(feature_groups, start=1):
            if group_name == "prior":
                features = feature_group.prior_features
            elif group_name == "fact":
                features = feature_group.fact_features
            elif group_name == "diagnostic":
                features = feature_group.diagnostic_features
            else:
                raise ValueError(f"Неизвестная группа признаков: {group_name}")

            row: TableRow = {
                "run_id": run_id,
                "scenario_id": feature_group.scenario_id,
            }
            row.update(features)
            rows.append(row)
        return rows

    def _all_feature_rows(
        self,
        feature_groups: Iterable[FeatureGroup],
        results: Iterable[SimulationResult],
    ) -> list[TableRow]:
        """Формирует объединенную таблицу признаков."""
        rows: list[TableRow] = []
        for run_id, (feature_group, result) in enumerate(
            zip(feature_groups, results),
            start=1,
        ):
            row: TableRow = {
                "run_id": run_id,
                "message_id": result.message.message_id,
            }
            row.update(feature_group_to_flat_row(feature_group))
            rows.append(row)
        return rows

    def _quality_rows(
        self,
        assessments: Iterable[QualityAssessment],
        results: Iterable[SimulationResult],
    ) -> list[TableRow]:
        """Формирует таблицу фактических целевых показателей качества."""
        rows: list[TableRow] = []
        base_rows = quality_assessments_to_rows(assessments)
        for run_id, (row, result) in enumerate(zip(base_rows, results), start=1):
            enriched_row: TableRow = {
                "run_id": run_id,
                "message_id": result.message.message_id,
            }
            enriched_row.update(row)
            rows.append(enriched_row)
        return rows

    def _build_summary(
        self,
        protocol_rows: list[TableRow],
        prior_rows: list[TableRow],
        fact_rows: list[TableRow],
        diagnostic_rows: list[TableRow],
        quality_rows: list[TableRow],
    ) -> dict[str, int | float | str]:
        """Формирует сводку по построенному датасету."""
        mean_integral_quality = self._mean_value(
            quality_rows,
            "integral_quality",
        )
        mean_q_acc = self._mean_value(quality_rows, "q_acc")
        mean_residual_error_rate = self._mean_value(
            fact_rows,
            "fact_residual_error_rate",
        )

        return {
            "run_count": len(protocol_rows),
            "protocol_row_count": len(protocol_rows),
            "prior_feature_row_count": len(prior_rows),
            "fact_feature_row_count": len(fact_rows),
            "diagnostic_feature_row_count": len(diagnostic_rows),
            "quality_row_count": len(quality_rows),
            "prior_feature_count": self._feature_count(prior_rows),
            "fact_feature_count": self._feature_count(fact_rows),
            "diagnostic_feature_count": self._feature_count(diagnostic_rows),
            "mean_integral_quality": self._round(mean_integral_quality),
            "mean_q_acc": self._round(mean_q_acc),
            "mean_residual_error_rate": self._round(mean_residual_error_rate),
        }

    def _write_csv(self, path: Path, rows: list[TableRow]) -> None:
        """Сохраняет строки таблицы в CSV-файл."""
        if path.exists() and not self.config.overwrite:
            raise FileExistsError(f"Файл уже существует: {path}")

        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = self._collect_fieldnames(rows)
        with path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _collect_fieldnames(self, rows: list[TableRow]) -> list[str]:
        """Собирает стабильный список столбцов для CSV-файла."""
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        return fieldnames

    def _feature_count(self, rows: list[TableRow]) -> int:
        """Считает число признаковых столбцов без идентификаторов."""
        if not rows:
            return 0
        excluded = {"run_id", "scenario_id", "message_id"}
        return len(set(rows[0]) - excluded)

    def _mean_value(self, rows: list[TableRow], key: str) -> float:
        """Рассчитывает среднее значение числового столбца."""
        if not rows:
            return 0.0
        values = [float(row[key]) for row in rows if key in row]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _round(self, value: float) -> float:
        """Округляет значение для отчетной сводки."""
        return round(float(value), self.config.round_digits)


def build_dataset(
    config: DatasetBuilderConfig | None = None,
) -> DatasetBuildResult:
    """Формирует и сохраняет датасет главы 3 с заданной конфигурацией."""
    return DatasetBuilder(config).build_and_save()


def main() -> None:
    """CLI-точка входа для ручного формирования датасета."""
    result = build_dataset()
    print("Датасет главы 3 сформирован.")
    print(f"Число прогонов моделирования: {result.run_count}")
    print(f"Средняя интегральная оценка качества: {result.summary['mean_integral_quality']}")
    print("Сохраненные файлы:")
    for name, path in result.saved_files.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
