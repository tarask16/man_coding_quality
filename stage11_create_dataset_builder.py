"""
Создание модуля формирования датасета DatasetBuilder.

Скрипт относится к этапу 11 программной реализации главы 3 диссертации.
Он создает модуль, который объединяет результаты интегрального симулятора,
извлеченные признаки X_prior, X_fact, X_diag и фактический вектор качества
q(A) в воспроизводимые табличные артефакты.

На этом этапе не выполняется токенизация признаков для LDA, не обучается
LDA-модель и не проверяется прогнозная достоверность метода. Формируется
только датасет, который будет входом для следующих глав.
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
    """Создает модуль DatasetBuilder, тесты и отчет этапа 11."""
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    write_text_file(
        SRC_DIR / "dataset_builder.py",
        '''
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
        ''',
    )

    append_text_once(
        SRC_DIR / "__init__.py",
        "from manual_coding_sim.dataset_builder import",
        '''
        # Экспорт этапа 11: формирование табличного датасета главы 3.
        from manual_coding_sim.dataset_builder import (
            DatasetBuildResult,
            DatasetBuilder,
            DatasetBuilderConfig,
            build_dataset,
        )
        ''',
    )

    write_text_file(
        TESTS_DIR / "test_stage11_dataset_builder.py",
        '''
        """Тесты этапа 11: формирование датасета главы 3."""

        from __future__ import annotations

        import csv
        import json
        from pathlib import Path

        import pytest

        from manual_coding_sim.dataset_builder import (
            DatasetBuildResult,
            DatasetBuilder,
            DatasetBuilderConfig,
            build_dataset,
        )


        def _make_config(tmp_path: Path, run_count: int = 3) -> DatasetBuilderConfig:
            """Создает тестовую конфигурацию с временными каталогами."""
            return DatasetBuilderConfig(
                run_count=run_count,
                output_dir=tmp_path / "data" / "processed",
                reports_dir=tmp_path / "reports" / "chapter3",
            )


        def test_dataset_builder_imports(tmp_path: Path) -> None:
            """Проверяет импортируемость построителя датасета."""
            builder = DatasetBuilder(_make_config(tmp_path))

            assert isinstance(builder.config, DatasetBuilderConfig)


        def test_build_returns_dataset_build_result(tmp_path: Path) -> None:
            """Проверяет формирование датасета в памяти."""
            result = DatasetBuilder(_make_config(tmp_path)).build()

            assert isinstance(result, DatasetBuildResult)
            assert result.run_count == 3
            assert result.saved_files == {}


        def test_dataset_rows_have_requested_count(tmp_path: Path) -> None:
            """Проверяет согласованность числа строк во всех таблицах."""
            result = DatasetBuilder(_make_config(tmp_path, run_count=4)).build()

            assert len(result.protocol_rows) == 4
            assert len(result.prior_feature_rows) == 4
            assert len(result.fact_feature_rows) == 4
            assert len(result.diagnostic_feature_rows) == 4
            assert len(result.quality_rows) == 4
            assert len(result.all_feature_rows) == 4


        def test_prior_rows_do_not_contain_fact_features(tmp_path: Path) -> None:
            """Проверяет отсутствие фактических признаков в X_prior."""
            result = DatasetBuilder(_make_config(tmp_path)).build()
            prior_keys = set(result.prior_feature_rows[0])

            assert not any(key.startswith("fact_") for key in prior_keys)
            assert "prior_step_count" in prior_keys


        def test_quality_rows_contain_quality_vector(tmp_path: Path) -> None:
            """Проверяет наличие всех компонентов q(A) в целевой таблице."""
            result = DatasetBuilder(_make_config(tmp_path)).build()
            quality_keys = set(result.quality_rows[0])

            assert {"q_acc", "q_time", "q_effort", "q_res", "q_rep", "q_fit"} <= quality_keys
            assert "integral_quality" in quality_keys


        def test_summary_contains_dataset_statistics(tmp_path: Path) -> None:
            """Проверяет отчетную сводку по датасету."""
            result = DatasetBuilder(_make_config(tmp_path, run_count=5)).build()

            assert result.summary["run_count"] == 5
            assert result.summary["prior_feature_count"] > 0
            assert result.summary["fact_feature_count"] > 0
            assert 0.0 <= result.summary["mean_integral_quality"] <= 1.0


        def test_build_and_save_creates_all_files(tmp_path: Path) -> None:
            """Проверяет сохранение CSV и JSON-артефактов."""
            result = DatasetBuilder(_make_config(tmp_path)).build_and_save()

            expected_keys = {
                "protocols",
                "prior_features",
                "fact_features",
                "diagnostic_features",
                "quality_targets",
                "all_features",
                "summary",
            }
            assert set(result.saved_files) == expected_keys
            assert all(Path(path).exists() for path in result.saved_files.values())


        def test_saved_prior_csv_has_no_fact_columns(tmp_path: Path) -> None:
            """Проверяет, что сохраненный prior_features.csv не содержит X_fact."""
            result = DatasetBuilder(_make_config(tmp_path)).build_and_save()
            path = Path(result.saved_files["prior_features"])

            with path.open("r", encoding="utf-8", newline="") as csv_file:
                reader = csv.reader(csv_file)
                header = next(reader)

            assert "prior_step_count" in header
            assert not any(column.startswith("fact_") for column in header)


        def test_saved_quality_csv_has_quality_columns(tmp_path: Path) -> None:
            """Проверяет столбцы CSV с фактическими показателями качества."""
            result = DatasetBuilder(_make_config(tmp_path)).build_and_save()
            path = Path(result.saved_files["quality_targets"])

            with path.open("r", encoding="utf-8", newline="") as csv_file:
                reader = csv.reader(csv_file)
                header = next(reader)

            assert "q_acc" in header
            assert "q_fit" in header
            assert "integral_quality" in header


        def test_saved_summary_json_contains_saved_files(tmp_path: Path) -> None:
            """Проверяет JSON-сводку по сформированному датасету."""
            result = DatasetBuilder(_make_config(tmp_path)).build_and_save()
            path = Path(result.saved_files["summary"])
            summary = json.loads(path.read_text(encoding="utf-8"))

            assert summary["run_count"] == 3
            assert "saved_files" in summary
            assert "quality_targets" in summary["saved_files"]


        def test_build_dataset_helper_saves_dataset(tmp_path: Path) -> None:
            """Проверяет вспомогательную функцию build_dataset()."""
            result = build_dataset(_make_config(tmp_path, run_count=2))

            assert result.run_count == 2
            assert Path(result.saved_files["protocols"]).exists()


        def test_invalid_config_is_rejected(tmp_path: Path) -> None:
            """Проверяет отклонение некорректной конфигурации."""
            with pytest.raises(ValueError):
                DatasetBuilderConfig(run_count=0, output_dir=tmp_path).validate()

            with pytest.raises(ValueError):
                DatasetBuilderConfig(run_count=1, round_digits=-1).validate()


        def test_overwrite_false_rejects_existing_files(tmp_path: Path) -> None:
            """Проверяет защиту от перезаписи сформированного датасета."""
            config = DatasetBuilderConfig(
                run_count=2,
                output_dir=tmp_path / "data" / "processed",
                reports_dir=tmp_path / "reports" / "chapter3",
                overwrite=False,
            )
            DatasetBuilder(config).build_and_save()

            with pytest.raises(FileExistsError):
                DatasetBuilder(config).build_and_save()
        ''',
    )

    created_files = [
        SRC_DIR / "dataset_builder.py",
        SRC_DIR / "__init__.py",
        TESTS_DIR / "test_stage11_dataset_builder.py",
    ]
    syntax_report = {
        str(path.relative_to(ROOT)): check_python_syntax(path)
        for path in created_files
    }

    report = {
        "stage": 11,
        "title": "Формирование датасета DatasetBuilder",
        "created_files": [str(path.relative_to(ROOT)) for path in created_files],
        "syntax_report": syntax_report,
        "scientific_scope": (
            "Формирование табличных артефактов protocols.csv, prior_features.csv, "
            "fact_features.csv, diagnostic_features.csv и quality_targets.csv "
            "по результатам моделирования сценариев A = {S, O, U, G, K}."
        ),
        "not_implemented_yet": [
            "токенизация признаков для LDA",
            "обучение LDA-модели",
            "построение априорного прогноза качества",
            "проверка достоверности прогноза на train/val/test разбиении",
        ],
    }

    report_path = REPORTS_DIR / "stage11_dataset_builder_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("ЭТАП 11. ФОРМИРОВАНИЕ ДАТАСЕТА DATASETBUILDER")
    print("=" * 70)
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
