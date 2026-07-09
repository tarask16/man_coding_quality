"""CLI-запуск главы 4 с предварительной проверкой расширенного корпуса.

Обычный ``chapter4_cli`` запускает LDA-pipeline на доступных данных. Этот
модуль добавляет методический предохранитель: перед запуском проверяется, что
корпус достаточен для финальных выводов главы 4. При недостаточном корпусе
запуск останавливается, если явно не задан ``--allow-insufficient-corpus``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from manual_coding_sim.lda.chapter4_runner import Chapter4LdaRunner
from manual_coding_sim.lda.config_loader import (
    Chapter4ConfigLoader,
    Chapter4ConfigOverrides,
)
from manual_coding_sim.lda.corpus_sufficiency import (
    CorpusSufficiencyAnalyzer,
    CorpusSufficiencyConfigLoader,
)
from manual_coding_sim.lda.paths import resolve_project_path


def build_extended_arg_parser() -> argparse.ArgumentParser:
    """Создать парсер аргументов расширенного запуска главы 4."""

    parser = argparse.ArgumentParser(
        description=(
            "Запуск главы 4 с проверкой достаточности расширенного корпуса."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Путь к YAML-конфигурации главы 4 и секции corpus.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Корень проекта, относительно которого разрешаются пути конфигурации.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Разрешить перезапись выходных артефактов.",
    )
    parser.add_argument(
        "--skip-diagnostic",
        action="store_true",
        help="Отключить построение LDA_diag и LDA_full.",
    )
    parser.add_argument(
        "--selected-k",
        type=int,
        default=None,
        help="Зафиксировать число латентных факторов K вместо значения из YAML.",
    )
    parser.add_argument(
        "--allow-insufficient-corpus",
        action="store_true",
        help=(
            "Разрешить запуск LDA даже при недостаточном корпусе. "
            "Использовать только для отладки, не для финальных выводов."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить расширенный CLI-запуск и вернуть код завершения."""

    parser = build_extended_arg_parser()
    args = parser.parse_args(argv)
    project_root = Path(args.project_root)

    try:
        overrides = Chapter4ConfigOverrides(
            overwrite=True if args.overwrite else None,
            skip_diagnostic=args.skip_diagnostic,
            selected_k=args.selected_k,
        )
        chapter_loader = Chapter4ConfigLoader()
        load_result = chapter_loader.load(args.config, overrides)
        runner_config = load_result.runner_config
        chapter_loader.validate_required_inputs(runner_config, project_root=project_root)

        corpus_config = CorpusSufficiencyConfigLoader().load(
            args.config,
            overwrite_override=True if args.overwrite else None,
        )
        prior_path = resolve_project_path(
            project_root,
            runner_config.chapter4.inputs.prior_features_path,
        )
        reports_dir = resolve_project_path(
            project_root,
            runner_config.chapter4.outputs.reports_dir,
        )
        sufficiency_result = CorpusSufficiencyAnalyzer(corpus_config).analyze_from_files(
            prior_features_path=prior_path,
            reports_dir=reports_dir,
        )
        if not sufficiency_result.passed and not args.allow_insufficient_corpus:
            print(
                "Корпус недостаточен для финальных выводов главы 4. "
                f"Отчет: {sufficiency_result.report_md_path}",
                file=sys.stderr,
            )
            return 2

        run_result = Chapter4LdaRunner(runner_config).run(project_root=project_root)

        dictionary_path = resolve_project_path(
            project_root,
            runner_config.chapter4.outputs.data_dir,
        ) / "dictionary.json"
        post_result = CorpusSufficiencyAnalyzer(corpus_config).analyze_from_files(
            prior_features_path=prior_path,
            dictionary_path=dictionary_path,
            reports_dir=reports_dir,
        )
        if not post_result.passed and not args.allow_insufficient_corpus:
            print(
                "После построения словаря корпус не прошел финальную проверку. "
                f"Отчет: {post_result.report_md_path}",
                file=sys.stderr,
            )
            return 3
    except Exception as exc:  # noqa: BLE001 - CLI преобразует ошибку в код завершения.
        print(f"Ошибка расширенного запуска главы 4: {exc}", file=sys.stderr)
        return 1

    print("Расширенный запуск главы 4 выполнен успешно.")
    print(f"Выбранное K: {run_result.selected_k}")
    print(f"Отчет достаточности корпуса: {post_result.report_md_path}")
    print(f"Итоговый отчет главы 4: {run_result.report_result.report_md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - точка входа проверяется через main().
    raise SystemExit(main())
