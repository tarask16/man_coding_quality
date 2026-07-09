"""CLI-запуск программного блока главы 4.

Модуль позволяет запускать полный LDA-pipeline из командной строки:
``python -m manual_coding_sim.lda.chapter4_cli --config configs/chapter4_lda.yaml``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from manual_coding_sim.lda.chapter4_runner import Chapter4LdaRunner
from manual_coding_sim.lda.config_loader import (
    Chapter4ConfigLoader,
    Chapter4ConfigOverrides,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Создать парсер аргументов командной строки."""

    parser = argparse.ArgumentParser(
        description="Запуск LDA-модуля главы 4 диссертационного исследования.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Путь к YAML-конфигурации главы 4.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Корень проекта, относительно которого разрешаются пути конфигурации.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Разрешить перезапись выходных артефактов независимо от YAML.",
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить CLI-запуск главы 4 и вернуть код завершения."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    loader = Chapter4ConfigLoader()
    try:
        load_result = loader.load(
            args.config,
            Chapter4ConfigOverrides(
                overwrite=True if args.overwrite else None,
                skip_diagnostic=args.skip_diagnostic,
                selected_k=args.selected_k,
            ),
        )
        runner_config = load_result.runner_config
        project_root = Path(args.project_root)
        loader.validate_required_inputs(runner_config, project_root=project_root)
        result = Chapter4LdaRunner(runner_config).run(project_root=project_root)
    except Exception as exc:  # noqa: BLE001 - CLI должен преобразовать любую ошибку в код 1.
        print(f"Ошибка запуска главы 4: {exc}", file=sys.stderr)
        return 1

    print("Глава 4: LDA-pipeline выполнен успешно.")
    print(f"Выбранное K: {result.selected_k}")
    print(f"Итоговый JSON-отчет: {result.report_result.report_json_path}")
    print(f"Итоговый Markdown-отчет: {result.report_result.report_md_path}")
    if result.diag_result is None and result.full_result is None:
        print("Диагностические модели: отключены")
    else:
        print("Диагностические модели: построены")
    return 0


if __name__ == "__main__":  # pragma: no cover - точка входа проверяется через main().
    raise SystemExit(main())
