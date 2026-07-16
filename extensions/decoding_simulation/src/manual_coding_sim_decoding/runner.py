"""CLI изолированного расширения моделирования декодирования."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from manual_coding_sim_decoding.base_adapter import ManualCodingSimAdapterContract
from manual_coding_sim_decoding.config import load_decoding_extension_config
from manual_coding_sim_decoding.encoded_message import EncodedMessageBuilder
from manual_coding_sim_decoding.paths import DecodingExtensionPaths


def build_argument_parser() -> argparse.ArgumentParser:
    """Создать аргументы CLI расширения."""
    parser = argparse.ArgumentParser(
        description="Проверка и запуск изолированного расширения декодирования."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Корень проекта manual_coding_quality.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Путь к YAML-конфигурации расширения.",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Показать проверенную конфигурацию и разрешенные пути.",
    )
    parser.add_argument(
        "--check-base-contract",
        action="store_true",
        help="Проверить публичные интерфейсы и baseline-хэши базы.",
    )
    parser.add_argument(
        "--build-encoded-message",
        action="store_true",
        help="Сформировать демонстрационное материальное сообщение C.",
    )
    parser.add_argument(
        "--message-id",
        default="M_STAGE2_DEMO",
        help="Идентификатор демонстрационного исходного сообщения.",
    )
    parser.add_argument(
        "--output",
        default="reports/stage2/encoded_message_demo.json",
        help=(
            "Путь результата относительно extensions/decoding_simulation "
            "или абсолютный путь внутри этой папки."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Вывести машинно-читаемый результат в JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Выполнить выбранные проверки или построение сообщения C."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if not (
        args.show_config
        or args.check_base_contract
        or args.build_encoded_message
    ):
        parser.error(
            "Нужно указать --show-config, --check-base-contract "
            "и/или --build-encoded-message."
        )

    project_root = Path(args.project_root).resolve()
    config_path = _resolve_config_path(project_root, args.config)
    config = load_decoding_extension_config(config_path)
    paths = DecodingExtensionPaths.from_config(project_root, config)

    if args.show_config:
        _print_config(config, paths)

    contract_result = None
    if args.check_base_contract or args.build_encoded_message:
        contract_result = ManualCodingSimAdapterContract(project_root, config).check()
        if not contract_result.is_compatible:
            if args.json:
                print(
                    json.dumps(
                        contract_result.to_dict(),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                _print_contract_result(contract_result.to_dict())
            return 2

    if args.check_base_contract:
        if args.json and not args.build_encoded_message:
            print(
                json.dumps(
                    contract_result.to_dict(),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif not args.json:
            _print_contract_result(contract_result.to_dict())

    if args.build_encoded_message:
        payload, output_path = _build_encoded_message(
            paths=paths,
            config=config,
            message_id=args.message_id,
            output_value=args.output,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            metadata = payload["encoded_message"]["metadata"]
            print("Материальное кодированное сообщение C сформировано.")
            print(f"Файл результата: {output_path}")
            print(
                "Нормативных / материализованных элементов: "
                f"{metadata['normative_element_count']} / "
                f"{metadata['materialized_element_count']}"
            )
            print(
                "Остаточных ошибок: "
                f"{metadata['residual_error_count']}"
            )
            print(
                "Соответствие нормативному плану: "
                f"{metadata['is_normative_equivalent']}"
            )

    return 0


def _build_encoded_message(
    paths: DecodingExtensionPaths,
    config,
    message_id: str,
    output_value: str,
) -> tuple[dict[str, object], Path]:
    """Выполнить один базовый прогон и материализовать его результат в C."""
    from manual_coding_sim.protocol_simulator import ProtocolSimulator

    simulation_result = ProtocolSimulator().simulate_once(message_id=message_id)
    builder = EncodedMessageBuilder(config.material_encoding)
    result = builder.build_from_simulation_result(simulation_result)
    payload = result.to_dict()
    output_path = _resolve_output_path(paths, output_value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    return payload, output_path


def _resolve_config_path(project_root: Path, value: str) -> Path:
    """Разрешить абсолютный или относительный путь конфигурации."""
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _resolve_output_path(paths: DecodingExtensionPaths, value: str) -> Path:
    """Разрешить путь вывода и запретить запись вне папки расширения."""
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = paths.extension_root / candidate
    resolved = candidate.resolve()
    extension_root = paths.extension_root.resolve()
    if resolved != extension_root and extension_root not in resolved.parents:
        raise ValueError("Файл результата должен находиться в папке расширения.")
    return resolved


def _print_config(config, paths: DecodingExtensionPaths) -> None:
    """Показать ключевые параметры без изменения файлов проекта."""
    print("Конфигурация расширения успешно загружена.")
    print(f"Имя расширения: {config.extension.name}")
    print(f"Пакет: {config.extension.package_name}")
    print(f"Версия: {config.extension.version}")
    print(f"Зерно расширения: {config.extension.random_seed}")
    print(f"Базовый пакет: {config.base_contract.package_name}")
    print(f"Корень расширения: {paths.extension_root}")
    print(f"Каталог отчетов: {paths.reports_dir}")
    print(f"Каталог данных: {paths.data_dir}")
    print(f"Baseline-манифест: {paths.baseline_manifest}")
    print(
        "Префикс кодированного сообщения: "
        f"{config.material_encoding.encoded_message_prefix}"
    )
    print(
        "Смещение при позиционной ошибке: "
        f"{config.material_encoding.position_shift}"
    )


def _print_contract_result(result: dict[str, object]) -> None:
    """Показать итог проверки адаптера в читаемом виде."""
    status = "совместим" if result["is_compatible"] else "несовместим"
    print(f"Статус контракта с базовым пакетом: {status}.")
    print(f"Базовый пакет импортируется: {result['base_package_importable']}")
    print(f"Файл базового пакета: {result['imported_package_file']}")
    print(f"Baseline-манифест найден: {result['baseline_manifest_found']}")
    print(f"Отсутствующие модули: {result['missing_modules']}")
    print(f"Отсутствующие символы: {result['missing_symbols']}")
    print(f"Отсутствующие baseline-файлы: {result['missing_baseline_files']}")
    print(f"Измененные baseline-файлы: {result['changed_baseline_files']}")


if __name__ == "__main__":
    raise SystemExit(main())
