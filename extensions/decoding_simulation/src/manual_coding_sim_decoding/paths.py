"""Пути изолированного расширения декодирования."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from manual_coding_sim_decoding.config import DecodingExtensionConfig


@dataclass(frozen=True)
class DecodingExtensionPaths:
    """Абсолютные пути расширения, вычисленные от корня проекта."""

    project_root: Path
    extension_root: Path
    reports_dir: Path
    data_dir: Path
    manifests_dir: Path
    baseline_manifest: Path

    @classmethod
    def from_config(
        cls,
        project_root: str | Path,
        config: DecodingExtensionConfig,
    ) -> "DecodingExtensionPaths":
        """Построить и проверить пути по конфигурации."""
        root = Path(project_root).resolve()
        extension_root = root / "extensions" / "decoding_simulation"
        instance = cls(
            project_root=root,
            extension_root=extension_root,
            reports_dir=root / config.paths.reports_dir,
            data_dir=root / config.paths.data_dir,
            manifests_dir=root / config.paths.manifests_dir,
            baseline_manifest=root / config.base_contract.baseline_manifest,
        )
        instance.validate()
        return instance

    def validate(self) -> None:
        """Запретить вывод артефактов за пределы папки расширения."""
        extension_root = self.extension_root.resolve()
        for path in (self.reports_dir, self.data_dir, self.manifests_dir):
            resolved = path.resolve()
            if extension_root != resolved and extension_root not in resolved.parents:
                raise ValueError(
                    f"Путь {resolved} находится за пределами папки расширения."
                )

    def ensure_output_directories(self) -> None:
        """Создать только разрешенные выходные каталоги расширения."""
        for path in (self.reports_dir, self.data_dir, self.manifests_dir):
            path.mkdir(parents=True, exist_ok=True)
