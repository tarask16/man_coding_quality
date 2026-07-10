"""Экспериментальные сценарии и корпусные запускатели диссертации.

Пакет содержит вспомогательные средства для формирования воспроизводимых
экспериментальных корпусов главы 3. Эти корпуса затем используются модулем
``manual_coding_sim.lda`` в главе 4.
"""

from manual_coding_sim.experiments.extended_corpus_plan import (
    ExtendedCorpusPlanBuilder,
    ExtendedCorpusPlanConfig,
    ExtendedScenario,
)
from manual_coding_sim.experiments.extended_corpus_runner import (
    ExtendedCorpusGenerationResult,
    ExtendedCorpusOutputPaths,
    ExtendedCorpusRunner,
    ExtendedCorpusRunnerConfig,
)

__all__ = [
    "ExtendedCorpusGenerationResult",
    "ExtendedCorpusOutputPaths",
    "ExtendedCorpusPlanBuilder",
    "ExtendedCorpusPlanConfig",
    "ExtendedCorpusRunner",
    "ExtendedCorpusRunnerConfig",
    "ExtendedScenario",
]
