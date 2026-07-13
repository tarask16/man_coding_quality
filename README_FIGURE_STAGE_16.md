# Этап 16 — рисунок 4.2

## Назначение

Этап формирует составной график документной частоты токенов итогового словаря `LDA_prior`.

Рисунок включает:

- ранжированное распределение `document_frequency` для всех 96 токенов;
- границы фильтрации `df_min = 2` и `df_max_ratio = 0,95`;
- гистограмму частот итогового словаря;
- сводку наблюдаемого диапазона и размера словаря.

Генератор использует фактические артефакты:

- `data/processed/lda/dictionary.json`;
- `data/processed/lda/corpus_metadata.json`.

## Создаваемые файлы

- `src/manual_coding_sim/dissertation_figures/figure_4_2_token_frequency_and_filtering.py`;
- `tests/dissertation_figures/test_figure_4_2_token_frequency_and_filtering.py`;
- `reports/chapter4/figures/figure_4_2_token_frequency_and_filtering.png`;
- `reports/chapter4/figures/figure_4_2_token_frequency_and_filtering.svg`.

## Генерация

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m manual_coding_sim.dissertation_figures.figure_4_2_token_frequency_and_filtering `
  --project-root . `
  --dpi 300
```

Для явно заданных входных файлов:

```powershell
python -m manual_coding_sim.dissertation_figures.figure_4_2_token_frequency_and_filtering `
  --project-root . `
  --dictionary .\data\processed\lda\dictionary.json `
  --metadata .\data\processed\lda\corpus_metadata.json `
  --dpi 300
```

## Локальные тесты этапа

Тесты подготовлены, но при формировании этапа автоматически не запускаются. Их выполняет пользователь локально:

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_4_2_token_frequency_and_filtering.py `
  -q
```

Полная локальная регрессия генераторов:

```powershell
python -m pytest tests\dissertation_figures -q
```

## Методическое ограничение

`dictionary.json` содержит уже отфильтрованный итоговый словарь. Поэтому рисунок достоверно показывает частоты 96 сохранённых токенов и их положение относительно заданных границ, но не используется для определения количества кандидатов, отброшенных до сохранения словаря.
