# Этап 15 — рисунок 4.1

## Назначение

Этап формирует блок-схему конвейера построения основной априорной модели `LDA_prior`:

`prior_features.csv → дискретизация → token_map.json → corpus_prior.csv → dictionary.json → LDA_prior → theta_prior.csv / topic_word.csv`.

На рисунке отражены фактические параметры расширенного корпуса: 150 документов, 96 токенов словаря, выбранное число факторов `K = 3`, `random_state = 11`.

## Создаваемые файлы

- `src/manual_coding_sim/dissertation_figures/figure_4_1_lda_pipeline.py`;
- `tests/dissertation_figures/test_figure_4_1_lda_pipeline.py`;
- `reports/chapter4/figures/figure_4_1_lda_pipeline.png`;
- `reports/chapter4/figures/figure_4_1_lda_pipeline.svg`.

## Генерация

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m manual_coding_sim.dissertation_figures.figure_4_1_lda_pipeline `
  --project-root . `
  --dpi 300
```

## Локальные тесты этапа

Тесты подготовлены, но в рамках формирования этапа автоматически не запускаются. Их выполняет пользователь локально:

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_4_1_lda_pipeline.py `
  -q
```

Полная локальная регрессия генераторов:

```powershell
python -m pytest tests\dissertation_figures -q
```

## Методическое ограничение

Основная модель `LDA_prior` обучается только по `prior_features.csv`. `fact_features.csv`, `quality_targets.csv` и диагностические расширения не могут участвовать в построении корпуса, словаря, матрицы и параметров основной априорной модели.
