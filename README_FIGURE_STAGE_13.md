# Этап 13. Рисунок 3.5 — проверка достаточности корпуса

## Назначение

Генератор формирует комбинированную диаграмму фактических и минимально допустимых показателей расширенного вычислительного корпуса.

На рисунке отображаются:

- число документов;
- число уникальных сценариев;
- число уникальных протоколов;
- число токенов словаря LDA;
- число уровней сложности сообщения;
- число уровней критичности сообщения;
- число уникальных значений расчётного времени оператора;
- число уникальных значений скорректированного времени условий.

## Входные данные

```text
data/processed/prior_features.csv
```

Число токенов словаря передаётся аргументом `--token-count`; значение по умолчанию — 96.

## Выходные артефакты

```text
reports/chapter3/figures/figure_3_5_corpus_sufficiency.png
reports/chapter3/figures/figure_3_5_corpus_sufficiency.svg
```

## Запуск

```powershell
python -m manual_coding_sim.dissertation_figures.figure_3_5_corpus_sufficiency `
  --project-root . `
  --token-count 96 `
  --dpi 300
```

## Тесты этапа

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_3_5_corpus_sufficiency.py `
  -q
```
