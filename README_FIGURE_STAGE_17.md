# Этап 17 — рисунок 4.3

## Назначение

Этап формирует комбинированный график выбора числа латентных факторов `K` для модели `LDA_prior`.

Рисунок включает четыре согласованные панели для `K = 3…8`:

- `perplexity` — меньшие значения предпочтительнее;
- `mean_coherence` — большие значения предпочтительнее;
- `topic_diversity` — большие значения предпочтительнее;
- нормированный `selection_score` — итоговый критерий выбора.

Рекомендованное значение `K = 3` выделяется единой вертикальной областью и специальным маркером во всех панелях.

Генератор использует фактический отчёт:

- `reports/chapter4/k_selection_report.csv`.

## Фактический результат выбора

Для `K = 3`:

- `perplexity = 94,8173`;
- `mean_coherence = -0,5300`;
- `topic_diversity = 0,9667`;
- `selection_score = 1,000`;
- `is_recommended = true`.

## Создаваемые файлы

- `src/manual_coding_sim/dissertation_figures/figure_4_3_k_selection_metrics.py`;
- `tests/dissertation_figures/test_figure_4_3_k_selection_metrics.py`;
- `reports/chapter4/figures/figure_4_3_k_selection_metrics.png`;
- `reports/chapter4/figures/figure_4_3_k_selection_metrics.svg`.

## Генерация

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m manual_coding_sim.dissertation_figures.figure_4_3_k_selection_metrics `
  --project-root . `
  --dpi 300
```

Для явно заданного отчёта:

```powershell
python -m manual_coding_sim.dissertation_figures.figure_4_3_k_selection_metrics `
  --project-root . `
  --input .\reports\chapter4\k_selection_report.csv `
  --dpi 300
```

## Локальные тесты этапа

Тесты подготовлены, но при формировании этапа автоматически не запускаются. Их выполняет пользователь локально:

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_4_3_k_selection_metrics.py `
  -q
```

Полная локальная регрессия генераторов:

```powershell
python -m pytest tests\dissertation_figures -q
```

## Методические ограничения

Отрицательное значение `mean_coherence` не трактуется как ошибка вычисления: для применённой логарифмической меры содержательно используется относительное сравнение кандидатных моделей. Нормированный `selection_score` агрегирует метрики выбора структуры LDA и не является показателем качества ручного кодирования или внешней валидности модели.
