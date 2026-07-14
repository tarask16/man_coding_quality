# Этап 33 — рисунок 6.7

## Назначение

Этап формирует группированную столбчатую диаграмму MAE и RMSE для четырёх вариантов прогноза:

- `mean baseline`;
- `prior-only baseline`;
- полной модели главы 5;
- `theta-only baseline`.

Выходная основа:

```text
reports/chapter6/figures/baseline_comparison
```

Меньшие значения MAE и RMSE трактуются как лучшие.

## Входные данные

```text
reports/chapter6/baseline_comparison.csv
```

Генератор поддерживает распространённые имена колонки модели:

```text
model
model_name
baseline
baseline_name
prediction_model
```

Обязательные метрики:

```text
mae
rmse
```

Поддерживаются алиасы имён моделей, включая `mean_baseline`, `prior_only`, `full_model`, `q_pred` и `theta_only`.

## Результаты расширенного корпуса

| Модель | MAE | RMSE |
|---|---:|---:|
| Mean baseline | 0,097227 | 0,112903 |
| Prior-only baseline | 0,111822 | 0,132113 |
| Полная модель главы 5 | 0,159244 | 0,194403 |
| Theta-only baseline | 0,304125 | 0,360544 |

Основные выводы:

- `mean baseline` имеет минимальные MAE и RMSE;
- `prior-only` занимает второе место;
- полная модель занимает третье место;
- полная модель лучше `theta-only`;
- полная модель хуже `mean baseline` и `prior-only` по обеим абсолютным метрикам;
- относительно `mean baseline` MAE полной модели выше на 63,8 %, RMSE — на 72,2 %;
- добавление латентного профиля в текущей схеме не обеспечило универсального снижения абсолютной ошибки.

Рисунок не отменяет ранее установленную высокую ранговую согласованность полной модели. MAE и RMSE характеризуют абсолютное приближение, а не сохранение порядка сценариев.

## Созданные файлы

```text
src/manual_coding_sim/dissertation_figures/figure_6_7_baseline_comparison.py
tests/dissertation_figures/test_figure_6_7_baseline_comparison.py
reports/chapter6/figures/baseline_comparison.png
reports/chapter6/figures/baseline_comparison.svg
docs/dissertation_figures_roadmap.md
README_FIGURE_STAGE_33.md
```

## Генерация

```powershell
python -m manual_coding_sim.dissertation_figures.figure_6_7_baseline_comparison `
  --project-root . `
  --dpi 300
```

Явное указание входного файла:

```powershell
python -m manual_coding_sim.dissertation_figures.figure_6_7_baseline_comparison `
  --project-root . `
  --input .\reports\chapter6\baseline_comparison.csv `
  --dpi 300
```

## Локальные тесты

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_6_7_baseline_comparison.py `
  -q
```

Ожидается:

```text
10 passed
```

Полная локальная проверка рисунков:

```powershell
python -m pytest tests\dissertation_figures -q
```

При сохранении ранее подтверждённых результатов ожидается:

```text
291 passed
```

## Выполненная проверка в рабочем окружении

Тесты не запускались. Проверены:

- успешное формирование PNG и SVG;
- размер PNG `6492 x 2234 px` при 300 dpi;
- наличие редактируемых подписей в SVG;
- корректное отображение четырёх моделей и двух метрик;
- выделение полной модели штриховкой;
- выделение лучшего абсолютного результата;
- читаемость диагностических карточек;
- отсутствие критических наложений, обрезки и выхода текста за панели.
