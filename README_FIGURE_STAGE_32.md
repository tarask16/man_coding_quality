# Этап 32 — рисунок 6.6

## Назначение

Этап формирует интервальный график сценариев, отсортированных по фактическому интегральному качеству. Рисунок показывает:

- диагностические интервалы `[q_pred_lower; q_pred_upper]`;
- центральный прогноз `q_pred`;
- фактическое значение `integral_quality`;
- покрытые наблюдения;
- промахи выше интервала;
- промахи ниже интервала;
- направленное расстояние фактического значения до ближайшей границы;
- сводные показатели покрытия и ширины интервалов.

Выходная основа:

```text
reports/chapter6/figures/prediction_intervals
```

## Входные данные

```text
reports/chapter5/prediction_uncertainty.csv
data/processed/quality_targets.csv
```

Объединение выполняется в режиме `one_to_one` по:

```text
scenario_id
protocol_id
```

## Условие покрытия

```text
q_pred_lower <= integral_quality <= q_pred_upper
```

Статусы:

- `covered` — фактическое значение находится внутри интервала;
- `miss_above` — фактическое значение выше `q_pred_upper`;
- `miss_below` — фактическое значение ниже `q_pred_lower`.

## Результаты расширенного корпуса

```text
N = 150
covered = 28
coverage_rate = 0.186667
miss_above = 111
miss_below = 11
mean_interval_width = 0.084183
median_interval_width = 0.097432
mean_distance_all = 0.121179
mean_distance_misses = 0.148990
max_distance = 0.353099
max_distance_scenario = scn_0041
```

Низкое эмпирическое покрытие подтверждает, что интервалы следует трактовать как диагностические. Они не являются абсолютно откалиброванными доверительными или предиктивными интервалами.

## Созданные файлы

```text
src/manual_coding_sim/dissertation_figures/figure_6_6_prediction_intervals.py
tests/dissertation_figures/test_figure_6_6_prediction_intervals.py
reports/chapter6/figures/prediction_intervals.png
reports/chapter6/figures/prediction_intervals.svg
docs/dissertation_figures_roadmap.md
README_FIGURE_STAGE_32.md
```

## Генерация

```powershell
python -m manual_coding_sim.dissertation_figures.figure_6_6_prediction_intervals `
  --project-root . `
  --dpi 300
```

Явное указание входных файлов:

```powershell
python -m manual_coding_sim.dissertation_figures.figure_6_6_prediction_intervals `
  --project-root . `
  --intervals .\reports\chapter5\prediction_uncertainty.csv `
  --q-fact .\data\processed\quality_targets.csv `
  --dpi 300
```

## Локальные тесты

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_6_6_prediction_intervals.py `
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

При сохранении ранее подтвержденных результатов ожидается:

```text
281 passed
```

## Выполненная проверка в рабочем окружении

Тесты не запускались. Проверены:

- успешное формирование PNG и SVG;
- размер PNG `6015 x 2590 px` при 300 dpi;
- наличие редактируемых подписей в SVG;
- корректное отображение покрытых и непокрытых наблюдений;
- раздельное представление промахов выше и ниже интервала;
- отсутствие критических наложений, обрезки и выхода текста за панели.
