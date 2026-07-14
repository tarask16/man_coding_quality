# Этап 27. Рисунок 6.1 — Q_pred против Q_fact

## Назначение

Этап формирует первый рисунок главы 6 для внешней проверки интегрального априорного индекса. Диаграмма сопоставляет `q_pred` из главы 5 с фактическим `integral_quality` расширенного корпуса.

## Входные артефакты

```text
reports/chapter5/q_pred.csv
data/processed/quality_targets.csv
```

Объединение выполняется в режиме `one_to_one` по составному ключу:

```text
scenario_id
protocol_id
```

## Содержание рисунка

Основная панель содержит:

- 150 пар `Q_pred` и `Q_fact`;
- линию идеального соответствия `y = x`;
- линейную тенденцию фактического качества относительно прогноза;
- среднюю точку корпуса;
- сценарий с максимальной абсолютной ошибкой.

Правая панель содержит:

- Pearson и Spearman;
- MAE, RMSE, медианную и максимальную абсолютную ошибку;
- Bias и средние значения `Q_pred`, `Q_fact`;
- численность недооценённых и переоценённых сценариев.

Методическая интерпретация разделяет сравнительную информативность индекса и абсолютную калибровку. Высокая корреляция не устраняет систематическое смещение относительно линии `y = x`.

## Фактические результаты расширенного корпуса

```text
N = 150
Pearson = 0.892053
Spearman = 0.881769
MAE = 0.159244
RMSE = 0.194403
Bias = -0.149399
Median |e| = 0.145240
Max |e| = 0.376663
Q_pred < Q_fact = 132
Q_pred > Q_fact = 18
```

Линейная тенденция:

```text
Q_fact = 0.469944 * Q_pred + 0.413814
```

## Созданные файлы

```text
src/manual_coding_sim/dissertation_figures/
    figure_6_1_q_pred_vs_q_fact.py

tests/dissertation_figures/
    test_figure_6_1_q_pred_vs_q_fact.py

reports/chapter6/figures/
    q_pred_vs_q_fact.png
    q_pred_vs_q_fact.svg

docs/
    dissertation_figures_roadmap.md

README_FIGURE_STAGE_27.md
```

## Генерация

```powershell
python -m manual_coding_sim.dissertation_figures.figure_6_1_q_pred_vs_q_fact `
  --project-root . `
  --dpi 300
```

Явное указание входов:

```powershell
python -m manual_coding_sim.dissertation_figures.figure_6_1_q_pred_vs_q_fact `
  --project-root . `
  --q-pred .\reports\chapter5\q_pred.csv `
  --q-fact .\data\processed\quality_targets.csv `
  --dpi 300
```

## Локальная проверка

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_6_1_q_pred_vs_q_fact.py `
  -q
```

Подготовлено 10 тестов. В рабочем окружении этапа они не запускались в соответствии с установленным правилом.
