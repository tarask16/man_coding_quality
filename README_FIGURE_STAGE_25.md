# Этап 25. Рисунок 5.5

## Назначение

Генератор строит диаграмму рассеяния `uncertainty_score` и `interval_radius` с маргинальными распределениями по данным:

```text
reports/chapter5/prediction_uncertainty.csv
reports/chapter5/prediction_uncertainty_report.json
```

Используется зависимость:

```text
interval_radius = delta * uncertainty_score
```

Для текущей конфигурации главы 5:

```text
delta = 0.15
```

Показатель неопределённости формируется по выражению:

```text
U = 0.50 * H_theta
  + 0.30 * (1 - mean_stability)
  + 0.20 * input_missing_share
```

На рисунке показаны:

- scatter-график `uncertainty_score` и `interval_radius`;
- теоретическая линия `r = 0,15 * U`;
- средняя точка корпуса;
- верхняя гистограмма `uncertainty_score`;
- правая гистограмма `interval_radius`;
- средние и медианы двух распределений;
- параметры расчёта и методическое ограничение.

Линейная зависимость задана алгоритмом и не является доказательством абсолютной калибровки или вероятностного покрытия интервала.

## Генерация

```powershell
python -m manual_coding_sim.dissertation_figures.figure_5_5_uncertainty_and_interval_radius `
  --project-root . `
  --dpi 300
```

При необходимости входные файлы задаются явно:

```powershell
python -m manual_coding_sim.dissertation_figures.figure_5_5_uncertainty_and_interval_radius `
  --project-root . `
  --input .\reports\chapter5\prediction_uncertainty.csv `
  --report .\reports\chapter5\prediction_uncertainty_report.json `
  --dpi 300
```

## Выходные файлы

```text
reports/chapter5/figures/figure_5_5_uncertainty_and_interval_radius.png
reports/chapter5/figures/figure_5_5_uncertainty_and_interval_radius.svg
```

## Локальная проверка

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_5_5_uncertainty_and_interval_radius.py `
  -q
```

Тесты в рабочем окружении подготовки этапа не запускаются.

## Фактический результат расширенного корпуса

Для 150 сценариев:

```text
uncertainty_score:
minimum = 0.099417
Q1      = 0.104548
median  = 0.324773
mean    = 0.280611
Q3      = 0.379353
maximum = 0.538694
std     = 0.136865

interval_radius:
minimum = 0.014913
Q1      = 0.015682
median  = 0.048716
mean    = 0.042092
Q3      = 0.056903
maximum = 0.080804
std     = 0.020530
```

Параметры:

```text
delta                = 0.15
mean_stability       = 0.84885
input_missing_share  = 0.0
Pearson(U, r)         = 1.0 по построению
```
