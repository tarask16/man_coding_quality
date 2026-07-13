# Этап 24. Рисунок 5.4

## Назначение

Генератор строит распределение интегрального априорного показателя `Q_pred` по данным `reports/chapter5/q_pred.csv`.

На рисунке показаны:

- гистограмма `Q_pred` в общей шкале `[0; 1]`;
- пороговые границы `0,45` и `0,70`;
- классы `low`, `medium`, `high`;
- численность и доля сценариев каждого класса;
- среднее, медиана, квартильный диапазон, минимум, максимум и стандартное отклонение.

Границы классов:

```text
low:    Q_pred < 0,45
medium: 0,45 <= Q_pred < 0,70
high:   Q_pred >= 0,70
```

## Генерация

```powershell
python -m manual_coding_sim.dissertation_figures.figure_5_4_q_pred_distribution `
  --project-root . `
  --dpi 300
```

При необходимости входной файл задаётся явно:

```powershell
python -m manual_coding_sim.dissertation_figures.figure_5_4_q_pred_distribution `
  --project-root . `
  --input .\reports\chapter5\q_pred.csv `
  --dpi 300
```

## Выходные файлы

```text
reports/chapter5/figures/figure_5_4_q_pred_distribution.png
reports/chapter5/figures/figure_5_4_q_pred_distribution.svg
```

## Локальная проверка

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_5_4_q_pred_distribution.py `
  -q
```

Тесты в рабочем окружении подготовки этапа не запускаются.

## Фактический результат расширенного корпуса

Для текущего `q_pred.csv` из 150 сценариев:

```text
minimum = 0.103445
Q1      = 0.304780
median  = 0.494024
mean    = 0.498844
Q3      = 0.688963
maximum = 0.889302
std     = 0.214068
```

Распределение классов:

```text
low    = 67 сценариев (44,7 %)
medium = 49 сценариев (32,7 %)
high   = 34 сценария   (22,7 %)
```
