# Этап 31. Рисунок 6.5 — матрица ошибок классов качества

## Назначение

Генератор строит абсолютную матрицу ошибок для классов интегрального качества `low`, `medium` и `high`. Строки соответствуют фактическому классу, столбцы — прогнозному классу.

Пороговые правила:

- `low`: `Q < 0.45`;
- `medium`: `0.45 <= Q < 0.70`;
- `high`: `Q >= 0.70`.

## Входные данные

- `reports/chapter5/q_pred.csv` — колонка `q_pred`;
- `data/processed/quality_targets.csv` — колонка `integral_quality`;
- объединение выполняется по `scenario_id` и `protocol_id` в режиме one-to-one.

## Выходные файлы

- `reports/chapter6/figures/confusion_matrix.png`;
- `reports/chapter6/figures/confusion_matrix.svg`.

## Фактический результат расширенного корпуса

Матрица в порядке строк и столбцов `low`, `medium`, `high`:

```text
[[ 1,  0,  0],
 [66, 33,  1],
 [ 0, 16, 33]]
```

Сводка:

- сценариев: 150;
- точных совпадений: 67, или 44,7 %;
- соседних ошибок: 83;
- критических ошибок `low -> high`: 0;
- критических ошибок `high -> low`: 0.

Отсутствие критических переходов не заменяет анализ непрерывных ошибок и абсолютной калибровки `Q_pred`.

## Генерация

```powershell
python -m manual_coding_sim.dissertation_figures.figure_6_5_confusion_matrix `
  --project-root . `
  --dpi 300
```

## Локальные тесты

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_6_5_confusion_matrix.py `
  -q
```

Ожидается 10 пройденных тестов.

Полная локальная проверка:

```powershell
python -m pytest tests\dissertation_figures -q
python -m pytest -q
```

Тесты в рабочем окружении подготовки этапа не запускались.
