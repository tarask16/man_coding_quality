# Этап 20. Рисунок 4.6 — распределение профилей theta_prior

## Назначение

Этап формирует составной рисунок для главы 4:

- boxplot компонент `theta_0`, `theta_1`, `theta_2` по всем сценариям;
- отметки среднего и медианы каждой компоненты;
- столбчатую диаграмму численности и долей доминирующих факторов;
- методическое примечание об относительной, а не причинной интерпретации профиля.

Доминирующая тема определяется как `argmax(theta_0, theta_1, theta_2)`.

## Входные данные

По умолчанию используется:

```text
reports/chapter4/theta_prior.csv
```

Обязательные колонки:

```text
scenario_id
protocol_id
theta_0
theta_1
theta_2
selected_k
```

Генератор проверяет:

- непустой набор строк;
- уникальность `scenario_id` и `protocol_id`;
- `selected_k = 3`;
- принадлежность компонент диапазону `[0; 1]`;
- равенство суммы `theta_0 + theta_1 + theta_2` единице с численным допуском.

## Фактические показатели расширенного корпуса

```text
N = 150
mean(theta_0) = 0.2654
mean(theta_1) = 0.3151
mean(theta_2) = 0.4195
```

Доминирующие факторы:

```text
theta_0: 34 сценария, 22.7 %
theta_1: 46 сценариев, 30.7 %
theta_2: 70 сценариев, 46.7 %
```

## Формируемые файлы

```text
reports/chapter4/figures/figure_4_6_theta_profile_distribution.png
reports/chapter4/figures/figure_4_6_theta_profile_distribution.svg
```

## Команда генерации

```powershell
python -m manual_coding_sim.dissertation_figures.figure_4_6_theta_profile_distribution `
  --project-root . `
  --dpi 300
```

При необходимости входной файл задаётся явно:

```powershell
python -m manual_coding_sim.dissertation_figures.figure_4_6_theta_profile_distribution `
  --project-root . `
  --input .\reports\chapter4\theta_prior.csv `
  --dpi 300
```

## Локальные тесты

Тесты подготовлены, но в рабочем окружении этапа не запускались.

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_4_6_theta_profile_distribution.py `
  -q
```

Ожидается 10 успешно пройденных тестов.

Полная локальная проверка:

```powershell
python -m pytest tests\dissertation_figures -q
python -m pytest -q
```
