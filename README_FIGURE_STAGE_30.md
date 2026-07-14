# Этап 30. Рисунок 6.4

## Назначение

Рисунок `partial_criteria_comparison` сопоставляет прогнозные и фактические значения шести частных критериев качества по четырём группам показателей:

- MAE;
- RMSE;
- Bias;
- ранговая корреляция Spearman.

## Входные данные

```text
reports/chapter5/q_pred_components.csv
data/processed/quality_targets.csv
```

Таблицы объединяются в режиме `one_to_one` по ключам:

```text
scenario_id
protocol_id
```

## Формируемые артефакты

```text
reports/chapter6/figures/partial_criteria_comparison.png
reports/chapter6/figures/partial_criteria_comparison.svg
```

## Генерация

```powershell
python -m manual_coding_sim.dissertation_figures.figure_6_4_partial_criteria_comparison `
  --project-root . `
  --dpi 300
```

## Локальная проверка

```powershell
python -m pytest `
  tests\dissertation_figures\test_figure_6_4_partial_criteria_comparison.py `
  -q
```

В рабочем окружении подготовки этапа тесты не запускались. Подготовлено 10 локальных тестов.

## Фактические результаты расширенного корпуса

| Критерий | MAE | RMSE | Bias | Spearman |
|---|---:|---:|---:|---:|
| `q_acc` | 0,208927 | 0,254995 | −0,190562 | 0,858321 |
| `q_time` | 0,170228 | 0,225267 | −0,112615 | 0,594474 |
| `q_effort` | 0,156064 | 0,184929 | −0,047225 | 0,611719 |
| `q_res` | 0,205666 | 0,256061 | −0,101341 | 0,632817 |
| `q_rep` | 0,224452 | 0,274526 | −0,157280 | 0,598172 |
| `q_fit` | 0,344288 | 0,388426 | −0,344288 | 0,873274 |

Средние показатели:

```text
mean MAE       = 0,218271
mean RMSE      = 0,264034
mean Bias      = −0,158885
mean Spearman  = 0,694796
```

Минимальная MAE получена для `q_effort`, максимальная — для `q_fit`. Критерий `q_fit` одновременно имеет наибольшую Spearman и наиболее выраженное абсолютное занижение. Это подчёркивает различие ранговой информативности и абсолютной калибровки.
