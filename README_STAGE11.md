# Этап 11. Диагностический анализ ошибок априорного прогноза

## Назначение

Этап выявляет сценарии и условия, связанные с наибольшими ошибками
зафиксированной модели главы 5. Фактические показатели используются только
для внешней проверки. Прогноз `q_pred`, веса, пороги и параметры модели не
изменяются.

## Изменяемые и создаваемые файлы

```text
src/manual_coding_sim/validation/prediction_error_analyzer.py
src/manual_coding_sim/validation/chapter6_runner.py
tests/validation/test_prediction_error_analyzer.py
README_STAGE11.md
STAGE11_MANIFEST.json
```

## Создаваемые расчетные артефакты

```text
reports/chapter6/top_prediction_errors.csv
reports/chapter6/error_group_analysis.csv
reports/chapter6/prediction_error_analysis.json
reports/chapter6/prediction_error_analysis.md
```

## Реализованные операции

1. Повторная проверка 150 строк, уникальности ключа
   `scenario_id + protocol_id`, числовых диапазонов и сохраненных классов.
2. Расчет `prediction_error = q_pred - q_fact`, абсолютной и квадратной
   ошибки.
3. Формирование top-10 по `absolute_error`.
4. Оценка связи signed- и absolute-error с:
   - `uncertainty_score`;
   - фактической длительностью;
   - числом ошибок, перепроверок и отказов;
   - фактической успешностью;
   - априорным уровнем шума;
   - априорным давлением времени.
5. Групповой анализ по:
   - фактическому и прогнозному классу;
   - доминирующему LDA-фактору;
   - квартилю неопределенности;
   - уровню шума и давления времени;
   - фактической успешности;
   - диапазонам фактических ошибок, перепроверок и отказов;
   - квартилю фактической длительности;
   - направлению ошибки.

## Контрольные результаты на расширенном корпусе

```text
Сценариев:                                      150
Top ошибок:                                      10
MAE:                                      0.1592441669
RMSE:                                     0.1944027110
Bias:                                    -0.1493986447
Максимальная абсолютная ошибка:           0.3766632080
Занижений:                                         132
Завышений:                                          18
Точных совпадений:                                   0
Доля суммарной абсолютной ошибки в top-10: 0.1488503378
```

Связь неопределенности с абсолютной ошибкой слабая:

```text
Pearson:  0.0859205578
Spearman: 0.1646704298
Kendall:  0.1017449664
```

Наиболее сильная диагностическая связь с абсолютной ошибкой получена для
фактической длительности выполнения:

```text
Spearman(fact_duration_sec, absolute_error) = 0.7468154140
```

Групповые результаты:

| Срез | Группа с наибольшей MAE | N | MAE | Bias |
|---|---|---:|---:|---:|
| Доминирующий фактор | `theta_0` | 34 | 0.2507012988 | -0.2507012988 |
| Прогнозный класс | `low` | 67 | 0.2584176927 | -0.2584176927 |
| Квартиль неопределенности | `Q3` | 37 | 0.1780086522 | -0.1780086522 |
| Давление времени | `low` | 72 | 0.2305493964 | -0.2289526569 |
| Фактическая успешность | `unsuccessful` | 103 | 0.2009176349 | -0.1951673397 |
| Число фактических ошибок | `6+` | 50 | 0.2632692218 | -0.2606411323 |
| Фактическая длительность | `Q4` | 38 | 0.2944493160 | -0.2944493160 |

## Команды Windows PowerShell

Все команды выполняются из корня проекта.

### 1. Применить архив

```powershell
Expand-Archive `
  -Path .\chapter6_stage11_error_analysis.zip `
  -DestinationPath . `
  -Force
```

### 2. Очистить кэш

```powershell
Get-ChildItem `
  .\src\manual_coding_sim `
  -Directory `
  -Recurse `
  -Filter __pycache__ |
Remove-Item -Recurse -Force

Remove-Item `
  .\.pytest_cache `
  -Recurse `
  -Force `
  -ErrorAction SilentlyContinue
```

### 3. Активировать окружение

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
```

### 4. Запустить этап 11

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --analyze-prediction-errors
```

Ожидаемый итог:

```text
Анализ ошибок априорного прогноза завершен.
Сценариев: 150
Сценариев в top-10: 10
Максимальная абсолютная ошибка: 0.3766632080
Занижений / завышений: 132 / 18
Spearman неопределенности с абсолютной ошибкой: 0.1646704298
Наиболее сильная диагностическая связь: fact_duration_sec = 0.7468154140
Наибольшая MAE по доминирующему фактору: theta_0 = 0.2507012988
Этап 11 выполнен. Переход к этапу 12 требует отдельного подтверждения.
```

### 5. Проверить JSON-отчет

```powershell
$report = Get-Content `
  .\reports\chapter6\prediction_error_analysis.json `
  -Raw `
  -Encoding UTF8 |
ConvertFrom-Json

$report.stage
$report.passed
$report.row_count
$report.top_error_count
$report.summary
$report.uncertainty_relation
$report.methodological_checks
```

Критические значения:

```text
stage = 11
passed = True
row_count = 150
top_error_count = 10
summary.underestimation_count = 132
summary.overestimation_count = 18
methodological_checks.chapter5_prediction_modified = False
methodological_checks.quality_thresholds_modified = False
```

### 6. Просмотреть top-10 ошибок

```powershell
Import-Csv `
  .\reports\chapter6\top_prediction_errors.csv |
Select-Object `
  error_rank,
  scenario_id,
  protocol_id,
  q_pred,
  q_fact,
  prediction_error,
  absolute_error,
  dominant_factor,
  uncertainty_score,
  fact_duration_sec,
  fact_error_count,
  fact_success |
Format-Table -AutoSize
```

### 7. Просмотреть групповой анализ

```powershell
Import-Csv `
  .\reports\chapter6\error_group_analysis.csv |
Sort-Object analysis_dimension, { [double]$_.mae } -Descending |
Select-Object `
  analysis_dimension,
  group,
  count,
  mae,
  rmse,
  bias,
  max_absolute_error |
Format-Table -AutoSize
```

Отдельно по доминирующему фактору:

```powershell
Import-Csv `
  .\reports\chapter6\error_group_analysis.csv |
Where-Object { $_.analysis_dimension -eq "dominant_factor" } |
Format-Table -AutoSize
```

### 8. Тесты этапа 11

```powershell
pytest `
  tests\validation\test_prediction_error_analyzer.py `
  -q
```

Ожидается:

```text
18 passed
```

### 9. Все тесты главы 6

```powershell
pytest tests\validation -q
```

Ожидается при текущем составе проекта:

```text
139 passed
```

### 10. Полная регрессия

```powershell
pytest -q
```

Ожидается при текущем составе проекта:

```text
456 passed
```

### 11. Повторный запуск после регрессии

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --analyze-prediction-errors
```

### 12. Совместный запуск этапов 10 и 11

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --bootstrap-analysis `
  --analyze-prediction-errors
```

Переход к этапу 12 выполняется только после успешного локального CLI,
тестов этапа, `tests/validation`, полной регрессии и повторного запуска.
