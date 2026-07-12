# Этап 6. Проверка частных прогнозных критериев

## Назначение

Этап выполняет внешнюю проверку шести частных прогнозных критериев,
сформированных в главе 5. Для каждого сценария сопоставляются пары:

```text
q_acc_pred    -> q_acc
q_time_pred   -> q_time
q_effort_pred -> q_effort
q_res_pred    -> q_res
q_rep_pred    -> q_rep
q_fit_pred    -> q_fit
```

Ошибка каждого частного критерия определяется как:

```text
criterion_error = q_j_pred - q_j_fact
```

Фактические критерии используются только для внешней проверки. На этапе 6
не изменяются веса, нормировки, направления латентных факторов, пороги
классов и формулы частных прогнозов главы 5.

## Рассчитываемые метрики

Для каждой пары рассчитываются:

```text
MAE
RMSE
Bias
Pearson
Spearman
Kendall tau-b
R²
прогнозное среднее
фактическое среднее
прогнозное стандартное отклонение
фактическое стандартное отклонение
максимальная абсолютная ошибка
```

Дополнительно формируется сводка с лучшим и худшим критерием по MAE и
Spearman.

## Создаваемые артефакты

```text
reports/chapter6/partial_criteria_validation.csv
reports/chapter6/partial_criteria_validation_report.json
reports/chapter6/partial_criteria_validation_report.md
```

## Файлы этапа

```text
src/manual_coding_sim/validation/partial_criteria_validator.py
src/manual_coding_sim/validation/chapter6_runner.py
tests/validation/test_partial_criteria_validator.py
README_STAGE6.md
STAGE6_MANIFEST.json
```

Файлы этапов 1–5 не заменяются, за исключением расширения общего CLI-файла
`chapter6_runner.py` новым флагом этапа 6.

## Запуск из Windows PowerShell

```powershell
Set-Location D:\Projects\man_coding_quality

Expand-Archive `
  -Path .\chapter6_stage6_partial_criteria.zip `
  -DestinationPath . `
  -Force

.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"

python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --validate-partial-criteria
```

## Ожидаемый результат для расширенного корпуса

```text
Сценариев: 150
Проверено критериев: 6
Среднее MAE: 0.2182707508
Средний Spearman: 0.6947963809
Наименьший MAE: q_effort = 0.1560640872
Наибольший MAE: q_fit = 0.3442877621
```

Основные результаты по критериям:

| Критерий | MAE | RMSE | Bias | Spearman | Kendall | R² |
|---|---:|---:|---:|---:|---:|---:|
| `q_acc` | 0.208927 | 0.254995 | -0.190562 | 0.858321 | 0.691816 | -3.453525 |
| `q_time` | 0.170228 | 0.225267 | -0.112615 | 0.594474 | 0.419060 | -3.331208 |
| `q_effort` | 0.156064 | 0.184929 | -0.047225 | 0.611719 | 0.455609 | -0.694342 |
| `q_res` | 0.205666 | 0.256061 | -0.101341 | 0.632817 | 0.518413 | 0.278714 |
| `q_rep` | 0.224452 | 0.274526 | -0.157280 | 0.598172 | 0.460266 | -0.316734 |
| `q_fit` | 0.344288 | 0.388426 | -0.344288 | 0.873274 | 0.690201 | -45.852658 |

`q_effort` имеет наименьшую абсолютную ошибку. `q_fit` лучше всего
сохраняет ранжирование сценариев, но имеет максимальную ошибку и сильное
систематическое занижение шкалы. Отрицательные значения R² являются
допустимыми результатами внешней проверки и не считаются программной
ошибкой.

## Проверка JSON-отчета

```powershell
$report = Get-Content `
  .\reports\chapter6\partial_criteria_validation_report.json `
  -Raw | ConvertFrom-Json

$report.stage
$report.passed
$report.row_count
$report.criterion_count
$report.error_definition
$report.summary
$report.metrics | Format-Table
```

Критические значения:

```text
stage = 6
passed = True
row_count = 150
criterion_count = 6
error_definition = criterion_error = q_j_pred - q_j_fact
```

## Проверка CSV-таблицы

```powershell
$metrics = @(
  Import-Csv .\reports\chapter6\partial_criteria_validation.csv
)

Write-Host "Критериев:" $metrics.Count

$metrics |
Sort-Object { [double]$_.mae } |
Select-Object `
  criterion,
  predicted_column,
  factual_column,
  mae,
  rmse,
  bias,
  spearman,
  kendall,
  r2 |
Format-Table -AutoSize
```

Ожидается:

```text
Критериев: 6
```

## Тесты этапа

```powershell
pytest tests\validation\test_partial_criteria_validator.py -q
```

Совместная проверка этапов 1–6:

```powershell
pytest tests\validation -q
```

Полная регрессия проекта:

```powershell
pytest -q
```

После полной регрессии повторно выполнить:

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --validate-partial-criteria
```

## Совмещенный запуск этапов 5 и 6

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --calculate-integral-metrics `
  --validate-partial-criteria
```

При необходимости можно пересоздать проверочный датасет и выполнить этапы
4–6 одной командой:

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --build-validation-dataset `
  --validate-integral-quality `
  --calculate-integral-metrics `
  --validate-partial-criteria
```

## Критерии закрытия

1. CLI завершается с кодом `0`.
2. Созданы CSV-, JSON- и Markdown-артефакты этапа 6.
3. В отчете `stage = 6`, `passed = true`, `row_count = 150`.
4. Проверены все шесть зафиксированных пар критериев.
5. Все метрики являются конечными числовыми значениями.
6. Тесты этапа 6 и все `tests/validation` проходят успешно.
7. Полная регрессия проекта проходит успешно.
8. Повторный CLI после регрессии завершается успешно.

Переход к этапу 7 выполняется только после отдельного подтверждения.
