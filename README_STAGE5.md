# Этап 5. Метрики интегрального априорного прогноза

## Назначение

Этап выполняет внешнюю проверку интегрального априорного прогноза,
сформированного в главе 5. Для каждого сценария сопоставляются:

```text
q_pred — неизмененный априорный прогноз главы 5;
q_fact — фактический интегральный показатель из validation_dataset.csv.
```

Ошибка определяется строго как:

```text
prediction_error = q_pred - q_fact
```

Фактические данные используются только для проверки. На этапе 5 не
изменяются веса, нормировки, направления латентных факторов, пороги классов
и формула `Q_pred`.

## Рассчитываемые метрики

```text
MAE
RMSE
Bias
Median Absolute Error
Max Absolute Error
Pearson
Spearman
Kendall tau-b
R²
```

Дополнительно фиксируются средние значения и стандартные отклонения
`q_pred` и `q_fact`.

## Создаваемые артефакты

```text
reports/chapter6/integral_prediction_errors.csv
reports/chapter6/validation_metrics.json
reports/chapter6/validation_metrics.md
```

CSV содержит:

```text
scenario_id
protocol_id
q_pred
q_fact
prediction_error
absolute_error
squared_error
```

## Файлы этапа

```text
src/manual_coding_sim/validation/integral_prediction_validator.py
src/manual_coding_sim/validation/chapter6_runner.py
tests/validation/test_integral_prediction_validator.py
README_STAGE5.md
STAGE5_MANIFEST.json
```

Файлы этапов 1–4 не заменяются, за исключением расширения общего CLI-файла
`chapter6_runner.py` новым флагом этапа 5.

## Запуск из Windows PowerShell

```powershell
Set-Location D:\Projects\man_coding_quality

Expand-Archive `
  -Path .\chapter6_stage5_integral_metrics.zip `
  -DestinationPath . `
  -Force

.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"

python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --calculate-integral-metrics
```

## Ожидаемый результат для расширенного корпуса

```text
Сценариев: 150
MAE: 0.1592441669
RMSE: 0.1944027110
Bias: -0.1493986447
Spearman: 0.8817689675
Kendall: 0.6993288591
R²: -1.9716122140
```

Дополнительные значения:

```text
Median Absolute Error: 0.1452399879
Max Absolute Error: 0.3766632080
Pearson: 0.8920527352
q_pred mean: 0.4988435686
q_fact mean: 0.6482422133
```

Отрицательный Bias указывает на систематическое занижение качества.
Отрицательный R² является допустимым результатом внешней проверки и не
считается программной ошибкой. Он означает, что по сумме квадратов ошибок
точечный прогноз уступает постоянному прогнозу средним фактическим
качеством. Одновременно высокие Spearman и Kendall показывают, что модель
в значительной степени сохраняет порядок сценариев по качеству.

## Проверка JSON-отчета

```powershell
$report = Get-Content `
  .\reports\chapter6\validation_metrics.json `
  -Raw | ConvertFrom-Json

$report.stage
$report.passed
$report.row_count
$report.error_definition
$report.metrics.mae
$report.metrics.rmse
$report.metrics.bias
$report.metrics.pearson
$report.metrics.spearman
$report.metrics.kendall
$report.metrics.r2
```

Критические значения:

```text
stage = 5
passed = True
row_count = 150
error_definition = prediction_error = q_pred - q_fact
```

## Проверка таблицы ошибок

```powershell
$errors = @(
  Import-Csv .\reports\chapter6\integral_prediction_errors.csv
)

Write-Host "Строк:" $errors.Count
Write-Host "Средняя абсолютная ошибка:" `
  (($errors.absolute_error | ForEach-Object { [double]$_ } |
    Measure-Object -Average).Average)

$errors |
Sort-Object { [double]$_.absolute_error } -Descending |
Select-Object -First 10 `
  scenario_id, protocol_id, q_pred, q_fact, `
  prediction_error, absolute_error
```

## Тесты этапа

```powershell
pytest tests\validation\test_integral_prediction_validator.py -q
```

Совместная проверка этапов 1–5:

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
  --calculate-integral-metrics
```

## Совмещенный запуск этапов 4 и 5

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --validate-integral-quality `
  --calculate-integral-metrics
```

При необходимости можно пересоздать датасет и выполнить этапы 4–5 одной
командой:

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --build-validation-dataset `
  --validate-integral-quality `
  --calculate-integral-metrics
```

## Критерии закрытия

1. CLI завершается с кодом `0`.
2. Созданы CSV-, JSON- и Markdown-артефакты этапа 5.
3. В отчете `stage = 5`, `passed = true`, `row_count = 150`.
4. Построчная ошибка рассчитана как `q_pred - q_fact`.
5. Все метрики являются конечными числовыми значениями.
6. Тесты этапа 5 и все `tests/validation` проходят успешно.
7. Полная регрессия проекта проходит успешно.
8. Повторный CLI после регрессии завершается успешно.

Переход к этапу 6 выполняется только после отдельного подтверждения.
