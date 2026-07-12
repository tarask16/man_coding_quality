# Этап 3. Формирование проверочного датасета главы 6

## Назначение

Этап формирует единый проверочный датасет для последующих расчетов главы 6.
Все входные таблицы повторно проходят проверки этапа 2, после чего
объединяются по составному ключу:

```text
scenario_id, protocol_id
```

Режим объединения — `one_to_one`. После каждого `merge` контролируется
сохранение ожидаемого числа сценариев.

## Методическое ограничение

Этап не изменяет и не пересчитывает прогноз главы 5. Фактические показатели
добавляются только как внешние проверочные данные. Колонка `q_fact` является
точной копией `integral_quality`.

## Создаваемый артефакт

```text
reports/chapter6/validation_dataset.csv
```

Для расширенного корпуса ожидается:

```text
строк: 150
колонок: 97
уникальных пар scenario_id, protocol_id: 150
прогнозные классы low/medium/high: 67/49/34
фактические классы low/medium/high: 1/100/49
```

Количество колонок фиксирует текущую версию входных артефактов глав 3–5.
Критическим приемочным условием является не число 97 само по себе, а наличие
обязательных колонок, отсутствие дублей и пропусков, а также 150 уникальных
сценариев.

## Файлы этапа

```text
src/manual_coding_sim/validation/validation_dataset_builder.py
src/manual_coding_sim/validation/chapter6_runner.py
tests/validation/test_validation_dataset_builder.py
README_STAGE3.md
STAGE3_MANIFEST.json
```

Конфигурационные файлы и загрузчик этапа 2 не заменяются.

## Запуск из Windows PowerShell

```powershell
Set-Location D:\Projects\man_coding_quality

Expand-Archive `
  -Path .\chapter6_stage3_validation_dataset.zip `
  -DestinationPath . `
  -Force

.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"

python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --build-validation-dataset
```

## Проверка созданного CSV

```powershell
$dataset = @(Import-Csv .\reports\chapter6\validation_dataset.csv)

$dataset.Count
($dataset | Select-Object scenario_id, protocol_id -Unique).Count
$dataset[0].PSObject.Properties.Name

$dataset |
  Group-Object q_pred_class |
  Select-Object Name, Count

$dataset |
  Group-Object q_fact_class |
  Select-Object Name, Count
```

Дополнительная автоматическая проверка:

```powershell
$requiredColumns = @(
    "scenario_id",
    "protocol_id",
    "q_pred",
    "q_fact",
    "integral_quality",
    "q_pred_class",
    "q_fact_class",
    "q_acc_pred",
    "q_time_pred",
    "q_effort_pred",
    "q_res_pred",
    "q_rep_pred",
    "q_fit_pred",
    "q_acc",
    "q_time",
    "q_effort",
    "q_res",
    "q_rep",
    "q_fit",
    "theta_0",
    "theta_1",
    "theta_2",
    "uncertainty_score",
    "q_pred_lower",
    "q_pred_upper"
)

$actualColumns = $dataset[0].PSObject.Properties.Name
$missingColumns = $requiredColumns |
  Where-Object { $_ -notin $actualColumns }

if ($dataset.Count -ne 150) {
    throw "Ожидалось 150 строк, получено $($dataset.Count)."
}

$uniqueKeys = $dataset |
  ForEach-Object { "$($_.scenario_id)|$($_.protocol_id)" } |
  Sort-Object -Unique

if ($uniqueKeys.Count -ne 150) {
    throw "Составной ключ не является уникальным."
}

if ($missingColumns) {
    throw "Отсутствуют колонки: $($missingColumns -join ', ')"
}

$mismatch = $dataset |
  Where-Object {
      [math]::Abs([double]$_.q_fact - [double]$_.integral_quality) -gt 1e-12
  }

if ($mismatch) {
    throw "q_fact не совпадает с integral_quality."
}

Write-Host "Проверочный датасет этапа 3 прошел контроль."
```

## Тесты этапа

```powershell
pytest tests\validation\test_validation_dataset_builder.py -q
```

Совместная проверка этапов 1–3:

```powershell
pytest tests\validation -q
```

Полная регрессия проекта:

```powershell
pytest -q
```

После полной регрессии повторно сформировать датасет:

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --build-validation-dataset
```

## Критерии закрытия

1. CLI завершается с кодом `0`.
2. Создан `reports/chapter6/validation_dataset.csv`.
3. В CSV содержится 150 строк и 150 уникальных составных ключей.
4. Обязательные прогнозные, фактические, латентные и интервальные колонки
   присутствуют.
5. `q_fact` точно совпадает с `integral_quality`.
6. Тесты этапа 3 и все `tests/validation` проходят успешно.
7. Полная регрессия проекта проходит успешно.
8. Повторный CLI после полной регрессии также завершается успешно.

Переход к этапу 4 выполняется только после отдельного подтверждения.
