# Этап 4. Проверка фактического интегрального качества

## Назначение

Этап проверяет внутреннюю согласованность фактического интегрального
показателя качества, сформированного программным контуром главы 3.

Основной фактический показатель остается неизменным:

```text
Q_fact = integral_quality
```

Дополнительно рассчитывается контрольная агрегация шести частных критериев:

```text
q_acc, q_time, q_effort, q_res, q_rep, q_fit
```

Контрольная величина используется только для диагностики и не подменяет
`integral_quality`.

## Весовая схема

Используется зафиксированная весовая схема главы 5:

| Критерий | Вес |
|---|---:|
| `q_acc` | 0.1666666667 |
| `q_time` | 0.1666666667 |
| `q_effort` | 0.1666666667 |
| `q_res` | 0.1666666667 |
| `q_rep` | 0.1666666666 |
| `q_fit` | 0.1666666666 |

Сумма весов равна 1,0.

## Допуск согласованности

Фиксированный диагностический допуск этапа 4:

```text
0.05
```

Проверка считается пройденной, если для всех сценариев абсолютное
расхождение между `integral_quality` и контрольной агрегацией не превышает
0,05. Допуск не используется для изменения весов, пересчета `Q_fact` или
корректировки модели главы 5.

## Создаваемые артефакты

```text
reports/chapter6/integral_quality_consistency.csv
reports/chapter6/integral_quality_consistency_report.json
reports/chapter6/integral_quality_consistency_report.md
```

CSV содержит:

```text
scenario_id
protocol_id
q_acc
q_time
q_effort
q_res
q_rep
q_fit
integral_quality
q_fact
q_fact_control
consistency_difference
absolute_difference
within_tolerance
```

## Файлы этапа

```text
src/manual_coding_sim/validation/integral_quality_validator.py
src/manual_coding_sim/validation/chapter6_runner.py
tests/validation/test_integral_quality_validator.py
README_STAGE4.md
STAGE4_MANIFEST.json
```

Конфигурационный каркас этапа 1, загрузчик этапа 2 и построитель этапа 3 не
заменяются.

## Запуск из Windows PowerShell

```powershell
Set-Location D:\Projects\man_coding_quality

Expand-Archive `
  -Path .\chapter6_stage4_integral_quality.zip `
  -DestinationPath . `
  -Force

.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"

python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --validate-integral-quality
```

## Ожидаемый результат для расширенного корпуса

```text
Сценариев: 150
Среднее абсолютное расхождение: 0.0149203755
Максимальное абсолютное расхождение: 0.0420206666
Сценариев вне допуска: 0
Статус: passed = true
```

## Проверка отчета

```powershell
$report = Get-Content `
  .\reports\chapter6\integral_quality_consistency_report.json `
  -Raw | ConvertFrom-Json

$report.stage
$report.passed
$report.row_count
$report.consistency_tolerance
$report.metrics.mean_absolute_difference
$report.metrics.max_absolute_difference
$report.metrics.outside_tolerance_count
$report.metrics.max_q_fact_alias_difference
```

Критические значения:

```text
stage = 4
passed = True
row_count = 150
consistency_tolerance = 0.05
outside_tolerance_count = 0
max_q_fact_alias_difference = 0
```

## Тесты этапа

```powershell
pytest tests\validation\test_integral_quality_validator.py -q
```

Совместная проверка этапов 1–4:

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
  --validate-integral-quality
```

## Совмещенный запуск этапов 3 и 4

При необходимости проверочный датасет можно пересоздать и сразу проверить:

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --build-validation-dataset `
  --validate-integral-quality
```

## Критерии закрытия

1. CLI завершается с кодом `0`.
2. Созданы CSV-, JSON- и Markdown-артефакты этапа 4.
3. В отчете `stage = 4` и `passed = true`.
4. Проверено 150 сценариев.
5. `q_fact` точно совпадает с `integral_quality`.
6. Сценариев вне допуска нет.
7. Тесты этапа 4 и все `tests/validation` проходят успешно.
8. Полная регрессия проекта проходит успешно.
9. Повторный CLI после регрессии завершается успешно.

Переход к этапу 5 выполняется только после отдельного подтверждения.
