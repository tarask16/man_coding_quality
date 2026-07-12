# Этап 14. Финальная приемка программного контура главы 6

## Назначение

Этап выполняет независимую повторную техническую и методическую приемку
результатов этапов 2–13. Научный статус основной гипотезы не изменяется:
технически корректный эксперимент может завершиться полным, частичным или
отрицательным подтверждением гипотезы.

## Создаваемые файлы

```text
reports/chapter6/chapter6_acceptance_report.json
reports/chapter6/chapter6_acceptance_report.md
```

## Изменяемые файлы проекта

```text
src/manual_coding_sim/validation/chapter6_acceptance.py
src/manual_coding_sim/validation/chapter6_runner.py
tests/validation/test_chapter6_acceptance.py
```

## Приемочные проверки

Финальный акт содержит 27 обязательных проверок:

1. наличие всех входных CSV- и JSON-файлов;
2. согласованность числа сценариев;
3. отсутствие потери строк при объединении;
4. уникальность составного ключа;
5. диапазон контролируемых показателей `[0; 1]`;
6. согласованность `integral_quality`;
7. наличие интегральных метрик;
8. проверка шести частных критериев;
9. корректность confusion matrix;
10. наличие классификационных метрик;
11. проверка прогнозных интервалов;
12. наличие четырех сравниваемых моделей;
13. отсутствие утечки в mean baseline;
14. выполнение cluster-bootstrap;
15. наличие 18 парных сравнений;
16. наличие top-10 ошибок;
17. наличие восьми рисунков;
18. наличие итогового JSON-отчета;
19. наличие итогового Markdown-отчета;
20. положительный статус проверок этапа 13;
21. соответствие SHA-256 источников этапа 13;
22. неизменность артефактов главы 5;
23. отсутствие подгонки параметров и порогов;
24. отсутствие целевой утечки;
25. отсутствие ручной подмены графических данных;
26. исключение синтетического holdout из основного эксперимента;
27. успешное завершение полного CLI-контура 2–13.

## Применение архива

Из корня проекта:

```powershell
Expand-Archive `
  -Path .\chapter6_stage14_final_acceptance.zip `
  -DestinationPath . `
  -Force
```

## Подготовка окружения

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
```

## Финальная приемка по существующим артефактам

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --run-acceptance
```

## Полный воспроизводимый запуск этапов 2–14

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --run-full-pipeline `
  --build-figures `
  --build-report `
  --run-acceptance
```

Ожидаемый итог:

```text
Полный программный контур главы 6 успешно завершен.
Выполнено шагов: 12/12
Артефакты главы 5 не изменены: True
Финальная приемка программного контура главы 6 завершена.
Сценариев: 150
Пройдено проверок: 27/27
Полный pipeline завершен: True
Модель главы 5 зафиксирована: True
Целевая утечка обнаружена: False
Статус основной гипотезы: hypothesis_partially_supported
Этап 14 выполнен. Программный контур главы 6 принят.
```

## Проверка JSON-акта

```powershell
$acceptance = Get-Content `
  .\reports\chapter6\chapter6_acceptance_report.json `
  -Raw `
  -Encoding UTF8 |
ConvertFrom-Json

$acceptance.stage
$acceptance.accepted
$acceptance.full_pipeline_completed
$acceptance.prediction_model_frozen
$acceptance.target_leakage_detected
$acceptance.row_count
$acceptance.hypothesis_status
$acceptance.check_count
$acceptance.passed_check_count
$acceptance.failed_check_count
```

Критические значения:

```text
stage = 14
accepted = True
full_pipeline_completed = True
prediction_model_frozen = True
target_leakage_detected = False
row_count = 150
hypothesis_status = hypothesis_partially_supported
check_count = 27
passed_check_count = 27
failed_check_count = 0
```

Проверка непройденных условий:

```powershell
$acceptance.checks |
Where-Object { $_.passed -ne $true } |
Format-Table id, description, passed -AutoSize
```

Ожидается отсутствие строк.

## Тесты этапа 14

```powershell
pytest tests\validation\test_chapter6_acceptance.py -q
```

Ожидается:

```text
20 passed
```

## Все тесты главы 6

```powershell
pytest tests\validation -q
```

После подтвержденных 167 тестов этапов 1–13 ожидается:

```text
187 passed
```

## Полная регрессия

```powershell
pytest -q
```

После подтвержденных 484 тестов этапов 1–13 ожидается:

```text
504 passed
```

Фактическое количество может быть больше при наличии дополнительных локальных
тестов. Критерий завершения — отсутствие падений.

## Повторная приемка после полной регрессии

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --run-acceptance
```

## Методическое ограничение

Статус `accepted = true` означает, что вычислительный контур технически
воспроизводим, артефакты согласованы, утечки и подгонка не обнаружены. Этот
статус не заменяет научный вывод `hypothesis_partially_supported` и не
превращает синтетический holdout в независимую фактическую выборку.
