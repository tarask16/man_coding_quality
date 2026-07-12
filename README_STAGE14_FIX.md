# Исправление этапа 14. Совместимость с отчетом этапа 2

## Причина исправления

Первая версия финальной приемки ожидала в
`chapter6_input_validation_report.json` два поля, которые не входят в
фактический контракт загрузчика этапа 2:

- `required_columns_present` для каждого CSV-артефакта;
- `key_alignment_reference` на верхнем уровне отчета.

Загрузчик этапа 2 подтверждает наличие обязательных колонок фактом успешного
завершения проверки и сохраняет для каждого артефакта признаки
`unique_keys`, `finite_values`, `unit_interval_values` и `key_set_aligned`.
Поэтому прежняя приемка формировала ложный отказ
`input_files_found, merge_without_row_loss` при корректных данных.

## Исправление

Финальная приемка теперь:

1. проверяет положительный статус отчета этапа 2;
2. сопоставляет `checked_csv_count` с числом записей `artifact_checks`;
3. проверяет наличие каждого CSV-файла, число строк и число колонок;
4. проверяет уникальность ключей, конечность значений, диапазон `[0; 1]` и
   согласованность множеств ключей;
5. использует `required_columns_present`, если поле присутствует, но не требует
   его наличия в старом контракте;
6. подтверждает отсутствие потери строк либо по явному
   `key_alignment_reference = q_pred`, либо по согласованности ключей всех
   восьми входных таблиц и положительной проверке `q_pred_consistency`.

Проверки не ослаблены: отсутствующий входной CSV, неверное число строк,
несогласованные ключи или отрицательный статус этапа 2 по-прежнему блокируют
приемку.

## Изменяемые файлы

```text
src/manual_coding_sim/validation/chapter6_acceptance.py
tests/validation/test_chapter6_acceptance.py
```

## Применение

```powershell
Expand-Archive `
  -Path .\chapter6_stage14_acceptance_contract_fix.zip `
  -DestinationPath . `
  -Force
```

## Целевые тесты

```powershell
pytest tests\validation\test_chapter6_acceptance.py -q
```

Ожидается:

```text
21 passed
```

## Финальная приемка

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --run-acceptance
```

Ожидаемый результат:

```text
Пройдено проверок: 27/27
Полный pipeline завершен: True
Модель главы 5 зафиксирована: True
Целевая утечка обнаружена: False
Статус основной гипотезы: hypothesis_partially_supported
```

## Полная проверка

```powershell
pytest tests\validation -q
pytest -q

python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --run-full-pipeline `
  --build-figures `
  --build-report `
  --run-acceptance
```
