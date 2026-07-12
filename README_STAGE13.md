# Этап 13. Итоговый отчет экспериментальной проверки главы 6

## Назначение

Этап завершает отчетный слой программного контура главы 6 и формирует:

```text
reports/chapter6/chapter6_validation_report.json
reports/chapter6/chapter6_validation_report.md
reports/chapter6/chapter6_pipeline_run_report.json
```

Итоговый отчет собирается только из зафиксированных артефактов основного
вычислительного корпуса из 150 сценариев. Независимый синтетический holdout,
расположенный в `synthetic_runs`, автоматически в итоговый отчет не включается
и не подменяет результаты основного эксперимента.

## Созданные и измененные файлы

```text
src/manual_coding_sim/validation/chapter6_report_builder.py
src/manual_coding_sim/validation/chapter6_pipeline.py
src/manual_coding_sim/validation/chapter6_runner.py
tests/validation/test_chapter6_report_builder.py
README_STAGE13.md
STAGE13_MANIFEST.json
```

## Реализованные функции

1. Повторная проверка `validation_dataset.csv`:
   - ожидаемое число строк;
   - уникальность ключа `scenario_id + protocol_id`;
   - наличие `q_pred` и `q_fact`;
   - конечность и диапазон `[0; 1]`.
2. Загрузка и проверка JSON-отчетов этапов 2, 4--12.
3. Проверка номера этапа, статуса `passed` и числа сценариев каждого отчета.
4. Проверка наличия восьми рисунков из манифеста этапа 12.
5. Фиксация SHA-256 всех источников итогового отчета.
6. Раздельная фиксация:
   - технического статуса программного контура;
   - статуса основной научной гипотезы.
7. Поддержка трех статусов гипотезы:

```text
hypothesis_supported
hypothesis_partially_supported
hypothesis_not_supported
```

8. Единый CLI-контур этапов 2--13.
9. Контроль неизменности зафиксированных артефактов главы 5.
10. Сохранение отрицательного pipeline-отчета при ошибке любого шага.

## Правило оценки гипотезы

Полное подтверждение требует статистически устойчивого преимущества полной
модели над `prior_only_baseline` одновременно по абсолютным ошибкам и
ранговым метрикам, а также преимущества по классификационным показателям без
критических ошибок.

Частичное подтверждение фиксируется, когда модель надежно ранжирует сценарии
или улучшает отдельные прикладные показатели, но не показывает одновременного
устойчивого преимущества по абсолютной точности и интервальной калибровке.

Техническое успешное выполнение программы само по себе не считается
подтверждением научной гипотезы.

## Контрольный результат на основном корпусе

```text
Сценариев: 150
Технический статус: True
Статус гипотезы: hypothesis_partially_supported
Шагов полного контура: 12 / 12
Артефакты главы 5 не изменены: True
```

Основные основания частичного статуса:

- Spearman и Kendall полной модели выше, чем у `prior_only_baseline` в
  точечной оценке;
- Balanced Accuracy и Macro F1 полной модели выше в точечной оценке;
- критические ошибки `low -> high` и `high -> low` отсутствуют;
- `prior_only_baseline` устойчиво лучше по MAE и RMSE;
- абсолютная шкала имеет отрицательное смещение;
- покрытие прогнозных интервалов недостаточно.

## Команды Windows PowerShell

Все команды выполняются из корня проекта.

### 1. Применить архив

```powershell
Expand-Archive `
  -Path .\chapter6_stage13_report_builder.zip `
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

### 4. Сформировать итоговый отчет по существующим артефактам

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --build-report
```

Ожидаемый итог:

```text
Итоговый отчет экспериментальной проверки главы 6 сформирован.
Сценариев: 150
Технический статус: True
Статус основной гипотезы: hypothesis_partially_supported
JSON-отчет: reports\chapter6\chapter6_validation_report.json
Markdown-отчет: reports\chapter6\chapter6_validation_report.md
Этап 13 выполнен. Переход к этапу 14 требует отдельного подтверждения.
```

### 5. Проверить итоговый JSON-отчет

```powershell
$report = Get-Content `
  .\reports\chapter6\chapter6_validation_report.json `
  -Raw `
  -Encoding UTF8 |
ConvertFrom-Json

$report.stage
$report.passed
$report.row_count
$report.hypothesis_status
$report.hypothesis.criteria
$report.technical_checks
$report.methodological_conclusion
$report.limitations
```

Критические значения основного корпуса:

```text
stage = 13
passed = True
row_count = 150
hypothesis_status = hypothesis_partially_supported
```

Все значения в `technical_checks` должны быть `True`.

### 6. Тесты этапа 13

```powershell
pytest `
  tests\validation\test_chapter6_report_builder.py `
  -q
```

Ожидается:

```text
16 passed
```

### 7. Все тесты главы 6

```powershell
pytest tests\validation -q
```

При локальном составе после подтвержденных 151 теста этапов 1--12 ожидается:

```text
167 passed
```

Критерий завершения — отсутствие падений; число может быть больше при наличии
дополнительных локальных тестов.

### 8. Полная регрессия

```powershell
pytest -q
```

До этапа 13 было подтверждено 468 тестов. После добавления 16 тестов этапа 13
ожидается:

```text
484 passed
```

### 9. Выполнить единый программный контур

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --run-full-pipeline `
  --build-figures `
  --build-report
```

Ожидаемый итог:

```text
Полный программный контур главы 6 успешно завершен.
Выполнено шагов: 12/12
Артефакты главы 5 не изменены: True
Отчет полного запуска: reports\chapter6\chapter6_pipeline_run_report.json
Статус основной гипотезы: hypothesis_partially_supported
Этап 13 выполнен. Переход к этапу 14 требует отдельного подтверждения.
```

### 10. Проверить отчет полного запуска

```powershell
$pipeline = Get-Content `
  .\reports\chapter6\chapter6_pipeline_run_report.json `
  -Raw `
  -Encoding UTF8 |
ConvertFrom-Json

$pipeline.stage
$pipeline.passed
$pipeline.full_pipeline_completed
$pipeline.completed_steps
$pipeline.methodological_checks
$pipeline.error
```

Критические значения:

```text
stage = 13
passed = True
full_pipeline_completed = True
prediction_model_frozen = True
chapter5_artifacts_unchanged = True
target_leakage_detected = False
factual_values_used_only_for_external_validation = True
quality_thresholds_modified = False
error = null
```

### 11. Повторная проверка отчета после регрессии

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --build-report
```

## Критерии закрытия этапа

- итоговые JSON- и Markdown-отчеты созданы;
- `stage = 13`;
- `passed = true`;
- проверено 150 сценариев;
- статус гипотезы определен по зафиксированному правилу;
- сформирован `chapter6_pipeline_run_report.json`;
- полный контур содержит 12 успешно завершенных шагов;
- артефакты главы 5 не изменены;
- тесты этапа, главы 6 и полная регрессия проходят;
- повторный CLI после регрессии завершается успешно.

Переход к этапу 14 выполняется только после локального подтверждения этих
условий.
