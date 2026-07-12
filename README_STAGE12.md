# Этап 12. Графические материалы главы 6

## Назначение

Этап формирует восемь воспроизводимых рисунков для вставки в диссертацию. Все изображения строятся только по расчетным CSV-артефактам этапов 3–11. Исходные таблицы, прогнозы главы 5, веса, пороги и фактические показатели не изменяются.

## Создаваемые рисунки

```text
reports/chapter6/figures/q_pred_vs_q_fact.png
reports/chapter6/figures/residuals_vs_q_fact.png
reports/chapter6/figures/absolute_error_distribution.png
reports/chapter6/figures/confusion_matrix.png
reports/chapter6/figures/baseline_comparison.png
reports/chapter6/figures/prediction_intervals.png
reports/chapter6/figures/error_by_dominant_topic.png
reports/chapter6/figures/partial_criteria_comparison.png
```

Дополнительно создаются:

```text
reports/chapter6/figures/figure_manifest.json
reports/chapter6/figures/figure_manifest.md
```

Манифест фиксирует:

- SHA-256 всех расчетных источников;
- SHA-256 каждого рисунка;
- размер изображения в пикселях;
- разрешение 300 DPI;
- число сценариев;
- отсутствие изменения исходных данных;
- отсутствие ручной подмены значений.

## Требования к окружению

Используются `matplotlib`, `numpy` и `pandas`. Проверка доступности:

```powershell
python -c "import matplotlib, numpy, pandas; print('Графические зависимости доступны')"
```

Если `matplotlib` отсутствует:

```powershell
python -m pip install matplotlib
```

## Применение архива

Выполнять из корня проекта:

```powershell
Expand-Archive `
  -Path .\chapter6_stage12_figures.zip `
  -DestinationPath . `
  -Force
```

## Очистка кэша

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

## Активация окружения

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
```

## Генерация рисунков

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --build-figures
```

Ожидаемый итог:

```text
Графические материалы главы 6 успешно сформированы.
Сценариев: 150
Рисунков: 8
Разрешение: 300 DPI
Этап 12 выполнен. Переход к этапу 13 требует отдельного подтверждения.
```

## Проверка манифеста

```powershell
$manifest = Get-Content `
  .\reports\chapter6\figures\figure_manifest.json `
  -Raw `
  -Encoding UTF8 |
ConvertFrom-Json

$manifest.stage
$manifest.passed
$manifest.row_count
$manifest.figure_count
$manifest.dpi
$manifest.source_data_modified
$manifest.manual_data_substitution
$manifest.figures |
Select-Object filename, width_px, height_px, size_bytes, sha256 |
Format-Table -AutoSize
```

Критические значения:

```text
stage = 12
passed = True
row_count = 150
figure_count = 8
dpi = 300
source_data_modified = False
manual_data_substitution = False
```

## Проверка полного перечня файлов

```powershell
$figureNames = @(
    "q_pred_vs_q_fact.png",
    "residuals_vs_q_fact.png",
    "absolute_error_distribution.png",
    "confusion_matrix.png",
    "baseline_comparison.png",
    "prediction_intervals.png",
    "error_by_dominant_topic.png",
    "partial_criteria_comparison.png"
)

$figuresDir = ".\reports\chapter6\figures"

foreach ($name in $figureNames) {
    $path = Join-Path $figuresDir $name
    if (-not (Test-Path $path)) {
        throw "Не сформирован рисунок: $path"
    }
    $file = Get-Item $path
    if ($file.Length -lt 10000) {
        throw "Рисунок имеет подозрительно малый размер: $path"
    }
    Write-Host "Создан: $path ($($file.Length) байт)"
}
```

## Тесты этапа 12

```powershell
pytest `
  tests\validation\test_chapter6_figure_builder.py `
  -q
```

Ожидается:

```text
12 passed
```

## Все тесты главы 6

```powershell
pytest tests\validation -q
```

С учетом ранее подтвержденных 139 тестов и 12 новых тестов ожидается:

```text
151 passed
```

## Полная регрессия

```powershell
pytest -q
```

С учетом ранее подтвержденных 456 тестов и 12 новых тестов ожидается:

```text
468 passed
```

Фактическое число может отличаться, если после этапа 11 в проект были добавлены или удалены другие тесты. Критический критерий — отсутствие падений.

## Повторная генерация после регрессии

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --build-figures
```

## Совместный запуск этапов 11 и 12

```powershell
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --analyze-prediction-errors `
  --build-figures
```

## Критерии завершения этапа

Этап 12 считается завершенным, если:

1. CLI завершился с кодом 0.
2. Сформированы все восемь PNG.
3. В манифесте `passed = true`.
4. В манифесте `row_count = 150`.
5. Разрешение указано как 300 DPI.
6. `source_data_modified = false`.
7. `manual_data_substitution = false`.
8. Тесты этапа 12 прошли.
9. Все тесты `tests/validation` прошли.
10. Полная регрессия прошла.
11. Повторный CLI после регрессии прошел.

Переход к этапу 13 выполняется только после отдельного подтверждения.
