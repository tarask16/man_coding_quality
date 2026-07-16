# Этап 1 — каркас изолированного расширения

## Применение архива

```powershell
Expand-Archive `
  -Path .\decoding_simulation_stage1_scaffold.zip `
  -DestinationPath . `
  -Force
```

## Активация окружения

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src;extensions\decoding_simulation\src"
```

## Проверка конфигурации и контракта

```powershell
python -m manual_coding_sim_decoding.runner `
  --project-root . `
  --config extensions\decoding_simulation\configs\decoding_stage1.yaml `
  --show-config `
  --check-base-contract
```

Ожидаемый итог:

```text
Статус контракта с базовым пакетом: совместим.
Измененные baseline-файлы: []
```

## Машинно-читаемый вывод

```powershell
python -m manual_coding_sim_decoding.runner `
  --project-root . `
  --config extensions\decoding_simulation\configs\decoding_stage1.yaml `
  --check-base-contract `
  --json
```

## Тест этапа

```powershell
python -m pytest `
  extensions\decoding_simulation\tests\test_stage1_package_scaffold.py `
  -q
```

## Полная регрессия

```powershell
python -m pytest -q
```

Тесты выполняются локально пользователем. Следующий этап начинается только
после подтверждения результатов.
