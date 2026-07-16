# Этап 2 — материальное кодированное сообщение C

## Применение архива

```powershell
Expand-Archive `
  -Path .\decoding_simulation_stage2_encoded_message.zip `
  -DestinationPath . `
  -Force
```

## Окружение

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src;extensions\decoding_simulation\src"
```

## Формирование демонстрационного C

```powershell
python -m manual_coding_sim_decoding.runner `
  --project-root . `
  --config extensions\decoding_simulation\configs\decoding_stage2.yaml `
  --show-config `
  --check-base-contract `
  --build-encoded-message
```

Результат:

```text
extensions/decoding_simulation/reports/stage2/encoded_message_demo.json
```

## Тест этапа

```powershell
python -m pytest `
  extensions\decoding_simulation\tests\test_stage2_encoded_message.py `
  -q
```

## Регрессия этапов расширения

```powershell
python -m pytest `
  extensions\decoding_simulation\tests `
  -q
```

## Полная регрессия проекта

```powershell
python -m pytest -q
```

Этап закрывается после подтверждения всех трех проверок пользователем.
