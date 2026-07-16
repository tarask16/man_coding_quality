# Этап 0 — аудит перед добавлением декодирования

Папка относится только к новому расширению. Файлы базового пакета
`src/manual_coding_sim/` не изменяются.

## Локальный запуск аудита

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python extensions\decoding_simulation\tools\audit_current_version.py `
  --project-root . `
  --report-dir extensions\decoding_simulation\reports\stage0
```

## Тест этапа

```powershell
python -m pytest extensions\decoding_simulation\tests\test_stage0_current_version_audit.py -q
```

## Полная регрессия

```powershell
python -m pytest -q
```

Прохождение тестов должно быть подтверждено локально пользователем.
