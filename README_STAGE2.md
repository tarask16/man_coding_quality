# Этап 2. Загрузка и проверка входных артефактов главы 6

## Назначение

Этап проверяет восемь CSV-артефактов и два JSON-отчета, необходимые для
внешней проверки прогноза главы 5. Проверочный датасет на этом этапе еще не
строится.

## Состав изменений

- `src/manual_coding_sim/validation/chapter6_data_loader.py` — загрузчик и валидатор;
- `src/manual_coding_sim/validation/chapter6_runner.py` — CLI этапов 1–2;
- `tests/validation/test_chapter6_data_loader.py` — тесты этапа 2;
- `tests/test_stage13_chapter3_report.py` — изоляция контрольного эксперимента
  главы 3 в `tmp_path`, чтобы полная регрессия не перезаписывала рабочий
  каталог `data/processed` сценарием `A_TEST_FINAL`.

Конфигурационные файлы этапа 1 не заменяются.

## Важное исправление регрессии

Ранее `tests/test_stage13_chapter3_report.py` запускал контрольный эксперимент
с `project_root=Path.cwd()`. В результате `pytest -q` мог заменить рабочие
`quality_targets.csv`, `fact_features.csv` и другие артефакты главы 3 тестовым
набором `A_TEST_FINAL`. Теперь эксперимент выполняется во временном каталоге.

Если рабочие данные уже были перезаписаны, до запуска этапа 2 восстановите их:

```powershell
git restore -- `
  data/processed/protocols.csv `
  data/processed/prior_features.csv `
  data/processed/fact_features.csv `
  data/processed/diagnostic_features.csv `
  data/processed/quality_targets.csv `
  data/processed/all_features.csv `
  data/processed/dataset_summary.json `
  reports/chapter3/experiment_run_report.json
```

После восстановления проверьте, что `quality_targets.csv` содержит 150 строк и
идентификаторы `scn_0000 ... scn_0149`, а не `A_TEST_FINAL`.

## Запуск

```powershell
$env:PYTHONPATH = "src"
python -m manual_coding_sim.validation.chapter6_runner `
  --project-root . `
  --config configs\chapter6.yaml `
  --validate-inputs
```

## Проверки

```powershell
pytest tests\validation\test_stage1_chapter6.py -q
pytest tests\validation\test_chapter6_data_loader.py -q
pytest tests\test_stage13_chapter3_report.py -q
pytest tests\validation -q
pytest -q
```

После полной регрессии повторный запуск `--validate-inputs` должен оставаться
успешным. Это подтверждает, что тесты больше не изменяют входные артефакты.
