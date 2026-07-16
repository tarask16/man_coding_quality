# Этап 4. Модель декодирующего оператора и условий

## Назначение

Этап вводит отдельные детерминированные модели декодирующего оператора
`O_d` и условий выполнения обратной процедуры `U_d`.

Модели принимают только формальный план `D_h(C)` этапа 3 и рассчитывают:

- ожидаемое время выполнения каждого шага;
- внимание и накопленную утомленность декодирующего оператора;
- трудоемкость, нагрузку обращения к инструкции и нагрузку неразрешенного токена;
- влияние шума, рабочей нагрузки, прерываний, освещения и дефицита времени;
- скорректированное время, внимание и индекс устойчивости.

## Границы этапа

На этапе 4:

- ошибки декодирования не генерируются;
- контроль декодирования не выполняется;
- восстановленное сообщение `M'` не создается;
- исходное сообщение и `source_value` не используются;
- файлы базового пакета `src/manual_coding_sim` не изменяются.

## Основная команда

```powershell
python -m manual_coding_sim_decoding.runner `
  --project-root . `
  --config extensions\decoding_simulation\configs\decoding_stage4.yaml `
  --show-config `
  --check-base-contract `
  --estimate-decoding-context `
  --message-id M_STAGE4_DEMO
```

Результат:

```text
extensions/decoding_simulation/reports/stage4/decoding_context_demo.json
```

## Тесты

```powershell
python -m pytest `
  extensions\decoding_simulation\tests\test_stage4_decoding_context.py `
  -q

python -m pytest extensions\decoding_simulation\tests -q

python -m pytest -q
```
