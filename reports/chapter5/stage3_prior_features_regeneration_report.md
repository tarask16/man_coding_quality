# Этап 3. Регенерация `data/processed/prior_features.csv`

## Статус

Этап 3 закрыт после регенерации расширенного корпуса главы 3.

## Выполненная команда

```bash
PYTHONPATH=src python -m manual_coding_sim.experiments.extended_corpus_cli --project-root . --config configs/chapter3_extended_corpus.yaml
```

## Проверка входов главы 5

```bash
PYTHONPATH=src python -m manual_coding_sim.prediction.chapter5_runner --project-root . --config configs/chapter5.yaml --validate-inputs
```

Результат проверки: данные главы 5 успешно загружены: 150 сценариев, 3 латентных фактора.

## Контрольные показатели

| Показатель | Значение |
|---|---:|
| Строк в `prior_features.csv` | 150 |
| Колонок в `prior_features.csv` | 36 |
| Строк в `theta_prior.csv` | 150 |
| Совпавших строк по `scenario_id` + `protocol_id` | 150 |
| Уникальных `scenario_id` | 150 |
| Уникальных `protocol_id` | 150 |
| Запрещенных фактических/целевых колонок | 0 |

## Проверка тестами

```text
tests/prediction: 24 passed in 0.31s
all tests: 272 passed in 3.47s
```

## Вывод

Файл `data/processed/prior_features.csv` приведен в соответствие с расширенным корпусом главы 4 и согласован с `reports/chapter4/theta_prior.csv`.

К этапу 4 не выполнялся переход.
