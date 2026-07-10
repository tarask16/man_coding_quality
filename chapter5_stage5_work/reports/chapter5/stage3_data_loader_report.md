# Этап 3. Загрузка и объединение входных данных главы 5

## Статус

Этап 3 реализован программно, но не может быть полностью закрыт на фактических данных текущего архива, потому что входной файл `data/processed/prior_features.csv` не соответствует расширенному корпусу главы 4.

## Что реализовано

Добавлен полноценный загрузчик `Chapter5DataLoader`, который выполняет:

1. чтение `data/processed/prior_features.csv`;
2. чтение `reports/chapter4/theta_prior.csv`;
3. чтение `reports/chapter4/topic_interpretation.json`;
4. проверку обязательных колонок;
5. проверку `selected_k = 3`;
6. проверку неотрицательности и суммы `theta_0 + theta_1 + theta_2 ≈ 1`;
7. проверку пригодности `LDA_prior` для априорного прогноза;
8. проверку дублей идентификаторов;
9. объединение таблиц по `scenario_id` или по паре `scenario_id`, `protocol_id`, если `protocol_id` присутствует в `prior_features.csv`.

Добавлен CLI-флаг:

```bash
PYTHONPATH=src python -m manual_coding_sim.prediction.chapter5_runner \
  --project-root . \
  --config configs/chapter5.yaml \
  --validate-inputs
```

## Проверка тестами

```text
pytest tests/prediction -q
24 passed in 0.31s

pytest -q
272 passed in 3.28s
```

## Проверка фактических данных текущего архива

| Файл | Строк | Статус |
|---|---:|---|
| `data/processed/prior_features.csv` | 2 | не соответствует расширенному корпусу |
| `reports/chapter4/theta_prior.csv` | 150 | соответствует расширенному корпусу |
| `reports/chapter4/topic_interpretation.json` | 3 темы | пригоден для априорного прогноза |

Проблема в `prior_features.csv`:

- найдено строк: 2;
- ожидается строк для расширенного корпуса: 150;
- `scenario_id` содержит дубли: ['A_TEST_FINAL'];
- `scenario_id` из `prior_features.csv` не совпадает со сценариями `scn_0000 ... scn_0149` из `theta_prior.csv`.

Фактическое сообщение проверки:

```text
Проверка входных данных главы 5 не пройдена: Во входном файле обнаружены дубли идентификаторов, объединение главы 5 неоднозначно. Ключи: ('scenario_id',). Примеры: [{'scenario_id': 'A_TEST_FINAL'}, {'scenario_id': 'A_TEST_FINAL'}]. Файл: data/processed/prior_features.csv
```

## Вывод

Код этапа 3 готов и покрыт тестами. Для полного закрытия этапа 3 нужно заменить `data/processed/prior_features.csv` на версию расширенного корпуса, по которой был построен `reports/chapter4/theta_prior.csv`.

Требование к корректному `prior_features.csv`:

1. 150 строк;
2. уникальные `scenario_id` вида `scn_0000 ... scn_0149`;
3. наличие априорных колонок `prior_*`;
4. отсутствие фактических и целевых колонок;
5. желательное наличие `protocol_id`, либо однозначное соответствие по `scenario_id`.

## Решение о переходе дальше

Переход к этапу 4 не рекомендуется до замены `prior_features.csv` на согласованную версию расширенного корпуса.
