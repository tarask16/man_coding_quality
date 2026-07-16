# Этап 0. Аудит текущей версии перед добавлением моделирования декодирования

## Итоговый статус

**ready_for_isolated_extension**

Проверено обязательных файлов ядра: 18; обязательных тестовых файлов: 15; обязательных CSV: 6.

- отсутствующие обязательные файлы ядра/тестов: нет;
- синтаксические ошибки в проверенных модулях: нет;
- компоненты декодирования: **absent**;
- тестовых файлов в снимке: 43; тестовых функций: 317;
- все обязательные CSV содержат 150 строк: True;
- исходники контура главы 6: **absent_in_uploaded_snapshot**.

## Проверка ключевых контрактов

| Файл | Статус | Отсутствующие символы |
|---|---|---|
| `src/manual_coding_sim/types.py` | ok | нет |
| `src/manual_coding_sim/procedure_model.py` | ok | нет |
| `src/manual_coding_sim/error_model.py` | ok | нет |
| `src/manual_coding_sim/control_model.py` | ok | нет |
| `src/manual_coding_sim/protocol_simulator.py` | ok | нет |
| `src/manual_coding_sim/feature_extractor.py` | ok | нет |
| `src/manual_coding_sim/quality_calculator.py` | ok | нет |

## Проверка табличных артефактов

| Артефакт | Строк | Столбцов | SHA-256 |
|---|---:|---:|---|
| `data/processed/protocols.csv` | 150 | 10 | `12cdb743199882ee31ace9782716ff4637dad5d1dd139433ef4a645a27364072` |
| `data/processed/prior_features.csv` | 150 | 36 | `24eec65576212ce74bbda16e0612a397ad148f2e16b4ca0ecefc4ec0ec9c8b69` |
| `data/processed/diagnostic_features.csv` | 150 | 8 | `9ce47d4a4ed3a312fca2f3ab0761194e5767ba5cb7d69212aec9203e190c5758` |
| `data/processed/fact_features.csv` | 150 | 8 | `daf45177b266e8235d2e3d517a4b3c1007f0b43e5fb2dda1942f2300f959bfcb` |
| `data/processed/quality_targets.csv` | 150 | 10 | `a11b2e856c81d2ac5912ff2b1cd54469d2f5b6f7f1af084bffacd4fe939ab4a5` |
| `data/processed/all_features.csv` | 150 | 53 | `1a76d64ec192d94ead904bbb8f04beaaf6068c31e051b2944c6fe0acf50e5ec9` |

## Архитектурные выводы

1. Событийный ProtocolSimulator завершает цепочку протоколом контроля и не формирует материальное кодированное сообщение C.
2. ProcedureModel создает нормативные абстрактные токены, но не выполняет отдельный слой фактического кодирования.
3. ErrorModel и ControlModel работают с исходами шагов, а не с изменяемой последовательностью кодированных элементов.
4. Типы EncodedMessage, DecodedMessage и восстановленное сообщение M_prime отсутствуют.
5. QualityCalculator рассчитывает q_acc по остаточной доле ошибок, а не по прямому расстоянию между M и M_prime.
6. Расширенный корпус из 150 сценариев формируется параметрическими формулами ExtendedCorpusRunner, а не событийным ProtocolSimulator.
7. Безопасная стратегия — отдельное расширение extensions/decoding_simulation без изменения исходного пакета manual_coding_sim.

## Решение по архитектуре расширения

Новые компоненты размещаются только в `extensions/decoding_simulation/`. Базовые файлы `src/manual_coding_sim/**`, исходные CSV и результаты глав 4–6 на этапах разработки не изменяются. Интеграция выполняется через адаптеры и новые выходные артефакты с отдельными именами.

## Критерий закрытия этапа 0

1. Аудит запускается локально и формирует JSON/Markdown-отчеты.
2. Обязательные файлы ядра и тестов присутствуют.
3. Зафиксированы SHA-256 базовых файлов ядра.
4. Подтверждено отсутствие декодирующих компонентов в базовом пакете.
5. Зафиксирован последовательный roadmap разработки.
