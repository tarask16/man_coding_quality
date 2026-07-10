# Итоговый отчет программной реализации главы 3

## Назначение

Программная реализация подтверждает возможность воспроизводимого компьютерного моделирования процессов ручного кодирования и формирования данных для дальнейшей априорной оценки качества.

## Реализованная цепочка

`S, O, U, G, K → M → E_h → C → D_h → M'`

## Состав научных компонентов

| Компонент | Модуль | Смысл |
|---|---|---|
| G | message_model.py | модель класса исходных сообщений M |
| S | procedure_model.py | модель средства ручного кодирования |
| O | operator_model.py | модель оператора ручного кодирования |
| U | condition_model.py | модель условий применения |
| ErrorModel | error_model.py | вероятностная модель ошибок |
| K | control_model.py | модель контрольных процедур |
| ProtocolSimulator | protocol_simulator.py | интегральный симулятор протоколов |
| FeatureExtractor | feature_extractor.py | разделение X_prior, X_fact и X_diag |
| QualityCalculator | quality_calculator.py | расчет q(A) |
| DatasetBuilder | dataset_builder.py | формирование табличного датасета |
| ExperimentRunner | experiment_runner.py | воспроизводимый запуск вычислительного эксперимента |

## Контроль файлов

Исходные файлы: 15 из 15.
Тестовые файлы: 13 из 13.
Артефакты датасета: 7 из 7.

## Исходные файлы

- [OK] `src/manual_coding_sim/__init__.py`
- [OK] `src/manual_coding_sim/config.py`
- [OK] `src/manual_coding_sim/types.py`
- [OK] `src/manual_coding_sim/message_model.py`
- [OK] `src/manual_coding_sim/procedure_model.py`
- [OK] `src/manual_coding_sim/operator_model.py`
- [OK] `src/manual_coding_sim/condition_model.py`
- [OK] `src/manual_coding_sim/error_model.py`
- [OK] `src/manual_coding_sim/control_model.py`
- [OK] `src/manual_coding_sim/protocol_simulator.py`
- [OK] `src/manual_coding_sim/feature_extractor.py`
- [OK] `src/manual_coding_sim/quality_calculator.py`
- [OK] `src/manual_coding_sim/dataset_builder.py`
- [OK] `src/manual_coding_sim/experiment_runner.py`
- [OK] `src/manual_coding_sim/chapter3_report.py`

## Тестовые файлы

- [OK] `tests/test_stage1_package_structure.py`
- [OK] `tests/test_stage2_message_model.py`
- [OK] `tests/test_stage3_procedure_model.py`
- [OK] `tests/test_stage4_operator_model.py`
- [OK] `tests/test_stage5_condition_model.py`
- [OK] `tests/test_stage6_error_model.py`
- [OK] `tests/test_stage7_control_model.py`
- [OK] `tests/test_stage8_protocol_simulator.py`
- [OK] `tests/test_stage9_feature_extractor.py`
- [OK] `tests/test_stage10_quality_calculator.py`
- [OK] `tests/test_stage11_dataset_builder.py`
- [OK] `tests/test_stage12_experiment_runner.py`
- [OK] `tests/test_stage13_chapter3_report.py`

## Артефакты датасета

- [OK] `protocols`
- [OK] `prior_features`
- [OK] `fact_features`
- [OK] `diagnostic_features`
- [OK] `quality_targets`
- [OK] `all_features`
- [OK] `summary`

## Воспроизводимый эксперимент

Эксперимент: `chapter3_final_control_experiment`
Сценарий: `A_001`
Число прогонов: `5`
random_seed: `42`
Контроль воспроизводимости: `True`
Контрольный хеш: `867988eb0e3f0b8ff9c494626da6cfd7c285fb267167771863edf4e774f5ae9c`

## Ограничение области реализации

- адаптированная LDA-модель латентных факторов качества
- построение априорной прогнозной оценки качества
- проверка достоверности прогноза по train/val/test-разбиению
