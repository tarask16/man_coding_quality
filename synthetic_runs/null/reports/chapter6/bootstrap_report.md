# Bootstrap-анализ статистической устойчивости

- Этап: **10**
- Статус: **пройден**
- Сценариев: **120**
- Bootstrap-повторов: **1000**
- Уровень доверия: **95.00%**
- Единица выборки: **scenario_id**
- Random seed: **42**

## Доверительные интервалы модели главы 5

| Метрика | Значение | Bootstrap mean | CI lower | CI upper |
|---|---:|---:|---:|---:|
| mae | 0.1002165271 | 0.0997406057 | 0.0860856587 | 0.1142732150 |
| rmse | 0.1245899990 | 0.1239968307 | 0.1107378506 | 0.1378278030 |
| spearman | 0.9071046600 | 0.9027931814 | 0.8675808976 | 0.9263291542 |
| kendall | 0.7198879552 | 0.7203567262 | 0.6777159110 | 0.7584041402 |
| accuracy | 0.8500000000 | 0.8498916667 | 0.7833333333 | 0.9083333333 |
| macro_f1 | 0.8155992844 | 0.8131160269 | 0.7335178009 | 0.8793740192 |

## Парные разности модели главы 5 и baseline

Разность определяется как `metric_chapter5_model - metric_baseline`.

| Baseline | Метрика | Δ | CI lower | CI upper | Вывод |
|---|---|---:|---:|---:|---|
| mean_baseline | mae | -0.1802763801 | -0.2023655804 | -0.1574060695 | chapter5_model_favored |
| mean_baseline | rmse | -0.1877042139 | -0.2074064934 | -0.1687756018 | chapter5_model_favored |
| mean_baseline | spearman | 1.0333258059 | 0.8301005614 | 1.2351971463 | chapter5_model_favored |
| mean_baseline | kendall | 0.8131362366 | 0.6557920300 | 0.9748748338 | chapter5_model_favored |
| mean_baseline | accuracy | 0.6916666667 | 0.5997916667 | 0.7833333333 | chapter5_model_favored |
| mean_baseline | macro_f1 | 0.7244721861 | 0.6477731976 | 0.7895953726 | chapter5_model_favored |
| prior_only_baseline | mae | 0.0761452173 | 0.0615900379 | 0.0906506137 | baseline_favored |
| prior_only_baseline | rmse | 0.0942434514 | 0.0800144855 | 0.1080932708 | baseline_favored |
| prior_only_baseline | spearman | -0.0832418918 | -0.1179256207 | -0.0642390796 | baseline_favored |
| prior_only_baseline | kendall | -0.1994397759 | -0.2379158625 | -0.1635863607 | baseline_favored |
| prior_only_baseline | accuracy | -0.1250000000 | -0.1916666667 | -0.0666666667 | baseline_favored |
| prior_only_baseline | macro_f1 | -0.1475237349 | -0.2240105257 | -0.0779661754 | baseline_favored |
| theta_only_baseline | mae | -0.1175205326 | -0.1344552364 | -0.1014063660 | chapter5_model_favored |
| theta_only_baseline | rmse | -0.1463548687 | -0.1626139095 | -0.1293822907 | chapter5_model_favored |
| theta_only_baseline | spearman | 0.3954649628 | 0.3048208533 | 0.5004347576 | chapter5_model_favored |
| theta_only_baseline | kendall | 0.4002801120 | 0.3356683893 | 0.4674307103 | chapter5_model_favored |
| theta_only_baseline | accuracy | 0.3250000000 | 0.2416666667 | 0.4083333333 | chapter5_model_favored |
| theta_only_baseline | macro_f1 | 0.3087245167 | 0.2260206974 | 0.3919959690 | chapter5_model_favored |

## Сводка статистически устойчивых различий

- Преимуществ модели главы 5: **12**
- Преимуществ baseline: **6**
- Неустойчивых различий: **0**

## Методическое ограничение

Использован парный percentile-bootstrap по сценариям. Внутри повторных выборок применялись уже зафиксированные out-of-fold прогнозы этапа 9. Модели не переобучались, прогноз главы 5 не изменялся.
