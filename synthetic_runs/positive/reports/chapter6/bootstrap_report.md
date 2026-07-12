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
| mae | 0.0202222143 | 0.0202385493 | 0.0175730680 | 0.0230566452 |
| rmse | 0.0253001877 | 0.0252621494 | 0.0220302433 | 0.0289205077 |
| spearman | 0.9888672825 | 0.9869401938 | 0.9797890328 | 0.9917131424 |
| kendall | 0.9137254902 | 0.9135009037 | 0.8925029273 | 0.9321566749 |
| accuracy | 1.0000000000 | 1.0000000000 | 1.0000000000 | 1.0000000000 |
| macro_f1 | 1.0000000000 | 1.0000000000 | 1.0000000000 | 1.0000000000 |

## Парные разности модели главы 5 и baseline

Разность определяется как `metric_chapter5_model - metric_baseline`.

| Baseline | Метрика | Δ | CI lower | CI upper | Вывод |
|---|---|---:|---:|---:|---|
| mean_baseline | mae | -0.1757400437 | -0.1964855076 | -0.1544481110 | chapter5_model_favored |
| mean_baseline | rmse | -0.2032325948 | -0.2198645906 | -0.1849208590 | chapter5_model_favored |
| mean_baseline | spearman | 1.2860483362 | 1.1138873567 | 1.4381420373 | chapter5_model_favored |
| mean_baseline | kendall | 1.1276663633 | 0.9993214944 | 1.2476879238 | chapter5_model_favored |
| mean_baseline | accuracy | 0.6666666667 | 0.5831250000 | 0.7500000000 | chapter5_model_favored |
| mean_baseline | macro_f1 | 0.8333333333 | 0.8038527692 | 0.8666666667 | chapter5_model_favored |
| prior_only_baseline | mae | -0.0761515080 | -0.0896339290 | -0.0638619566 | chapter5_model_favored |
| prior_only_baseline | rmse | -0.0941245187 | -0.1076689835 | -0.0806729331 | chapter5_model_favored |
| prior_only_baseline | spearman | 0.0733384263 | 0.0489280752 | 0.1119223318 | chapter5_model_favored |
| prior_only_baseline | kendall | 0.1644257703 | 0.1167309685 | 0.2133915183 | chapter5_model_favored |
| prior_only_baseline | accuracy | 0.2583333333 | 0.1831250000 | 0.3416666667 | chapter5_model_favored |
| prior_only_baseline | macro_f1 | 0.2615330182 | 0.1853267328 | 0.3455612123 | chapter5_model_favored |
| theta_only_baseline | mae | -0.0973274858 | -0.1132657709 | -0.0821720343 | chapter5_model_favored |
| theta_only_baseline | rmse | -0.1187781499 | -0.1343616252 | -0.1034994044 | chapter5_model_favored |
| theta_only_baseline | spearman | 0.2010486839 | 0.1594147165 | 0.2663820765 | chapter5_model_favored |
| theta_only_baseline | kendall | 0.3492997199 | 0.2948595843 | 0.4087962877 | chapter5_model_favored |
| theta_only_baseline | accuracy | 0.2833333333 | 0.2083333333 | 0.3666666667 | chapter5_model_favored |
| theta_only_baseline | macro_f1 | 0.2849515557 | 0.2084376042 | 0.3646349074 | chapter5_model_favored |

## Сводка статистически устойчивых различий

- Преимуществ модели главы 5: **18**
- Преимуществ baseline: **0**
- Неустойчивых различий: **0**

## Методическое ограничение

Использован парный percentile-bootstrap по сценариям. Внутри повторных выборок применялись уже зафиксированные out-of-fold прогнозы этапа 9. Модели не переобучались, прогноз главы 5 не изменялся.
