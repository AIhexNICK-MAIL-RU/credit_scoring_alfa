# Кредитный скоринг — Альфа-Банк × МФТИ

Предсказание вероятности выхода клиента в дефолт по истории карточных/кредитных
транзакций. Метрика — **ROC-AUC**. Формат сабмита: `id, flag` (вероятность).

## Данные (не включены в репозиторий — слишком большие)
- `train_data.parquet` (~1 ГБ, 18.3M строк) — история кредитных продуктов, 59 признаков (бинаризованные/закодированные категориальные коды).
- `test_data.parquet` (~460 МБ, 7.8M строк).
- `train_target.csv` — `id, flag` (2.1M клиентов, доля дефолтов 3.55%).
- `sample_submission.csv` — образец сабмита.

## Подход

**1. Градиентный бустинг (CatBoost)**
- `build_features.py` — статистические агрегаты по `id` (mean/max/min/std/last).
- `build_features2.py` — count/fraction-кодирование значений категорий по истории клиента (1075 признаков).
- `train_cat_v2.py`, `train_cat_strong.py` — CatBoost (depth 6–7, count/fraction-фичи). Лучший одиночный CatBoost: **fold0 AUC ≈ 0.769**.

**2. Sequence-нейросеть (PyTorch, CPU)**
- `prep_seq.py` — паддинг истории в тензоры `[N, L=25, F=59]` (последние 25 кредитных продуктов).
- `train_nn_final2.py` / `train_nn_full.py` — эмбеддинги категорий (единая таблица со смещениями) → `Conv1d` → masked mean/last пулинг → MLP. Каждая модель **fold AUC ≈ 0.778–0.780**.
- Несколько прогонов на разных seed/фолдах → **bagging**.

**3. Ансамбль**
- Ранг-усреднение нейросетей + CatBoost. Веса подобраны на кросс-валидации.
- Достигнутая валидация: **ROC-AUC ≈ 0.78** (улучшение с baseline 0.7587 на ~+0.022).

## Файлы
| Файл | Назначение |
|---|---|
| `build_features*.py` | фичеинжиниринг (агрегаты, count/fraction) |
| `prep_seq.py` | подготовка последовательностей для NN |
| `train_cat_*.py` | обучение CatBoost |
| `train_nn_*.py` | обучение sequence-нейросетей |
| `predict_nn*.py` | предсказание из сохранённых весов |
| `run_after_reboot.sh` | драйвер полноценного 5-fold bagging |
| `submission.csv` | финальный сабмит |
| `CHAT_LOG.md` | подробный лог всей работы |

## Запуск
```bash
pip install -r requirements.txt
python build_features2.py        # фичи
python prep_seq.py               # последовательности
python train_cat_strong.py       # CatBoost
python train_nn_full.py 0 42     # нейросеть (fold 0, seed 42)
# ... ансамбль предсказаний -> submission.csv
```

## Аппаратные замечания
Нейросети обучались на CPU: на Apple Silicon фоновые процессы душатся macOS на GPU (Metal/MPS),
поэтому использован CPU + компактная загрузка данных (int8/int16) для обхода нехватки RAM.
Для 0.79+ рекомендуется GPU с CUDA и полноценный 5-fold bagging.
