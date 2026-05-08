# Feedback Sorting System

Система анализа тональности отзывов с Wildberries на основе `scikit-learn`.  
Классифицирует отзывы как **позитивные** или **негативные** с указанием уверенности модели.

## Стек

- Python 3.10+
- [scikit-learn](https://scikit-learn.org/) — обучение модели (TF-IDF + Logistic Regression)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets) — загрузка датасетов
- [joblib](https://joblib.readthedocs.io/) — сохранение и загрузка модели

## Датасеты

| Назначение | Ссылка |
|---|---|
| Обучение | [nyuuzyou/wb-feedbacks](https://huggingface.co/datasets/nyuuzyou/wb-feedbacks) |
| Тестирование | [Roaoch/urfu_ecom_wb](https://huggingface.co/datasets/Roaoch/urfu_ecom_wb) |

## Структура проекта

```
src/
├── main.py       — точка входа
├── train.py      — загрузка данных и обучение модели
├── predict.py    — загрузка модели и предсказания
├── cli.py        — интерактивный интерфейс в терминале
└── config.py     — константы (пути, параметры датасета)
```

## Установка

```bash
git clone <repo-url>
cd Feedback-Sorting-System

pip install scikit-learn datasets joblib
```

## Запуск

```bash
cd src
python main.py
```

При первом запуске система автоматически загрузит датасет и обучит модель.  
При повторных запусках предложит использовать сохранённую модель или переобучить.

## Как это работает

```
отзыв → TfidfVectorizer → LogisticRegression → позитивный / негативный
```

Обучающий датасет содержит отзывы с оценками 1–5. Оценки конвертируются в метки:

- `4–5` → `positive`
- `1–2` → `negative`
- `3` → пропускается (неоднозначно)

Модель обучается на `100 000` примерах с разбивкой `80/20` на train/test.

## Пример работы

```
Введите отзыв (или 'выход' для завершения):

Отзыв: Отличный товар, очень доволен покупкой!
  → Позитивный (уверенность: 94.3%)

Отзыв: Пришёл бракованный, требую возврата
  → Негативный (уверенность: 88.7%)
```
