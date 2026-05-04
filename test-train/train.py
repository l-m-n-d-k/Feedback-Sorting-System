from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib
import itertools

print("Загружаем датасет (стриминг)...")
ds = load_dataset("nyuuzyou/wb-feedbacks", streaming=True)

MAX_SAMPLES = 50_000
texts, labels = [], []

for example in itertools.islice(ds["train"], MAX_SAMPLES):
    rating = example["productValuation"]
    text = example["text"].strip()

    if not text:
        continue
    if rating >= 4:
        labels.append("positive")
    elif rating <= 2:
        labels.append("negative")
    else:
        continue

    texts.append(text)

print(f"Собрано примеров: {len(texts)}")
print(f"  позитивных: {labels.count('positive')}")
print(f"  негативных: {labels.count('negative')}")

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=100_000,
        sublinear_tf=True,
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=1.0,
    )),
])

print("Обучаем модель...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("\nРезультаты на тестовой выборке:")
print(classification_report(y_test, y_pred))

joblib.dump(pipeline, "sentiment_model.pkl")
print("Модель сохранена в sentiment_model.pkl")
