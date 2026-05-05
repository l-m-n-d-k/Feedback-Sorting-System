import itertools
import joblib
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from config import DATASET_NAME, MAX_SAMPLES, MODEL_PATH


class DataLoader:
    def __init__(self):
        self.dataset_name = DATASET_NAME
        self.max_samples = MAX_SAMPLES

    def load(self) -> tuple[list, list]:
        print(f"Загружаем датасет '{self.dataset_name}' (стриминг)...")
        ds = load_dataset(self.dataset_name, streaming=True)

        texts, labels = [], []

        for example in itertools.islice(ds["train"], self.max_samples):
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

        self._print_stats(labels)
        return texts, labels

    def _print_stats(self, labels: list):
        print(f"Собрано примеров: {len(labels)}")
        print(f"  позитивных: {labels.count('positive')}")
        print(f"  негативных: {labels.count('negative')}")


class FeedbackTrainer:
    def __init__(self):
        self.pipeline: Pipeline | None = None

    def train(self, texts: list, labels: list):
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        self.pipeline = Pipeline([
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
        self.pipeline.fit(X_train, y_train)

        print("\nРезультаты на тестовой выборке:")
        y_pred = self.pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))

    def save(self):
        joblib.dump(self.pipeline, MODEL_PATH)
        print(f"Модель сохранена в {MODEL_PATH}")
