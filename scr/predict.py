import joblib
from config import MODEL_PATH


class FeedbackPredictor:
    LABELS = {
        "positive": "Позитивный",
        "negative": "Негативный",
    }

    def __init__(self):
        self.pipeline = joblib.load(MODEL_PATH)
        print("Модель загружена.")

    def predict(self, text: str) -> dict:
        label = self.pipeline.predict([text])[0]
        confidence = round(max(self.pipeline.predict_proba([text])[0]) * 100, 1)

        return {
            "label": label,
            "display": self.LABELS[label],
            "confidence": confidence,
        }
