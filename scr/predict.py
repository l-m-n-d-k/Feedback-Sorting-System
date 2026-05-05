import joblib

pipeline = joblib.load("sentiment_model.pkl")

def predict(text: str) -> dict:
    label = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    confidence = round(max(proba) * 100, 1)

    icons = {"positive": "Позитивный", "negative": "Негативный"}
    return {
        "label": label,
        "display": icons[label],
        "confidence": confidence,
    }

print("Введите отзыв (или 'выход' для завершения):\n")
while True:
    text = input("Отзыв: ").strip()
    if text.lower() in ("выход", "exit", "quit"):
        break
    if not text:
        continue

    result = predict(text)
    print(f"  → {result['display']} (уверенность: {result['confidence']}%)\n")
