from predict import FeedbackPredictor


class CLI:
    def __init__(self, predictor: FeedbackPredictor):
        self.predictor = predictor

    def run(self):
        print("\nВведите отзыв (или 'выход' для завершения):\n")
        while True:
            text = input("Отзыв: ").strip()
            if text.lower() in ("выход", "exit", "quit"):
                print("До свидания!")
                break
            if not text:
                continue

            result = self.predictor.predict(text)
            self._display(result)
            print()

    def _display(self, result: dict):
        print(f"  → {result['display']} (уверенность: {result['confidence']}%)")
