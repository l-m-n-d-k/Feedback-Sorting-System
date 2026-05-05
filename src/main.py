import os
from config import MODEL_PATH
from train import DataLoader, FeedbackTrainer
from predict import FeedbackPredictor
from cli import CLI


def train_and_save():
    texts, labels = DataLoader().load()
    trainer = FeedbackTrainer()
    trainer.train(texts, labels)
    trainer.save()


if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        print("Найдена сохранённая модель.")
        answer = input("Переобучить? (да/нет): ").strip().lower()
        if answer == "да":
            train_and_save()
    else:
        print("Модель не найдена — запускаем обучение.")
        train_and_save()

    CLI(FeedbackPredictor()).run()
