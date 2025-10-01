import pickle
import json
import yaml
import sys
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score


# Функция для загрузки параметров из yaml файла
def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


# Функция для загрузки метрик
def load_metrics():
    metrics_path = Path("metrics/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


# Функция для проверки модели
def validate_model():
    params = load_params()
    accuracy_min = params.get("accuracy_min", 0.0)

    # Загружаем метрики
    metrics = load_metrics()
    if metrics and "accuracy" in metrics:
        accuracy = metrics["accuracy"]
        print(f"Используем метрику из metrics.json: accuracy = {accuracy:.4f}")
    else:
        # Если метрик нет то пересчет accuracy
        print("metrics.json не найден или accuracy отсутствует")

        # Загрузка модели
        with open("models/model.pkl", "rb") as f:
            model = pickle.load(f)

        # Загрузка данных
        df = pd.read_csv("data/processed/dataset.csv")
        X = df[["total_bill", "size"]]
        y = df["high_tip"]
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"Пересчитанная accuracy = {accuracy:.4f}")

    print(f"Порог accuracy_min = {accuracy_min:.4f}")

    # Проверка порога
    if accuracy < accuracy_min:
        print(f"Модель не прошла минимальный порог ({accuracy} < {accuracy_min})")
        sys.exit(1)
    else:
        print(f"Модель прошла минимальный порог ({accuracy} < {accuracy_min})")
        sys.exit(0)


if __name__ == "__main__":
    validate_model()
