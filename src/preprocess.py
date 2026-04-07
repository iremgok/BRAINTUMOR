import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def get_data_path():
    """Proje yapısına göre data klasörünün yolunu dinamik olarak döndürür."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    return os.path.join(project_root, 'data')


def load_data(img_size=(224, 224)):
    """Verileri okur ve numpy dizilerine çevirir."""
    data_dir = get_data_path()
    images = []
    labels = []
    categories = {'no': 0, 'yes': 1}

    for category, label in categories.items():
        path = os.path.join(data_dir, category)
        if not os.path.exists(path):
            print(f"Uyarı: {path} dizini bulunamadı!")
            continue

        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Hata: {img_name} okunamadı. {e}")

    return np.array(images), np.array(labels)


def preprocess_and_split(images, labels, test_size=0.2):
    """Normalizasyon ve veri setini bölme işlemini yapar."""
    images = images.astype('float32') / 255.0
    return train_test_split(images, labels, test_size=test_size, random_state=42)


if __name__ == "__main__":
    # Test amaçlı çalıştırma
    X, y = load_data()
    print(f"Veriler yüklendi. Toplam görsel: {len(X)}")