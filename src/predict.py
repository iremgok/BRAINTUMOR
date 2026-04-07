import tensorflow as tf
import numpy as np
import cv2
import os


def predict_tumor(image_path):
    # 1. Modeli Yükle
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(os.path.dirname(current_dir), 'brain_tumor_model.h5')
    model = tf.keras.models.load_model(model_path)

    # 2. Resmi Hazırla (Ön işleme ile aynı olmalı)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Model 4 boyutlu bekler (batch boyutu)

    # 3. Tahmin Yap
    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        print(f"Sonuç: TÜMÖR TESPİT EDİLDİ (Güven Oranı: %{prediction[0][0] * 100:.2f})")
    else:
        print(f"Sonuç: SAĞLIKLI BEYİN (Güven Oranı: %{(1 - prediction[0][0]) * 100:.2f})")


if __name__ == "__main__":
    # Test etmek için 'data/pred' klasöründen bir resim yolu ver
    # Örnek: r"C:\Users\irem\PycharmProjects\BRAINTUMOR\data\pred\pred10.jpg"
    test_image = input("Test edilecek resmin tam yolunu girin: ")
    predict_tumor(test_image)