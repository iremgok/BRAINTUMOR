import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_shape=(224, 224, 3)):
    model = models.Sequential([
        # 1. Katman: Resimdeki basit çizgileri/kenarları yakalar
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        # 2. Katman: Daha karmaşık şekilleri yakalar
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # 3. Katman
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Düzleştirme: 2 Boyutlu resmi tek bir listeye çevirir
        layers.Flatten(),

        # Tam Bağlantılı Katman (Karar verme aşaması)
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Ezberlemeyi (overfitting) önlemek için bazı nöronları kapatır

        # Çıkış Katmanı: 0 (Yok) veya 1 (Var) sonucunu verir
        layers.Dense(1, activation='sigmoid')
    ])

    # Modeli derleyelim (Hata ölçer ve iyileştirici seçimi)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model