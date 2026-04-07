import os
import matplotlib.pyplot as plt
from preprocess import load_data, preprocess_and_split
from model import build_model


def plot_history(history):
    """Eğitim sürecindeki başarı ve kayıp grafiklerini çizer."""
    plt.figure(figsize=(12, 4))

    # Accuracy Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Başarısı')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Başarısı')
    plt.title('Model Başarısı (Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()

    # Loss Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.title('Model Kaybı (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()

    plt.tight_layout()
    plt.show()


def run_training():
    # 1. Verileri Hazırla
    print("--- 1. AŞAMA: Veriler Hazırlanıyor ---")
    X, y = load_data()

    if len(X) == 0:
        print("Hata: Veri seti bulunamadı!")
        return

    X_train, X_test, y_train, y_test = preprocess_and_split(X, y)
    print(f"Eğitim seti: {X_train.shape}, Test seti: {X_test.shape}")

    # 2. Modeli Kur
    print("\n--- 2. AŞAMA: Model İnşa Ediliyor ---")
    model = build_model(input_shape=(224, 224, 3))
    model.summary()

    # 3. Eğitimi Başlat
    print("\n--- 3. AŞAMA: Eğitim Başlıyor ---")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2
    )

    # --- GRAFİKLERİ GÖSTER ---
    # Eğitim biter bitmez grafikleri ekrana basar
    plot_history(history)

    # 4. Modeli Kaydet
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    save_path = os.path.join(project_root, 'brain_tumor_model.h5')

    model.save(save_path)
    print(f"\nİşlem Tamamlandı! Model buraya kaydedildi: {save_path}")

if __name__ == "__main__":
    run_training()