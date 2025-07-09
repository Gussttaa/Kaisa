# retrain_pipeline.py

import tensorflow as tf
import numpy as np
import os
import cv2
from datetime import datetime
import shutil

# CONFIGS
IMG_SIZE = (224, 224)
NUM_CLASSES = 3  # Altere conforme sua aplicação
BUFFER_DIR = "Buffer/"
BACKUPS_DIR = "Backups/"
MODEL_PATH = "modelo_base.h5"
TFLITE_EXPORT_PATH = "modelo_atualizado.tflite"
BACKUP_H5_DIR = os.path.join(BACKUPS_DIR, "Models-h5")
BACKUP_TFLITE_DIR = os.path.join(BACKUPS_DIR, "Update models-tflite")


def load_buffer_data():
    images = []
    labels = []

    if not os.path.exists(BUFFER_DIR):
        return np.array([]), np.array([])

    for fname in os.listdir(BUFFER_DIR):
        if fname.endswith(".jpg") or fname.endswith(".png"):
            path = os.path.join(BUFFER_DIR, fname)
            label = int(fname.split("_")[0])  # Nome do arquivo: "0_img123.jpg"
            img = cv2.imread(path)
            img = cv2.resize(img, IMG_SIZE)
            img = img.astype(np.float32) / 255.0
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)


def build_model():
    base = tf.keras.applications.MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
    base.trainable = False
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def retrain():
    print("[INFO] Carregando modelo base...")
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        model = build_model()

    print("[INFO] Carregando dados do buffer...")
    x, y = load_buffer_data()

    if len(x) == 0:
        print("[INFO] Nenhuma nova imagem para treinar. Abortando.")
        return

    print(f"[INFO] Iniciando fine-tuning com {len(x)} novas amostras...")
    model.fit(x, y, epochs=10000, batch_size=8, validation_split=0.1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    new_model_path = f"modelo_retreinado_{timestamp}.h5"
    print(f"[INFO] Salvando novo modelo em: {new_model_path}")
    model.save(new_model_path)

    # Backup do modelo .h5
    os.makedirs(BACKUP_H5_DIR, exist_ok=True)
    backup_h5_path = os.path.join(BACKUP_H5_DIR, f"modelo_{timestamp}.h5")
    shutil.copy(new_model_path, backup_h5_path)
    print(f"[INFO] Backup .h5 salvo em: {backup_h5_path}")

    # Exporta para TFLite
    print("[INFO] Exportando para .tflite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(TFLITE_EXPORT_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"[INFO] Modelo .tflite atualizado salvo em: {TFLITE_EXPORT_PATH}")

    # Backup do modelo .tflite
    os.makedirs(BACKUP_TFLITE_DIR, exist_ok=True)
    backup_tflite_path = os.path.join(BACKUP_TFLITE_DIR, f"modelo_{timestamp}.tflite")
    shutil.copy(TFLITE_EXPORT_PATH, backup_tflite_path)
    print(f"[INFO] Backup .tflite salvo em: {backup_tflite_path}")

    # (Opcional) Limpa o buffer após treino
    for f in os.listdir(BUFFER_DIR):
        file_path = os.path.join(BUFFER_DIR, f)
        if os.path.isfile(file_path) and (f.endswith(".jpg") or f.endswith(".png")):
            os.remove(file_path)
    print("[INFO] Buffer limpo.")


if __name__ == '__main__':
    retrain()
