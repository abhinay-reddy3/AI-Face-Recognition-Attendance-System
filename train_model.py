import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras import layers, models

STUDENTS_DIR = "dataset/Students"
KAGGLE_DIR = "dataset/Kaggle_Faces"
MODEL_DIR = "models"
TARGET_SIZE = (160, 160)

os.makedirs(MODEL_DIR, exist_ok=True)

print("[INFO] Loading MobileNetV2 feature extractor...")
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(160, 160, 3),
    pooling="avg"
)

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml" if hasattr(cv2, 'data') else "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise RuntimeError("Could not load Haar Cascade classifier.")

def get_embedding_from_rgb(rgb_face):
    face_resized = cv2.resize(rgb_face, TARGET_SIZE)
    face_arr = face_resized.astype(np.float32)
    face_arr = preprocess_input(face_arr)
    face_tensor = np.expand_dims(face_arr, axis=0)
    embedding = base_model(face_tensor, training=False).numpy().flatten()
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding

def extract_embeddings(image_path, augment=False):
    img = cv2.imread(image_path)
    if img is None:
        return []

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=4)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = rgb_img[y:y+h, x:x+w]
    else:
        face = rgb_img

    embeddings = [get_embedding_from_rgb(face)]

    # Augment student photos (horizontal flip + slight brightness shifts)
    if augment:
        flipped = cv2.flip(face, 1)
        embeddings.append(get_embedding_from_rgb(flipped))
        bright = np.clip(face.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
        embeddings.append(get_embedding_from_rgb(bright))
        dark = np.clip(face.astype(np.float32) * 0.85, 0, 255).astype(np.uint8)
        embeddings.append(get_embedding_from_rgb(dark))

    return embeddings

def main():
    X = []
    y = []
    class_names = []

    if not os.path.exists(STUDENTS_DIR):
        raise RuntimeError(f"Directory '{STUDENTS_DIR}' not found!")

    student_folders = sorted([d for d in os.listdir(STUDENTS_DIR) if os.path.isdir(os.path.join(STUDENTS_DIR, d))])
    
    total_student_samples = 0
    for idx, folder_name in enumerate(student_folders):
        class_names.append(folder_name)
        folder_path = os.path.join(STUDENTS_DIR, folder_name)
        count = 0
        for img_name in os.listdir(folder_path):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                embs = extract_embeddings(os.path.join(folder_path, img_name), augment=True)
                for emb in embs:
                    X.append(emb)
                    y.append(idx)
                    count += 1
        total_student_samples += count
        print(f"[STUDENT] Class {idx}: '{folder_name}' -> {count} augmented samples loaded.")

    if len(class_names) == 0 or total_student_samples == 0:
        raise RuntimeError("No student images found inside 'dataset/Students/'! Check your folder path.")

    # Balance Kaggle samples to ~1.5x total student samples so it never overpowers students
    unknown_idx = len(class_names)
    class_names.append("Unknown")
    max_kaggle = max(60, int(total_student_samples * 1.5))
    kaggle_count = 0

    if os.path.exists(KAGGLE_DIR):
        for root, _, files in os.walk(KAGGLE_DIR):
            for img_name in files:
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    embs = extract_embeddings(os.path.join(root, img_name), augment=False)
                    for emb in embs:
                        X.append(emb)
                        y.append(unknown_idx)
                        kaggle_count += 1
                    if kaggle_count >= max_kaggle:
                        break
            if kaggle_count >= max_kaggle:
                break
    print(f"[BACKGROUND] Class {unknown_idx}: 'Unknown' (Kaggle Dataset) -> {kaggle_count} balanced samples loaded.")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    print("\n[INFO] Training balanced classifier head...")
    classifier = models.Sequential([
        layers.Input(shape=(1280,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(len(class_names), activation="softmax")
    ])

    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    classifier.fit(X, y, epochs=40, batch_size=16, verbose=1)

    classifier.save(os.path.join(MODEL_DIR, "face_classifier.keras"))
    with open(os.path.join(MODEL_DIR, "class_names.pkl"), "wb") as f:
        pickle.dump(class_names, f)

    print(f"\n[SUCCESS] Training complete! Output labels: {class_names}")

if __name__ == "__main__":
    main()