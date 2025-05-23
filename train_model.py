import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# === Configuration des chemins et des hyperparamètres ===
train_dir = 'dataset/train'
val_dir = 'dataset/val'

img_size = 150
batch_size = 32
epochs = 10

# === Générateur personnalisé qui ignore les images corrompues ===
def custom_image_generator(directory, batch_size, img_size):
    classes = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    class_indices = {cls_name: idx for idx, cls_name in enumerate(classes)}
    image_paths = []
    labels = []

    for cls in classes:
        cls_dir = os.path.join(directory, cls)
        for fname in os.listdir(cls_dir):
            path = os.path.join(cls_dir, fname)
            image_paths.append(path)
            labels.append(class_indices[cls])
    
    num_samples = len(image_paths)

    while True:
        batch_images = []
        batch_labels = []
        count = 0

        for i in range(num_samples):
            if count == batch_size:
                yield (np.array(batch_images), np.array(batch_labels))
                batch_images = []
                batch_labels = []
                count = 0

            path = image_paths[i]
            label = labels[i]

            try:
                img = load_img(path, target_size=(img_size, img_size))
                img = img_to_array(img) / 255.0
                batch_images.append(img)
                batch_labels.append(label)
                count += 1
            except Exception as e:
                print(f"Image corrompue ignorée : {path}")

        if batch_images:
            yield (np.array(batch_images), np.array(batch_labels))

# === Calcul du nombre d'images dans chaque dossier ===
def count_images(directory):
    total = 0
    for root, dirs, files in os.walk(directory):
        total += len([f for f in files if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))])
    return total

train_num = count_images(train_dir)
val_num = count_images(val_dir)

# === Création des générateurs ===
train_data = custom_image_generator(train_dir, batch_size, img_size)
val_data = custom_image_generator(val_dir, batch_size, img_size)

# === Définition du modèle CNN ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Classification binaire
])

# === Compilation du modèle ===
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# === Entraînement du modèle ===
history = model.fit(
    train_data,
    steps_per_epoch=train_num // batch_size,
    validation_data=val_data,
    validation_steps=val_num // batch_size,
    epochs=epochs
)

# === Sauvegarde du modèle entraîné ===
os.makedirs('model', exist_ok=True)
model.save('model/fracture_model.h5')
print("Modèle sauvegardé dans le fichier 'model/fracture_model.h5'")
