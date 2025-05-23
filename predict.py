import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Chargement des données
train_dir = 'dataset/train'
val_dir = 'dataset/val'

img_size = 150
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(img_size, img_size),
                                               batch_size=batch_size,
                                               class_mode='binary')

val_data = val_datagen.flow_from_directory(val_dir,
                                           target_size=(img_size, img_size),
                                           batch_size=batch_size,
                                           class_mode='binary')

# Modèle CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # binaire
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=10)

# Sauvegarde
os.makedirs('model', exist_ok=True)
model.save('model/fracture_model.h5')
