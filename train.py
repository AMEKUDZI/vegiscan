import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ── 1. Download dataset from Kaggle ──────────────────────────────────────────
# Dataset: https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset
# Make sure ~/.kaggle/kaggle.json exists with your Kaggle API credentials

DATASET_DIR = 'dataset'

if not os.path.exists(DATASET_DIR):
    print("Downloading dataset from Kaggle...")
    os.system('kaggle datasets download -d misrakahmed/vegetable-image-dataset --unzip -p dataset')
    print("Download complete.")
else:
    print("Dataset already exists, skipping download.")

# ── 2. Locate train/test directories ─────────────────────────────────────────
TRAIN_DIR = os.path.join(DATASET_DIR, 'Vegetable Images', 'train')
VAL_DIR   = os.path.join(DATASET_DIR, 'Vegetable Images', 'validation')
TEST_DIR  = os.path.join(DATASET_DIR, 'Vegetable Images', 'test')

IMG_SIZE  = (224, 224)
BATCH     = 32
EPOCHS    = 10

# ── 3. Data generators ────────────────────────────────────────────────────────
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.1
)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')
val_data   = val_gen.flow_from_directory(VAL_DIR,   target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')

NUM_CLASSES = len(train_data.class_indices)
print(f"\nClasses ({NUM_CLASSES}): {list(train_data.class_indices.keys())}")

# Save class names for use in app.py
import json
with open('class_names.json', 'w') as f:
    json.dump(train_data.class_indices, f)
print("Saved class_names.json")

# ── 4. Build model (MobileNetV2 transfer learning) ────────────────────────────
base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
base.trainable = False  # freeze base

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── 5. Train ──────────────────────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint('vegiscan_model.h5', save_best_only=True, monitor='val_accuracy', verbose=1),
    EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
]

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\n✅ Training complete. Model saved as vegiscan_model.h5")

# ── 6. Evaluate on test set ───────────────────────────────────────────────────
test_data = val_gen.flow_from_directory(TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')
loss, acc  = model.evaluate(test_data)
print(f"Test Accuracy: {acc*100:.2f}%")
