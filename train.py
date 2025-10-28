import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Path to dataset
dataset_dir = r"C:\xampp\htdocs\signbridge\dataset"

# Check dataset exists
if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"Dataset not found at {dataset_dir}")

print("✅ Dataset directory found:", dataset_dir)

# Preprocess & augment data
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,   # 80% train, 20% validation
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

# Training data generator
train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),    # Resize images to 64x64
    batch_size=32,
    class_mode="sparse",     # labels are integer encoded
    subset="training"
)

# Validation data generator
val_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode="sparse",
    subset="validation"
)

print(f"📂  Training samples: {train_gen.samples}")
print(f"📂 Validation samples: {val_gen.samples}")
print(f"📂 Classes found: {train_gen.class_indices}")

# Build CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(train_gen.num_classes, activation="softmax")
])

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train for 1 epoch (quick test run)
history = model.fit(
    train_gen,
    epochs=1,
    validation_data=val_gen
)

# Save trained model
model.save("sign_model.h5")
print("✅ Model saved as sign_model.h5")
