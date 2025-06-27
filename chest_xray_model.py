import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

# Paths
train_dir = "chest_x_ray_images/train"
test_dir = "chest_x_ray_images/test"
model_path = "chest_xray_model.keras"

# Image generators
train_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, horizontal_flip=True)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=(150,150), batch_size=32, class_mode='binary')
test_data = test_gen.flow_from_directory(test_dir, target_size=(150,150), batch_size=32, class_mode='binary')

# Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_data, validation_data=test_data, epochs=10)

# Save
model.save(model_path)
print(f"✅ Model trained and saved at: {model_path}")
