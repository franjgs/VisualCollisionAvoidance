#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 15:53:29 2025

@author: fran
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

from tensorflow.keras.applications import VGG16, MobileNetV2, MobileNetV3Small, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess_input
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenetv3_preprocess_input
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
import efficientnet.keras as efn

# Import functions from the utils folder
from utils.data_processing import (
    create_image_directories,
    generate_paired_file_lists,
    process_and_save_frames
)
from utils.plotting_utils import (
    example_errors,
    plot_training_history
)

# Configure GPU memory growth
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# --- Define Model Choice ---
MODEL_NAME = "EfficientNetB0"  # Options: "VGG16", "MobileNetV2", "MobileNetV3Small", "EfficientNetB0", "ResNet50"

# --- Main Script ---
# Setup directories
output_base_dir = 'image_data'
train_dir, test_dir = create_image_directories(output_base_dir)

# Load and process data
video_files, excel_files = generate_paired_file_lists(range_min=70, range_max=93)
all_frames, all_labels, all_filenames = process_and_save_frames(
    excel_files, video_files, output_base_dir, target_size=(224, 224)
)

# --- Data Generators ---
batch_size = 20
datagen_train = ImageDataGenerator(
    preprocessing_function=None,  # Preprocessing handled by model-specific function
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=[0.9, 1.1],
    fill_mode='nearest'
)
datagen_test = ImageDataGenerator(
    preprocessing_function=None,  # Preprocessing handled by model-specific function
)

def get_preprocessing_function(model_name):
    if model_name == "VGG16":
        return vgg16_preprocess_input
    elif model_name in ["MobileNetV2", "MobileNetV3Small"]:
        return mobilenetv2_preprocess_input
    elif model_name.startswith("EfficientNet"):
        return efn.preprocess_input
    elif model_name == "ResNet50":
        return resnet_preprocess_input
    else:
        return None

preprocess_function = get_preprocessing_function(MODEL_NAME)
if preprocess_function:
    datagen_train.preprocessing_function = preprocess_function
    datagen_test.preprocessing_function = preprocess_function
else:
    print(f"Warning: No specific preprocessing function found for {MODEL_NAME}. Using default.")

generator_train = datagen_train.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=batch_size, class_mode='sparse', shuffle=True
)
generator_test = datagen_test.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=batch_size, class_mode='sparse', shuffle=False
)

steps_test = int(np.ceil(generator_test.n / batch_size))
class_names = list(generator_train.class_indices.keys())

# --- Class Weights ---
class_weight = dict(enumerate(
    compute_class_weight(class_weight='balanced', classes=np.unique(generator_train.classes), y=generator_train.classes)
))

# --- Model Setup ---
input_shape = (224, 224, 3)
base_model = None
top_model = None
model = None

if MODEL_NAME == "VGG16":
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    top_model = Sequential([GlobalAveragePooling2D(), Dense(1024, activation='relu'), Dropout(0.5), Dense(len(class_names), activation='softmax')])
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
elif MODEL_NAME == "MobileNetV2":
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    top_model = Sequential([GlobalAveragePooling2D(), Dense(1024, activation='relu'), Dropout(0.5), Dense(len(class_names), activation='softmax')])
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
elif MODEL_NAME == "MobileNetV3Small":
    base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=input_shape)
    top_model = Sequential([GlobalAveragePooling2D(), Dense(1024, activation='relu'), Dropout(0.5), Dense(len(class_names), activation='softmax')])
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
elif MODEL_NAME.startswith("EfficientNet"):
    base_model_fn = getattr(efn, MODEL_NAME)
    base_model = base_model_fn(weights='imagenet', include_top=False, input_shape=input_shape)
    top_model = Sequential([GlobalAveragePooling2D(), Dense(1024, activation='relu'), Dropout(0.5), Dense(len(class_names), activation='softmax')])
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
elif MODEL_NAME == "ResNet50":
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    top_model = Sequential([GlobalAveragePooling2D(), Dense(1024, activation='relu'), Dropout(0.5), Dense(len(class_names), activation='softmax')])
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
else:
    raise ValueError(f"Model name '{MODEL_NAME}' not recognized.")

# --- Initial Training (Freeze Base Layers for CNNs) ---
if base_model is not None:
    base_model.trainable = False
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    epochs = 20
    steps_per_epoch = int(np.ceil(generator_train.n / batch_size))

    history = model.fit(
        generator_train,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=generator_test,
        validation_steps=steps_test,
        class_weight=class_weight
    )

    # --- Fine-Tuning (Unfreeze some layers for CNNs) ---
    if MODEL_NAME == "VGG16":  # Apply fine-tuning only for VGG16
        base_model.trainable = True
        for layer in base_model.layers:
            layer.trainable = 'block5' in layer.name or 'block4' in layer.name

        model.compile(optimizer=Adam(learning_rate=2e-8), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history_fine = model.fit(
            generator_train,
            epochs=epochs * 2,
            initial_epoch=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=generator_test,
            validation_steps=steps_test,
            class_weight=class_weight
        )
    else:
        history_fine = None # Set history_fine to None for other models

else:
    print(f"Skipping initial training/fine-tuning for {MODEL_NAME}.")
    history = None
    history_fine = None

# --- Evaluation (for single frame CNNs) ---
if model is not None:
    model.evaluate(generator_test, steps=steps_test)
    generator_test.reset()
    y_pred = model.predict(generator_test, steps=steps_test)
    cls_pred = np.argmax(y_pred, axis=1)
    cls_true = generator_test.classes
    example_errors(cls_true, cls_pred, generator_test, class_names)

    # --- Save Results ---
    timestamp = int(time.time())
    name = f"{MODEL_NAME}-collision-avoidance-{timestamp}"
    output_dir = "models"  # Output directory for models and plots
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    if history:
        fig = plot_training_history(history, MODEL_NAME, history_fine, save_path=f"{output_dir}/{name}.pdf")
        plt.show()  # Optional: Display plot (remove in non-interactive environments)

    model.save(f"{output_dir}/{name}.keras")
    if history_fine:
        with open(f"{output_dir}/trainHistoryDict_fine_{timestamp}.pkl", 'wb') as f:
            pickle.dump(history_fine.history, f)
    elif history:
        with open(f"{output_dir}/trainHistoryDict_{timestamp}.pkl", 'wb') as f:
            pickle.dump(history.history, f)

session.close()