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
import psutil
import gc

from tensorflow.keras.applications import EfficientNetB0, VGG16, MobileNetV2, MobileNetV3Small, ResNet50
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess_input
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenetv3_preprocess_input
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

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

# Memory logging
def log_memory():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / 1024**2:.2f} MB")
    
    
# --- Define Model Choice ---
MODEL_LIST = ["EfficientNet", "MobileNetV2", "ResNet50", "EfficientNetB0", "MobileNetV3Small"] # "ResNet50" # Options: "VGG16", "MobileNetV2", "MobileNetV3Small", "EfficientNet",

# --- Main Script ---
# Setup directories
output_base_dir = 'image_data'
train_dir, test_dir = create_image_directories(output_base_dir)

output_dir = "models"  # Output directory for models and plots
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Load and process data
log_memory()
video_files, excel_files = generate_paired_file_lists(range_min=1, range_max=93)
all_filenames = process_and_save_frames(
    excel_files, video_files, output_base_dir, target_size=(224, 224)
)
log_memory()

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
        return efficientnet_preprocess_input # Use the TF built-in preprocess_input
    elif model_name == "ResNet50":
        return resnet_preprocess_input
    else:
        return None

# --- Model Setup ---
input_shape = (224, 224, 3)
    
for MODEL_NAME in MODEL_LIST:
    print(f"\nTraining model: {MODEL_NAME}")
    log_memory()
    
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

    base_model = None
    top_model = None
    model = None

    # Model creation code remains the same
    if MODEL_NAME == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        top_model = Sequential([GlobalAveragePooling2D(), Dense(1024, activation='relu'), Dropout(0.5), Dense(len(class_names), activation='softmax')])
        model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
        epochs = 20
    elif MODEL_NAME == "MobileNetV2":
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        top_model = Sequential([GlobalAveragePooling2D(), Dense(1024, activation='relu'), Dropout(0.5), Dense(len(class_names), activation='softmax')])
        model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
        epochs = 20
    elif MODEL_NAME == "MobileNetV3Small":
        base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=input_shape)
        top_model = Sequential([GlobalAveragePooling2D(), Dense(1024, activation='relu'), Dropout(0.5), Dense(len(class_names), activation='softmax')])
        model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
        epochs = 20
    elif MODEL_NAME.startswith("EfficientNet"):
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        top_model = Sequential([GlobalAveragePooling2D(), Dense(1024, activation='relu'), Dropout(0.5), Dense(len(class_names), activation='softmax')])
        model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
        epochs = 40
    elif MODEL_NAME == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        top_model = Sequential([GlobalAveragePooling2D(), Dense(1024, activation='relu'), Dropout(0.5), Dense(len(class_names), activation='softmax')])
        model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
        epochs = 40
    else:
        raise ValueError(f"Model name '{MODEL_NAME}' not recognized.")

    # Training and fine-tuning code remains the same
    if base_model is not None:
        base_model.trainable = False
        model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        steps_per_epoch = int(np.ceil(generator_train.n / batch_size))

        history = model.fit(
            generator_train,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=generator_test,
            validation_steps=steps_test,
            class_weight=class_weight
        )

        base_model.trainable = True
        if MODEL_NAME == "VGG16":
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
        elif MODEL_NAME == "MobileNetV2":
            for layer in base_model.layers:
                layer.trainable = 'block16' in layer.name or 'Conv_1' in layer.name
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
        elif MODEL_NAME == "ResNet50":
            for layer in base_model.layers:
                layer.trainable = 'conv5' in layer.name
            decay_steps = steps_per_epoch * epochs
            cosine_lr = CosineDecay(initial_learning_rate=1e-4, decay_steps=decay_steps, alpha=0.1)
            plateau_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-9, verbose=1)
            model.compile(optimizer=Adam(learning_rate=2e-8), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            history_fine = model.fit(
                generator_train,
                epochs=epochs * 2,
                initial_epoch=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=generator_test,
                validation_steps=steps_test,
                class_weight=class_weight,
                callbacks=[plateau_cb]
            )
        else:
            history_fine = None

    # Evaluation and saving code remains the same
    if model is not None:
        model.evaluate(generator_test, steps=steps_test)
        generator_test.reset()
        y_pred = model.predict(generator_test, steps=steps_test)
        cls_pred = np.argmax(y_pred, axis=1)
        cls_true = generator_test.classes
        example_errors(cls_true, cls_pred, generator_test, class_names)

        timestamp = int(time.time())
        name = f"{MODEL_NAME}-collision-avoidance-{timestamp}"
        if history:
            fig = plot_training_history(history, MODEL_NAME, history_fine, save_path=f"{output_dir}/{name}.pdf")
            plt.show()

        model.save(f"{output_dir}/{name}.keras")
        if history_fine:
            with open(f"{output_dir}/trainHistoryDict_fine_{timestamp}.pkl", 'wb') as f:
                pickle.dump(history_fine.history, f)
        elif history:
            with open(f"{output_dir}/trainHistoryDict_{timestamp}.pkl", 'wb') as f:
                pickle.dump(history.history, f)

    # Clear memory
    tf.keras.backend.clear_session()
    del model, base_model, top_model, generator_train, generator_test
    gc.collect()
    log_memory()

session.close()