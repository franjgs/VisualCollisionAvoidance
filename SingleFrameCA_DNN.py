#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for training and evaluating collision avoidance Deep Neural Networks.
Supports single-frame CNNs models.

Created on Mon May 19 16:30:54 2025
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
# Use tf.keras.optimizers.legacy.Adam if on M1/M2 Mac
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers.legacy import Adam # Using legacy Adam for broader compatibility

from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

# Import functions from the utils folder
from utils.data_processing import (
    create_image_directories,
    generate_paired_file_lists,
    process_and_save_frames,
    process_and_save_frames_cars_dataset, # Use the new function
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
    
    
# --- Main Configuration Section ---
# Choose which dataset to use: "drone" or "cars"
DATASET_TO_USE = "cars" # "drones" #  <--- IMPORTANT: Change this to "drones" or "cars" as needed

# --- Common Setup (applies to both datasets) ---
MODEL_LIST = ["VGG16", "ResNet50", "EfficientNet", "MobileNetV2", "EfficientNetB0"]
# MODEL_LIST = ["VGG16"]

output_base_dir_for_images = os.path.join(DATASET_TO_USE,f'image_data_{DATASET_TO_USE}')
output_dir_for_models = os.path.join(DATASET_TO_USE,"models")
os.makedirs(output_dir_for_models, exist_ok=True)
output_dir_for_results = os.path.join(DATASET_TO_USE,"results")
os.makedirs(output_dir_for_results, exist_ok=True)

# Setup directories
train_dir = os.path.join(output_base_dir_for_images, 'train')
test_dir = os.path.join(output_base_dir_for_images, 'test')


TARGET_IMAGE_SIZE = (224, 224)
TRAIN_VALIDATION_SPLIT_RATIO = 0.8


# --- Dataset-Specific Configuration and Processing Logic ---
print(f"--- Processing data for the '{DATASET_TO_USE}' dataset ---")
log_memory()

all_filenames = [] # Initialize all_filenames here

if DATASET_TO_USE == "drones":
    print("Configuring for Drone Collision Dataset...")
    # Drone dataset specific parameters
    DRONE_RANGE_MIN = 1
    DRONE_RANGE_MAX = 93
    # Removed DRONE_FRAMES_PER_SECOND as process_and_save_frames does not use it.
    # The Excel files define which frames are processed.

    # Load and process drone data
    video_files, excel_files = generate_paired_file_lists(
        range_min=DRONE_RANGE_MIN, range_max=DRONE_RANGE_MAX,
        base_dir = f'{DATASET_TO_USE}'
    )
    all_filenames = process_and_save_frames(
        excel_files,
        video_files,
        output_base_dir_for_images,
        target_size=TARGET_IMAGE_SIZE,
        train_ratio=TRAIN_VALIDATION_SPLIT_RATIO # Pass train_ratio explicitly
    )


elif DATASET_TO_USE == "cars":
    print("Configuring for Car Crashes Dataset...")
    # Car crashes dataset specific parameters
    CAR_LABELS_CSV_FILE = os.path.join(DATASET_TO_USE,'data_labels.csv')
    CAR_SOURCE_VIDEOS_DIRECTORY = os.path.join(DATASET_TO_USE,'videos') # Points to the folder containing car videos
    CAR_FRAMES_PER_SECOND = 15 # Orginally, 30 fps
    CAR_NUM_COLLISION_VIDEOS = 25
    CAR_NUM_NO_COLLISION_VIDEOS = 15

    # Call the updated function for car crashes dataset
    all_filenames = process_and_save_frames_cars_dataset(
        CAR_LABELS_CSV_FILE,
        CAR_SOURCE_VIDEOS_DIRECTORY,
        output_base_dir_for_images,
        target_size=TARGET_IMAGE_SIZE,
        train_ratio=TRAIN_VALIDATION_SPLIT_RATIO,
        num_collision_videos_to_process=CAR_NUM_COLLISION_VIDEOS,
        num_no_collision_videos_to_process=CAR_NUM_NO_COLLISION_VIDEOS,
        frames_per_second=CAR_FRAMES_PER_SECOND
    )

else:
    raise ValueError(f"Unknown DATASET_TO_USE: {DATASET_TO_USE}. Please choose 'drone' or 'cars'.")

log_memory()
print(f"Finished data processing for {DATASET_TO_USE} dataset. Total frames saved: {len(all_filenames)}")

# --- Data Generators ---
batch_size = 20 # 8 # 20
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
n_hidden = 1024

model_configs = {
    "binary_sigmoid": {
        "num_output_units": 1,
        "activation_function": "sigmoid",
        "loss_function": "binary_crossentropy"
    },
    "multiclass_softmax": {
        "num_output_units": 2, # len(class_names), # This will be 2 for binary classification
        "activation_function": "softmax",
        "loss_function": "sparse_categorical_crossentropy" # Or 'categorical_crossentropy' if your labels were one-hot encoded
    }
}
selected_config = model_configs['binary_sigmoid']

  
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

    # --- Model creation --- 
    if MODEL_NAME == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        model = Sequential([
            base_model,
            Flatten(),
            Dense(n_hidden, activation='relu'),
            Dropout(0.5),
            Dense(
                selected_config["num_output_units"], # Use selected_config here
                activation=selected_config["activation_function"]
            )
        ])
        epochs = 20
    elif MODEL_NAME == "MobileNetV2":
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        model = Sequential([
            base_model,
            Flatten(),
            Dense(n_hidden, activation='relu'),
            Dropout(0.5),
            Dense(
                selected_config["num_output_units"], # Use selected_config here
                activation=selected_config["activation_function"]
            )
        ])
        epochs = 20
    elif MODEL_NAME == "MobileNetV3Small":
        base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=input_shape)
        model = Sequential([
            base_model,
            Flatten(),
            Dense(n_hidden, activation='relu'),
            Dropout(0.5),
            Dense(
                selected_config["num_output_units"], # Use selected_config here
                activation=selected_config["activation_function"]
            )
        ])
        epochs = 40
    elif MODEL_NAME.startswith("EfficientNet"):
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        model = Sequential([
            base_model,
            Flatten(),
            Dense(n_hidden, activation='relu'),
            Dropout(0.5),
            Dense(
                selected_config["num_output_units"], # Use selected_config here
                activation=selected_config["activation_function"]
            )
        ])
        epochs = 60
    elif MODEL_NAME == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        model = Sequential([
           base_model,
           Flatten(),
           Dense(n_hidden, activation='relu'),
           Dropout(0.5),
           Dense(
               selected_config["num_output_units"], # Use selected_config here
               activation=selected_config["activation_function"]
           )
       ])
        epochs = 20
    else:
        raise ValueError(f"Model name '{MODEL_NAME}' not recognized.")

    # --- Initial Training ---
    if base_model is not None:
        base_model.trainable = False
        model.compile(optimizer=Adam(learning_rate=1e-5), 
                      loss=selected_config["loss_function"],
                      metrics=['accuracy'])
        steps_per_epoch = int(np.ceil(generator_train.n / batch_size))
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(
            generator_train,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=generator_test,
            validation_steps=steps_test,
            class_weight=class_weight,
            # callbacks=[early_stopping]
        )

        # --- Fine-Tuning ---
        base_model.trainable = True
        fine_tune_epochs = epochs * 2
        if MODEL_NAME == "VGG16":
            for layer in base_model.layers:
                layer.trainable = 'block5' in layer.name or 'block4' in layer.name
                
            # Use the legacy Adam optimizer for M1/M2 Mac compatibility
            # Always create a NEW optimizer instance when recompiling after changing trainable status
            fine_tune_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-8)
        
            model.compile(optimizer=fine_tune_optimizer,
                          loss=selected_config["loss_function"],
                          metrics=['accuracy'])
            
            history_fine = model.fit(
                generator_train,
                epochs=fine_tune_epochs,
                initial_epoch=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=generator_test,
                validation_steps=steps_test,
                class_weight=class_weight,
                callbacks=[early_stopping]
                # Better, early stopping
            )
        elif MODEL_NAME == "MobileNetV2":
            # Works well
            for layer in base_model.layers:
                layer.trainable = 'block16' in layer.name or 'Conv_1' in layer.name
                
            # Always create a NEW optimizer instance when recompiling after changing trainable status
            fine_tune_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
        
            model.compile(optimizer=fine_tune_optimizer,
                          loss=selected_config["loss_function"],
                          metrics=['accuracy'])

            history_fine = model.fit(
                generator_train,
                epochs=fine_tune_epochs,
                initial_epoch=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=generator_test,
                validation_steps=steps_test,
                class_weight=class_weight,
                # callbacks=[early_stopping]
            )
        elif MODEL_NAME == "ResNet50":
            # Set all layers in the base model to trainable=True by default first.
            # Then, selectively set some to False and handle BatchNormalization layers.
            for layer in base_model.layers:
                layer.trainable = True # Start by making all base model layers trainable

            # Now, iterate again to specifically set trainability based on block name
            # and to explicitly keep BatchNormalization layers frozen.
            for layer in base_model.layers:
                # Unfreeze 'conv5' blocks (the last stage of ResNet50)
                # If a layer's name contains 'conv5', it should be trainable.
                # All other layers will remain trainable=False if not explicitly set to True above,
                # but we'll ensure only conv5 is trainable if that's the strategy.
                if 'conv5' in layer.name:
                    layer.trainable = True
                else:
                    layer.trainable = False # Explicitly freeze layers outside conv5

                # Crucially, keep BatchNormalization layers frozen.
                # This prevents instability from updating BN stats with potentially different data distributions or small batch sizes.
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False

            # Calculate decay steps for the CosineDecay learning rate schedule.
            decay_steps = steps_per_epoch * fine_tune_epochs

            # Define the CosineDecay learning rate schedule.
            # Start with a significantly lower learning rate for fine-tuning (e.g., 1e-6 or even 1e-7)
            # This makes updates smaller and helps prevent "forgetting" pre-trained weights.
            cosine_lr = CosineDecay(initial_learning_rate=1e-6, decay_steps=decay_steps, alpha=0.1)

            # Create a NEW instance of the optimizer for the fine-tuning phase.
            # This is essential to prevent issues with the optimizer's internal state
            # when the model's trainable parameters change.
            # Use the legacy Adam optimizer for better compatibility on M1/M2 Macs.
            fine_tune_optimizer = tf.keras.optimizers.Adam(learning_rate=cosine_lr)

            # Recompile the model with the new optimizer (which uses the CosineDecay schedule).
            # Recompilation is necessary for the changes in layer.trainable status to take effect.
            model.compile(optimizer=fine_tune_optimizer,
                          loss=selected_config["loss_function"],
                          metrics=['accuracy'])

            # Continue training the model with the unfrozen layers and new learning rate schedule.
            history_fine = model.fit(
                generator_train,
                epochs=fine_tune_epochs,
                initial_epoch=epochs, # Start fine-tuning from where initial training left off.
                steps_per_epoch=steps_per_epoch,
                validation_data=generator_test,
                validation_steps=steps_test,
                class_weight=class_weight,
                # Callbacks like early_stopping can be added here if desired.
                # However, for observing the transient, you might temporarily remove them.
            )
        else:
            class DummyHistory:
                """
                A simple class to mimic the structure of a Keras History object.
                """
                def __init__(self):
                    self.history = {} # Initialize the 'history' attribute as an empty dictionary
            
            # To create your dummy object:
            history_fine = DummyHistory()

    # --- Evaluation ---
    if model is not None:
        model.evaluate(generator_test, steps=steps_test)
        generator_test.reset()
        y_pred = model.predict(generator_test, steps=steps_test)
        # --- Adapt cls_pred calculation based on the chosen model configuration ---
        if selected_config["activation_function"] == "sigmoid":
            # For sigmoid output (binary_sigmoid config), threshold at 0.5
            cls_pred = (y_pred > 0.5).astype(int).flatten()
        elif selected_config["activation_function"] == "softmax":
            # For softmax output (multiclass_softmax config), take argmax
            cls_pred = np.argmax(y_pred, axis=1)
        else:
            # Fallback or error if an unexpected activation is used
            print(f"Warning: Unexpected activation function '{selected_config['activation_function']}'. Defaulting to argmax for cls_pred.")
            cls_pred = np.argmax(y_pred, axis=1)
        
        cls_true = generator_test.classes
        # Pass output_dir and MODEL_NAME to example_errors
        example_errors(cls_true, cls_pred, generator_test, class_names, output_dir=output_dir_for_results, model_name=MODEL_NAME)
        # --- Save Results ---
        timestamp = int(time.time())
        name = f"{MODEL_NAME}_collision_avoidance_{timestamp}"
        if any(history_fine.history):
            fig = plot_training_history(history, MODEL_NAME, history_fine, save_path=f"{output_dir_for_results}/{name}.pdf")
        elif any(history.history):
            fig = plot_training_history(history, MODEL_NAME, None, save_path=f"{output_dir_for_results}/{name}.pdf")
        plt.show()
        
        model.save(f"{output_dir_for_models}/{name}.keras")
        if any(history_fine.history):
            with open(f"{output_dir_for_models}/{MODEL_NAME}_trainHistoryDict_fine_{timestamp}.pkl", 'wb') as f:
                pickle.dump(history_fine.history, f)
        elif any(history.history):
            with open(f"{output_dir_for_models}/{MODEL_NAME}_trainHistoryDict_{timestamp}.pkl", 'wb') as f:
                pickle.dump(history.history, f)

    # --- Clear memory --- 
    tf.keras.backend.clear_session()
    del model, base_model, top_model, generator_train, generator_test
    gc.collect()
    log_memory()

session.close()
