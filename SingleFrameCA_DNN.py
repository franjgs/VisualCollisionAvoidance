#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for training and evaluating collision avoidance Deep Neural Networks.
Supports single-frame CNNs models.

This refactored version allows:
1. Training and inference with 'drones', 'cars', or a 'mixed' dataset, controlled by a single variable.
2. Configurable inference on a potentially different dataset when skipping training.

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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

# Import functions from the utils folder
from utils.data_processing import (
    create_image_directories,
    generate_paired_file_lists,
    process_and_save_frames,
    process_and_save_frames_cars_dataset,
    process_and_save_mixed_frames,
    get_model_preprocessing_function
)
from utils.plotting_utils import (
    example_errors,
    plot_training_history
)

# Configure GPU memory growth
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Memory logging utility
def log_memory():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / 1024**2:.2f} MB")

# --- Global Configuration Section ---
# Set to True to train a new model, False to skip training
TRAIN_MODEL = True
# Set to True to run inference after training (or with a loaded model)
RUN_INFERENCE = True

# --- Primary Dataset Mode for Training (and default for Inference if TRAIN_MODEL is True) ---
# Choose the dataset mode for training: "drones", "cars", or "mixed"
GLOBAL_DATASET_MODE = "mixed" # "cars" #  "drones" #

# --- Inference Specific Configuration (ONLY if TRAIN_MODEL is False) ---
# If TRAIN_MODEL is False, this specifies the dataset to run inference on.
# If TRAIN_MODEL is True, INFERENCE_DATASET_TYPE will automatically be set to GLOBAL_DATASET_MODE.
# Set this to None if TRAIN_MODEL is True and you want inference on the same dataset as training.
# Set this explicitly (e.g., "cars", "drones", "mixed") if TRAIN_MODEL is False
# and you want to perform cross-dataset inference or just inference on a specific dataset.
INFERENCE_DATASET_TYPE_OVERRIDE = None # Example: "cars" if you want to infer on cars data when not training

# If GLOBAL_DATASET_MODE is "drones" or "mixed"
DRONES_TRAIN_RANGE_MIN = 2 
DRONES_TRAIN_RANGE_MAX = 93

# If GLOBAL_DATASET_MODE is "cars" or "mixed"
CARS_TRAIN_NUM_COLLISION_VIDEOS = 25
CARS_TRAIN_NUM_NO_COLLISION_VIDEOS = 25
CARS_TRAIN_FRAMES_PER_SECOND = 15 # Sampling rate for car data during training

# Path to load pre-trained weights for inference IF TRAIN_MODEL is False.
# If TRAIN_MODEL is True, the newly trained model will be used for inference.
# IMPORTANT: If TRAIN_MODEL is False, this path MUST be set to a valid .keras model file
# that is compatible with the selected GLOBAL_DATASET_MODE (if training) or
# INFERENCE_DATASET_TYPE_OVERRIDE (if only inferring).
MODEL_WEIGHTS_FOR_INFERENCE = "drones/models_singleframe/ResNet50_collision_avoidance_1710500000.keras" 


# --- Common Model and Data Processing Parameters ---
MODEL_LIST = ["ResNet50"] # Models to train/evaluate
MODEL_LIST = ["VGG16"] # , "EfficientNet", "MobileNetV2", "EfficientNetB0"]

TARGET_IMAGE_SIZE = (224, 224)
TRAIN_VALIDATION_SPLIT_RATIO = 0.8
BATCH_SIZE = 20
N_HIDDEN_DENSE = 1024 # Number of units in the hidden dense layer

# Model output configuration (binary classification)
MODEL_OUTPUT_CONFIG = {
    "num_output_units": 1,
    "activation_function": "sigmoid",
    "loss_function": "binary_crossentropy"
}

# --- Base Directories for Datasets ---
DRONES_BASE_DIR = "drones"
CARS_BASE_DIR = "cars"

# Paths relative to DRONES_BASE_DIR or CARS_BASE_DIR for raw data
DRONES_RAW_VIDEO_DIR = os.path.join(DRONES_BASE_DIR, 'videos')
DRONES_RAW_DATAFRAMES_DIR = os.path.join(DRONES_BASE_DIR, 'dataframes')

# CARS_RAW_LABELS_CSV_FILE is defined here, around line 83
CARS_RAW_LABELS_CSV_FILE = os.path.join(CARS_BASE_DIR, 'data_labels.csv')
CARS_RAW_VIDEOS_DIR = os.path.join(CARS_BASE_DIR, 'videos') # Points to the folder containing car videos

# Dynamic output directories based on GLOBAL_DATASET_MODE (primarily for training outputs)
if GLOBAL_DATASET_MODE == "drones":
    OUTPUT_BASE_DIR_FOR_IMAGES = os.path.join(DRONES_BASE_DIR, 'image_data_drones')
    OUTPUT_DIR_FOR_MODELS = os.path.join(DRONES_BASE_DIR, "models_SF")
    OUTPUT_DIR_FOR_RESULTS = os.path.join(DRONES_BASE_DIR, "results_SF")
elif GLOBAL_DATASET_MODE == "cars":
    OUTPUT_BASE_DIR_FOR_IMAGES = os.path.join(CARS_BASE_DIR, 'images_SF_cars')
    OUTPUT_DIR_FOR_MODELS = os.path.join(CARS_BASE_DIR, "models_SF")
    OUTPUT_DIR_FOR_RESULTS = os.path.join(CARS_BASE_DIR, "results_SF")
elif GLOBAL_DATASET_MODE == "mixed":
    OUTPUT_BASE_DIR_FOR_IMAGES = "mixed_data/image_data_mixed"
    OUTPUT_DIR_FOR_MODELS = "mixed_data/models_SF"
    OUTPUT_DIR_FOR_RESULTS = "mixed_data/results_SF"
else:
    raise ValueError(f"Unknown GLOBAL_DATASET_MODE: {GLOBAL_DATASET_MODE}. Choose 'drones', 'cars', or 'mixed'.")

os.makedirs(OUTPUT_DIR_FOR_MODELS, exist_ok=True)
os.makedirs(OUTPUT_DIR_FOR_RESULTS, exist_ok=True)


# --- Function to build the model architecture ---
def build_model_architecture(model_name, input_shape, n_hidden_dense, output_config):
    """
    Builds the CNN model architecture based on the specified base model.
    """
    base_model = None
    if model_name == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "MobileNetV2":
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "MobileNetV3Small":
        base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name.startswith("EfficientNet"):
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Model name '{model_name}' not recognized.")

    if base_model:
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(), # Using GlobalAveragePooling2D for better feature aggregation
            Dense(n_hidden_dense, activation='relu'),
            Dropout(0.5),
            Dense(
                output_config["num_output_units"],
                activation=output_config["activation_function"]
            )
        ])
    else:
        model = None # Indicate failure to build
    return model, base_model

# --- Training Function ---
def train_single_frame_model(model_name, output_base_dir_for_images, output_dir_for_models, output_dir_for_results,
                             target_image_size, batch_size, n_hidden_dense, model_output_config,
                             train_validation_split_ratio, dataset_mode):
    
    print(f"\n--- Starting Training for model: {model_name} on {dataset_mode} dataset ---")
    log_memory()

    preprocess_function = get_model_preprocessing_function(model_name)
    if preprocess_function:
        datagen_train = ImageDataGenerator(
            preprocessing_function=preprocess_function,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=[0.9, 1.1],
            fill_mode='nearest'
        )
        datagen_test = ImageDataGenerator(
            preprocessing_function=preprocess_function,
        )
    else:
        print(f"Warning: No specific preprocessing function found for {model_name}. Using default ImageDataGenerator.")
        datagen_train = ImageDataGenerator(
            width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=[0.9, 1.1], fill_mode='nearest'
        )
        datagen_test = ImageDataGenerator()

    train_dir = os.path.join(output_base_dir_for_images, 'train')
    test_dir = os.path.join(output_base_dir_for_images, 'test')

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"Error: Training or testing data directories not found at {train_dir} or {test_dir}. Skipping training.")
        return None, None, None, None

    generator_train = datagen_train.flow_from_directory(
        train_dir, target_size=target_image_size, batch_size=batch_size, class_mode='sparse', shuffle=True
    )
    generator_test = datagen_test.flow_from_directory(
        test_dir, target_size=target_image_size, batch_size=batch_size, class_mode='sparse', shuffle=False
    )

    if generator_train.n == 0 or generator_test.n == 0:
        print(f"No images found in training or testing directories. Training skipped for {model_name}.")
        return None, None, None, None

    steps_test = int(np.ceil(generator_test.n / batch_size))
    class_names = list(generator_train.class_indices.keys())

    # --- Class Weights ---
    class_weight = dict(enumerate(
        compute_class_weight(class_weight='balanced', classes=np.unique(generator_train.classes), y=generator_train.classes)
    ))

    model, base_model = build_model_architecture(model_name, (*target_image_size, 3), n_hidden_dense, model_output_config)

    if model is None:
        print(f"Failed to build model architecture for {model_name}. Skipping training.")
        return None, None, None, None

    # Determine epochs based on model type (can be customized further)
    epochs_initial = 20
    epochs_fine_tune = 40
    if model_name.startswith("EfficientNet"):
        epochs_initial = 30
        epochs_fine_tune = 60
    elif model_name == "MobileNetV3Small":
        epochs_initial = 20
        epochs_fine_tune = 40


    # --- Initial Training (Feature Extractor) ---
    print("\n--- Training (Feature Extractor Phase) ---")
    base_model.trainable = False
    optimizer_initial = Adam(learning_rate=1e-5, clipnorm=1.0)
    model.compile(optimizer=optimizer_initial, 
                    loss=model_output_config["loss_function"],
                    metrics=['accuracy'])
    steps_per_epoch = int(np.ceil(generator_train.n / batch_size))
    
    early_stopping_initial = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        generator_train,
        epochs=epochs_initial,
        steps_per_epoch=steps_per_epoch,
        validation_data=generator_test,
        validation_steps=steps_test,
        class_weight=class_weight,
        callbacks=[early_stopping_initial]
    )

    # --- Fine-Tuning ---
    print("\n--- Fine-Tuning Phase ---")
    base_model.trainable = True

    if model_name == "VGG16":
        for layer in base_model.layers:
            layer.trainable = 'block5' in layer.name or 'block4' in layer.name
    elif model_name == "MobileNetV2":
        for layer in base_model.layers:
            layer.trainable = 'block_16' in layer.name or 'Conv_1' in layer.name
    elif model_name == "ResNet50":
        for layer in base_model.layers:
            if 'conv5' in layer.name:
                layer.trainable = True
            else:
                layer.trainable = False
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
    else:
        for layer in base_model.layers:
            layer.trainable = True
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
    
    fine_tune_optimizer = Adam(learning_rate=1e-6, clipnorm=1.0)
    if model_name == "ResNet50":
        decay_steps = steps_per_epoch * (epochs_fine_tune - epochs_initial)
        cosine_lr = CosineDecay(initial_learning_rate=1e-6, decay_steps=decay_steps, alpha=0.1)
        fine_tune_optimizer = Adam(learning_rate=cosine_lr, clipnorm=1.0)

    model.compile(optimizer=fine_tune_optimizer,
                    loss=model_output_config["loss_function"],
                    metrics=['accuracy'])
    
    early_stopping_fine_tune = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history_fine = model.fit(
        generator_train,
        epochs=epochs_fine_tune,
        initial_epoch=epochs_initial,
        steps_per_epoch=steps_per_epoch,
        validation_data=generator_test,
        validation_steps=steps_test,
        class_weight=class_weight,
        callbacks=[early_stopping_fine_tune]
    )

    combined_history = {}
    for key in history.history.keys():
        combined_history[key] = history.history[key] + history_fine.history[key]

    # --- Save Results ---
    timestamp = int(time.time())
    model_save_name = f"{model_name}_collision_avoidance_{timestamp}"
    
    model_path = os.path.join(output_dir_for_models, f'{model_save_name}.keras')
    model.save(model_path)
    print(f"Model saved to {model_path}")

    metrics_save_path = os.path.join(output_dir_for_results, f'{model_save_name}_training_metrics.pkl')
    with open(metrics_save_path, 'wb') as f:
        pickle.dump(combined_history, f)
    print(f"Training metrics saved to {metrics_save_path}")

    plot_training_history(history, model_name, history_fine, save_path=os.path.join(output_dir_for_results, f"{model_save_name}_curves.pdf"))
    plt.show()

    # --- Evaluation on test set after training ---
    print(f"\n--- Evaluating {model_name} on its test set ---")
    loss, accuracy = model.evaluate(generator_test, steps=steps_test, verbose=1)
    print(f"Test Loss for {model_name}: {loss:.4f}")
    print(f"Test Accuracy for {model_name}: {accuracy:.4f}")

    generator_test.reset()
    y_pred_raw = model.predict(generator_test, steps=steps_test, verbose=1)

    if model_output_config["activation_function"] == "sigmoid":
        y_prob = y_pred_raw.ravel()
        cls_pred = (y_prob > 0.5).astype(int)
    elif model_output_config["activation_function"] == "softmax":
        y_prob = y_pred_raw[:, 1] if y_pred_raw.shape[1] == 2 else y_pred_raw.ravel()
        cls_pred = np.argmax(y_pred_raw, axis=1)
    else:
        print(f"Warning: Unexpected activation function '{model_output_config['activation_function']}'. Defaulting to argmax for cls_pred.")
        y_prob = y_pred_raw.ravel()
        cls_pred = np.argmax(y_pred_raw, axis=1)
    
    cls_true = generator_test.classes
    
    example_errors(cls_true, cls_pred, generator_test, class_names, 
                   output_dir=output_dir_for_results, model_name=model_save_name)

    filenames = [os.path.basename(f) for f in generator_test.filenames]
    pd.DataFrame({
        "file": filenames,
        "p_collision": y_prob,
        "collision_pred": cls_pred,
        "true": cls_true
    }).to_csv(f"{output_dir_for_results}/{model_save_name}_predictions_{dataset_mode}.csv", index=False)

    print(f"Saved predictions to {output_dir_for_results}/{model_save_name}_predictions_{dataset_mode}.csv")

    log_memory()
    return model, combined_history, class_names, model_path


# --- Inference Function ---
def run_inference(model_path, inference_dataset_type, output_dir_for_results,
                  target_image_size, batch_size, model_output_config, base_dir_drones, base_dir_cars):
    """
    Performs inference using a pre-trained single-frame CNN model on a specified dataset.

    Args:
        model_path (str): Full path to the pre-trained Keras model (.keras file).
        inference_dataset_type (str): Type of dataset to run inference on ("drones", "cars", or "mixed").
        output_dir_for_results (str): Directory to save inference results (predictions, error examples).
        target_image_size (tuple): Expected (height, width) of input images for the model.
        batch_size (int): Batch size for inference.
        model_output_config (dict): Dictionary containing 'activation_function' and 'num_output_units'
                                    of the model's output layer.
        base_dir_drones (str): Base directory for drone dataset.
        base_dir_cars (str): Base directory for car dataset.
    """
    print(f"\n--- Starting Inference on dataset: '{inference_dataset_type}' ---")
    log_memory()

    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}. Cannot perform inference.")
        return

    model_name_for_results = os.path.basename(model_path).split('.keras')[0]
    
    base_model_name = model_name_for_results.split('_')[0] 
    preprocess_function = get_model_preprocessing_function(base_model_name)
    
    if preprocess_function:
        datagen_inference = ImageDataGenerator(preprocessing_function=preprocess_function)
    else:
        print(f"Warning: No specific preprocessing function found for {base_model_name}. Using default ImageDataGenerator.")
        datagen_inference = ImageDataGenerator()

    # Determine the correct directory for inference data based on type
    inference_data_root_dir = ""
    if inference_dataset_type == "drones":
        inference_data_root_dir = os.path.join(base_dir_drones, 'image_data_drones')
    elif inference_dataset_type == "cars":
        inference_data_root_dir = os.path.join(base_dir_cars, 'images_SF_cars')
    elif inference_dataset_type == "mixed":
        inference_data_root_dir = "mixed_data/image_data_mixed"
    else:
        print(f"Error: Unknown inference dataset type: {inference_dataset_type}. Skipping inference.")
        return

    inference_data_test_dir = os.path.join(inference_data_root_dir, 'test')

    if not os.path.exists(inference_data_test_dir):
        print(f"Error: Inference data directory not found at {inference_data_test_dir}. Skipping inference.")
        return
    
    inference_generator = datagen_inference.flow_from_directory(
        inference_data_test_dir,
        target_size=target_image_size,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False
    )

    if inference_generator.n == 0:
        print(f"No images found in inference directory {inference_data_test_dir}. Skipping inference.")
        return

    steps_inference = int(np.ceil(inference_generator.n / batch_size))
    class_names_inference = list(inference_generator.class_indices.keys())

    print(f"\n--- Evaluating {model_name_for_results} on {inference_dataset_type} test set ---")
    loss, accuracy = model.evaluate(inference_generator, steps=steps_inference, verbose=1)
    print(f"Inference Loss: {loss:.4f}")
    print(f"Inference Accuracy: {accuracy:.4f}")

    inference_generator.reset()
    y_pred_raw = model.predict(inference_generator, steps=steps_inference, verbose=1)

    if model_output_config["activation_function"] == "sigmoid":
        y_prob = y_pred_raw.ravel()
        cls_pred = (y_prob > 0.5).astype(int)
    elif model_output_config["activation_function"] == "softmax":
        y_prob = y_pred_raw[:, 1] if y_pred_raw.shape[1] == 2 else y_pred_raw.ravel()
        cls_pred = np.argmax(y_pred_raw, axis=1)
    else:
        print(f"Warning: Unexpected activation function '{model_output_config['activation_function']}'. Defaulting to argmax for cls_pred.")
        y_prob = y_pred_raw.ravel()
        cls_pred = np.argmax(y_pred_raw, axis=1)
    
    cls_true = inference_generator.classes
    
    example_errors(cls_true, cls_pred, inference_generator, class_names_inference, 
                   output_dir=output_dir_for_results, model_name=model_name_for_results)

    filenames = [os.path.basename(f) for f in inference_generator.filenames]
    pd.DataFrame({
        "file": filenames,
        "p_collision": y_prob,
        "collision_pred": cls_pred,
        "true": cls_true
    }).to_csv(f"{output_dir_for_results}/{model_name_for_results}_predictions_{inference_dataset_type}.csv", index=False)

    print(f"Saved inference predictions to {output_dir_for_results}/{model_name_for_results}_predictions_{inference_dataset_type}.csv")

    log_memory()


# --- Main Execution ---

# Determine the actual dataset type for inference
actual_inference_dataset_type = GLOBAL_DATASET_MODE
if not TRAIN_MODEL and INFERENCE_DATASET_TYPE_OVERRIDE is not None:
    actual_inference_dataset_type = INFERENCE_DATASET_TYPE_OVERRIDE

print(f"--- Processing data for training mode: '{GLOBAL_DATASET_MODE}' ---")
log_memory()

if GLOBAL_DATASET_MODE == "drones":
    print("Configuring for Drone Collision Dataset...")
    video_files, excel_files = generate_paired_file_lists(
        range_min=DRONES_TRAIN_RANGE_MIN, range_max=DRONES_TRAIN_RANGE_MAX,
        base_dir=DRONES_BASE_DIR
    )
    all_processed_filenames = process_and_save_frames(
        excel_files,
        video_files,
        OUTPUT_BASE_DIR_FOR_IMAGES,
        target_size=TARGET_IMAGE_SIZE,
        train_ratio=TRAIN_VALIDATION_SPLIT_RATIO
    )
elif GLOBAL_DATASET_MODE == "cars":
    print("Configuring for Car Crashes Dataset...")
    all_processed_filenames = process_and_save_frames_cars_dataset(
        CARS_RAW_LABELS_CSV_FILE,
        CARS_RAW_VIDEOS_DIR,
        OUTPUT_BASE_DIR_FOR_IMAGES,
        target_size=TARGET_IMAGE_SIZE,
        train_ratio=TRAIN_VALIDATION_SPLIT_RATIO,
        num_collision_videos_to_process=CARS_TRAIN_NUM_COLLISION_VIDEOS,
        num_no_collision_videos_to_process=CARS_TRAIN_NUM_NO_COLLISION_VIDEOS,
        frames_per_second=CARS_TRAIN_FRAMES_PER_SECOND
    )
elif GLOBAL_DATASET_MODE == "mixed":
    print("Configuring for Mixed (Drones + Cars) Dataset...")
    all_processed_filenames = process_and_save_mixed_frames(
        drone_video_range_min=DRONES_TRAIN_RANGE_MIN,
        drone_video_range_max=DRONES_TRAIN_RANGE_MAX,
        drone_base_dir=DRONES_BASE_DIR,
        car_labels_csv_path=CARS_RAW_LABELS_CSV_FILE,
        car_source_videos_directory=CARS_RAW_VIDEOS_DIR,
        output_base_dir=OUTPUT_BASE_DIR_FOR_IMAGES,
        target_size=TARGET_IMAGE_SIZE,
        train_ratio=TRAIN_VALIDATION_SPLIT_RATIO,
        car_num_collision_videos_to_process=CARS_TRAIN_NUM_COLLISION_VIDEOS,
        car_num_no_collision_videos_to_process=CARS_TRAIN_NUM_NO_COLLISION_VIDEOS,
        car_frames_per_second=CARS_TRAIN_FRAMES_PER_SECOND
    )
else:
    raise ValueError(f"Invalid GLOBAL_DATASET_MODE: {GLOBAL_DATASET_MODE}")

print(f"Finished data processing. Total frames saved: {len(all_processed_filenames)}")
log_memory()

last_trained_model_path = None 
class_names_trained = None 

if TRAIN_MODEL:
    for model_name_to_train in MODEL_LIST:
        trained_model, history_data, class_names_trained, saved_model_path = train_single_frame_model(
            model_name_to_train, OUTPUT_BASE_DIR_FOR_IMAGES, OUTPUT_DIR_FOR_MODELS, OUTPUT_DIR_FOR_RESULTS,
            TARGET_IMAGE_SIZE, BATCH_SIZE, N_HIDDEN_DENSE, MODEL_OUTPUT_CONFIG,
            TRAIN_VALIDATION_SPLIT_RATIO, GLOBAL_DATASET_MODE # Pass GLOBAL_DATASET_MODE as dataset_mode
        )
        if trained_model:
            last_trained_model_path = saved_model_path
            print(f"Last trained model path: {last_trained_model_path}")
        
        tf.keras.backend.clear_session()
        del trained_model
        gc.collect()
        log_memory()

if RUN_INFERENCE:
    inference_model_path = None
    if TRAIN_MODEL and last_trained_model_path:
        inference_model_path = last_trained_model_path # Use the model just trained
    elif not TRAIN_MODEL and MODEL_WEIGHTS_FOR_INFERENCE:
        inference_model_path = MODEL_WEIGHTS_FOR_INFERENCE # Use the pre-specified model
    else:
        print("No valid model path specified or available for inference. Skipping inference.")

    if inference_model_path:
        run_inference(
            inference_model_path, 
            actual_inference_dataset_type, # Use the dynamically determined inference type
            OUTPUT_DIR_FOR_RESULTS,
            TARGET_IMAGE_SIZE, 
            BATCH_SIZE, 
            MODEL_OUTPUT_CONFIG,
            DRONES_BASE_DIR, # Pass DRONES_BASE_DIR
            CARS_BASE_DIR # Pass CARS_BASE_DIR
        )
    else:
        print("Inference was requested but no model could be loaded or found. Skipping inference.")

session.close()
