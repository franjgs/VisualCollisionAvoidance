#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for training and evaluating collision avoidance Deep Neural Networks.
Supports both single-frame CNNs and multi-frame CNN-LSTM models.

Created on Mon May 19 16:30:54 2025
@author: fran
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, TimeDistributed, LSTM, Reshape
# Use tf.keras.optimizers.legacy.Adam if on M1/M2 Mac
from tensorflow.keras.optimizers import Adam 
# from tensorflow.keras.optimizers.legacy import Adam # Using legacy Adam for broader compatibility

from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import pickle

# For GPU memory growth configuration
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# Import utility functions for data processing and plotting
from utils.data_processing import (
    load_labeled_hdf5_sequence_filepaths,
    HDF5DataLoaderForTFData,
    get_preprocessing_function,
)
from utils.plotting_utils import plot_training_history, example_errors


# Clear Keras session to reset any previous model states
tf.keras.backend.clear_session()

# Configure GPU memory growth to prevent TensorFlow from allocating all GPU memory at once
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# --- Global Model Output Configurations ---
# Defines standard output layer configurations for binary (sigmoid) and multiclass (softmax) tasks.
model_configs = {
    "binary_sigmoid": {
        "num_output_units": 1,
        "activation_function": "sigmoid",
        "loss_function": "binary_crossentropy"
    },
    "multiclass_softmax": {
        "num_output_units": 2, # Specifically for binary classification (collision/no_collision) with softmax
        "activation_function": "softmax",
        "loss_function": "sparse_categorical_crossentropy" # Used for integer labels
    }
}

# --- USER CONFIGURATION ---
DATASET_TYPE = 'drones' # 'cars' # 

# Set this to 'single_frame' for a CNN model or 'multi_frame' for a CNN-LSTM model.
MODEL_TYPE = 'Multiframe' # Options: 'Singleframe', 'Multiframe'

# Common Parameters for data and training
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32 # Adjust based on GPU memory
EPOCHS = 40
LEARNING_RATE = 1e-5

# Data Generation Parameters (also define model input shape)
# SEQUENCE_LENGTH: Number of frames in each sequence.
# STRIDE: Number of frames to advance between consecutive sequences.
# These parameters directly influence the name of the HDF5 data directory.
SEQUENCE_LENGTH = 3 # Example: 1 for single-frame, 3 or 5 for multi-frame
STRIDE = 1          # Example: 1 for single-frame, 3 or 5 for multi-frame

# Model-specific parameters
SF_PRETRAINED_MODEL = 'ResNet50' # Pretrained backbone for single-frame model
SF_N_HIDDEN_DENSE = 1024 # Number of units in the hidden dense layer for single-frame CNN

MF_PRETRAINED_MODEL = 'MobileNetV2' # 'ResNet50' # Pretrained backbone for multi-frame model
MF_N_DENSE_BOTTLENECK = 256 # Units in TimeDistributed Dense bottleneck before LSTM
MF_N_HIDDEN_LSTM = 128 # Units in the LSTM layer


# --- Data Directory Definition ---
# Constructs the path to the HDF5 data based on configured SEQUENCE_LENGTH and STRIDE.
# This directory must match the 'output_dir' used when generating data via create_labeled_sequences_from_annotations.
DATA_DIR_FOR_LOADING = os.path.join(DATASET_TYPE, f'labeled_sequences_len_{SEQUENCE_LENGTH}_stride_{STRIDE}')



# --- DYNAMIC PARAMETER SETUP BASED ON MODEL_TYPE ---
# Configures model-specific settings and data loader behavior based on the chosen MODEL_TYPE.
if MODEL_TYPE == 'Singleframe':
    # For single-frame models, the dataset will always load individual frames (sequence length 1).
    SEQUENCE_LENGTH_FOR_LOADING = SEQUENCE_LENGTH # Will typically be 1
    STRIDE_FOR_LOADING = STRIDE                   # Will typically be 1
    
    SELECTED_PRETRAINED_MODEL = SF_PRETRAINED_MODEL
    CURRENT_MODEL_OUTPUT_CONFIG = model_configs['binary_sigmoid'] # Single-frame uses sigmoid output
    OUTPUT_DIR = os.path.join(DATASET_TYPE,'models_SF_tfdata') # Output directory for saved models and plots
    MODEL_NAME = f'CNN_Singleframe_{SELECTED_PRETRAINED_MODEL}_len_{SEQUENCE_LENGTH}_stride_{STRIDE}_TFData'
    SHOULD_DATASET_OUTPUT_4D = True # Dataset will output (H,W,C) for this model type

elif MODEL_TYPE == 'Multiframe':
    # For multi-frame models, the dataset will load sequences of frames.
    SEQUENCE_LENGTH_FOR_LOADING = SEQUENCE_LENGTH # Uses the global SEQUENCE_LENGTH for sequence definition
    STRIDE_FOR_LOADING = STRIDE                   # Uses the global STRIDE for sequence definition
    
    SELECTED_PRETRAINED_MODEL = MF_PRETRAINED_MODEL
    CURRENT_MODEL_OUTPUT_CONFIG = model_configs['multiclass_softmax'] # Multi-frame uses softmax output
    OUTPUT_DIR = os.path.join(DATASET_TYPE,'models_MF_tfdata') # Output directory for saved models and plots
    MODEL_NAME = f'CNN_Multiframe_{SELECTED_PRETRAINED_MODEL}_len_{SEQUENCE_LENGTH}_stride_{STRIDE}_TFData'
    SHOULD_DATASET_OUTPUT_4D = False # Dataset will output (SeqLen,H,W,C) for this model type

else:
    raise ValueError("Invalid MODEL_TYPE specified. Choose 'single_frame' or 'multi_frame'.")

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Data Collection ---
print(f"Collecting HDF5 file paths for {MODEL_TYPE} model from {DATA_DIR_FOR_LOADING}...")
train_df_raw, test_df_raw, class_names = load_labeled_hdf5_sequence_filepaths(DATA_DIR_FOR_LOADING)

# Exit if no data is loaded
if train_df_raw.empty and test_df_raw.empty:
    print(f"Error: No data loaded from {DATA_DIR_FOR_LOADING}. Please ensure data is generated and paths are correct.")
    session.close()
    exit()

# Determine the number of classes from the loaded training labels
num_classes = len(np.unique(train_df_raw['labels'].values))


# --- Calculate Class Weights ---
# Computes class weights to handle imbalanced datasets, giving more importance to under-represented classes.
print("Calculating Class Weights...")
class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(train_df_raw['labels'].values), y=train_df_raw['labels'].values
)
class_weight_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weight_dict)


# --- Split Training Data for Validation ---
print("Splitting training data for validation...")
train_df_for_model, val_df = train_test_split(
    train_df_raw, test_size=0.2, stratify=train_df_raw['labels'], random_state=42
)


# --- Create tf.data.Dataset instances ---
# Initializes the custom HDF5 data loader wrapper for TensorFlow Datasets.
hdf5_data_loader_tf = HDF5DataLoaderForTFData(
    expected_seq_len=SEQUENCE_LENGTH_FOR_LOADING,
    img_h=IMG_HEIGHT, img_w=IMG_WIDTH,
    model_name_for_preprocess=SELECTED_PRETRAINED_MODEL,
    output_as_4d_if_single_frame=SHOULD_DATASET_OUTPUT_4D # Instructs loader on output tensor shape
)

def create_dataset(dataframe: pd.DataFrame, shuffle: bool = False, repeat: bool = False) -> tf.data.Dataset:
    """
    Creates a tf.data.Dataset from a Pandas DataFrame containing file paths and labels.
    The dataset is mapped to load HDF5 data, batch, shuffle, and prefetch.
    """
    if CURRENT_MODEL_OUTPUT_CONFIG["num_output_units"] == 1:
        labels_tensor_dtype = tf.float32 # For binary_crossentropy loss
    else:
        labels_tensor_dtype = tf.int32 # For sparse_categorical_crossentropy loss
    
    filepaths_tensor = tf.constant(dataframe['filepaths'].values)
    labels_tensor = tf.constant(dataframe['labels'].values, dtype=labels_tensor_dtype)
        
    dataset = tf.data.Dataset.from_tensor_slices((filepaths_tensor, labels_tensor))

    if shuffle:
        # Shuffles the dataset with a buffer size equal to the dataframe length for better randomization
        dataset = dataset.shuffle(buffer_size=len(dataframe), reshuffle_each_iteration=True)
    
    # Maps the custom HDF5 loader function to each element in the dataset
    dataset = dataset.map(hdf5_data_loader_tf.load_fn_tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(BATCH_SIZE) # Batches the elements
    if repeat:
        dataset = dataset.repeat() # Repeats the dataset indefinitely for training
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE) # Prefetches elements for performance
    return dataset

print("\nCreating tf.data.Dataset for training, validation, and testing...")
train_dataset = create_dataset(train_df_for_model, shuffle=True, repeat=True)
val_dataset = create_dataset(val_df, shuffle=False, repeat=False)
test_dataset = create_dataset(test_df_raw, shuffle=False, repeat=False)


# --- Model Building Functions ---

def build_singleframe_pretrained_cnn(
    img_height: int,
    img_width: int,
    pretrained_model_name: str = 'ResNet50',
    n_hidden_dense: int = 1024,
    model_output_config: dict = model_configs['binary_sigmoid']
) -> tf.keras.Model:
    """
    Constructs a single-frame CNN model using a pre-trained CNN as a feature extractor.
    The model's input expects preprocessed (H, W, C) image data from the tf.data.Dataset.
    """
    if pretrained_model_name == 'VGG16':
        base_cnn = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    elif pretrained_model_name == 'ResNet50':
        base_cnn = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    elif pretrained_model_name == 'MobileNetV2':
        base_cnn = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    elif pretrained_model_name == 'EfficientNetB0':
        base_cnn = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    else:
        raise ValueError(f"Pre-trained model '{pretrained_model_name}' not supported.")

    base_cnn.trainable = False # Freeze the pre-trained base CNN layers

    model = Sequential([
        Input(shape=(img_height, img_width, 3)), # Expects 4D input (H,W,C) per sample
        base_cnn, # Processes the 4D input
        Flatten(), # Flattens the output of the CNN base
        Dense(n_hidden_dense, activation='relu'), # Custom dense layer for classification
        Dropout(0.5), # Dropout for regularization
        Dense(
            model_output_config["num_output_units"],
            activation=model_output_config["activation_function"] # Output layer (e.g., sigmoid for binary)
        )
    ])
    return model


def build_multiframe_pretrained_cnn_lstm_simplified(
    sequence_length: int, img_height: int, img_width: int, num_classes: int,
    pretrained_model_name: str = 'ResNet50', 
    n_dense_bottleneck: int = 256, 
    n_hidden_lstm: int = 64,
    n_hidden_single_frame_dense: int = 1024 # Used if sequence_length is 1 for a single-frame equivalent head
) -> tf.keras.Model:
    """
    Constructs a multi-frame CNN-LSTM model for sequence prediction.
    It uses a pre-trained CNN as a TimeDistributed feature extractor,
    followed by a conditional classification head (either a dense head for
    sequence_length=1 or an LSTM head for sequence_length > 1).
    The model's input expects preprocessed (SeqLen, H, W, C) data from the tf.data.Dataset.
    """
    if pretrained_model_name == 'VGG16':
        base_cnn = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    elif pretrained_model_name == 'ResNet50':
        base_cnn = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    elif pretrained_model_name == 'MobileNetV2':
        base_cnn = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    elif pretrained_model_name == 'EfficientNetB0':
        base_cnn = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    else:
        raise ValueError(f"Pre-trained model '{pretrained_model_name}' not supported.")
    
    base_cnn.trainable = False # Freeze the pre-trained base CNN layers

    # Defines a sub-network to extract features from a single frame
    single_frame_feature_extractor = Sequential([
        base_cnn, # Processes individual (H,W,C) frames
        Flatten() # Flattens features for each frame
    ], name="base_feature_extractor_per_frame")

    # Main model layers for sequence processing
    model_layers = [
        Input(shape=(sequence_length, img_height, img_width, 3), name="sequence_input"), # Expects 5D input (SeqLen,H,W,C)
        # Applies the feature extractor to each frame in the sequence
        TimeDistributed(single_frame_feature_extractor, name="time_distributed_cnn_features"),
    ]

    # Conditional branch for sequence_length = 1 (behaves like a single-frame model, but within multi-frame architecture)
    if sequence_length == 1:
        model_layers.extend([
            # Reshapes (None, 1, flattened_features) to (None, flattened_features)
            Reshape((single_frame_feature_extractor.output_shape[-1],), name="reshape_for_single_frame_head"), 
            Dense(n_hidden_single_frame_dense, activation='relu', name="single_frame_dense_head"), 
            Dropout(0.5, name="single_frame_dropout"),
            # Output layer matching the multi-class softmax configuration (e.g., 2 units, softmax)
            Dense(num_classes, 
                  activation=model_configs['multiclass_softmax']["activation_function"], 
                  name="single_frame_output_head") 
        ])
    else: # Branch for sequence_length > 1 (true multi-frame LSTM)
        model_layers.extend([
            TimeDistributed(Dense(n_dense_bottleneck, activation='relu', kernel_regularizer=l2(0.001)), name="time_distributed_bottleneck"), 
            Dropout(0.5, name="bottleneck_dropout"),
            tf.keras.layers.LSTM(n_hidden_lstm, activation='tanh', kernel_regularizer=l2(0.001), name="lstm_layer"),
            Dropout(0.5, name="lstm_dropout"),
            Dense(num_classes, activation=model_configs['multiclass_softmax']["activation_function"], kernel_regularizer=l2(0.001), name="multiframe_output_head") 
        ])
    model = Sequential(model_layers, name=f"CNN_LSTM_{pretrained_model_name}_Seq{sequence_length}")
    return model

# --- Build Model (Dynamically selects and builds the model based on MODEL_TYPE) ---
print(f"Building {MODEL_TYPE} model: {MODEL_NAME}...")
if MODEL_TYPE == 'single_frame':
    model = build_singleframe_pretrained_cnn(
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        pretrained_model_name=SELECTED_PRETRAINED_MODEL,
        n_hidden_dense=SF_N_HIDDEN_DENSE,
        model_output_config=CURRENT_MODEL_OUTPUT_CONFIG
    )
elif MODEL_TYPE == 'multi_frame':
    model = build_multiframe_pretrained_cnn_lstm_simplified(
        sequence_length=SEQUENCE_LENGTH_FOR_LOADING, # Uses the sequence length dynamically set for data loading
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        num_classes=CURRENT_MODEL_OUTPUT_CONFIG["num_output_units"], 
        pretrained_model_name=SELECTED_PRETRAINED_MODEL,
        n_dense_bottleneck=MF_N_DENSE_BOTTLENECK,
        n_hidden_lstm=MF_N_HIDDEN_LSTM,
        n_hidden_single_frame_dense=SF_N_HIDDEN_DENSE # Passed for the sequence_length=1 case within multi-frame
    )
else:
    raise ValueError("MODEL_TYPE must be 'single_frame' or 'multi_frame'.")

# Compile the model with the chosen optimizer, loss function, and metrics
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=CURRENT_MODEL_OUTPUT_CONFIG["loss_function"],
    metrics=['accuracy']
)
model.summary()

# --- Callbacks for Training ---
# EarlyStopping: Stops training if validation loss doesn't improve for a number of epochs.
# ReduceLROnPlateau: Reduces learning rate when validation loss stops improving.
early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1
)

# --- Calculate steps per epoch for tf.data.Dataset ---
# Required when using an infinitely repeating dataset (train_dataset) with model.fit().
train_steps_per_epoch = len(train_df_for_model) // BATCH_SIZE
if train_steps_per_epoch == 0:
    if not train_df_for_model.empty:
        train_steps_per_epoch = 1
    else:
        print("Warning: Training dataset is empty. steps_per_epoch set to 0.")
        train_steps_per_epoch = 0

val_steps_per_epoch = len(val_df) // BATCH_SIZE
if val_steps_per_epoch == 0:
    if not val_df.empty:
        val_steps_per_epoch = 1
    else:
        print("Warning: Validation dataset is empty. validation_steps set to 0.")
        val_steps_per_epoch = 0

test_steps = len(test_df_raw) // BATCH_SIZE
if test_steps == 0:
    if not test_df_raw.empty:
        test_steps = 1
    else:
        print("Warning: Test dataset is empty. test_steps set to 0.")
        test_steps = 0


# --- Train the Model ---
print(f"\n--- Starting {MODEL_TYPE} Model Training ---")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    steps_per_epoch=train_steps_per_epoch,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr]
)

# --- Evaluate the Model ---
print(f"\n--- Model Evaluation on Test Set ---")
loss, accuracy = model.evaluate(test_dataset, steps=test_steps)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# --- Get Predictions for Confusion Matrix and Metrics ---
print(f"\n--- Generating Predictions for Metrics ---")
test_predictions_prob = model.predict(test_dataset, steps=test_steps)

# Convert probabilities to class labels based on output configuration
if CURRENT_MODEL_OUTPUT_CONFIG["num_output_units"] > 1: # Softmax output for multi-class
    test_predicted_classes = np.argmax(test_predictions_prob, axis=1)
else: # Sigmoid output for binary classification
    test_predicted_classes = (test_predictions_prob > 0.5).astype(int).flatten()

# Extract true labels from the test_dataset for metrics calculation
true_labels_from_dataset = []
for _, label_value in test_dataset.unbatch().as_numpy_iterator():
    true_labels_from_dataset.append(label_value)

cls_true = np.array(true_labels_from_dataset).flatten()

# Handle potential length mismatch between true labels and predictions due to batching
if len(cls_true) != len(test_predictions_prob):
    print(f"Warning: True labels ({len(cls_true)}) and predicted probabilities ({len(test_predictions_prob)}) have different lengths. This might indicate an issue with dataset prediction completeness. Truncating true labels to match predictions.")
    cls_true = cls_true[:len(test_predictions_prob)]

# --- Call example_errors to get Confusion Matrix and Metrics ---
model_name_results = f'{MODEL_TYPE}_{SELECTED_PRETRAINED_MODEL}_len_{SEQUENCE_LENGTH}_stride_{STRIDE}'
# This function generates and saves the confusion matrix and classification report.
example_errors(cls_true=cls_true,
               cls_pred=test_predicted_classes,
               generator_test=None, # Not applicable for visualizing individual images from tf.data.Dataset
               class_names=class_names,
               output_dir=OUTPUT_DIR,
               model_name=model_name_results)


# --- Plot Training History ---
fig = plot_training_history(history, model_name_results, None, save_path=os.path.join(OUTPUT_DIR,f"{model_name_results}.pdf"))
plt.show()
# --- Save the Model and Class Names ---
print(f"\nSaving {MODEL_TYPE} model...")
model.save(os.path.join(OUTPUT_DIR, f'{model_name_results}.keras'))
with open(os.path.join(OUTPUT_DIR, f'{model_name_results}_classes.pkl'), 'wb') as f:
    pickle.dump(class_names, f)
print(f"Model and class names saved to {OUTPUT_DIR}")

# Close the TensorFlow session
session.close()