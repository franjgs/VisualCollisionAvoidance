#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 20:33:44 2025

@author: fran
"""

import numpy as np
import sys
import os
import cv2

import pandas as pd


from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


(CV2_VERSION, minor_ver, subminor_ver) = (cv2.__version__).split('.')

def open_video(video):
    return cv2.VideoCapture(video)

def check_video(cap, video_path):
    """
    Checks if the video capture object is successfully opened and if the video file exists.

    Args:
        cap (cv2.VideoCapture): The video capture object.
        video_path (str): The path to the video file.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return False  # Indicate file not found
    if not cap.isOpened():
        print(f"Error: Could not open video at '{video_path}'")
        return False  # Indicate unable to open
    success, frame = cap.read()
    if not success:
        print(f"Warning: Could not read the first frame of the video at '{video_path}'.")
        cap.release()
        return False  # Indicate unable to read first frame
    return True

def close_cap(cap):
    cap.release()
    #cv2.destroyAllWindows() #Close viewing of frames


"""##### Generic functions"""

def getFrameRate(video):
    fps = 30 # default frame rate - this values should be replace in the next if/else evaluation
    if int(CV2_VERSION)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
    return fps

def generate_out_videoname(vid_base):
    if "video-" in vid_base:
        return vid_base.split('.')[0]
    out_video = 'video-00001'
    try:
        collision_number = vid_base.split('collision')[1]
        numb = collision_number
        s_numb = str(numb).zfill(5)
        return f"video-{s_numb}"
    except IndexError:
        print(f"Warning: Unexpected video filename format: '{vid_base}'. Returning default: '{out_video}'")
        return out_video
    except Exception as e:
        print(f"Exception generating new video name for '{vid_base}': {e}. Returning default: '{out_video}'")
        return out_video

def generate_framename(video_num, pos_frame):
    s_outvid = str(video_num).zfill(5)
    s_frame = str(pos_frame).zfill(5)
    return f"video-{s_outvid}-frame-{s_frame}"

def generate_video_num(out_videoname):
    return int(out_videoname.split('-')[1])


"""##### Data Loading and Preprocessing for VGG16"""
def load_data_from_files(excel_files, video_files, target_size=(300, 300)):
    all_frames = []
    all_labels = []

    for excel_file, video_file in zip(excel_files, video_files):
        if not os.path.exists(excel_file) or not os.path.exists(video_file):
            print(f"Warning: Skipping pair - Excel file '{excel_file}' or Video file '{video_file}' not found.")
            continue

        df = pd.read_excel(excel_file)
        frames_data = []
        labels = []

        cap = open_video(video_file)
        if not check_video(cap, video_file):
            continue

        frame_dict = {row['file']: row['collision'] for index, row in df.iterrows()}

        frame_count = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            video_base_name = os.path.basename(video_file).split('.')[0]
            out_video_name_base = generate_out_videoname(video_base_name)
            frame_name = f"{out_video_name_base}-frame-{str(frame_count + 1).zfill(5)}"

            if frame_name in frame_dict:
                label = frame_dict[frame_name]
                resized_frame = cv2.resize(frame, target_size)
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                normalized_frame = rgb_frame / 255.0
                frames_data.append(normalized_frame)
                labels.append(label)

            frame_count += 1
        close_cap(cap)
        all_frames.extend(frames_data)
        all_labels.extend(labels)

    return np.array(all_frames), np.array(all_labels)

"""##### Create the Modified VGG16 Model"""
def create_modified_vgg16(input_shape=(300, 300, 3), num_classes=2):
    """
    Generates a modified VGG16 model for binary classification (collision/no collision).

    Args:
        input_shape (tuple): The expected input shape of the images (height, width, channels).
                             VGG16 typically expects (224, 224, 3).
        num_classes (int): The number of output classes (in this case, 2).

    Returns:
        tf.keras.Model: The modified VGG16 model.
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    x = Flatten()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output_layer)
    return model

def generate_paired_file_lists(video_prefix='collision', excel_prefix='video-', range_min=1, range_max=94):
    """
    Generates explicitly paired lists of video and Excel filenames based on their numerical identifier.

    Args:
        video_prefix (str): The prefix for the video filenames (e.g., 'collision').
        excel_prefix (str): The prefix for the Excel filenames (e.g., 'video-').
        range_min (int): The starting numerical identifier of the files (inclusive).
        range_max (int): The ending numerical identifier of the files (inclusive).

    Returns:
        tuple: A tuple containing two lists:
               - video_files_list (list): List of video filenames.
               - excel_files_list (list): List of Excel filenames.
    """
    video_files_list = []
    excel_files_list = []
    for i in range(range_min, range_max + 1):
        video_num_str = str(i).zfill(2)
        excel_num_str = str(i).zfill(5)
        video_files_list.append(f'{video_prefix}{video_num_str}.mp4')
        excel_files_list.append(f'{excel_prefix}{excel_num_str}.xlsx')
    return video_files_list, excel_files_list

if __name__ == '__main__':
    # Define paths and parameters
    w = 300
    h = 300
    input_shape = (w, h, 3)
    num_classes = 2
    batch_size = 20
    epochs_per_subset_phase1 = 5  # Train each subset for fewer epochs initially
    epochs_per_subset_phase2 = 5
    steps_per_epoch = 100
    cwd = os.getcwd()
    video_path_folder = os.path.join(cwd, 'videos')
    excel_path_folder = os.path.join(cwd, 'dataframes')

    # Define the range of all files
    start_num = 1
    end_num = 93  # Adjust to your total number of collision files

    video_files, excel_files = generate_paired_file_lists(
        video_prefix='collision',
        excel_prefix='video-',
        range_min=start_num,
        range_max=end_num
    )

    # Create the modified VGG16 model (create it only once)
    model = create_modified_vgg16(input_shape=input_shape, num_classes=num_classes)

    # --- Phase 1 Training (Freeze base layers) ---
    print("\n--- Phase 1 Training (Freezing base layers) ---")
    optimizer_phase1 = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer_phase1, loss='categorical_crossentropy', metrics=['accuracy'])

    # Process files in chunks (you can adjust the chunk size)
    chunk_size = 10
    for i in range(0, len(video_files), chunk_size):
        video_chunk = [os.path.join(video_path_folder, vf) for vf in video_files[i:i + chunk_size]]
        excel_chunk = [os.path.join(excel_path_folder, ef) for ef in excel_files[i:i + chunk_size]]

        # Load data from the current chunk
        X_chunk, y_chunk = load_data_from_files(excel_chunk, video_chunk, target_size=input_shape[:2])

        if X_chunk.shape[0] > 0:
            # Split the chunk data into training and a small validation set
            X_train_chunk, X_val_chunk, y_train_chunk, y_val_chunk = train_test_split(
                X_chunk, y_chunk, test_size=0.1, stratify=y_chunk, random_state=42
            )
            y_train_categorical_chunk = to_categorical(y_train_chunk, num_classes=num_classes)
            y_val_categorical_chunk = to_categorical(y_val_chunk, num_classes=num_classes)

            print(f"\nTraining on files {i+1} to {min(i + chunk_size, len(video_files))}")
            history_phase1_chunk = model.fit(
                X_train_chunk, y_train_categorical_chunk,
                validation_data=(X_val_chunk, y_val_categorical_chunk),
                epochs=epochs_per_subset_phase1,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch // (len(video_files) // chunk_size + 1) # Adjust steps per chunk
            )
        else:
            print(f"Warning: No valid data found in files {i+1} to {min(i + chunk_size, len(video_files))}. Skipping.")

    # --- Phase 2 Training (Fine-tuning all layers) ---
    print("\n--- Phase 2 Training (Fine-tuning all layers) ---")
    for layer in model.layers:
        layer.trainable = True
    optimizer_phase2 = Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer_phase2, loss='categorical_crossentropy', metrics=['accuracy'])

    for i in range(0, len(video_files), chunk_size):
        video_chunk = [os.path.join(video_path_folder, vf) for vf in video_files[i:i + chunk_size]]
        excel_chunk = [os.path.join(excel_path_folder, ef) for ef in excel_files[i:i + chunk_size]]

        X_chunk, y_chunk = load_data_from_files(excel_chunk, video_chunk, target_size=input_shape[:2])

        if X_chunk.shape[0] > 0:
            X_train_chunk, X_val_chunk, y_train_chunk, y_val_chunk = train_test_split(
                X_chunk, y_chunk, test_size=0.1, stratify=y_chunk, random_state=42
            )
            y_train_categorical_chunk = to_categorical(y_train_chunk, num_classes=num_classes)
            y_val_categorical_chunk = to_categorical(y_val_chunk, num_classes=num_classes)

            print(f"\nFine-tuning on files {i+1} to {min(i + chunk_size, len(video_files))}")
            history_phase2_chunk = model.fit(
                X_train_chunk, y_train_categorical_chunk,
                validation_data=(X_val_chunk, y_val_categorical_chunk),
                epochs=epochs_per_subset_phase2,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch // (len(video_files) // chunk_size + 1) # Adjust steps per chunk
            )
        else:
            print(f"Warning: No valid data found in files {i+1} to {min(i + chunk_size, len(video_files))}. Skipping fine-tuning.")

    # Evaluate the model on a separate validation set (if you have one that covers all data)
    # If not, the validation done on each chunk provides some indication of performance.
    print("\nEvaluating the model (on the last validation chunk):")
    if X_val_chunk.shape[0] > 0:
        loss, accuracy = model.evaluate(X_val_chunk, y_val_categorical_chunk, verbose=0)
        print(f"Validation Loss: {loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
    else:
        print("No validation data available for final evaluation.")
