#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for processing video data, creating labeled sequences,
and preparing them for deep learning model training. This module supports
both frame-level and sequence-level data generation and loading.

Created on Mon May 19 15:48:03 2025
@author: fran
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import math

from sklearn.model_selection import train_test_split
import h5py
import shutil

import tensorflow as tf 
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input


def open_video(video: str) -> cv2.VideoCapture:
    """
    Opens a video file using OpenCV.
    
    Args:
        video (str): Path to the video file.
        
    Returns:
        cv2.VideoCapture: The video capture object.
    """
    return cv2.VideoCapture(video)

def check_video(cap: cv2.VideoCapture, video_path: str) -> bool:
    """
    Verifies if a video capture object is opened and can read its first frame.
    
    Args:
        cap (cv2.VideoCapture): The video capture object.
        video_path (str): The path to the video file being checked.
        
    Returns:
        bool: True if the video is successfully opened and readable, False otherwise.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video not found at '{video_path}'")
        return False
    if not cap.isOpened():
        print(f"Error: Cannot open video at '{video_path}'")
        return False
    success, _ = cap.read()
    if not success:
        print(f"Error: Cannot read first frame of '{video_path}'")
        cap.release()
        return False
    return True

def close_cap(cap: cv2.VideoCapture):
    """
    Releases a video capture object, freeing up resources.
    
    Args:
        cap (cv2.VideoCapture): The video capture object to release.
    """
    cap.release()

def getFrameRate(video: cv2.VideoCapture) -> float:
    """
    Retrieves the frame rate (frames per second) of an opened video.
    
    Args:
        video (cv2.VideoCapture): The video capture object.
        
    Returns:
        float: The frame rate of the video.
    """
    return video.get(cv2.CAP_PROP_FPS)

def generate_framename(video_num: int, pos_frame: int) -> str:
    """
    Generates a standardized frame name string from a video number and frame position.
    
    Args:
        video_num (int): The numerical ID of the video.
        pos_frame (int): The 1-based index of the frame within the video.
        
    Returns:
        str: A formatted string like "video-000XX-frame-000YY".
    """
    return f"video-{str(video_num).zfill(5)}-frame-{str(pos_frame).zfill(5)}"

def generate_video_num(out_videoname: str) -> int:
    """
    Extracts the numerical video ID from a standardized video name string.
    
    Args:
        out_videoname (str): A formatted video name string (e.g., "video-000XX").
        
    Returns:
        int: The extracted video number.
    """
    return int(out_videoname.split('-')[1])

def create_image_directories(output_base_dir: str) -> tuple[str, str]:
    """
    Creates the necessary directory structure for storing processed individual image frames:
    `output_base_dir/train/0`, `output_base_dir/train/1`, `output_base_dir/test/0`, `output_base_dir/test/1`.
    Directories are created if they don't exist.
    
    Args:
        output_base_dir (str): The root directory where the structure will be created.
        
    Returns:
        tuple[str, str]: Paths to the created 'train' and 'test' directories.
    """
    train_dir = os.path.join(output_base_dir, 'train')
    test_dir = os.path.join(output_base_dir, 'test')

    os.makedirs(os.path.join(train_dir, '0'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, '1'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, '0'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, '1'), exist_ok=True)
    return train_dir, test_dir

def generate_paired_file_lists(range_min: int = 70, range_max: int = 93, base_dir=None) -> tuple[list[str], list[str]]:
    """
    Generates lists of paired video and Excel annotation file paths based on a numerical range.
    Only includes pairs where both the video and Excel files exist.
    
    Args:
        range_min (int): The starting video ID (inclusive).
        range_max (int): The ending video ID (inclusive).
        
    Returns:
        tuple[list[str], list[str]]: Two lists, one for video file paths and one for Excel file paths.
    """
    video_dir = os.path.join(base_dir,'videos')
    excel_dir = os.path.join(base_dir,'dataframes')
    
    video_files = [os.path.join(video_dir, f'collision{i:02d}.mp4') for i in range(range_min, range_max + 1)]
    excel_files = [os.path.join(excel_dir, f'video-{i:05d}.xlsx') for i in range(range_min, range_max + 1)]
    
    paired_files = [(v, e) for v, e in zip(video_files, excel_files) if os.path.exists(v) and os.path.exists(e)]
    video_files_out, excel_files_out = zip(*paired_files) if paired_files else ([], [])
    
    return list(video_files_out), list(excel_files_out)

def generate_out_videoname(video_base: str) -> str:
    """
    Generates a standardized output video name string from a base video filename.
    Formats the extracted video number to five digits, matching Excel annotation format.
    
    Args:
        video_base (str): The base name of the video file (e.g., 'collision02').
        
    Returns:
        str: A formatted string like "video-00002".
    """
    video_num = video_base.replace('collision', '')
    return f"video-{int(video_num):05d}"

def process_and_save_frames(excel_files: list[str], video_files: list[str], output_dir: str, 
                            target_size: tuple[int, int] = (224, 224), train_ratio: float = 0.8) -> list[str]:
    """
    Processes video files, extracts individual frames, resizes them, labels them
    based on Excel annotations, and saves them as PNG images into a train/test
    directory structure. This function is typically used for single-frame models.
    
    If the target output directories already exist and appear complete, the function
    will skip processing to avoid redundant work. Otherwise, it clears and recreates
    the directories before processing.

    Args:
        excel_files (list[str]): List of paths to Excel annotation files.
        video_files (list[str]): List of paths to video files.
        output_dir (str): Base directory where 'train' and 'test' folders will be created,
                          containing subfolders for '0' (no collision) and '1' (collision) labels.
        target_size (tuple[int, int]): Target (width, height) for resizing frames.
        train_ratio (float): Ratio of frames to assign to the training set (0.0 to 1.0).

    Returns:
        list[str]: A list of file paths to the newly saved frame images. Returns an empty
                   list if processing is skipped (directories already existed).
    """
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    
    train_collision_dir = os.path.join(train_dir, '1')
    train_no_collision_dir = os.path.join(train_dir, '0')
    test_collision_dir = os.path.join(test_dir, '1')
    test_no_collision_dir = os.path.join(test_dir, '0')

    # Check if the full expected directory structure for saved frames already exists
    all_dirs_exist = (os.path.exists(train_collision_dir) and
                      os.path.exists(train_no_collision_dir) and
                      os.path.exists(test_collision_dir) and
                      os.path.exists(test_no_collision_dir))

    if all_dirs_exist:
        print(f"Output directories already exist and are complete at '{output_dir}'. Skipping frame processing.")
        return [] # Return an empty list as no new files were processed/saved

    # If the full structure is not found or incomplete, proceed to clear and recreate
    print(f"Output directories not fully found or are incomplete at '{output_dir}'. Clearing and recreating the structure.")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir) # Remove the entire base output directory and its contents
        
    create_image_directories(output_dir) # Recreate the fresh directory structure
    
    filenames = []
    np.random.seed(42) # Set seed for reproducibility of train/test split at frame level

    for excel_file, video_file in tqdm(zip(excel_files, video_files), total=len(excel_files), desc="Processing videos"):
        if not (os.path.exists(excel_file) and os.path.exists(video_file)):
            print(f"Skipping: File missing - Excel: {excel_file}, Video: {video_file}")
            continue
        try:
            df = pd.read_excel(excel_file)
        except Exception as e:
            print(f"Error reading Excel file {excel_file}: {e}")
            continue

        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Error opening video file: {video_file}")
            cap.release()
            continue

        frame_dict = {row['file']: int(row['collision']) for _, row in df.iterrows()}

        video_base = os.path.basename(video_file).split('.')[0]
        out_video_name = generate_out_videoname(video_base)

        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_name = f"{out_video_name}-frame-{str(frame_count + 1).zfill(5)}"
            if frame_name in frame_dict:
                label = frame_dict[frame_name]
                if label not in [0, 1]:
                    print(f"Skipping frame {frame_name} with invalid label: {label}")
                    frame_count += 1
                    continue
                
                frame = cv2.resize(frame, target_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB before saving as PNG
                
                split = 'train' if np.random.rand() < train_ratio else 'test'
                label_dir = os.path.join(output_dir, split, str(label))
                save_path = os.path.join(label_dir, f"{frame_name}.png")
                
                if cv2.imwrite(save_path, frame): # Save the RGB frame as PNG
                    filenames.append(save_path)
                else:
                    print(f"Failed to save: {save_path}")
            
            frame_count += 1
            
        cap.release()

    return filenames


def process_and_save_frames_cars_dataset(labels_csv_path, source_videos_directory, output_dir,
                                         target_size=(224, 224), train_ratio=0.8,
                                         num_collision_videos_to_process=50, # Renamed
                                         num_no_collision_videos_to_process=50, # Renamed
                                         frames_per_second=1):
    """Processes video frames from the cars dataset, saves them to train/test/0/1.
    It clears and recreates output directories before processing.

    Args:
        labels_csv_path (str): Path to the CSV file containing video IDs and labels (e.g., train.csv).
        source_videos_directory (str): The directory containing the video files to be processed.
                                       (e.g., 'train' folder containing all source videos).
        output_dir (str): Base directory where 'train' and 'test' folders for images will be created.
        target_size (tuple): Target (height, width) for resizing frames.
        train_ratio (float): Ratio of extracted frames to assign to the training set.
        num_collision_videos_to_process (int): Number of collision videos to select from the source.
        num_no_collision_videos_to_process (int): Number of no-collision videos to select from the source.
        frames_per_second (int): How many frames to extract per second from the video.
    Returns:
        list: List of paths to the saved frame images.
    """
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    
    train_collision_dir = os.path.join(train_dir, '1')
    train_no_collision_dir = os.path.join(train_dir, '0')
    test_collision_dir = os.path.join(test_dir, '1')
    test_no_collision_dir = os.path.join(test_dir, '0')

    # Check if the full expected directory structure for saved frames already exists
    all_dirs_exist = (os.path.exists(train_collision_dir) and
                      os.path.exists(train_no_collision_dir) and
                      os.path.exists(test_collision_dir) and
                      os.path.exists(test_no_collision_dir))

    if all_dirs_exist:
        print(f"Output directories already exist and are complete at '{output_dir}'. Skipping frame processing.")
        return [] # Return an empty list as no new files were processed/saved

    # If the full structure is not found or incomplete, proceed to clear and recreate
    print(f"Output directories not fully found or are incomplete at '{output_dir}'. Clearing and recreating the structure.")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir) # Remove the entire base output directory and its contents
        
    create_image_directories(output_dir) # Recreate the fresh directory structure

    filenames = []
    np.random.seed(42)

    try:
        df_labels = pd.read_csv(labels_csv_path) # Changed to df_labels
    except Exception as e:
        print(f"Error reading labels CSV file {labels_csv_path}: {e}")
        return []

    df_labels['time_of_alert'] = pd.to_numeric(df_labels['time_of_alert'], errors='coerce')
    df_labels['time_of_event'] = pd.to_numeric(df_labels['time_of_event'], errors='coerce')

    collision_video_ids = df_labels[df_labels['target'] == 1]['id'].tolist()
    no_collision_video_ids = df_labels[df_labels['target'] == 0]['id'].tolist()

    np.random.shuffle(collision_video_ids)
    np.random.shuffle(no_collision_video_ids)

    selected_collision_ids = collision_video_ids[:num_collision_videos_to_process]
    selected_no_collision_ids = no_collision_video_ids[:num_no_collision_videos_to_process]

    all_selected_videos_to_process = selected_collision_ids + selected_no_collision_ids
    np.random.shuffle(all_selected_videos_to_process)

    print(f"Processing {len(selected_collision_ids)} collision-event videos and "
          f"{len(selected_no_collision_ids)} no-collision-event videos from '{source_videos_directory}' for data generation...")

    # --- Process Selected Videos from the SINGLE SOURCE Directory ---
    # The 'source_videos_directory' now directly points to the folder containing videos (e.g., 'train/')
    # So we don't need os.path.join(base_video_dir, 'train') anymore.
    # It's just 'source_videos_directory' itself.

    for video_id in tqdm(all_selected_videos_to_process, desc="Processing selected videos"):
        video_filename = f"{video_id:05d}.mp4"
        video_file_path = os.path.join(source_videos_directory, video_filename) # Corrected path here

        if not os.path.exists(video_file_path):
            print(f"Warning: Video file not found: {video_file_path}. Skipping.")
            continue

        video_data = df_labels[df_labels['id'] == video_id].iloc[0]
        if video_data.empty:
            print(f"Warning: No data found for video ID {video_id} in CSV. Skipping.")
            continue

        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_file_path}. Skipping.")
            cap.release()
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print(f"Warning: Could not get FPS for {video_file_path}. Skipping.")
            cap.release()
            continue

        time_of_alert = video_data['time_of_alert'] if pd.notna(video_data['time_of_alert']) else -1
        time_of_event = video_data['time_of_event'] if pd.notna(video_data['time_of_event']) else -1
        has_collision_target = video_data['target']

        frame_interval = max(1, round(fps / frames_per_second))
        frame_idx = 0

        stop_frame_idx = float('inf')
        if has_collision_target == 1 and time_of_event != -1:
            stop_frame_idx = math.floor(time_of_event * fps) + 1

        while cap.isOpened():
            success, frame = cap.read()
            if not success or frame_idx >= stop_frame_idx:
                break

            current_time = frame_idx / fps

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            # --- Binary Labeling Logic (0 or 1) ---
            label = 0 # Default: No Collision

            if has_collision_target == 1:
                if time_of_alert != -1 and time_of_event != -1:
                    if current_time >= time_of_alert and current_time <= time_of_event:
                        label = 1
                elif time_of_event != -1:
                    if current_time >= time_of_event:
                        label = 1
                elif time_of_alert != -1:
                    if current_time >= time_of_alert:
                        label = 1

            frame_base_name = f"{video_id:05d}-frame-{str(frame_idx + 1).zfill(5)}"

            frame_resized = cv2.resize(frame, target_size)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            split = 'train' if np.random.rand() < train_ratio else 'test'
            label_dir = os.path.join(output_dir, split, str(label))
            os.makedirs(label_dir, exist_ok=True)
            save_path = os.path.join(label_dir, f"{frame_base_name}.png")

            try:
                if cv2.imwrite(save_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)):
                    filenames.append(save_path)
                else:
                    print(f"Failed to save: {save_path}")
            except Exception as e:
                print(f"Error saving image {save_path}: {e}")

            frame_idx += 1
        cap.release()

    return filenames

def get_preprocessing_function(model_name: str):
    """
    Returns the appropriate `preprocess_input` function for a given Keras Application model.
    These functions normalize pixel values and handle format conversions (e.g., BGR to RGB)
    as required by the specific pre-trained model's training regime.
    
    Args:
        model_name (str): The name of the pre-trained model (e.g., 'VGG16', 'ResNet50', 'MobileNetV2', 'EfficientNetB0').
        
    Returns:
        Callable: The corresponding `preprocess_input` function, or None if the model name is not supported.
    """
    if model_name == "VGG16":
        return vgg16_preprocess_input
    elif model_name == "ResNet50":
        return resnet50_preprocess_input
    elif model_name == "MobileNetV2":
        return mobilenetv2_preprocess_input
    elif model_name == "EfficientNetB0": 
        return efficientnet_preprocess_input 
    else:
        return None 


class HDF5DataLoaderForTFData:
    """
    A callable class designed to load image/sequence data from HDF5 files
    and apply model-specific preprocessing. It's built for seamless integration
    with `tf.data.Dataset.map()` using `tf.py_function`.
    """
    def __init__(self, expected_seq_len: int, img_h: int, img_w: int, 
                 model_name_for_preprocess: str, 
                 output_as_4d_if_single_frame: bool):
        """
        Initializes the HDF5 data loader with expected data dimensions and preprocessing details.
        
        Args:
            expected_seq_len (int): The expected number of frames in each sequence as stored in HDF5.
                                    For single-frame data, this should be 1.
            img_h (int): Target image height.
            img_w (int): Target image width.
            model_name_for_preprocess (str): Name of the pre-trained model to fetch the correct preprocessing function.
            output_as_4d_if_single_frame (bool): If True, and `expected_seq_len` is 1, the output tensor
                                                 will be shaped as (H, W, C) (4D). If False, or
                                                 `expected_seq_len` > 1, the output will be (SeqLen, H, W, C) (5D).
                                                 This flag is crucial for matching the input requirements of the target model.
        """
        self.expected_seq_len = expected_seq_len
        self.img_h = img_h
        self.img_w = img_w
        self.model_name_for_preprocess = model_name_for_preprocess 
        self.preprocess_func = get_preprocessing_function(model_name_for_preprocess)
        self.output_as_4d_if_single_frame = output_as_4d_if_single_frame 

        if not self.preprocess_func:
            print(f"Warning (HDF5DataLoaderForTFData.init): No specific preprocessing function found for {model_name_for_preprocess}. Data will be loaded as [0, 255] float32.")

    def _load_and_process_single_file(self, filepath_tensor: tf.Tensor) -> np.ndarray:
        """
        Internal function to load and process a single HDF5 file.
        It decodes the filepath, loads data, converts BGR to RGB, applies conditional
        reshaping based on `output_as_4d_if_single_frame`, and then applies
        model-specific preprocessing. This function runs in Python eager mode.
        
        Args:
            filepath_tensor (tf.Tensor): A TensorFlow string tensor containing the HDF5 file path.
            
        Returns:
            np.ndarray: The processed image data as a NumPy array, ready for conversion to a TensorFlow tensor.
        """
        filepath = filepath_tensor.numpy().decode('utf-8') 
        try:
            with h5py.File(filepath, 'r') as hf:
                data = hf['sequences'][:] # Load data (expected uint8, BGR, [0,255])
                data = data.astype(np.float32)

                # Convert BGR (OpenCV default) to RGB (Keras applications typically expect RGB)
                if data.ndim == 4: # If it's a sequence (S, H, W, C)
                    data_for_preprocess = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in data])
                elif data.ndim == 3: # If it's a single frame (H, W, C)
                    data_for_preprocess = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                else:
                    data_for_preprocess = data 

                # Conditional reshaping based on whether the target model expects 4D or 5D input.
                # HDF5 files internally store (SeqLen, H, W, C) even if SeqLen=1.
                expected_output_shape = None 
                
                if self.output_as_4d_if_single_frame and self.expected_seq_len == 1:
                    # For single-frame CNNs (expecting 4D: H, W, C).
                    if data_for_preprocess.ndim == 4 and data_for_preprocess.shape[0] == 1:
                        data_for_preprocess = data_for_preprocess[0] # Strip the leading 1 dimension
                    
                    expected_output_shape = (self.img_h, self.img_w, 3)
                
                else: # For multi-frame CNN-LSTMs (always expecting 5D: SeqLen, H, W, C).
                    expected_output_shape = (self.expected_seq_len, self.img_h, self.img_w, 3)

                # Validate the shape after all adjustments, before preprocessing function call
                if data_for_preprocess.shape != expected_output_shape:
                    raise ValueError(f"Shape mismatch: Data from {filepath} has unexpected shape {data_for_preprocess.shape} after BGR2RGB and conditional reshape. Expected {expected_output_shape}")

                # Apply the model-specific preprocessing function
                if self.preprocess_func:
                    processed_data = self.preprocess_func(data_for_preprocess) 
                else:
                    processed_data = data_for_preprocess 

                return processed_data
            
        except Exception as e:
            # Error handling for issues during file loading or processing
            print(f"\n--- ERROR IN HDF5DataLoaderForTFData._load_and_process_single_file ---")
            print(f"File: {filepath}")
            print(f"Exception Type: {type(e).__name__}")
            print(f"Error Message: {e}")
            print(f"Current expected_seq_len: {self.expected_seq_len}")
            print(f"Current img_h: {self.img_h}, img_w: {self.img_w}")
            print(f"Model for preprocess: {self.model_name_for_preprocess}")
            print(f"----------------------------------------------------")
            
            # Return a placeholder array on error to prevent the dataset from crashing.
            # The shape and dtype must match the expected output of the `load_fn_tf_wrapper`.
            if self.output_as_4d_if_single_frame and self.expected_seq_len == 1:
                output_shape_on_error = (self.img_h, self.img_w, 3)
            else:
                output_shape_on_error = (self.expected_seq_len, self.img_h, self.img_w, 3)
            
            return np.zeros(output_shape_on_error, dtype=np.float32)


    @tf.function 
    def load_fn_tf_wrapper(self, filepath_tensor: tf.Tensor, label_tensor: tf.Tensor):
        """
        A TensorFlow function wrapper that invokes `_load_and_process_single_file`
        using `tf.py_function` and explicitly sets the static output shape.
        This function is designed to be mapped over a `tf.data.Dataset`.
        
        Args:
            filepath_tensor (tf.Tensor): The TensorFlow tensor containing the file path.
            label_tensor (tf.Tensor): The TensorFlow tensor containing the label.
            
        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple of the processed image tensor and the label tensor.
        """
        # Determine the expected output shape for `set_shape` based on the configuration
        if self.output_as_4d_if_single_frame and self.expected_seq_len == 1:
            # For single-frame models, the Dataset will output (H, W, C)
            image_output_shape = tf.TensorShape([self.img_h, self.img_w, 3]) 
        else:
            # For multi-frame models, the Dataset will output (SeqLen, H, W, C)
            image_output_shape = tf.TensorShape([self.expected_seq_len, self.img_h, self.img_w, 3])

        # Execute the Python loading/processing function within the TensorFlow graph
        processed_image_tensor = tf.py_function(
            func=self._load_and_process_single_file,
            inp=[filepath_tensor],
            Tout=tf.float32, 
            name="load_and_preprocess_image_op" 
        )
        # It's crucial to set the static shape of the output tensor for efficient graph compilation.
        processed_image_tensor.set_shape(image_output_shape)
        
        return processed_image_tensor, label_tensor


def load_labeled_hdf5_sequence_filepaths(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Identifies and loads file paths to labeled HDF5 sequence files from a specified
    directory structure (expected: `data_dir/[train|test]/[0|1]/`).
    Performs basic integrity checks on HDF5 files to ensure they are valid and readable.
    
    Args:
        data_dir (str): The root directory containing 'train' and 'test' subdirectories,
                        which in turn contain '0' and '1' class label subdirectories with HDF5 files.
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, list[str]]: 
            - train_df (pd.DataFrame): DataFrame with 'filepaths' and 'labels' for the training set.
            - test_df (pd.DataFrame): DataFrame with 'filepaths' and 'labels' for the testing set.
            - class_names (list[str]): List of class names (e.g., ['no_collision', 'collision']).
    """
    train_filepaths, train_labels = [], []
    test_filepaths, test_labels = [], []
    class_names = ['no_collision', 'collision']

    print(f"Collecting HDF5 file paths from: {data_dir}")
    total_files_found = 0
    total_files_valid = 0

    for split_type in ['train', 'test']:
        for class_label_str in ['0', '1']:
            current_dir_path = os.path.join(data_dir, split_type, class_label_str)
            
            if not os.path.isdir(current_dir_path):
                print(f"Warning: Directory not found: {current_dir_path}. Skipping.")
                continue
            
            h5_files = [f for f in os.listdir(current_dir_path) if f.endswith('.hdf5')]
            for h5_file_name in tqdm(h5_files, desc=f"Checking {split_type}/{class_label_str} files from {os.path.basename(current_dir_path)}"):
                total_files_found += 1
                
                actual_h5_path = os.path.join(current_dir_path, h5_file_name)
                
                if not os.path.exists(actual_h5_path):
                    print(f"Error: File path does not exist on disk: {actual_h5_path}")
                    continue
                
                try:
                    # Attempt to open the HDF5 file to check its integrity
                    with h5py.File(actual_h5_path, 'r') as hf:
                        _ = hf['sequences'] # Access 'sequences' dataset to ensure it's a valid HDF5 file
                    
                    label = int(class_label_str)

                    if split_type == 'train':
                        train_filepaths.append(actual_h5_path)
                        train_labels.append(label)
                    else: 
                        test_filepaths.append(actual_h5_path)
                        test_labels.append(label)
                    total_files_valid += 1
                
                except Exception as e:
                    print(f"Error: Could not open or read HDF5 file {actual_h5_path} (potentially corrupted/invalid HDF5): {e}. Skipping file.")
    
    train_df = pd.DataFrame({'filepaths': train_filepaths, 'labels': train_labels})
    test_df = pd.DataFrame({'filepaths': test_filepaths, 'labels': test_labels})

    print(f"\n--- HDF5 File Verification Summary ---")
    print(f"Total HDF5 files found on disk: {total_files_found}")
    print(f"Total HDF5 files successfully loaded and validated: {total_files_valid}")
    print(f"Files that caused errors/were not found: {total_files_found - total_files_valid}")
    print(f"DataFrame: Found {len(train_df)} training files and {len(test_df)} testing files.")
    print(f"------------------------------------")

    return train_df, test_df, class_names

def create_labeled_sequences_from_annotations(
    dataset_type: str,
    base_dataset_dir: str, # e.g., 'drones', 'cars'
    output_dir: str, 
    sequence_length: int = 10, 
    target_size: tuple[int, int] = (224, 224), 
    train_test_split_ratio: float = 0.8, 
    stride: int = 1,
    num_collision_videos_to_process: int = 0, 
    num_no_collision_videos_to_process: int = 0
) -> tuple[str, str, list[str]]:
    """
    Processes video files and their corresponding annotations to create labeled sequences of frames.
    This function is flexible to handle both 'drones' (Excel annotations) and 'cars' (CSV annotations) datasets.
    Each extracted sequence is saved as an individual HDF5 file. The sequences are then
    split into training and testing sets, ensuring stratification by collision label.
    
    For 'cars' dataset videos with a collision target, frame extraction for sequences
    will stop at or near the `time_of_event` to focus on the pre-collision context.
    
    Args:
        dataset_type (str): Specifies the dataset to process ('drones' or 'cars').
        base_dataset_dir (str): The root directory for the specific dataset (e.g., './drones' or './cars').
                                 Expected sub-structure depends on `dataset_type`.
        output_dir (str): The base directory where the processed HDF5 sequences will be saved.
                          A subdirectory structure like `output_dir/train/0/`, `output_dir/test/1/`
                          will be created to organize the HDF5 files.
        sequence_length (int): The number of consecutive frames to include in each sequence.
        target_size (tuple[int, int]): A tuple `(width, height)` specifying the desired dimensions
                                       to resize each frame to.
        train_test_split_ratio (float): The proportion of sequences to allocate to the
                                        training set (a value between 0.0 and 1.0).
        stride (int): The step size in frames for the sliding window that extracts sequences.
                      A stride of 1 means every possible sequence is extracted.
                      A stride equal to `sequence_length` means non-overlapping sequences.
        num_collision_videos_to_process (int): For the 'cars' dataset, specifies the maximum number of
                                               collision videos to process from the training split.
                                               Set to 0 or negative to process all available collision videos.
                                               Ignored for 'drones' dataset.
        num_no_collision_videos_to_process (int): For the 'cars' dataset, specifies the maximum number of
                                                  no-collision videos to process from the training split.
                                                  Set to 0 or negative to process all available no-collision videos.
                                                  Ignored for 'drones' dataset.
                      
    Returns:
        tuple[str, str, list[str]]: 
            - train_output_dir (str): Path to the 'train' directory containing sequence HDF5 files.
            - test_output_dir (str): Path to the 'test' directory containing sequence HDF5 files.
            - class_names (list[str]): List of class names (e.g., ['no_collision', 'collision']).
    """
    train_output_dir = os.path.join(output_dir, 'train')
    test_output_dir = os.path.join(output_dir, 'test')
    class_names = ['no_collision', 'collision']

    # --- Check for existing sequences to avoid re-processing ---
    # Check if the expected HDF5 directories exist and contain at least some files
    all_target_dirs = [os.path.join(train_output_dir, '0'), os.path.join(train_output_dir, '1'),
                       os.path.join(test_output_dir, '0'), os.path.join(test_output_dir, '1')]
    
    all_dirs_exist_and_not_empty = True
    for d in all_target_dirs:
        if not os.path.exists(d) or not any(f.endswith('.hdf5') for f in os.listdir(d)):
            all_dirs_exist_and_not_empty = False
            break

    if all_dirs_exist_and_not_empty:
        print(f"Detected existing HDF5 sequences in '{output_dir}'. Skipping sequence generation.")
        # Return existing directories if generation is skipped
        return train_output_dir, test_output_dir, class_names

    # If directories are not complete, proceed to clear and recreate
    print(f"HDF5 sequences not fully found or incomplete in '{output_dir}'. Clearing and regenerating.")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    os.makedirs(os.path.join(train_output_dir, '0'), exist_ok=True)
    os.makedirs(os.path.join(train_output_dir, '1'), exist_ok=True)
    os.makedirs(os.path.join(test_output_dir, '0'), exist_ok=True)
    os.makedirs(os.path.join(test_output_dir, '1'), exist_ok=True)
    print(f"Created new output directory structure at {output_dir}")

    all_videos_info = [] # Will store structured info for videos to be processed

    if dataset_type == "drones":
        drones_video_dir = os.path.join(base_dataset_dir, 'videos')
        drones_annotation_dir = os.path.join(base_dataset_dir, 'dataframes')
        
        annotation_files_dict = {f.replace('.xlsx', '').split('-')[-1]: os.path.join(drones_annotation_dir, f) 
                                 for f in os.listdir(drones_annotation_dir) if f.endswith('.xlsx')}

        print(f"Collecting video and annotation info for 'drones' from: {drones_video_dir} and {drones_annotation_dir}")
        for video_id_key in tqdm(sorted(annotation_files_dict.keys()), desc="Collecting Drone Video Info"):
            video_id_for_filename = str(int(video_id_key)).zfill(2) # e.g., '01' from '00001'
            video_file_name = f"collision{video_id_for_filename}.mp4" 
            video_path = os.path.join(drones_video_dir, video_file_name)
            annotation_path = annotation_files_dict.get(video_id_key)

            if not os.path.exists(video_path) or not annotation_path:
                print(f"Warning: Missing video ({video_path}) or annotation ({annotation_path}) for ID: {video_id_key}. Skipping.")
                continue

            try:
                df_anno = pd.read_excel(annotation_path)
                frame_collision_labels = {}
                for _, row in df_anno.iterrows():
                    try:
                        filename_in_excel = str(row['file']) # e.g., 'video-00001-frame-00001'
                        frame_number = int(filename_in_excel.split('-frame-')[1].split('.')[0])
                        collision_label = int(row['collision'])
                        frame_collision_labels[frame_number] = collision_label
                    except (IndexError, ValueError):
                        print(f"Warning: Could not parse frame or label from '{row.get('file', 'N/A')}' in {annotation_path}. Skipping row.")
                
                video_id_for_hdf5_name = video_id_key 
                all_videos_info.append({
                    'video_path': video_path,
                    'video_id_for_hdf5_name': video_id_for_hdf5_name,
                    'annotations': frame_collision_labels, # Dict: {frame_number: label}
                    'frame_rate': None # Not directly used for labeling logic in drones for sequence generation
                })
            except Exception as e:
                print(f"Error processing drone video {video_id_key}: {e}. Skipping.")

    elif dataset_type == "cars":
        cars_videos_source_dir_train = os.path.join(base_dataset_dir, 'videos') # Base directory (e.g., ./cars/videos)
        cars_csv_path = os.path.join(base_dataset_dir, 'data_labels.csv') # Assuming data_labels.csv

        if not os.path.exists(cars_videos_source_dir_train):
            print(f"Error: Car videos source directory not found at '{cars_videos_source_dir_train}'. Exiting.")
            return train_output_dir, test_output_dir, class_names
        if not os.path.exists(cars_csv_path):
            print(f"Error: Car annotation CSV not found at '{cars_csv_path}'. Exiting.")
            return train_output_dir, test_output_dir, class_names

        try:
            annotations_df_cars = pd.read_csv(cars_csv_path)
            annotations_df_cars['id'] = annotations_df_cars['id'].astype(int) 
            annotations_df_cars['time_of_alert'] = pd.to_numeric(annotations_df_cars['time_of_alert'], errors='coerce')
            annotations_df_cars['time_of_event'] = pd.to_numeric(annotations_df_cars['time_of_event'], errors='coerce')
        except Exception as e:
            print(f"Error reading car annotation CSV '{cars_csv_path}': {e}. Exiting.")
            return train_output_dir, test_output_dir, class_names

        collision_video_ids = annotations_df_cars[annotations_df_cars['target'] == 1]['id'].tolist()
        no_collision_video_ids = annotations_df_cars[annotations_df_cars['target'] == 0]['id'].tolist()

        np.random.seed(42) # For reproducible selection of car videos
        np.random.shuffle(collision_video_ids)
        np.random.shuffle(no_collision_video_ids)

        selected_collision_ids = collision_video_ids[:num_collision_videos_to_process] if num_collision_videos_to_process > 0 else collision_video_ids
        selected_no_collision_ids = no_collision_video_ids[:num_no_collision_videos_to_process] if num_no_collision_videos_to_process > 0 else no_collision_video_ids

        all_selected_videos_to_process = selected_collision_ids + selected_no_collision_ids
        np.random.shuffle(all_selected_videos_to_process) 

        print(f"Processing {len(selected_collision_ids)} collision videos and "
              f"{len(selected_no_collision_ids)} no-collision videos from '{cars_videos_source_dir_train}' for sequence generation...")

        if not all_selected_videos_to_process:
            print("No videos selected for processing from the 'cars' dataset. Exiting.")
            return train_output_dir, test_output_dir, class_names

        print(f"Collecting video details for 'cars' from: {cars_videos_source_dir_train}")
        for video_id in tqdm(all_selected_videos_to_process, desc="Collecting Car Video Info"):
            video_filename = f"{video_id:05d}.mp4" 
            video_path = os.path.join(cars_videos_source_dir_train, video_filename)
            
            if not os.path.exists(video_path):
                print(f"Warning: Video file not found for ID {video_id} at {video_path}. Skipping.")
                continue

            video_annotation_row = annotations_df_cars[annotations_df_cars['id'] == video_id].iloc[0]
            
            video_id_for_hdf5_name = str(video_id).zfill(5)

            temp_cap = cv2.VideoCapture(video_path)
            temp_frame_rate = temp_cap.get(cv2.CAP_PROP_FPS)
            temp_cap.release() 
            
            if temp_frame_rate <= 0:
                print(f"Warning: Invalid frame rate for {video_path}. Skipping.")
                continue

            all_videos_info.append({
                'video_path': video_path,
                'video_id_for_hdf5_name': video_id_for_hdf5_name,
                'annotations': video_annotation_row, # Pandas Series with time_of_event, time_of_alert, target
                'frame_rate': temp_frame_rate
            })
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Choose 'drones' or 'cars'.")

    if not all_videos_info:
        print("No valid videos found or selected for processing for any dataset type. Exiting.")
        return train_output_dir, test_output_dir, class_names
    
    # --- Main loop for sequence extraction (Unified Logic) ---
    all_sequences_metadata = []
    print("Extracting sequences from selected videos...")
    for video_info in tqdm(all_videos_info, desc="Extracting Sequences"):
        video_path = video_info['video_path']
        video_id_for_hdf5_name = video_info['video_id_for_hdf5_name']
        annotations = video_info['annotations'] 
        frame_rate = video_info['frame_rate'] 

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}. Skipping.")
            continue

        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # --- Determine the frame limit for extraction based on dataset type ---
        max_frame_to_process = total_frames_in_video
        if dataset_type == "cars":
            video_overall_target = annotations.get('target', 0)
            time_of_event = annotations.get('time_of_event')
            
            # If it's a collision video and time_of_event is valid, limit frame processing
            if video_overall_target == 1 and pd.notna(time_of_event) and time_of_event >= 0:
                # Calculate the frame index corresponding to time_of_event
                # Sequences should not start if their end would go past time_of_event
                # Or, more simply, stop iterating through frames once time_of_event is reached.
                # `math.floor(time_of_event * frame_rate)` gives the 0-based index of the frame AT the event time.
                # +1 to include that frame.
                frame_idx_at_event = math.floor(time_of_event * frame_rate) + 1
                # We need to ensure that the *start_frame_idx* for a sequence, plus its length,
                # does not exceed this `frame_idx_at_event`.
                # So, sequences can only start up to `frame_idx_at_event - sequence_length`.
                max_frame_to_process = min(total_frames_in_video, frame_idx_at_event)
                # Ensure it's at least 1, otherwise range might become negative and fail.
                max_frame_to_process = max(0, max_frame_to_process) 
                
        sequence_index_in_video = 0 

        # The loop now runs up to max_frame_to_process, considering the sequence length.
        for start_frame_idx in range(0, max_frame_to_process - sequence_length + 1, stride):
            frames_in_sequence = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx) 
            
            for _ in range(sequence_length):
                ret, frame = cap.read() 
                if not ret:
                    break
                frame_resized = cv2.resize(frame, target_size)
                frames_in_sequence.append(frame_resized.astype(np.uint8)) 
            
            if len(frames_in_sequence) == sequence_length:
                is_collision_in_sequence = False
                
                # --- Dataset-specific labeling logic for sequence ---
                if dataset_type == "drones":
                    frame_collision_labels = annotations # This is the dict {frame_number: label}
                    for frame_offset in range(sequence_length):
                        current_frame_number_in_video = start_frame_idx + frame_offset + 1 # 1-based index
                        if current_frame_number_in_video in frame_collision_labels and frame_collision_labels[current_frame_number_in_video] == 1:
                            is_collision_in_sequence = True
                            break
                elif dataset_type == "cars":
                    video_overall_target = annotations.get('target', 0)
                    time_of_alert = annotations.get('time_of_alert')
                    time_of_event = annotations.get('time_of_event')

                    if video_overall_target == 1: # Only check collision interval if video is labeled as positive
                        for frame_offset in range(sequence_length):
                            current_frame_time = (start_frame_idx + frame_offset) / frame_rate # 0-based frame_idx to time
                            
                            # A sequence is labeled '1' if any frame within it is between time_of_alert and time_of_event.
                            if pd.notna(time_of_alert) and pd.notna(time_of_event) and \
                               time_of_alert <= current_frame_time <= time_of_event:
                                is_collision_in_sequence = True
                                break
                
                data_to_save = np.array(frames_in_sequence) # Shape: (sequence_length, H, W, C) - always 5D here

                all_sequences_metadata.append({
                    'video_id': video_id_for_hdf5_name, 
                    'sequence_idx_in_video': sequence_index_in_video,
                    'data': data_to_save, 
                    'label': 1 if is_collision_in_sequence else 0
                })
                sequence_index_in_video += 1 
        
        cap.release()

    if not all_sequences_metadata:
        print("No sequences extracted from any video.")
        return train_output_dir, test_output_dir, class_names

    # --- Perform SEQUENCE-LEVEL Train/Test Split ---
    print("\nPerforming sequence-level train/test split...")
    sequences_labels = [item['label'] for item in all_sequences_metadata]
    
    train_indices, test_indices, _, _ = train_test_split(
        range(len(all_sequences_metadata)), 
        sequences_labels, 
        test_size=(1 - train_test_split_ratio), 
        stratify=sequences_labels, 
        random_state=42
    )

    # --- Save split sequences to individual HDF5 files ---
    print("Saving split sequences to individual HDF5 files...")
    for idx_type, indices_list in {'train': train_indices, 'test': test_indices}.items():
        for original_idx in tqdm(indices_list, desc=f"Saving {idx_type} sequences"):
            item = all_sequences_metadata[original_idx]
            
            target_label_dir = os.path.join(output_dir, idx_type, str(item['label']))
            
            filename = f"video-{item['video_id']}-seq-{str(item['sequence_idx_in_video']).zfill(5)}.hdf5"
            h5_path = os.path.join(target_label_dir, filename)

            try:
                with h5py.File(h5_path, 'w') as hf:
                    hf.create_dataset('sequences', data=item['data'],
                                      compression='gzip', compression_opts=4)
                    hf.create_dataset('label', data=item['label'])
            except Exception as e:
                print(f"Error saving {h5_path}: {e}")

    print("\nSequence generation and saving complete.")
    return train_output_dir, test_output_dir, class_names