#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for processing video data, creating labeled frames or sequences,
and preparing them for deep learning model training. This module supports
different dataset structures and annotation formats.

Created on Mon May 19 15:48:03 2025
@author: fran
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import h5py
import shutil
import math # Import math for floor function

import tensorflow as tf 
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess_input
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenetv3_preprocess_input # Added for MobileNetV3Small
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input


def open_video(video):
    """
    Opens a video file using OpenCV's VideoCapture.
    
    Args:
        video (str): Path to the video file.
        
    Returns:
        cv2.VideoCapture: The video capture object.
    """
    return cv2.VideoCapture(video)

def check_video(cap, video_path):
    """
    Verifies if a video capture object is successfully opened and can read its first frame.
    
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

def close_cap(cap):
    """
    Releases a video capture object, freeing associated resources.
    
    Args:
        cap (cv2.VideoCapture): The video capture object to release.
    """
    cap.release()

def getFrameRate(video):
    """
    Retrieves the frame rate (frames per second) of an opened video.
    
    Args:
        video (cv2.VideoCapture): The video capture object.
        
    Returns:
        float: The frame rate of the video.
    """
    return video.get(cv2.CAP_PROP_FPS)

def generate_framename(video_num, pos_frame):
    """
    Generates a standardized frame name string from a video number and frame position.
    This format is typically used for annotations and saved image filenames.
    
    Args:
        video_num (int): The numerical ID of the video.
        pos_frame (int): The 1-based index of the frame within the video.
        
    Returns:
        str: A formatted string like "video-000XX-frame-000YY".
    """
    return f"video-{str(video_num).zfill(5)}-frame-{str(pos_frame).zfill(5)}"

def generate_video_num(out_videoname):
    """
    Extracts the numerical video ID from a standardized video name string.
    
    Args:
        out_videoname (str): A formatted video name string (e.g., "video-000XX").
        
    Returns:
        int: The extracted video number.
    """
    return int(out_videoname.split('-')[1])

def create_image_directories(output_base_dir):
    """
    Creates the necessary directory structure for storing processed individual image frames:
    `output_base_dir/train/0`, `output_base_dir/train/1`, `output_base_dir/test/0`, `output_base_dir/test/1`.
    Directories are created if they don't exist. This function is typically called
    after any necessary cleanup (e.g., `shutil.rmtree`) has already occurred.
    
    Args:
        output_base_dir (str): The root directory where the structure will be created.
    """
    train_dir = os.path.join(output_base_dir, 'train')
    test_dir = os.path.join(output_base_dir, 'test')

    os.makedirs(os.path.join(train_dir, '0'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, '1'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, '0'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, '1'), exist_ok=True)

def generate_paired_file_lists(range_min = 70, range_max = 93, base_dir = ''):
    """
    Generates lists of paired video and Excel annotation file paths for a specific
    range of video IDs, assuming the old dataset structure (used for 'drones' dataframes).
    Only includes pairs where both the video and Excel files physically exist.
    
    Args:
        range_min (int): The starting video ID (inclusive).
        range_max (int): The ending video ID (inclusive).
        base_dir (str): The base directory where 'videos' and 'dataframes' subdirectories are located.
                        If empty, assumes they are in the current working directory.
        
    Returns:
        tuple[list[str], list[str]]: Two lists, one for video file paths and one for Excel file paths.
                                     Returns empty lists if no valid pairs are found.
    """
    video_dir_path = os.path.join(base_dir, 'videos') if base_dir else 'videos'
    excel_dir_path = os.path.join(base_dir, 'dataframes') if base_dir else 'dataframes'
    
    video_files = [os.path.join(video_dir_path, f'collision{i:02d}.mp4') for i in range(range_min, range_max + 1)]
    excel_files = [os.path.join(excel_dir_path, f'video-{i:05d}.xlsx') for i in range(range_min, range_max + 1)]
    
    paired_files = [(v, e) for v, e in zip(video_files, excel_files) if os.path.exists(v) and os.path.exists(e)]
    video_files_out, excel_files_out = zip(*paired_files) if paired_files else ([], [])
    
    return list(video_files_out), list(excel_files_out)

def generate_out_videoname(video_base):
    """
    Generates a standardized output video name string from a base video filename.
    Formats the extracted video number to five digits, matching typical Excel annotation format.
    
    Args:
        video_base (str): The base name of the video file (e.g., 'collision02').
        
    Returns:
        str: A formatted string like "video-00002".
    """
    video_num = video_base.replace('collision', '')
    return f"video-{int(video_num):05d}"

def process_and_save_frames(excel_files, video_files, output_dir, 
                            target_size = (224, 224), train_ratio = 0.8):
    """
    Processes video frames from the original 'drones' dataset structure, labels them based on Excel annotations,
    resizes, and saves them as individual PNG images into a train/test/label directory structure.
    This function is primarily used for single-frame models and older data formats.

    This function ASSUMES the `output_dir` structure has already been created
    (e.g., by `create_image_directories` or `process_and_save_mixed_frames`).
    It will NOT clear the directory itself.

    Args:
        excel_files (list): List of full paths to Excel annotation files.
        video_files (list): List of full paths to video files.
        output_dir (str): Base directory where 'train' and 'test' folders are located,
                          containing subfolders for '0' (no collision) and '1' (collision) labels.
        target_size (tuple): Target (width, height) for resizing frames.
        train_ratio (float): Ratio of frames to assign to the training set.

    Returns:
        list: A list of file paths to the newly saved frame images.
    """
    filenames = []
    np.random.seed(42) # Set seed for reproducibility of frame-level train/test split

    print(f"Processing drone videos for output to: {output_dir}")
    for excel_file, video_file in tqdm(zip(excel_files, video_files), total=len(excel_files), desc="Processing Drone videos"):
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
                os.makedirs(label_dir, exist_ok=True) # Ensure directory exists
                save_path = os.path.join(label_dir, f"{frame_name}.png")
                
                # OpenCV's imwrite expects BGR, so convert back from RGB for saving
                if cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                    filenames.append(save_path)
                else:
                    print(f"Failed to save: {save_path}")
            
            frame_count += 1
            
        cap.release()

    return filenames


def process_and_save_frames_cars_dataset(labels_csv_path, source_videos_directory, output_dir,
                                          target_size = (224, 224), train_ratio = 0.8,
                                          num_collision_videos_to_process = 50,
                                          num_no_collision_videos_to_process = 50,
                                          frames_per_second = 1):
    """
    Processes video frames from the cars dataset, saving them to train/test/0/1 subdirectories.

    This function ASSUMES the `output_dir` structure has already been created
    (e.g., by `create_image_directories` or `process_and_save_mixed_frames`).
    It will NOT clear the directory itself.
    
    Args:
        labels_csv_path (str): Full path to the CSV file containing video IDs and labels (e.g., cars/data_labels.csv).
        source_videos_directory (str): Full path to the directory containing the video files to be processed
                                       (e.g., cars/videos).
        output_dir (str): Base directory where 'train' and 'test' folders for images are located.
        target_size (tuple): Target (height, width) for resizing frames.
        train_ratio (float): Ratio of extracted frames to assign to the training set.
        num_collision_videos_to_process (int): Number of collision videos to select from the source.
                                               Set to 0 or negative to process all available collision videos.
        num_no_collision_videos_to_process (int): Number of no-collision videos to select from the source.
                                                  Set to 0 or negative to process all available no-collision videos.
        frames_per_second (int): How many frames to extract per second from the video. This is
                                 a sampling rate, not the original video's FPS.
    
    Returns:
        list: List of paths to the saved frame images.
    """
    filenames = []
    np.random.seed(42) # Set seed for reproducibility

    print(f"Processing car videos for output to: {output_dir}")

    try:
        df_labels = pd.read_csv(labels_csv_path)
    except Exception as e:
        print(f"Error reading labels CSV file {labels_csv_path}: {e}")
        return []

    # Convert time columns to numeric, coercing errors to NaN
    df_labels['time_of_alert'] = pd.to_numeric(df_labels['time_of_alert'], errors='coerce')
    df_labels['time_of_event'] = pd.to_numeric(df_labels['time_of_event'], errors='coerce')

    collision_video_ids = df_labels[df_labels['target'] == 1]['id'].tolist()
    no_collision_video_ids = df_labels[df_labels['target'] == 0]['id'].tolist()

    # Shuffle to ensure random selection if subsetting
    np.random.shuffle(collision_video_ids)
    np.random.shuffle(no_collision_video_ids)

    # Select specified number of videos for each class
    selected_collision_ids = collision_video_ids[:num_collision_videos_to_process] if num_collision_videos_to_process > 0 else collision_video_ids
    selected_no_collision_ids = no_collision_video_ids[:num_no_collision_videos_to_process] if num_no_collision_videos_to_process > 0 else no_collision_video_ids

    all_selected_videos_to_process = selected_collision_ids + selected_no_collision_ids
    np.random.shuffle(all_selected_videos_to_process) # Shuffle combined list for mixed processing

    print(f"Processing {len(selected_collision_ids)} collision-event videos and "
          f"{len(selected_no_collision_ids)} no-collision-event videos from '{source_videos_directory}' for data generation...")

    for video_id in tqdm(all_selected_videos_to_process, desc="Processing selected Car videos"):
        video_filename = f"{video_id:05d}.mp4"
        video_file_path = os.path.join(source_videos_directory, video_filename)

        if not os.path.exists(video_file_path):
            print(f"Warning: Video file not found: {video_file_path}. Skipping.")
            continue

        video_data_row = df_labels[df_labels['id'] == video_id]
        if video_data_row.empty:
            print(f"Warning: No data found for video ID {video_id} in CSV. Skipping.")
            continue
        video_data = video_data_row.iloc[0] # Get the specific row for this video ID

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

        # Determine frame interval for sampling (e.g., if fps=30 and frames_per_second=15, sample every 2 frames)
        frame_sampling_interval = max(1, round(fps / frames_per_second))
        frame_idx_in_video = 0 # This tracks the actual frame index in the video

        stop_frame_idx_limit = float('inf')
        if has_collision_target == 1 and time_of_event != -1:
            stop_frame_idx_limit = math.floor(time_of_event * fps) + 1 # Convert time_of_event to frame index

        while cap.isOpened():
            success, frame = cap.read()
            if not success or frame_idx_in_video >= stop_frame_idx_limit:
                break

            current_time = frame_idx_in_video / fps

            # Only process frames according to the sampling rate (frames_per_second)
            if frame_idx_in_video % frame_sampling_interval != 0:
                frame_idx_in_video += 1
                continue

            # --- Binary Labeling Logic (0 or 1) for the current frame ---
            label = 0 # Default: No Collision

            if has_collision_target == 1:
                # If both alert and event times are valid, label if current_time is within the interval
                if pd.notna(time_of_alert) and pd.notna(time_of_event):
                    if current_time >= time_of_alert and current_time <= time_of_event:
                        label = 1
                # If only event time is valid, label from event time onwards (less common for 'alert' datasets)
                elif pd.notna(time_of_event):
                    if current_time >= time_of_event:
                        label = 1
                # If only alert time is valid, label from alert time onwards (less common)
                elif pd.notna(time_of_alert):
                    if current_time >= time_of_alert:
                        label = 1

            frame_base_name = f"video-{video_id:05d}-frame-{str(frame_idx_in_video + 1).zfill(5)}"

            frame_resized = cv2.resize(frame, target_size)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) # Convert to RGB

            split = 'train' if np.random.rand() < train_ratio else 'test'
            label_dir = os.path.join(output_dir, split, str(label))
            os.makedirs(label_dir, exist_ok=True) # Ensure directory exists before saving
            save_path = os.path.join(label_dir, f"{frame_base_name}.png")

            try:
                # OpenCV's imwrite expects BGR, so convert back from RGB for saving
                if cv2.imwrite(save_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)):
                    filenames.append(save_path)
                else:
                    print(f"Failed to save: {save_path}")
            except Exception as e:
                print(f"Error saving image {save_path}: {e}")

            frame_idx_in_video += 1
        cap.release()

    print(f"Finished processing and saving frames to {output_dir}.")
    return filenames


def process_and_save_mixed_frames(
    drone_video_range_min,
    drone_video_range_max,
    drone_base_dir, # e.g., "drones"
    car_labels_csv_path, # e.g., "cars/data_labels.csv"
    car_source_videos_directory, # e.g., "cars/videos"
    output_base_dir, # e.g., "mixed_data/image_data_mixed"
    target_size = (224, 224),
    train_ratio = 0.8,
    car_num_collision_videos_to_process = 0,
    car_num_no_collision_videos_to_process = 0,
    car_frames_per_second = 1
):
    """
    Processes and saves frames from both 'drones' and 'cars' datasets into a single
    mixed output directory structure (train/0, train/1, test/0, test/1).

    This function coordinates the calls to `process_and_save_frames` (for drones)
    and `process_and_save_frames_cars_dataset` (for cars), ensuring all processed
    images are placed into the `output_base_dir`.

    Args:
        drone_video_range_min (int): Min video ID for drones (inclusive).
        drone_video_range_max (int): Max video ID for drones (inclusive).
        drone_base_dir (str): Base directory for drone dataset (e.g., "drones").
                              Expected sub-structure: `drones_base_dir/videos` and `drones_base_dir/dataframes`.
        car_labels_csv_path (str): Full path to the car dataset's labels CSV (e.g., "cars/data_labels.csv").
        car_source_videos_directory (str): Full path to the car dataset's raw videos (e.g., "cars/videos").
        output_base_dir (str): The root directory where mixed processed image data will be saved.
                               Expected output structure: `output_base_dir/[train|test]/[0|1]/`.
        target_size (tuple): Target (width, height) for resizing frames.
        train_ratio (float): Ratio of frames to assign to the training set for the combined data.
        car_num_collision_videos_to_process (int): Number of collision videos to select from the car source.
        car_num_no_collision_videos_to_process (int): Number of no-collision videos to select from the car source.
        car_frames_per_second (int): Sampling rate for car data.

    Returns:
        list: A list of file paths to all newly saved frame images from both datasets.
                   Returns an empty list if processing is skipped (directories already existed).
    """
    print(f"\n--- Starting processing for MIXED dataset ---")
    
    # Define output directories for the combined dataset
    train_dir = os.path.join(output_base_dir, 'train')
    test_dir = os.path.join(output_base_dir, 'test')
    
    train_collision_dir = os.path.join(train_dir, '1')
    train_no_collision_dir = os.path.join(train_dir, '0')
    test_collision_dir = os.path.join(test_dir, '1')
    test_no_collision_dir = os.path.join(test_dir, '0')

    # Check if the full expected directory structure for saved frames already exists and is non-empty
    # This check is for the *combined* output_base_dir, not for individual dataset processing.
    all_dirs_exist = (os.path.exists(train_collision_dir) and
                      os.path.exists(train_no_collision_dir) and
                      os.path.exists(test_collision_dir) and
                      os.path.exists(test_no_collision_dir) and
                      len(os.listdir(train_collision_dir)) > 0 and # Check if directories are non-empty
                      len(os.listdir(train_no_collision_dir)) > 0 and
                      len(os.listdir(test_collision_dir)) > 0 and
                      len(os.listdir(test_no_collision_dir)) > 0)

    if all_dirs_exist:
        print(f"Output directories for mixed data already exist and are complete at '{output_base_dir}'. Skipping frame processing.")
        return [] # Return an empty list as no new files were processed/saved

    # If the full structure is not found or incomplete, proceed to clear and recreate
    print(f"Output directories for mixed data not fully found or are incomplete at '{output_base_dir}'. Clearing and recreating the structure.")
    if os.path.exists(output_base_dir):
        shutil.rmtree(output_base_dir) # Remove the entire base output directory and its contents
        
    create_image_directories(output_base_dir) # Recreate the fresh directory structure

    all_processed_filenames = []
    
    # --- Process Drones Data ---
    print("\nProcessing Drones data for mixed dataset...")
    # generate_paired_file_lists now needs base_dir
    drone_video_files, drone_excel_files = generate_paired_file_lists(
        range_min=drone_video_range_min, range_max=drone_video_range_max,
        base_dir=drone_base_dir # Pass the base directory for drones
    )
    
    drone_filenames = process_and_save_frames(
        drone_excel_files, # These are already full paths from generate_paired_file_lists
        drone_video_files, # These are already full paths from generate_paired_file_lists
        output_base_dir, # Save directly to the mixed output directory
        target_size=target_size,
        train_ratio=train_ratio
    )
    all_processed_filenames.extend(drone_filenames)
    print(f"Drones data processed. Total drone frames: {len(drone_filenames)}")

    # --- Process Cars Data ---
    print("\nProcessing Cars data for mixed dataset...")
    car_filenames = process_and_save_frames_cars_dataset(
        car_labels_csv_path,
        car_source_videos_directory,
        output_base_dir, # Save directly to the mixed output directory
        target_size=target_size,
        train_ratio=train_ratio,
        num_collision_videos_to_process=car_num_collision_videos_to_process,
        num_no_collision_videos_to_process=car_num_no_collision_videos_to_process,
        frames_per_second=car_frames_per_second # Corrected keyword argument name
    )
    all_processed_filenames.extend(car_filenames)
    print(f"Cars data processed. Total car frames: {len(car_filenames)}")

    print(f"\nFinished processing mixed dataset. Total combined frames: {len(all_processed_filenames)}")
    return all_processed_filenames


# --- Function to get the model-specific preprocessing function ---
def get_model_preprocessing_function(model_name):
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
    elif model_name == "MobileNetV3Small": # Added for MobileNetV3Small
        return mobilenetv3_preprocess_input
    elif model_name.startswith("EfficientNet"): 
        return efficientnet_preprocess_input 
    else:
        return None 


# --- CUSTOM HDF5 LOADER FOR tf.data.Dataset ---
class HDF5DataLoaderForTFData:
    """
    A callable class designed to load image/sequence data from HDF5 files
    and apply model-specific preprocessing. It's built for seamless integration
    with `tf.data.Dataset.map()` using `tf.py_function`.
    """
    def __init__(self, expected_seq_len, img_h, img_w, 
                 model_name_for_preprocess, 
                 should_dataset_output_4d):
        """
        Initializes the HDF5 data loader with expected data dimensions and preprocessing details.
        
        Args:
            expected_seq_len (int): The expected number of frames in each sequence as stored in HDF5.
                                    For single-frame data, this should be 1.
            img_h (int): Target image height.
            img_w (int): Target image width.
            model_name_for_preprocess (str): Name of the pre-trained model to fetch the correct preprocessing function.
            should_dataset_output_4d (bool): If True, and `expected_seq_len` is 1, the output tensor
                                                 will be shaped as (H, W, C) (4D). If False, or
                                                 `expected_seq_len` > 1, the output will be (SeqLen, H, W, C) (5D).
                                                 This flag is crucial for matching the input requirements of the target model.
        """
        self.expected_seq_len = expected_seq_len
        self.img_h = img_h
        self.img_w = img_w
        self.model_name_for_preprocess = model_name_for_preprocess 
        self.preprocess_func = get_model_preprocessing_function(model_name_for_preprocess) # Use get_model_preprocessing_function

        if not self.preprocess_func:
            print(f"Warning (HDF5DataLoaderForTFData.init): No specific preprocessing function found for {model_name_for_preprocess}. Data will be loaded as [0, 255] float32.")

    def _load_and_process_single_file(self, filepath_tensor):
        """
        Internal function to load and process a single HDF5 file.
        It decodes the filepath, loads data, converts BGR to RGB, applies conditional
        reshaping based on `should_dataset_output_4d`, and then applies
        model-specific preprocessing. This function runs in Python eager mode.
        
        Args:
            filepath_tensor (tf.Tensor): A TensorFlow string tensor containing the HDF5 file path.
            
        Returns:
            np.ndarray: The processed image data as a NumPy array, ready for conversion to a TensorFlow tensor.
        """
        filepath = filepath_tensor.numpy().decode('utf-8') 
        try:
            with h5py.File(filepath, 'r') as hf:
                # Load data from HDF5 (expected uint8, BGR, [0,255])
                data = hf['sequences'][:] 
                
                # Convert to float32 (still [0, 255] range)
                data = data.astype(np.float32)

                # --- Shape Handling and Validation ---
                if self.expected_seq_len == 1:
                    if data.ndim == 4 and data.shape[0] == 1:
                        data = data[0] # Convert (1, H, W, C) to (H, W, C)
                    expected_shape_before_preprocess = (self.img_h, self.img_w, 3)
                else: # self.expected_seq_len > 1
                    expected_shape_before_preprocess = (self.expected_seq_len, self.img_h, self.img_w, 3)

                if data.shape != expected_shape_before_preprocess:
                    print(f"CRITICAL ERROR (HDF5DataLoaderForTFData._load_and_process_single_file): HDF5 data from {filepath} has unexpected shape {data.shape} before preprocessing. Expected {expected_shape_before_preprocess}")
                    # Raise a ValueError to indicate a problem with the HDF5 data itself
                    raise ValueError(f"Shape mismatch for {filepath}")
                
                # Apply the model-specific preprocessing function
                if self.preprocess_func:
                    processed_data = self.preprocess_func(data)
                else:
                    processed_data = data # If no specific func, pass as is

                return processed_data
        
        except Exception as e:
            # Log the specific error for debugging
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
            output_shape = (self.img_h, self.img_w, 3) if self.expected_seq_len == 1 else \
                           (self.expected_seq_len, self.img_h, self.img_w, 3)
            return np.zeros(output_shape, dtype=np.float32) # Return zeros of float32


    @tf.function # This is the outer tf.function that wraps the py_function
    def load_fn_tf_wrapper(self, filepath_tensor, label_tensor):
        # Call the Python function `_load_and_process_single_file`
        # `Tout` specifies the dtype of the output of `func`
        # `processed_image_tensor` will be a tf.Tensor
        
        # Determine the expected output shape for `set_shape`
        if self.expected_seq_len == 1:
            image_output_shape = tf.TensorShape([self.img_h, self.img_w, 3])
        else:
            image_output_shape = tf.TensorShape([self.expected_seq_len, self.img_h, self.img_w, 3])

        processed_image_tensor = tf.py_function(
            func=self._load_and_process_single_file,
            inp=[filepath_tensor], # Pass the filepath EagerTensor as input
            Tout=tf.float32, # Output type of the image data
            name="load_and_preprocess_image_op" # A descriptive name for the TF operation
        )
        # It's crucial to set the static shape of the output tensor for efficient graph compilation.
        # Without this, the downstream layers might get Unknown (None) shapes.
        processed_image_tensor.set_shape(image_output_shape)
        
        return processed_image_tensor, label_tensor


# --- FUNCTION TO COLLECT HDF5 FILEPATHS ---
def load_labeled_hdf5_sequence_filepaths(data_dir):
    """
    Identifies paths to labeled HDF5 sequence files and their corresponding labels.
    This function is designed to work with the output of `create_labeled_sequences_from_annotations`.
    
    Args:
        data_dir (str): The root directory containing 'train' and 'test' subdirectories,
                        which in turn contain '0' and '1' class label subdirectories with HDF5 files.
        
    Returns:
        tuple: 
            - train_df (pd.DataFrame): DataFrame with 'filepaths' and 'labels' for the training set.
            - test_df (pd.DataFrame): DataFrame with 'filepaths' and 'labels' for the testing set.
            - class_names (list): List of class names (e.g., ['no_collision', 'collision']).
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
                    with h5py.File(actual_h5_path, 'r') as hf:
                        _ = hf['sequences'] 
                    
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


# --- REVISED create_labeled_sequences_from_annotations FUNCTION ---
def create_labeled_sequences_from_annotations(video_dir, annotation_dir, output_dir, 
                                            sequence_length = 10, target_size = (224, 224), 
                                            train_test_split_ratio = 0.8, stride = 1):
    """
    Creates labeled sequences of frames for collision detection.
    Performs a SEQUENCE-LEVEL train/test split. Each individual sequence is saved
    as a separate HDF5 file within train/test/class_label directories.

    Args:
        video_dir (str): Directory containing the .mp4 video files.
        annotation_dir (str): Directory containing the Excel (.xlsx) annotation files.
        output_dir (str): Base directory where 'train' and 'test' folders will be created,
                          each containing HDF5 files per sequence.
        sequence_length (int): Number of consecutive frames to extract and label per sequence.
        target_size (tuple): Target (height, width) for resizing frames.
        train_test_split_ratio (float): Ratio of sequences to assign to the training set.
        stride (int): Step size for sliding window.

    Returns:
        tuple: (train_output_dir, test_output_dir, class_names)
            - train_output_dir: Path to the 'train' directory containing sequence HDF5 files.
            - test_output_dir: Path to the 'test' directory containing sequence HDF5 files.
            - class_names: List of class names (e.g., ['no_collision', 'collision']).
    """
    train_output_dir = os.path.join(output_dir, 'train')
    test_output_dir = os.path.join(output_dir, 'test')
    class_names = ['no_collision', 'collision']

    # --- Prepare output directory structure ---
    # Delete existing directories to ensure a clean slate for the new split logic
    if os.path.exists(output_dir):
        print(f"Clearing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    os.makedirs(os.path.join(train_output_dir, '0'), exist_ok=True) # train/no_collision
    os.makedirs(os.path.join(train_output_dir, '1'), exist_ok=True) # train/collision
    os.makedirs(os.path.join(test_output_dir, '0'), exist_ok=True)  # test/no_collision
    os.makedirs(os.path.join(test_output_dir, '1'), exist_ok=True)   # test/collision
    print(f"Created new output directory structure at {output_dir}")

    # The keys in annotation_files_dict (e.g., '00001') are used to form video filenames
    annotation_files_dict = {f.replace('.xlsx', '').split('-')[-1]: os.path.join(annotation_dir, f) 
                             for f in os.listdir(annotation_dir) if f.endswith('.xlsx')}

    # Collect all sequences and their metadata (original video ID, sequence index, label)
    # We collect metadata to reconstruct the desired filename later.
    all_sequences_metadata = [] 

    print("Extracting all sequences from videos...")
    # Iterate through annotation file keys, as they contain the video IDs with correct padding
    for video_id_from_annotation_key in tqdm(sorted(annotation_files_dict.keys()), desc="Extracting Sequences from Videos"):
        # --- CRITICAL CORRECTION HERE ---
        # video_id_from_annotation_key is like '00001', '00002', etc.
        # We need to format it to match 'collisionXX.mp4'
        # Convert to int, then back to string with 2 digits for collisionXX.mp4
        video_id_for_filename = str(int(video_id_from_annotation_key)).zfill(2) 
        video_file_name = f"collision{video_id_for_filename}.mp4" 
        video_path = os.path.join(video_dir, video_file_name)
        annotation_path = annotation_files_dict.get(video_id_from_annotation_key) # Use the key directly to get path

        if not os.path.exists(video_path) or not annotation_path:
            print(f"Warning: Missing video ({video_path}) or annotation ({annotation_path}) for ID: {video_id_from_annotation_key}. Skipping.")
            continue

        try:
            df = pd.read_excel(annotation_path)
            frame_collision_labels = {}
            for _, row in df.iterrows():
                try:
                    filename = str(row['file'])
                    frame_number = int(filename.split('-frame-')[1].split('.')[0])
                    collision_label = int(row['collision'])
                    frame_collision_labels[frame_number] = collision_label
                except (IndexError, ValueError):
                    print(f"Warning: Could not parse frame or label from '{row.get('file', 'N/A')}' in {annotation_path}. Skipping row.")
        except Exception as e:
            print(f"Error reading annotation file for {video_id_from_annotation_key}: {e}. Skipping video.")
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}. Skipping.")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        sequence_index_in_video = 0 # To generate sequential numbers for HDF5 filenames

        for start_frame_idx in range(0, total_frames - sequence_length + 1, stride):
            frames_in_sequence = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx) # Set video capture position
            
            for _ in range(sequence_length):
                ret, frame = cap.read() # frame is read as BGR by OpenCV
                if not ret:
                    break
                frame_resized = cv2.resize(frame, target_size)
                # Store frames in BGR format and uint8 (CRITICAL for ResNet/VGG preprocess_input)
                frames_in_sequence.append(frame_resized.astype(np.uint8)) 
            
            if len(frames_in_sequence) == sequence_length:
                is_collision_in_sequence = False
                for frame_offset in range(sequence_length):
                    current_frame_number_in_video = start_frame_idx + frame_offset + 1 # 1-based index
                    if current_frame_number_in_video in frame_collision_labels and frame_collision_labels[current_frame_number_in_video] == 1:
                        is_collision_in_sequence = True
                        break
                
                # Store sequence data along with its determined label and original video ID/sequence index
                data_to_save = np.array(frames_in_sequence) 

                # Special handling for sequence_length = 1 to match single-frame expectations if needed
                if sequence_length == 1:
                    # If it's a single frame, ensure its shape is (H, W, C) not (1, H, W, C)
                    if data_to_save.ndim == 4 and data_to_save.shape[0] == 1:
                        data_to_save = data_to_save[0]
                    elif data_to_save.ndim != 3: # If it's not (H,W,C) after attempting to strip 1st dim
                        print(f"Warning: Single-frame data for {video_id_from_annotation_key}-seq-{sequence_index_in_video} has unexpected shape {data_to_save.shape} for seq_len=1. Expected (1,H,W,C) or (H,W,C). Saving as is.")


                all_sequences_metadata.append({
                    'video_id': video_id_from_annotation_key, # Use the video_id_from_annotation_key directly for consistent naming
                    'sequence_idx_in_video': sequence_index_in_video,
                    'data': data_to_save,
                    'label': 1 if is_collision_in_sequence else 0
                })
                sequence_index_in_video += 1 # Increment sequence counter for unique naming
        
        cap.release()

    if not all_sequences_metadata:
        print("No sequences extracted from any video.")
        return train_output_dir, test_output_dir, class_names # Return empty directories

    # --- Perform SEQUENCE-LEVEL Train/Test Split ---
    print("\nPerforming sequence-level train/test split...")
    # Extract data and labels for splitting
    # We only need the labels for stratification, not the actual sequence data itself for the split
    sequences_labels = [item['label'] for item in all_sequences_metadata]
    
    # Split indices to keep track of original metadata
    train_indices, test_indices, _, _ = train_test_split(
        range(len(all_sequences_metadata)), 
        sequences_labels, # Stratify based on labels
        test_size=(1 - train_test_split_ratio), 
        stratify=sequences_labels, # Ensure stratified split
        random_state=42
    )

    # --- Save split sequences to individual HDF5 files ---
    print("Saving split sequences to individual HDF5 files...")
    for idx_type, indices_list in {'train': train_indices, 'test': test_indices}.items():
        for original_idx in tqdm(indices_list, desc=f"Saving {idx_type} sequences"):
            item = all_sequences_metadata[original_idx]
            
            # Construct output path (e.g., output_dir/train/1/video-00002-seq-00005.hdf5)
            target_label_dir = os.path.join(output_dir, idx_type, str(item['label']))
            
            # Generate filename: video-XXXXX-seq-YYYYY.hdf5
            # YYYYY is the sequence's original index within its video
            filename = f"video-{item['video_id']}-seq-{str(item['sequence_idx_in_video']).zfill(5)}.hdf5"
            h5_path = os.path.join(target_label_dir, filename)

            try:
                with h5py.File(h5_path, 'w') as hf:
                    # Sequences are stored as (sequence_length, img_height, img_width, 3)
                    # For sequence_length=1, it will be (H,W,C) after the data_to_save conversion
                    hf.create_dataset('sequences', data=item['data'],
                                      compression='gzip', compression_opts=4)
                    hf.create_dataset('label', data=np.array(item['label'], dtype=np.int32), # Save single label
                                      compression='gzip', compression_opts=4)
            except Exception as e:
                print(f"Error saving {h5_path}: {e}")

    print("\nSequence generation and saving complete.")
    return train_output_dir, test_output_dir, class_names
