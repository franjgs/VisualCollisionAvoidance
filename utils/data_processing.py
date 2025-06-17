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

def generate_paired_file_lists(range_min: int = 70, range_max: int = 93) -> tuple[list[str], list[str]]:
    """
    Generates lists of paired video and Excel annotation file paths based on a numerical range.
    Only includes pairs where both the video and Excel files exist.
    
    Args:
        range_min (int): The starting video ID (inclusive).
        range_max (int): The ending video ID (inclusive).
        
    Returns:
        tuple[list[str], list[str]]: Two lists, one for video file paths and one for Excel file paths.
    """
    video_dir = 'videos'
    excel_dir = 'dataframes'
    
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

def create_labeled_sequences_from_annotations(video_dir: str, annotation_dir: str, output_dir: str, 
                                            sequence_length: int = 10, target_size: tuple[int, int] = (224, 224), 
                                            train_test_split_ratio: float = 0.8, stride: int = 1):
    """
    Processes video files and their corresponding annotations to create labeled sequences of frames.
    Each extracted sequence is saved as an individual HDF5 file. The sequences are then
    split into training and testing sets, ensuring stratification by collision label.
    
    Args:
        video_dir (str): Directory containing the .mp4 video files (e.g., 'videos/').
        annotation_dir (str): Directory containing the Excel (.xlsx) annotation files (e.g., 'dataframes/').
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
                      
    Returns:
        tuple[str, str, list[str]]: 
            - train_output_dir (str): Path to the 'train' directory containing sequence HDF5 files.
            - test_output_dir (str): Path to the 'test' directory containing sequence HDF5 files.
            - class_names (list[str]): List of class names (e.g., ['no_collision', 'collision']).
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
        # Convert the video_id from annotation key (e.g., '00001') to an integer,
        # then format it back to a two-digit string (e.g., '01').
        video_id_for_filename = str(int(video_id_from_annotation_key)).zfill(2)
        video_file_name = f"collision{video_id_for_filename}.mp4" 
        video_path = os.path.join(video_dir, video_file_name)
        annotation_path = annotation_files_dict.get(video_id_from_annotation_key) # Use the original key for lookup

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
                
                # --- CRITICAL CHANGE HERE: Handle single frame data saving ---
                # If sequence_length is 1, save as (H, W, C) directly for ImageDataGenerator.
                # Otherwise, save as (seq_len, H, W, C) for multi-frame.
                data_to_save = np.array(frames_in_sequence) # This is (seq_len, H, W, C)

                if sequence_length == 1:
                    # Remove the sequence dimension if it's 1 for ImageDataGenerator compatibility
                    # E.g., (1, 224, 224, 3) becomes (224, 224, 3)
                    if data_to_save.ndim == 4 and data_to_save.shape[0] == 1:
                        data_to_save = data_to_save[0] # Take the single frame out of the sequence dimension
                    elif data_to_save.ndim != 3: # If not (1,H,W,C) nor (H,W,C) after processing 1 frame
                        print(f"Warning: Single-frame data for {video_id_from_annotation_key}-seq-{sequence_index_in_video} has unexpected shape {data_to_save.shape} for seq_len=1. Expected (1,H,W,C) or (H,W,C). Saving as is.")
                # --- END CRITICAL CHANGE ---

                # Store sequence data along with its determined label and original video ID/sequence index
                all_sequences_metadata.append({
                    'video_id': video_id_from_annotation_key, # Use the video_id_base directly for consistent naming
                    'sequence_idx_in_video': sequence_index_in_video,
                    'data': data_to_save, # Use the potentially adjusted data
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
    sequences_data = [item['data'] for item in all_sequences_metadata]
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
                    # Save 'sequences' dataset with the adjusted shape (H,W,C) or (seq_len, H,W,C)
                    # No maxshape, compression is fine for the dataset itself
                    hf.create_dataset('sequences', data=item['data'],
                                      compression='gzip', compression_opts=4)
                    hf.create_dataset('label', data=item['label']) # Corrected: no chunk/filter for scalar
            except Exception as e:
                print(f"Error saving {h5_path}: {e}")

    print("\nSequence generation and saving complete.")
    return train_output_dir, test_output_dir, class_names