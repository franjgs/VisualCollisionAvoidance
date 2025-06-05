#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 15:48:03 2025

@author: fran
"""

# data_processing.py
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import h5py

def open_video(video):
    """Opens a video file for processing."""
    return cv2.VideoCapture(video)

def check_video(cap, video_path):
    """Verifies if a video can be opened and read."""
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
    """Releases a video capture object."""
    cap.release()

def getFrameRate(video):
    """Retrieves the frame rate of a video."""
    return video.get(cv2.CAP_PROP_FPS)

def generate_framename(video_num, pos_frame):
    """Generates a frame name from video number and frame position."""
    return f"video-{str(video_num).zfill(5)}-frame-{str(pos_frame).zfill(5)}"

def generate_video_num(out_videoname):
    """Extracts the video number from a video name."""
    return int(out_videoname.split('-')[1])

def create_image_directories(output_base_dir):
    train_dir = os.path.join(output_base_dir, 'train')
    test_dir = os.path.join(output_base_dir, 'test')
    
    for dir_path in [train_dir, test_dir]:
        for label in ['0', '1']:
            os.makedirs(os.path.join(dir_path, label), exist_ok=True)
    
    return train_dir, test_dir

def generate_paired_file_lists(range_min=70, range_max=93):
    video_dir = 'videos'
    excel_dir = 'dataframes'
    
    video_files = [os.path.join(video_dir, f'collision{i:02d}.mp4') for i in range(range_min, range_max + 1)]
    excel_files = [os.path.join(excel_dir, f'video-{(i):05d}.xlsx') for i in range(range_min, range_max + 1)]
    
    paired_files = [(v, e) for v, e in zip(video_files, excel_files) if os.path.exists(v) and os.path.exists(e)]
    video_files, excel_files = zip(*paired_files) if paired_files else ([], [])
    
    return list(video_files), list(excel_files)

def generate_out_videoname(video_base):
    """Generate output video name from base name, matching Excel format."""
    # Extract video number (e.g., '02' from 'collision02') and format as five digits
    video_num = video_base.replace('collision', '')
    return f"video-{int(video_num):05d}"

def process_and_save_frames(excel_files, video_files, output_dir, target_size=(224, 224), train_ratio=0.8):
    """Processes video frames, saves them to directories, and returns filenames."""
    create_image_directories(output_dir)
    
    filenames = []
    
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
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                split = 'train' if np.random.rand() < train_ratio else 'test'
                label_dir = os.path.join(output_dir, split, str(label))
                save_path = os.path.join(label_dir, f"{frame_name}.png")
                
                # Skip if file exists
                if os.path.exists(save_path):
                    filenames.append(save_path)
                    frame_count += 1
                    continue
                
                if cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                    filenames.append(save_path)
                else:
                    print(f"Failed to save: {save_path}")
            
            frame_count += 1
        
        cap.release()

    return filenames

# --- Load and Preprocess Data ---
def create_labeled_sequences_from_annotations(video_dir, annotation_dir, output_dir, sequence_length=10, frames_per_sequence=10, target_size=(224, 224), train_test_split_ratio=0.8, collision_window=5):
    """
    Creates labeled sequences of frames for collision detection based on annotations
    in Excel files.

    Args:
        video_dir (str): Directory containing the .mp4 video files.
        annotation_dir (str): Directory containing the Excel (.xlsx) annotation files.
        output_dir (str): Directory to save the train and test labeled frame sequences.
        sequence_length (int): Number of consecutive frames to extract for each sequence.
        frames_per_sequence (int): The fixed length of each sequence.
        target_size (tuple): Target (height, width) for resizing frames.
        train_test_split_ratio (float): Ratio of data to use for training.
        collision_window (int): Number of frames before and after a collision frame
                                to consider the sequence as a collision.

    Returns:
        tuple: (train_dir, test_dir, class_names)
    """
    os.makedirs(output_dir, exist_ok=True)
    train_output_dir = os.path.join(output_dir, 'train')
    test_output_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    annotation_files = {f.replace('.xlsx', '').split('-')[1]: os.path.join(annotation_dir, f) for f in os.listdir(annotation_dir) if f.endswith('.xlsx')}
    class_names = ['no_collision', 'collision']

    video_ids = [f.replace('collision', '').replace('.mp4', '') for f in video_files]
    train_ids, test_ids = train_test_split(video_ids, test_size=(1 - train_test_split_ratio), random_state=42)

    for video_id_base in tqdm(video_ids, desc="Processing Videos"):
        # Determine the appropriate padding length based on your annotation keys
        padding_length = len(list(annotation_files.keys())[0]) if annotation_files else 1  # Default to 1 if no annotations

        # Format video_id_base with leading zeros
        formatted_video_id = video_id_base.zfill(padding_length)

        video_file = f"collision{video_id_base}.mp4" # Keep the original ID for filename
        video_path = os.path.join(video_dir, video_file)
        annotation_path = annotation_files.get(formatted_video_id) # Use formatted ID for lookup

        if not os.path.exists(video_path) or not annotation_path:
            print(f"Warning: Could not find video or annotation for ID: {formatted_video_id} (original: {video_id_base})")
            continue

        try:
            df = pd.read_excel(annotation_path)
            # Create a dictionary mapping frame number to collision label
            frame_collision_labels = {}
            for index, row in df.iterrows():
                try:
                    filename = row['file']
                    frame_part = filename.split('-frame-')[1]
                    frame_number = int(frame_part.split('.')[0])
                    collision_label = int(row['collision'])
                    frame_collision_labels[frame_number] = collision_label
                except IndexError:
                    print(f"Warning: Could not extract frame number from filename: {filename} in {annotation_path}")
                except ValueError:
                    print(f"Warning: Invalid frame or collision label format in {annotation_path}")
        except Exception as e:
            print(f"Error reading annotation file for {video_id_base}: {e}")
            continue

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        sequences = []
        labels = []

        for start_frame in range(total_frames - frames_per_sequence + 1):
            frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for i in range(frames_per_sequence):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_resized = cv2.resize(frame, target_size)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb / 255.0)
            if len(frames) == frames_per_sequence:
                sequences.append(np.array(frames))

                # Determine label for the sequence based on the 'collision' column
                is_collision = False
                for frame_offset in range(frames_per_sequence):
                    current_frame_number = start_frame + frame_offset + 1 # Frame numbers are 1-based
                    if current_frame_number in frame_collision_labels and frame_collision_labels[current_frame_number] == 1:
                        is_collision = True
                        break
                labels.append(1 if is_collision else 0)

        cap.release()

        # Save sequences and labels to an HDF5 file per video
        output_base = os.path.join(train_output_dir if video_id_base in train_ids else test_output_dir, f"video_{video_id_base}")
        os.makedirs(output_base, exist_ok=True)
        h5_path = os.path.join(output_base, "data.hdf5")

        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset('sequences', data=np.array(sequences), compression='gzip', compression_opts=4)
            hf.create_dataset('labels', data=np.array(labels), dtype=np.int32, compression='gzip', compression_opts=4)
            
    return train_output_dir, test_output_dir, class_names