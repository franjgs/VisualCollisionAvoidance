#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 17:43:35 2025

@author: fran
"""
import os

from utils.data_processing import (
    create_labeled_sequences_from_annotations
)

DATASET_TYPE = 'drones' # 'cars' # 

sequence_length = 2
stride = 1
target_size = (224, 224)
train_test_ratio = 0.8
output_directory = os.path.join(DATASET_TYPE, f'labeled_sequences_len_{sequence_length}_stride_{stride}')

train_dir, test_dir, class_names = create_labeled_sequences_from_annotations(
    dataset_type = DATASET_TYPE,
    base_dataset_dir = DATASET_TYPE, # e.g., 'drones', 'cars'
    output_dir = output_directory, 
    sequence_length = sequence_length, 
    target_size = target_size, 
    train_test_split_ratio = train_test_ratio, 
    stride = stride,
    num_collision_videos_to_process = 25, # For 'cars' dataset to limit processing
    num_no_collision_videos_to_process = 15 # For 'cars' dataset to limit processing
) 

print(f"Training sequences saved to: {train_dir}")
print(f"Testing sequences saved to: {test_dir}")
print(f"Class names: {class_names}")