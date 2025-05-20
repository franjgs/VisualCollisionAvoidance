#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 17:43:35 2025

@author: fran
"""

from utils.data_processing import (
    create_labeled_sequences_from_annotations
)

video_directory = 'videos'  # Replace with your video directory
annotation_directory = 'dataframes' # Replace with your dataframe directory
output_directory = 'labeled_sequences'
sequence_length = 15
target_size = (224, 224)
train_test_ratio = 0.8
collision_temporal_window = 5

train_dir, test_dir, class_names = create_labeled_sequences_from_annotations(
    video_directory,
    annotation_directory,
    output_directory,
    sequence_length=sequence_length,
    target_size=target_size,
    train_test_split_ratio=train_test_ratio,
    collision_window=collision_temporal_window
)

print(f"Training sequences saved to: {train_dir}")
print(f"Testing sequences saved to: {test_dir}")
print(f"Class names: {class_names}")