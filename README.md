# Visual Collision Avoidance for Autonomous Systems

This repository contains code for a research project on visual collision avoidance for autonomous platforms, such as Unmanned Aerial Vehicles (UAVs) and robots, conducted at Universidad Carlos III de Madrid. It uses deep learning, specifically a fine-tuned VGG16 model, to detect potential collisions from camera feeds and enable safe navigation in real-time.

The project is inspired by the [uav-collision-avoidance](https://github.com/dario-pedro/uav-collision-avoidance) repository by dario-pedro, incorporating similar dataset structures and evaluation methodologies.

## Project Overview

Autonomous navigation in dynamic environments requires robust collision avoidance to ensure safety. Traditional sensors like LiDAR are costly and heavy, making camera-based solutions an attractive alternative. This project leverages visual data processed by a VGG16-based convolutional neural network to identify obstacles and assess collision risks, supporting safe navigation. The codebase processes video and Excel-based datasets, trains and fine-tunes the VGG16 model, and evaluates performance through metrics and visualizations.

## Key Features

### Implemented
- **Dataset Processing**: Extracts and preprocesses frames from video files (`videos/`) and Excel annotations (`dataframes/`), saving them to `image_data/train/` and `image_data/test/`.
- **VGG16 Model Training**: Fine-tunes VGG16 for binary classification (collision vs. no collision) using TensorFlow and Keras.
- **Model Evaluation**: Computes accuracy, plots training history, and visualizes confusion matrices and misclassified images.
- **Data Augmentation**: Applies image preprocessing with augmentation (e.g., shear, zoom) for robust training.

### Planned
- **Real-Time Object Detection**: Develop algorithms for detecting and tracking obstacles in live camera feeds.
- **Collision Risk Prediction**: Create models to estimate obstacle distance, velocity, and collision likelihood.
- **Path Planning**: Integrate collision avoidance with trajectory planning for autonomous navigation.