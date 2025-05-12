# Visual Collision Avoidance for Autonomous Systems

This repository contains code for developing visual collision avoidance systems for autonomous platforms, such as Unmanned Aerial Vehicles (UAVs) and robots. It uses deep learning, specifically a fine-tuned VGG16 model, to detect potential collisions from camera feeds and enable safe navigation in real-time.

The project is inspired by and may incorporate elements from the [uav-collision-avoidance](https://github.com/dario-pedro/uav-collision-avoidance) repository by dario-pedro, particularly in dataset structures and evaluation methodologies.

## Project Overview

Autonomous navigation in dynamic environments demands robust collision avoidance to ensure safety. Traditional sensors like LiDAR can be costly and heavy, making camera-based solutions an attractive alternative. This project leverages visual data processed by a VGG16-based convolutional neural network to identify obstacles, assess collision risks, and support avoidance maneuvers.

The codebase includes scripts for processing video and Excel-based datasets, training and fine-tuning the VGG16 model, and evaluating performance through metrics and visualizations like confusion matrices and misclassification plots.

## Key Features

### Implemented
- **Dataset Processing**: Scripts to extract and preprocess frames from video files and corresponding Excel annotations, saving them into train/test directories (`image_data/`).
- **VGG16 Model Training**: Fine-tuned VGG16 model for binary classification (collision vs. no collision) using TensorFlow and Keras.
- **Model Evaluation**: Tools to compute accuracy, plot training history, and visualize misclassified images and confusion matrices.
- **Data Augmentation**: Image preprocessing with augmentation techniques (e.g., shear, zoom) for robust model training.

### Planned
- **Real-Time Object Detection**: Algorithms for detecting and tracking obstacles in live camera feeds.
- **Collision Risk Prediction**: Models to estimate obstacle distance, velocity, and collision likelihood.
- **Path Planning**: Integration of collision avoidance with trajectory planning for autonomous navigation.
- **Simulation Testing**: Support for environments like Gazebo or AirSim to validate the system.
- **Hardware Integration**: Guidelines for deploying the model on UAVs or robotic platforms.

## Repository Structure

- `dataframes/`: Excel files with frame annotations (e.g., collision labels).
- `videos/`: Video files containing raw footage for frame extraction.
- `image_data/`: Processed frames organized into `train/` and `test/` directories.
- `models/`: Saved models, weights, and training history files.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/franjgs/VisualCollisionAvoidance.git
   cd VisualCollisionAvoidance
   ```

2. **Install Dependencies**:
   Create a virtual environment and install required packages:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn pillow
   ```

3. **Prepare the Dataset**:
   - Place video files (e.g., `collision01.mp4`) in `videos/`.
   - Place corresponding Excel files (e.g., `video-00001.xlsx`) in `dataframes/`.
   - Ensure file naming follows the pattern: `collisionXX.mp4` and `video-XXXXX.xlsx`, where `XX` and `XXXXX` align (e.g., `collision01.mp4` pairs with `video-00001.xlsx`).

## Usage

1. **Process Videos and Annotations**:
   The `VGG16_CA.py` script processes videos and Excel files, extracts frames, and saves them to `image_data/train/` and `image_data/test/`:
   ```bash
   python Code_develop/VGG16_CA.py
   ```
   - Ensure `videos/` and `dataframes/` contain the required files.
   - The script uses frames 70 to 93 by default (adjust `range_min` and `range_max` in the script if needed).

2. **Train the Model**:
   The same script trains and fine-tunes the VGG16 model:
   - Outputs: Trained model (`models/VGG-collision-avoidance-<timestamp>.keras`), weights, and training history (`trainHistoryDict_fine.pkl`).
   - Training includes 20 epochs of initial training and 20 epochs of fine-tuning.

3. **Evaluate Results**:
   The script evaluates the model on the test set and generates:
   - Accuracy metrics.
   - Plots of training/validation accuracy and loss (`VGG-collision-avoidance-<timestamp>.pdf`).
   - Confusion matrix and misclassified image visualizations.

4. **Customize Parameters** (Optional):
   Edit `Code_develop/VGG16_CA.py` to adjust:
   - `range_min` and `range_max` for video file range.
   - `target_size` for frame resolution (default: 224x224).
   - `train_ratio` for train/test split (default: 0.8).
   - `epochs` or learning rates for training.

## Dependencies

tensorflow==2.19.0

opencv-python==4.11.0.86

numpy==1.26.4

scipy==1.15.2

matplotlib==3.10.1

pyyaml==6.0

torch==2.5.1

torchvision==0.20.1

**Libraries for a specific simulation environment:**

airsim==1.9.0


**Robotics-related libraries:**

rospy==1.16.0

message-generation==0.11.14

## Datasets

* Information about the datasets used in this project will be provided here, potentially referencing publicly available datasets or custom datasets generated for this research.
* Consider exploring datasets related to UAV collision avoidance, such as those potentially utilized or referenced in the [uav-collision-avoidance](https://github.com/dario-pedro/uav-collision-avoidance) repository.

## Contributing

* Contributions are welcome! If you have ideas, improvements, or bug fixes, please feel free to submit pull requests or open issues. Please follow the contribution guidelines (if any) outlined in the repository.

## License

* [Specify the license under which this project is released]

## Acknowledgements

* This project draws inspiration and potentially utilizes concepts and data structures from the [uav-collision-avoidance](https://github.com/dario-pedro/uav-collision-avoidance) repository by dario-pedro. We acknowledge their valuable work in the field.
* [Mention any other individuals, institutions, or resources that contributed to this project.]
