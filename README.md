# Visual Collision Avoidance for Autonomous Systems

This repository contains code for a research project on visual collision avoidance for autonomous platforms, such as Unmanned Aerial Vehicles (UAVs) and robots. It uses deep learning, specifically a fine-tuned VGG16 model, to detect potential collisions from camera feeds and enable safe navigation in real-time.

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
- **Simulation Testing**: Support validation in environments like Gazebo or AirSim.
- **Hardware Integration**: Provide guidelines for deploying the model on UAVs or robotic platforms.

## Repository Structure

- `VGG16_CA.py`: Main script for dataset processing, model training, and evaluation.
- `dataframes/`: Excel files (e.g., `video-00001.xlsx`) with frame-level collision annotations.
- `videos/`: Video files (e.g., `collision01.mp4`) for frame extraction.
- `image_data/`: Processed frames in `train/` and `test/` directories, organized by class (`0` for no collision, `1` for collision).
- `models/`: Saved models, training history, and plots (automatically created if missing).
- `requirements.txt`: Lists required Python packages with versions.
- `.gitignore`: Excludes temporary files, macOS metadata (e.g., `.DS_Store`), and large directories.
- `LICENSE`: MIT License with citation requirement.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/franjgs/VisualCollisionAvoidance.git
   cd VisualCollisionAvoidance
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Install the required packages with specific versions for reproducibility:
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, install manually:
   ```bash
   pip install tensorflow==2.12.0 opencv-python==4.11.0 numpy==1.24.3 pandas==2.0.3 matplotlib==3.7.2 scikit-learn==1.3.0 pillow==10.0.0
   ```

4. **Prepare the Dataset**:
   - Place video files (e.g., `collision01.mp4`) in `videos/`.
   - Place corresponding Excel files (e.g., `video-00001.xlsx`) in `dataframes/`.
   - Ensure filenames align: `collisionXX.mp4` pairs with `video-XXXXX.xlsx` (e.g., `collision01.mp4` with `video-00001.xlsx`).

## Usage

1. **Run the Main Script**:
   Process videos, train, and evaluate the VGG16 model:
   ```bash
   python VGG16_CA.py
   ```
   - **Inputs**: Videos in `videos/` and Excel files in `dataframes/`.
   - **Outputs**:
     - Frames in `image_data/train/` and `image_data/test/`.
     - Model in `models/` (e.g., `VGG-collision-avoidance-<timestamp>.keras`).
     - Training history in `models/` (e.g., `trainHistoryDict_fine_<timestamp>.pkl`).
     - Accuracy/loss plots and confusion matrix in `models/` (e.g., `VGG-collision-avoidance-<timestamp>.pdf`).

2. **Customize Parameters**:
   Edit `VGG16_CA.py` to adjust:
   - `range_min`, `range_max`: Video file range (default: 70 to 93).
   - `target_size`: Frame resolution (default: 224x224).
   - `train_ratio`: Train/test split (default: 0.8).
   - `epochs`: Training duration (default: 20 initial + 20 fine-tuning).
   - `batch_size`: Batch size (default: 20).
   - `output_dir`: Output directory for models and plots (default: `models`).

3. **View Results**:
   - Check `models/` for saved models, history, and plots.
   - Open `models/VGG-collision-avoidance-<timestamp>.pdf` for training and evaluation visuals.

## Requirements

- **Python**: 3.8 or higher
- **Libraries**:
  - `tensorflow==2.12.0`
  - `opencv-python==4.11.0`
  - `numpy==1.24.3`
  - `pandas==2.0.3`
  - `matplotlib==3.7.2`
  - `scikit-learn==1.3.0`
  - `pillow==10.0.0`
- **Hardware**: GPU (optional, for faster training; CPU fallback supported)

## Datasets

This project uses custom datasets developed for research at Universidad Carlos III de Madrid, consisting of video files and corresponding Excel annotations:
- **Videos**: Stored in `videos/` (e.g., `collision01.mp4`), containing footage for collision avoidance scenarios.
- **Annotations**: Stored in `dataframes/` (e.g., `video-00001.xlsx`), providing frame-level collision labels.

Details about the dataset (e.g., source, size, or public availability) will be added as the research progresses. For related work, consider exploring datasets used in the [uav-collision-avoidance](https://github.com/dario-pedro/uav-collision-avoidance) repository, which may include publicly available UAV collision avoidance data.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please adhere to PEP 8 guidelines, include tests where applicable, and cite the author (Francisco J. González) if using this code, as per the License terms.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. If you use this code in your work, please cite the author, Francisco J. Gonzalez, and acknowledge the research conducted at Universidad Carlos III de Madrid.

## Citation

If you use this code in your research or projects, please cite:

<<<<<<< HEAD
**Francisco J. Gonzalez, Universidad Carlos III de Madrid**

Example citation format:
```
Gonzalez-Serrano, Francisco J. (2025). Visual Collision Avoidance for Autonomous Systems. Universidad Carlos III de Madrid.
```
=======
>>>>>>> c0290dd7f169043f781d9ed5110189e292ce0449

## Acknowledgments

- Inspired by the [uav-collision-avoidance](https://github.com/dario-pedro/uav-collision-avoidance) repository by dario-pedro.
- Built with open-source libraries: TensorFlow, OpenCV, NumPy, Pandas, Matplotlib, scikit-learn, and Pillow.
