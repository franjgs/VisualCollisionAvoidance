# Visual Collision Avoidance for Autonomous Systems

This repository contains code for a research project on visual collision avoidance for autonomous platforms, such as Unmanned Aerial Vehicles (UAVs) and robots. It uses deep learning to detect potential collisions from camera feeds and enable safe navigation in real-time. The approach includes both single-frame and multi-frame analysis.

The project is inspired by the [uav-collision-avoidance](https://github.com/dario-pedro/uav-collision-avoidance) repository by dario-pedro, incorporating similar dataset structures and evaluation methodologies.

## Project Overview

Autonomous navigation in dynamic environments requires robust collision avoidance to ensure safety. Traditional sensors like LiDAR are costly and heavy, making camera-based solutions an attractive alternative. This project leverages visual data processed by deep learning models to identify obstacles and assess collision risks, supporting safe navigation. The codebase processes video and Excel-based datasets, trains and evaluates models, and provides tools for data preparation.

## Key Features

### Implemented

-   **Dataset Processing**:
    -   Extracts and preprocesses frames from video files (`videos/`) and Excel annotations (`dataframes/`).
    -   Generates multi-frame sequences from processed frames using `GenerateMultiFrameData.py`.
-   **Model Training**:
    -   Supports training of single-frame (`SingleFrameCA_DNN.py`) and multi-frame (`MultiFrameCA_DNN.py`) collision avoidance models.
    -   Uses TensorFlow and Keras.
-   **Model Evaluation**: Computes accuracy and plots training history.
-   **Data Augmentation**: Applies image preprocessing with augmentation (e.g., shear, zoom) for robust training.

### Planned

-   **Real-Time Object Detection**: Develop algorithms for detecting and tracking obstacles in live camera feeds.
-   **Collision Risk Prediction**: Create models to estimate obstacle distance, velocity, and collision likelihood.
-   **Path Planning**: Integrate collision avoidance with trajectory planning for autonomous navigation.
-   **Simulation Testing**: Support validation in environments like Gazebo or AirSim.
-   **Hardware Integration**: Provide guidelines for deploying the model on UAVs or robotic platforms.

## Repository Structure

The project is structured to handle multiple datasets and differentiate between single-frame and multi-frame processing outputs.

-   `SingleFrameCA_DNN.py`: Main script for training and evaluating a single-frame collision avoidance Deep Neural Network. This script is configurable to work with either the "Drones" or "Cars" dataset.
-   `MultiFrameCA_DNN.py`: Main script for training and evaluating a multi-frame collision avoidance Deep Neural Network (CNN-LSTM). This script is also configurable for different datasets.
-   `GenerateMultiFrameData.py`: Dedicated script for generating multi-frame sequences in HDF5 format from raw video and annotation data, specifically for multi-frame model training.

-   `utils/`: Contains essential utility scripts:
    -   `data_processing.py`: Core functions for video processing, including frame/sequence extraction, image preprocessing, and efficient data loading for TensorFlow datasets from various annotation formats (Excel, CSV).
    -   `plotting_utils.py`: Provides helper functions for visualizing model training history (e.g., loss/accuracy curves) and evaluating performance metrics (e.g., confusion matrices).

-   `cars/`: **Root directory for the "Cars" ([Kaggle Nexar Collision Prediction](https://www.kaggle.com/competitions/nexar-collision-prediction/data)) dataset.**
    -   `videos/`: Contains the raw video files for the 'cars' dataset, typically organized into `train/` and `test/` subfolders with `.mp4` files (e.g., `00058.mp4`).
    -   `data_labels.csv`: The primary annotation file for the 'cars' dataset, specifying event and alert times for collisions.
    -   `image_data_cars/`: **(Generated Output)** Directory containing processed single-frame images derived from the 'cars' dataset. This structure is created by `SingleFrameCA_DNN.py` when `DATASET_TO_USE` is set to "cars".
    -   `models/`: **(Generated Output)** Directory where trained model files (`.keras` format) specific to the 'cars' dataset are saved.
    -   `results/`: **(Generated Output)** Directory for evaluation plots (e.g., training history, confusion matrices) generated for the 'cars' dataset.

-   `drones/`: **Root directory for the [Drones](https://github.com/dario-pedro/uav-collision-avoidance)dataset.**
    -   `videos/`: Contains the raw video files for the 'drones' dataset (e.g., `collision01.mp4`).
    -   `dataframes/`: Contains Excel annotation files (`.xlsx`) for the 'drones' dataset (e.g., `video-00001.xlsx`), providing frame-level collision annotations.
    -   `image_data_drones/`: **(Generated Output)** Directory containing processed single-frame images derived from the 'drones' dataset. This structure is created by `SingleFrameCA_DNN.py` when `DATASET_TO_USE` is set to "drones".
    -   `models/`: **(Generated Output)** Directory where trained model files (`.keras` format) specific to the 'drones' dataset are saved.
    -   `results/`: **(Generated Output)** Directory for evaluation plots (e.g., training history, confusion matrices) generated for the 'drones' dataset.

-   `requirements.txt`: Lists all required Python packages and their versions to set up the development environment.
-   `.gitignore`: Specifies intentionally untracked files that Git should ignore (e.g., temporary files, large generated data, system files).
-   `LICENSE`: Details the licensing terms for the project, which is the MIT License with a citation requirement.

## Installation

1.  **Clone the Repository**:

    ```bash
    git clone [https://github.com/franjgs/VisualCollisionAvoidance.git](https://github.com/franjgs/VisualCollisionAvoidance.git)
    cd VisualCollisionAvoidance
    ```

2.  **Set Up a Virtual Environment**:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:

    Install the required packages with specific versions for reproducibility:

    ```bash
    pip install -r requirements.txt
    ```

    Alternatively, install manually:

    ```bash
    pip install tensorflow==2.12.0 opencv-python==4.11.0 numpy==1.24.3 pandas==2.0.3 matplotlib==3.7.2 scikit-learn==1.3.0 pillow==10.0.0 h5py
    ```

4.  **Prepare the Dataset**:

    -   Place video files (e.g., `collision01.mp4`) in `videos/`.
    -   Place corresponding Excel files (e.g., `video-00001.xlsx`) in `dataframes/`.
    -   Ensure filenames align: `collisionXX.mp4` pairs with `video-XXXXX.xlsx` (e.g., `collision01.mp4` with `video-00001.xlsx`).
    -   If using `MultiFrameCA_DNN.py`, run `GenerateMultiFrameData.py` to create the required HDF5 files in `labeled_sequences/`.

## Usage

1.  **Generate Multi-Frame Data (for MultiFrameCA_DNN.py):**

    If you intend to use the multi-frame model (`MultiFrameCA_DNN.py`), you first need to generate the multi-frame sequences from your video and annotation data.

    ```bash
    python GenerateMultiFrameData.py --video_dir <path/to/videos> --annotation_dir <path/to/dataframes> --output_dir <path/to/output_dir>
    ```

    This will create the `labeled_sequences/` directory containing the processed data in HDF5 format.

2.  **Run the Main Scripts**:

    Run either `SingleFrameCA_DNN.py` or `MultiFrameCA_DNN.py` to train and evaluate the respective model.

    ```bash
    python SingleFrameCA_DNN.py
    ```

    or

    ```bash
    python MultiFrameCA_DNN.py
    ```

    The scripts will:

    -   Load the preprocessed data.
    -   Build the model.
    -   Train the model.
    -   Evaluate the model on the test set.
    -   Save the trained model and training history.

## Output

The scripts will save:

-   Trained model files (.keras).
-   Training history plots (accuracy and loss) as PDF files.
-   Pickled files containing the class names.

## Requirements

-   **Python**: 3.8 or higher
-   **Libraries**:
    -   `tensorflow==2.12.0`
    -   `opencv-python==4.11.0`
    -   `numpy==1.24.3`
    -   `pandas==2.0.3`
    -   `matplotlib==3.7.2`
    -   `scikit-learn==1.3.0`
    -   `pillow==10.0.0`
    -   `h5py`
-   **Hardware**: GPU (optional, for faster training; CPU fallback supported)

## Datasets

This project uses custom datasets developed for research at Universidad Carlos III de Madrid, consisting of video files and corresponding Excel annotations:

-   **Videos**: Stored in `videos/` (e.g., `collision01.mp4`), containing footage for collision avoidance scenarios.
-   **Annotations**: Stored in `dataframes/` (e.g., `video-00001.xlsx`), providing frame-level collision labels.

Details about the dataset (e.g., source, size, or public availability) will be added as the research progresses.

## Contributing

Contributions are welcome! To contribute:

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/your-feature`).
3.  Commit changes (`git commit -m "Add your feature"`).
4.  Push to the branch (`git push origin feature/your-feature`).
5.  Open a pull request.

Please adhere to PEP 8 guidelines, include tests where applicable, and cite the author (Francisco J. González) if using this code, as per the License terms.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. If you use this code in your work, please cite the author, Francisco J. Gonzalez, and acknowledge the research conducted at Universidad Carlos III de Madrid.

## Citation

If you use this code in your research or projects, please cite:

**Francisco J. Gonzalez, Universidad Carlos III de Madrid**

Example citation format:
```
Gonzalez-Serrano, Francisco J. (2025). Visual Collision Avoidance for Autonomous Systems. Universidad Carlos III de Madrid.
```

## Acknowledgments

- Inspired by the [uav-collision-avoidance](https://github.com/dario-pedro/uav-collision-avoidance) repository by dario-pedro.
- Built with open-source libraries: TensorFlow, OpenCV, NumPy, Pandas, Matplotlib, scikit-learn, and Pillow.
