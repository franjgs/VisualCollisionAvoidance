# Visual Collision Avoidance for Autonomous Systems

This repository contains code and resources for developing visual collision avoidance capabilities for autonomous systems, such as Unmanned Aerial Vehicles (UAVs) and robots. The project leverages visual information from onboard cameras to detect and avoid potential collisions with obstacles.

This work draws inspiration and may incorporate elements from the [uav-collision-avoidance](https://github.com/dario-pedro/uav-collision-avoidance) repository by dario-pedro, particularly concerning dataset structures and evaluation methodologies.

## Project Overview

Autonomous navigation in complex environments requires robust collision avoidance mechanisms. Relying solely on traditional sensors like LiDAR or sonar can be limiting in terms of cost, weight, or environmental conditions. This project explores the use of visual data as a primary source for detecting and predicting potential collisions, enabling autonomous systems to navigate safely.

The core idea is to train machine learning models to understand the visual cues that indicate an impending collision. By processing camera feeds in real-time, the system can identify obstacles, estimate their distance and velocity relative to the autonomous agent, and generate appropriate avoidance maneuvers.

## Key Features (Planned and Implemented)

* **Dataset Handling:** Tools and scripts for processing and utilizing visual collision avoidance datasets, potentially compatible with the structure of datasets used in the [uav-collision-avoidance](https://github.com/dario-pedro/uav-collision-avoidance) repository.
* **Object Detection and Tracking:** Implementation of algorithms for detecting and tracking relevant obstacles in the visual field.
* **Distance and Velocity Estimation:** Methods for estimating the distance and relative velocity of detected obstacles based on visual cues.
* **Collision Risk Assessment:** Development of models to predict the likelihood and severity of potential collisions based on the estimated obstacle properties.
* **Path Planning and Avoidance Maneuver Generation:** Algorithms for generating collision-free trajectories or control commands to avoid detected obstacles.
* **Simulation Environment:** Integration with simulation environments (e.g., Gazebo, AirSim) for training and evaluating the collision avoidance system.
* **Model Training and Evaluation:** Scripts and tools for training machine learning models for various components of the system (e.g., object detection, collision prediction) and evaluating their performance using relevant metrics.
* **Integration with Autonomous Platforms:** Guidelines and examples for integrating the developed collision avoidance system with actual autonomous platforms.

## Getting Started

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Set up the environment:**

    * Install the necessary dependencies (see `requirements.txt`):

        ```bash
        pip install -r requirements.txt
        ```

    * If using a simulation environment, follow the installation instructions for that specific simulator.

3.  **Explore the codebase:**

    * Familiarize yourself with the different modules and scripts within the repository.
    * Refer to any available documentation or tutorials.

4.  **Dataset preparation:**

    * Download or generate visual collision avoidance datasets.
    * Follow the provided scripts or guidelines to preprocess the data into a suitable format for training.

5.  **Model training:**

    * Run the training scripts for the desired components of the collision avoidance system.
    * Configure training parameters (e.g., network architecture, hyperparameters) as needed.

6.  **Evaluation:**

    * Use the evaluation scripts to assess the performance of the trained models in simulation or on real-world data.
    * Analyze the evaluation metrics to identify areas for improvement.

## Dependencies
tensorflow==2.19.0

opencv-python==4.11.0.86

numpy==1.26.4

scipy==1.15.2

matplotlib==3.10.1

pyyaml==6.0

torch==2.5.1

torchvision==0.20.1

Libraries for a specific simulation environment 
airsim==1.9.0

Robotics-related libraries 
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
