Computer Vision to Support the Training of Circles: Prototype
=============================

A prototype to support the training of circles with the scope of popular sport using computer vision (CV).

- [Project Description](#project-description)
- [Restrictions](#restrictions)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Running the Code](#running-the-code)
  - [Data Analysis](#data-analysis)
  - [Data Preparation](#data-preparation)
  - [Data Featurization](#data-featurization)
  - [Baseline Model Training](#baseline-model-training)
  - [Baseline Model Evaluation](#baseline-model-evaluation)
  - [Presenting the Results](#presenting-the-results)
- [Licence](#licence)

## Project Description

The project contains corresponding Python scripts for data analysis, preparation and feature selection. In addition, scripts are available for modelling in order to train and evaluate corresponding baseline models. Furthermore, the first visualisations of the results can be found in the context of the first operationalization scripts, which are intended to demonstrate a possible use of the models. Suitable helper scripts enable standardised visualisations and logging and provide centralised possibilities for parameter checking, CSV operations and advanced exception handling.

In the context of the developed pipeline, the available training videos are first analysed and corresponding metadata is stored. Then the videos are prepared for further processing and suitable features are extracted in the form of time series data of the positions of body features. These time series data are then used for training and evaluating baseline models based on classification (machine learning) and mathematical considerations. The operationalisation of the models in the prototype focuses on an initial usage of the system to existing videos, whereby the results are visualised to present the functional scope of the prototype, identify weaknesses and define the next steps towards a productive system. The results are visualised using the Bokeh library and persisted in CSV files across the entire pipeline. All trained models are saved as joblib files and the program run is logged centrally in individual log files.

This prototype contains 5 tasks that are related in a corresponding hierarchy:
| No. | Task | Description |
|-----|------|-------------|
| 1   | Pose estimation | Estimation of the human pose within the video streams |
| 2   | Detection of circles | Detect circles within a training attempt on the basis of the pose estimation in parts of the video |
| 3a  | Evaluation of circles | Extension of the detection to include an analysis of the extent to which an evaluation of the circles can be implemented under the given environmental parameters |
| 3b  | Anomaly detection | Development of a model for detecting anomalies in sections where circles could be detected |
| 3c  | Timing support | Integration of acoustic support for training in relation to executed circles |

## Restrictions

- This is an initial prototype to assess the applicability of CV to training videos in gymnastics in the context of popular sport. The focus is not on productively applicable models, but rather on initial baseline models that can be used as a basis for comparison in the context of further development.
- The restriction of the hardware includes the use of a standard notebook for training and standard smartphones as target devices.
- Only 2D videos are processed and multiple perspectives are not supported.
- The video images are processed exclusively frame by frame; the frames are not grouped.
- The video data is not pushed into this repository, so a clone of the project is not fully functional! All scripts for processing the videos cannot be executed. However, the extracted time series data of the positions of the body features are available within the repository, so all subsequent scripts can be executed out of the box.

## Prerequisites

To get the code running, only the following steps are required:
1. Clone the repository
    ```bash
    git clone https://github.com/finkbefl/circles-with-cv_prototype.git
    ```
2. Use conda to create a virtual environment for this python project and install all dependencies:
    ```bash
    conda env create --file program_requirements.yml
    ```
3. The virtual environment must be activated:
    ```bash
    conda activate circles-with-cv
    ```

## Project Structure

```
.
├── data                >>> includes all data (videos, time series data, persisted models, metadata)
│   ├── raw             >>> all raw videos and the collected metadata
│       ├── <video-num>.mp4
│       ├── ...
│       ├── deployment  >>> dir contains the videos used for showing the functionality
│       ├── PoC         >>> dir for videos used within the initial experiments
│       ├── training_videos.csv
│       └── training_videos_with_metadata.csv
│   ├── processed       >>> all preprocessed videos
│   │   ├── anomaly-detection
│   │   │   ├── <trial-num>.mp4
│   │   │   ...
│   │   ├── circles-detection
│   │   │   ├── <video-num>.mp4
│   │   │   ...
│   │   ├── circles-performance
│   │   │   ├── <trial-num>.mp4
│   │   │   ...
│   │   └── timing-support
│   │   │   ├── <trial-num>.mp4
│   │   │   ...
│   └── modeling        >>> data for the modelling processes (time series data for training and testing and the persisted models)
│   │   ├── anomaly-detection
│   │   │   ├── baseline-svm-model.joblib
│   │   │   ├── data_test_<video-num>.csv
│   │   │   ...
│   │   │   ├── data_test.csv
│   │   │   ├── data_train.csv
│   │   │   ├── features_<trial-num>.csv
│   │   │   ...
│   │   ├── circles-detection
│   │   │   ├── baseline-svm-model.joblib
│   │   │   ├── data_test_<video-num>.csv
│   │   │   ...
│   │   │   ├── data_test.csv
│   │   │   ├── data_train.csv
│   │   │   ├── features_<video-num>.csv
│   │   │   ...
│   │   ├── circles-performance
│   │   │   ├── baseline-svm-model.joblib
│   │   │   ├── data_test_<video-num>.csv
│   │   │   ...
│   │   │   ├── data_test.csv
│   │   │   ├── data_train.csv
│   │   │   ├── features_<trial-num>.csv
│   │   │   ...
│   │   └── timing-support
│   │   │   ├── baseline-svm-model.joblib
│   │   │   ├── data_test_<video-num>.csv
│   │   │   ...
│   │   │   ├── data_test.csv
│   │   │   ├── data_train.csv
│   │   │   ├── features_<trial-num>.csv
│   │   │   ...
├── src                 >>> the python scripts
    ├── data            >>> scripts for data preparation and feature selection
    │   ├── anomaly-detection
    │   │   ├── data-featurization.py
    │   │   └── data-preparation.py
    │   ├── circles-detection
    │   │   ├── data-featurization.py
    │   │   └── data-preparation.py
    │   ├── circles-performance
    │   │   ├── data-featurization.py
    │   │   └── data-preparation.py
    │   ├── data-analysis.py
    │   └── timing-support
    │       ├── data-featurization.py
    │       └── data-preparation.py
    ├── model           >>> scripts for the training and testing of the models
    │   ├── anomaly-detection
    │   │   ├── baseline-eval.py
    │   │   └── baseline-train.py
    │   ├── circles-detection
    │   │   ├── baseline-eval.py
    │   │   └── baseline-train.py
    │   ├── circles-performance
    │   │   ├── baseline-eval.py
    │   │   └── baseline-train.py
    │   ├── overall-eval-visualization.py
    │   └── timing-support
    │       ├── baseline-eval.py
    │       ├── baseline-prep-eda.py
    │       └── baseline-train.py
    ├── operationalization      >>> scripts for presenting the functional scope of the prototype
    │   ├── anomaly-detection
    │   │   └── baseline-deploy.py
    │   ├── circles-detection
    │   │   └── baseline-deploy.py
    │   ├── circles-performance
    │   │   └── baseline-deploy.py
    │   └── timing-support
    │       └── baseline-deploy.py
    ├── spike           >>> initial experiments
    └── utils           >>> helper scripts
        ├── check_parameter.py
        ├── csv_operations.py
        ├── own_exceptions.py
        ├── own_logging.py
        ├── plot_data.py
├── logs                >>> generated logging files
│   ├── anomaly-detection_baseline-deploy.log
│   ├── anomaly-detection_baseline-eval.log
│   ├── anomaly-detection_baseline-train.log
│   ├── anomaly-detection_data-featurization.log
│   ├── anomaly-detection_data-preparation.log
│   ├── circles-detection_baseline-deploy.log
│   ├── circles-detection_baseline-eval.log
│   ├── circles-detection_baseline-train_grid-search.log
│   ├── circles-detection_baseline-train.log
│   ├── circles-detection_data-featurization.log
│   ├── circles-detection_data-preparation.log
│   ├── circles-performance_baseline-deploy.log
│   ├── circles-performance_baseline-eval.log
│   ├── circles-performance_baseline-train.log
│   ├── circles-performance_data-featurization.log
│   ├── circles-performance_data-preparation.log
│   ├── data-analysis.log
│   ├── overall-eval-visualization_overall-eval-visualization.log
│   ├── timing-support_baseline-deploy.log
│   ├── timing-support_baseline-eval.log
│   ├── timing-support_baseline-prep-eda.log
│   ├── timing-support_baseline-train.log
│   ├── timing-support_data-featurization.log
│   └── timing-support_data-preparation.log
├── output              >>> generated output/ visualizations
│   ├── anomaly-detection
│   │   ├── baseline-deploy.html
│   │   ├── baseline-eval.html
│   │   ├── baseline-train.html
│   │   └── data-featurization.html
│   ├── circles-detection
│   │   ├── baseline-eval.html
│   │   ├── baseline-train.html
│   │   └── data-featurization.html
│   ├── circles-performance
│   │   ├── baseline-eval.html
│   │   ├── baseline-train.html
│   │   └── data-featurization.html
│   ├── data-analysis.html
│   ├── overall-eval-visualization
│   │   └── baseline-eval.html
│   └── timing-support
│       ├── baseline-deploy.html
│       ├── baseline-eval.html
│       ├── baseline-preparation-eda.html
│       ├── baseline-train.html
│       └── data-featurization.html
├── LICENSE                     >>> The license file for this project
├── program_requirements.yml    >>> The required libs for the env   
└── README.md                   >>> This README file
```

## Running the Code

The pipeline is represented by various directories, which can be used to select the appropriate step:
- `/src/data`
- `/src/model`
- `/src/operationalization`

> Note: The video data is not pushed into this repository. Scripts that access it are therefore not executable.

Each task can be selected by choosing the corresponding directory:
- Detection of circles: `/circles-detection`
- Evaluation of circles: `/circles-performance`
- Anomaly detection: `/anomaly-detection`
- Timing support: `/timing-support`

To run the whole pipeline, the following scripts must be called up for the corresponding task.

### Data Analysis

```bash
python <project-path>/src/data/data-analysis.py
```

### Data Preparation

```bash
python <project-path>/src/data/<task-dir>/data-preparation.py
```

### Data Featurization

```bash
python <project-path>/src/data/<task-dir>/data-featurization.py
```

### Baseline Model Training

```bash
python <project-path>/src/model/<task-dir>/baseline-train.py
```

### Baseline Model Evaluation

```bash
python <project-path>/src/model/<task-dir>/baseline-eval.py
```

### Presenting the Results

```bash
python <project-path>/src/operationalization/<task-dir>/baseline-deploy.py
```

## Licence

[GNU GPLv3](LICENSE)