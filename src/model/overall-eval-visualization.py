# Script for the Evaluation of the Baseline Model

# Import the external packages
# Operating system functionalities
import sys
import os
# Regular expressions
import re
from pathlib import Path
# To handle pandas data frames
import pandas as pd
# sklearn: Metrics
from sklearn import metrics
# Numpy
import numpy as np
# Serialization of the trained model
from joblib import load

# Import internal packages/ classes
# Import the src-path to sys path that the internal modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
# To handle the Logging for all modules in the same way
from utils.own_logging import OwnLogging, log_overview_data_frame
# To handle csv files
from utils.csv_operations import load_data,  save_data
# To plot data with bokeh
from utils.plot_data import PlotMultipleLayers, PlotMultipleFigures, figure_vbar, figure_hist, figure_hist_as_layers, figure_time_series_data_as_layers, figure_vbar_as_layers

#########################################################

# Initialize the logger
__own_logger = OwnLogging("overall-eval-visualization_" + Path(__file__).stem).logger

#########################################################
#########################################################
#########################################################

# When this script is called directly...
if __name__ == "__main__":
    # ...then calling the functions

    __own_logger.info("########## START ##########")

    # Create a plot for multiple figures
    file_name = "baseline-eval.html"
    file_title = "Evaluation of the the baseline model"
    __own_logger.info("Plot %s as multiple figures to file %s", file_title, file_name)
    plot = PlotMultipleFigures(os.path.join("output/overall-eval-visualization",file_name), file_title)

    # Join the filepaths for the data
    data_raw_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw")
    data_modeling_path_detection = os.path.join(os.path.dirname(__file__), "..", "..", "data", "modeling", "circles-detection")
    data_modeling_path_performance = os.path.join(os.path.dirname(__file__), "..", "..", "data", "modeling", "circles-performance")
    data_modeling_path_anomaly = os.path.join(os.path.dirname(__file__), "..", "..", "data", "modeling", "anomaly-detection")

    __own_logger.info("Path of the raw input data: %s", data_raw_path)
    __own_logger.info("Path of the modeling input data - detection: %s", data_modeling_path_detection)
    __own_logger.info("Path of the modeling input data - performance: %s", data_modeling_path_performance)
    __own_logger.info("Path of the modeling input data - anomaly: %s", data_modeling_path_anomaly)

    # Get the testing Data
    data_test_detection = load_data(data_modeling_path_detection, "data_test.csv")
    data_test_performance = load_data(data_modeling_path_performance, "data_test.csv")
    data_test_anomaly = load_data(data_modeling_path_anomaly, "data_test.csv")

    # Evaluation
    detection_arr = []
    performance_arr = []
    anomaly_arr = []
    # Model Accuracy (Percentage of correct predictions): Number of correct predicitons / Number of all predictions
    # Good when classes are well balanced
    # Optimizations with regard to this metric means, that as many correct predictions as possible are made without placing a greater emphasis on a particular class
    #accuracy_arr = []
    detection_arr.append(metrics.accuracy_score(data_test_detection.circles_running, data_test_detection.prediction))
    performance_arr.append(metrics.accuracy_score(data_test_performance.amplitude_lack, data_test_performance.prediction))
    anomaly_arr.append(metrics.accuracy_score(data_test_anomaly.anomaly, data_test_anomaly.prediction))

    # Model Precision (Ratio of true positives correctly predicted) : Number of predicted true positives / Number of all items matched positive by the algorithm (predicted true positives + predicted false pisitives)
    # Good if we want to be very sure that the positive prediction is correct
    # Optimizations with regard to this metric means, that emphasis is placed on ensuring that the positive predictions are actually positive
    #precision_arr = []
    detection_arr.append(metrics.precision_score(data_test_detection.circles_running, data_test_detection.prediction))
    performance_arr.append(metrics.precision_score(data_test_performance.amplitude_lack, data_test_performance.prediction))
    anomaly_arr.append(metrics.precision_score(data_test_anomaly.anomaly, data_test_anomaly.prediction))

    # Model Recall (how well the model is able to identify positive outcomes): Number of predicted true positives / Number of actual real positives (predicted true positives + predicted false negatives)
    # Good if we want to identify as many positive results as possible
    # Optimizations with regard to this metric means, that emphasis is placed on identifying as many actual positives as possible
    #recall_arr = []
    detection_arr.append(metrics.recall_score(data_test_detection.circles_running, data_test_detection.prediction))
    performance_arr.append(metrics.recall_score(data_test_performance.amplitude_lack, data_test_performance.prediction))
    anomaly_arr.append(metrics.recall_score(data_test_anomaly.anomaly, data_test_anomaly.prediction))

    # Model F1-Score
    #f1_arr = []
    detection_arr.append(metrics.f1_score(data_test_detection.circles_running, data_test_detection.prediction))
    performance_arr.append(metrics.f1_score(data_test_performance.amplitude_lack, data_test_performance.prediction))
    anomaly_arr.append(metrics.f1_score(data_test_anomaly.anomaly, data_test_anomaly.prediction))

    # Visualize the metrics
    # Create dict for visualization data
    dict_visualization_data = {
        "layer": ["circles_detect", "circles_performance", "anomaly_detect"],
        "label": ["accuracy", "precision", "recall", "f1"],
        "value": [detection_arr, performance_arr, anomaly_arr]
    }
    # Create a bar chart
    figure_evaluation = figure_vbar_as_layers(__own_logger, "Evaluierung", "Wert der Metrik", dict_visualization_data.get('layer'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), set_x_range=True, single_x_range=True, width=0.2, x_offset=0.2, fill_alpha=1, legend_location='center_left')
    # Append the figure to the plot
    plot.appendFigure(figure_evaluation.getFigure())

    # Show the plot in responsive layout, but only stretch the width
    plot.showPlotResponsive('stretch_width')


