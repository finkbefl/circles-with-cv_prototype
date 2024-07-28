# Script for the Training of a Baseline Model

# Import the external packages
# Operating system functionalities
import sys
import os
# Regular expressions
import re
from pathlib import Path
# To handle pandas data frames
import pandas as pd
# sklearn: SVM Model
from sklearn import svm
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.preprocessing import StandardScaler
# sklearn: Metrics
from sklearn import metrics
# Numpy
import numpy as np
# Serialization of the trained model
from joblib import dump
# Time series transformation with seglearn
import seglearn as sgl

# Import internal packages/ classes
# Import the src-path to sys path that the internal modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))
# To handle the Logging for all modules in the same way
from utils.own_logging import OwnLogging, log_overview_data_frame
# To handle csv files
from utils.csv_operations import load_data,  save_data
# To plot data with bokeh
from utils.plot_data import PlotMultipleLayers, PlotMultipleFigures, figure_vbar, figure_hist, figure_hist_as_layers, figure_time_series_data_as_layers

#########################################################

# Initialize the logger
__own_logger = OwnLogging("anomaly-detection_" + Path(__file__).stem).logger

#########################################################
#########################################################
#########################################################

# When this script is called directly...
if __name__ == "__main__":
    # ...then calling the functions

    __own_logger.info("########## START ##########")

    # Create a plot for multiple figures
    file_name = "baseline-train.html"
    file_title = "Training the baseline model"
    __own_logger.info("Plot %s as multiple figures to file %s", file_title, file_name)
    plot = PlotMultipleFigures(os.path.join("output/anomaly-detection",file_name), file_title)

    # Join the filepaths for the data
    data_raw_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw")
    data_modeling_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "modeling", "anomaly-detection")

    __own_logger.info("Path of the raw input data: %s", data_raw_path)
    __own_logger.info("Path of the modeling input data: %s", data_modeling_path)

    # Get csv file, which was created during data collection and adapted during data analysis as DataFrame
    metadata = load_data(data_raw_path, 'training_videos_with_metadata.csv')

    # Get the Training Data
    # Some variable initializations
    data_training_arr = []
    # Iterate over all data where selected as training data (tagged in metadata column 'validation' with 'no')
    for video_idx in metadata.index[metadata.validation == 'no']:
        # The filename of the video contains also a number, but starting from 1
        video_name_num = video_idx + 1
        # Get all seperated training data (features) per raw video
        regex = re.compile('features_{}_.\.csv'.format(video_name_num))
        for dirpath, dirnames, filenames in os.walk(data_modeling_path):
            for train_data_file_name in filenames:
                if regex.match(train_data_file_name):
                    __own_logger.info("Training data detected: %s", train_data_file_name)
                    # Get the data related to the specific video
                    try:
                            data_specific_video = load_data(data_modeling_path, train_data_file_name)
                    except FileNotFoundError as error:
                        __own_logger.error("########## Error when trying to access training data ##########", exc_info=error)
                    # Merge all data in one array
                    data_training_arr.append(data_specific_video)

    __own_logger.info("Find %d Videos for Training", len(data_training_arr))

    # Get the Validation Data
    # Some variable initializations
    data_validation_arr = []
    # Iterate over all data where selected as validation data (tagged in metadata column 'validation' with 'yes')
    for video_idx in metadata.index[metadata.validation == 'yes']:
        # The filename of the video contains also a number, but starting from 1
        video_name_num = video_idx + 1
        # Get all seperated training data (features) per raw video
        regex = re.compile('features_{}_.\.csv'.format(video_name_num))
        for dirpath, dirnames, filenames in os.walk(data_modeling_path):
            for valid_data_file_name in filenames:
                if regex.match(valid_data_file_name):
                    __own_logger.info("Validation data detected: %s", valid_data_file_name)
                    # Get the data related to the specific video
                    try:
                            data_specific_video = load_data(data_modeling_path, valid_data_file_name)
                    except FileNotFoundError as error:
                        __own_logger.error("########## Error when trying to access training data ##########", exc_info=error)
                    # Merge all data in one array
                    data_validation_arr.append(data_specific_video)
                    
    __own_logger.info("Find %d Videos for Validation", len(data_validation_arr))

    # Concatenate the data in one frame by simply chain together the time series rows, but ignore the index of the rows to add so that we generate a continuous increasing index
    data_training = pd.concat(data_training_arr, ignore_index=True)
    log_overview_data_frame(__own_logger, data_training)
    data_validation = pd.concat(data_validation_arr, ignore_index=True)
    log_overview_data_frame(__own_logger, data_validation)

    # Handling missing data (frames with no detected landmarks)
    __own_logger.info("Training Data: Detected missing data: %s", data_training.isna().sum())
    __own_logger.info("Validation Data: Detected missing data: %s", data_validation.isna().sum())
    # Backward filling (take the next observation and fill bachward) for rows which where initially labeled as missing-data
    data_training = data_training.mask(data_training.missing_data == True, data_training.fillna(method='bfill'))
    data_validation = data_validation.mask(data_validation.missing_data == True, data_validation.fillna(method='bfill'))

    # Visualize the training data
    # Create dict for visualization data
    dict_visualization_data = {
        "label": data_training.columns.values, # Take all columns for visualization in dataframe
        "value": [data_training[data_training.columns.values][col] for col in data_training[data_training.columns.values]],
        # As x_data generate a consecutive number: a frame number for the whole merged time series, so the index + 1 can be used
        "x_data": data_training.index + 1
    }
    # Create a Line-Circle Chart
    figure_training_data = figure_time_series_data_as_layers(__own_logger, "Trainingsdaten", "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Frame")
    # Append the figure to the plot
    plot.appendFigure(figure_training_data.getFigure())

    # Visualize the validation data
    # Create dict for visualization data
    dict_visualization_data = {
        "label": data_validation.columns.values, # Take all columns for visualization in dataframe
        "value": [data_validation[data_validation.columns.values][col] for col in data_validation[data_validation.columns.values]],
        # As x_data generate a consecutive number: a frame number for the whole merged time series, so the index + 1 can be used
        "x_data": data_validation.index + 1
    }
    # Create a Line-Circle Chart
    figure_validation_data = figure_time_series_data_as_layers(__own_logger, "Validierungsdaten", "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Frame")
    # Append the figure to the plot
    plot.appendFigure(figure_validation_data.getFigure())

    # Hyperparameter Tuning with ParameterGrid, as GridSearch is not useable with dedicated validation set: But restrict to rbf kernel
    # Using a pipeline: Scale the data and then propagate the data to the model (when segmentation is needed, then we need the sklearn pipeline extension from seglearn, only scaler and model would also be possible wit sklearn pipeline)
    # For value of gamma: For 'scale' it uses 1 / (n_features * X.var()), for 'auto' 1 / n_features
    tuning_parameters = [{'kernel': ['rbf'], 'gamma': ['scale','auto',1e-2,1e-3, 1e-4], 'nu': [0.05, 0.1, 0.15, 0.2]}]
    # Define the ParameterGrid
    grid = ParameterGrid(tuning_parameters)
    # Iterate over the tuning parameter
    # Some variable initializations
    accuracy_arr = []
    precision_arr = []
    recall_arr = []
    f1_arr = []
    for params in grid:
        __own_logger.info("Evaluate tuning parameter: %s",params)
        pipeline = sgl.Pype([
            ("scaler", StandardScaler()), 
            ("svc", svm.OneClassSVM(kernel=params['kernel'], gamma=params['gamma'], nu=params['nu']))])
        #Fit the model using the training sets
        pipeline.fit(data_training.drop(['missing_data'], axis=1).to_numpy(), [])
        # Predict anomalies for the validation: Returns -1 for outliers and 1 for inliers.
        anomalies_pred = (pipeline.predict(data_validation.drop(['missing_data', 'anomaly'], axis=1).to_numpy()) == -1)
        # Use a copy for evaluation, so that the original validation data is every loop the same
        data_evaluation = data_validation.copy()
        data_evaluation['prediction'] = anomalies_pred
        # Evaluation
        # Calculate some metrics
        # Model Accuracy (Percentage of correct predictions): Number of correct predicitons / Number of all predictions
        # Good when classes are well balanced
        # Optimizations with regard to this metric means, that as many correct predictions as possible are made without placing a greater emphasis on a particular class
        accuracy = metrics.accuracy_score(data_evaluation.anomaly, data_evaluation.prediction)
        __own_logger.info("Accuracy: %s",accuracy)
        accuracy_arr.append(accuracy)
        # Model Precision (Ratio of true positives correctly predicted) : Number of predicted true positives / Number of all items matched positive by the algorithm (predicted true positives + predicted false pisitives)
        # Good if we want to be very sure that the positive prediction is correct
        # Optimizations with regard to this metric means, that emphasis is placed on ensuring that the positive predictions are actually positive
        precision = metrics.precision_score(data_evaluation.anomaly, data_evaluation.prediction)
        __own_logger.info("Precision: %s",precision)
        precision_arr.append(precision)
        # Model Recall (how well the model is able to identify positive outcomes): Number of predicted true positives / Number of actual real positives (predicted true positives + predicted false negatives)
        # Good if we want to identify as many positive results as possible
        # Optimizations with regard to this metric means, that emphasis is placed on identifying as many actual positives as possible
        recall = metrics.recall_score(data_evaluation.anomaly, data_evaluation.prediction)
        __own_logger.info("Recall: %s",recall)
        recall_arr.append(recall)
        # Model F1-Score
        f1 = metrics.f1_score(data_evaluation.anomaly, data_evaluation.prediction)
        __own_logger.info("F1-Score: %s",f1)
        f1_arr.append(f1)
    # Visualize the evaluation criteria values for the params
    # Create dict for visualization data
    dict_visualization_data = {
        "label": ['accuracy', 'precision', 'recall', 'f1'],
        "value": [accuracy_arr, precision_arr, recall_arr, f1_arr],
        "x_data": list(range(0, len(grid)))
    }
    # # Create a Line-Circle Chart
    figure_hyperparam_optimization_eval = figure_time_series_data_as_layers(__own_logger, "Hyperparameter Tuning mit ParameterGrid: Metriken", "Wert der Metrik", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "ParameterGrid-Index", legend_location='top_left')
    # Detect the best params: Using accuracy as criteria
    best_param_value = max(accuracy_arr)
    best_param_index = accuracy_arr.index(max(accuracy_arr))
    __own_logger.info("Evaluation based on accuracy: Max value %s with param (index %s) %s", best_param_value, best_param_index, grid[best_param_index - 1])
    # VIsualize the best params
    figure_hyperparam_optimization_eval.add_vertical_line(best_param_index, 1.1)
    figure_hyperparam_optimization_eval.add_annotation(best_param_index-0.5, 1.05, "Accuracy: {:.2f}".format(best_param_value), text_align='right')
    figure_hyperparam_optimization_eval.add_annotation(best_param_index-0.5, 0, "Index: {}".format(best_param_index), text_align='right')
    # # Append the figure to the plot
    plot.appendFigure(figure_hyperparam_optimization_eval.getFigure())

    # Using the best parameters and create a pipeline
    pipeline = sgl.Pype([
        ("scaler", StandardScaler()), 
        ("svc", svm.OneClassSVM(kernel=grid[best_param_index - 1]['kernel'], gamma=grid[best_param_index - 1]['gamma'], nu=grid[best_param_index - 1]['nu']))])

    #Fit the model using the training sets
    pipeline.fit(data_training.drop(['missing_data'], axis=1).to_numpy(), [])

    # Predict anomalies within the whole data set: Returns -1 for outliers and 1 for inliers.
    anomalies_pred = (pipeline.predict(data_training.drop(['missing_data'], axis=1).to_numpy()) == -1)
    __own_logger.info("Number of detected outliers: %s", anomalies_pred.sum())

    # Get the anomalies per video
    # Iterate over the single videos and using the single training data
    for idx, data_train_single_arr in enumerate(data_training_arr):
        data_train_single = pd.DataFrame(data_train_single_arr)
        # Handling missing data (frames with no detected landmarks): Backward filling (take the next observation and fill backward) for rows which where initially labeled as missing-data
        data_train_single = data_train_single.mask(data_train_single.missing_data == True, data_train_single.fillna(method='bfill'))
        # For missing data at the end, the bfill mechanism not work, so do now a ffill
        data_train_single = data_train_single.mask(data_train_single.missing_data == True, data_train_single.fillna(method='ffill'))
        # Predict the anomalies: Returns -1 for outliers and 1 for inliers.
        anomalies_pred_single = (pipeline.predict(data_train_single.drop(['missing_data'], axis=1).to_numpy()) == -1)
        # Merge the data for visualization
        data_visualization = data_train_single
        data_visualization['anomalies_pred'] = anomalies_pred_single
        # Create dict for visualization data
        dict_visualization_data = {
            "label": data_visualization.columns.values, # Take all columns for visualization in dataframe
            "value": [data_visualization[data_visualization.columns.values][col] for col in data_visualization[data_visualization.columns.values]],
            "x_data": list(range(1, anomalies_pred_single.size + 1))
        }
        # # Create a Line-Circle Chart
        figure_train_data_single = figure_time_series_data_as_layers(__own_logger, "Trainingsdaten Video {}: Vorhersage der Anomalien".format(idx+1), "Anomalien detektiert", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Frame")
        # # Append the figure to the plot
        plot.appendFigure(figure_train_data_single.getFigure())
    # Iterate over the single videos and using the single validation data
    for idx, data_validation_single_arr in enumerate(data_validation_arr):
        data_valid_single = pd.DataFrame(data_validation_single_arr)
        # Handling missing data (frames with no detected landmarks): Backward filling (take the next observation and fill backward) for rows which where initially labeled as missing-data
        data_valid_single = data_valid_single.mask(data_valid_single.missing_data == True, data_valid_single.fillna(method='bfill'))
        # For missing data at the end, the bfill mechanism not work, so do now a ffill
        data_valid_single = data_valid_single.mask(data_valid_single.missing_data == True, data_valid_single.fillna(method='ffill'))
        # Predict the anomalies: Returns -1 for outliers and 1 for inliers.
        anomalies_pred_single = (pipeline.predict(data_valid_single.drop(['missing_data', 'anomaly'], axis=1).to_numpy()) == -1)
        # Merge the data for visualization
        data_visualization = data_valid_single
        data_visualization['anomalies_pred'] = anomalies_pred_single
        # Create dict for visualization data
        dict_visualization_data = {
            "label": data_visualization.columns.values, # Take all columns for visualization in dataframe
            "value": [data_visualization[data_visualization.columns.values][col] for col in data_visualization[data_visualization.columns.values]],
            "x_data": list(range(1, anomalies_pred_single.size + 1))
        }
        # # Create a Line-Circle Chart
        figure_valid_data_single = figure_time_series_data_as_layers(__own_logger, "Validierungsdaten Video {}: Vorhersage der Anomalien".format(idx+1), "Anomalien detektiert", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Frame")
        # # Append the figure to the plot
        plot.appendFigure(figure_valid_data_single.getFigure())
        
    # Show the plot in responsive layout, but only stretch the width
    plot.showPlotResponsive('stretch_width')

    # Save the Data to CSV
    # Training Data
    save_data(data_training, data_modeling_path, "data_train.csv")

    # Save the trained model with joblib
    file_path = os.path.join(data_modeling_path, 'baseline-svm-model.joblib')
    dump(pipeline, file_path)
