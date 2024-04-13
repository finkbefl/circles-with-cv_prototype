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
# For standardized features
from sklearn.preprocessing import StandardScaler
# Numpy
import numpy as np
# Serialization of the trained model
from joblib import dump
# Time series transformation with seglearn
import seglearn as sgl
# Using statsmodel for decomposition analysis
from statsmodels.tsa.seasonal import seasonal_decompose

# Import internal packages/ classes
# Import the src-path to sys path that the internal modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))
# To handle the Logging for all modules in the same way
from utils.own_logging import OwnLogging, log_overview_data_frame
# To handle csv files
from utils.csv_operations import load_data,  save_data, convert_series_into_date
# To plot data with bokeh
from utils.plot_data import PlotMultipleLayers, PlotMultipleFigures, figure_vbar, figure_hist, figure_hist_as_layers, figure_time_series_data_as_layers

#########################################################

# Initialize the logger
__own_logger = OwnLogging("timing-support_" + Path(__file__).stem).logger

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
    plot = PlotMultipleFigures(os.path.join("output/timing-support",file_name), file_title)

    # Join the filepaths for the data
    data_raw_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw")
    data_modeling_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "modeling", "timing-support")

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

    # Get the data for the specific video which should be analyzed in detail
    data_video_to_analyze = load_data(data_modeling_path, 'features_2_0.csv')

    # Convert the timestamps (in ms) into DateTime (raise an exception when parsing is invalid) and set it as index
    data_video_to_analyze = data_video_to_analyze.set_index(convert_series_into_date(data_video_to_analyze.timestamp, unit='ms'))
    # Remove the timestamp column
    data_video_to_analyze.drop('timestamp', axis=1, inplace=True)
    log_overview_data_frame(__own_logger, data_video_to_analyze)

    # Concatenate the data in one frame by simply chain together the time series rows, but ignore the index of the rows to add so that we generate a continuous increasing index
    data_training = pd.concat(data_training_arr, ignore_index=True)
    # Remove the timestamp column
    data_training.drop('timestamp', axis=1, inplace=True)
    log_overview_data_frame(__own_logger, data_training)

    # Handling missing data (frames with no detected landmarks)
    __own_logger.info("Training Data: Detected missing data: %s", data_training.isna().sum())
    # Backward filling (take the next observation and fill bachward) for rows which where initially labeled as missing-data
    data_training = data_training.mask(data_training.missing_data == True, data_training.fillna(method='bfill'))
    __own_logger.info("Data to Analyze: Detected missing data: %s", data_video_to_analyze.isna().sum())
    # Backward filling (take the next observation and fill bachward) for rows which where initially labeled as missing-data
    data_video_to_analyze = data_video_to_analyze.mask(data_video_to_analyze.missing_data == True, data_video_to_analyze.fillna(method='bfill'))

    # Visualize the training data
    # Create dict for visualization data
    dict_visualization_data = {
        "label": data_training.columns.values, # Take all columns for visualization in dataframe
        "value": [data_training[data_training.columns.values][col] for col in data_training[data_training.columns.values]],
        # As x_data generate a consecutive number: a frame number for the whole merged time series, so the index + 1 can be used
        "x_data": data_training.index + 1
    }
    # Create a Line-Circle Chart
    figure_training_data = figure_time_series_data_as_layers(__own_logger, "Trainingsdaten: Positionen der Füße und Handgelenke", "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Frame")
    # Append the figure to the plot
    plot.appendFigure(figure_training_data.getFigure())

    # Visualize the data to analyze in detail
    # Create dict for visualization data
    dict_visualization_data = {
        "label": data_video_to_analyze.columns.values, # Take all columns for visualization in dataframe
        "value": [data_video_to_analyze[data_video_to_analyze.columns.values][col] for col in data_video_to_analyze[data_video_to_analyze.columns.values]],
        # As x_data generate a consecutive number: a frame number for the whole merged time series, so the index + 1 can be used
        "x_data": data_video_to_analyze.index
    }
    # Create a Line-Circle Chart
    figure_analyze_data = figure_time_series_data_as_layers(__own_logger, "Datenanalyse des Videos 2_0: Positionen der Füße und Handgelenke", "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Laufzeit des Videos", x_axis_type='datetime')
    # Append the figure to the plot
    plot.appendFigure(figure_analyze_data.getFigure())

    # Analyze the specific video in detail
    # TODO: Detect frequency of features?
    # Decompose the data columns
    # right_wrist_y_pos
    #res = seasonal_decompose(data_video_to_analyze.right_wrist_y_pos, model='additive' period=40)
    # Visualize the decomposition
    # TODO





        
    # Show the plot in responsive layout, but only stretch the width
    plot.showPlotResponsive('stretch_width')
