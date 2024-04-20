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
# For plotting the learning curve
from sklearn.model_selection import learning_curve
# For auto_arima model
import pmdarima as pm
# Numpy
import numpy as np
# Serialization of the trained model
from joblib import dump
# Time series transformation with seglearn
import seglearn as sgl
# Using statsmodel for detecting stationarity
from statsmodels.tsa.stattools import adfuller, kpss
# Using statsmodel for decomposition analysis
from statsmodels.tsa.seasonal import seasonal_decompose
# Detection of relative minima of data
from scipy.signal import argrelmin
# Using scipy for calculating the spectrum
from scipy import signal
# Fractions
from fractions import Fraction

# Import internal packages/ classes
# Import the src-path to sys path that the internal modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))
# To handle the Logging for all modules in the same way
from utils.own_logging import OwnLogging, log_overview_data_frame
# To handle csv files
from utils.csv_operations import load_data,  save_data, convert_series_into_date
# To plot data with bokeh
from utils.plot_data import PlotMultipleLayers, PlotMultipleFigures, figure_vbar, figure_vbar_as_layers, figure_hist, figure_hist_as_layers, figure_time_series_data_as_layers

#########################################################

# Initialize the logger
__own_logger = OwnLogging("timing-support_" + Path(__file__).stem).logger

#########################################################

def stationarity_test(df):
    """
    Function to test the data columns for stationarity (Augmented Dickey-Fuller and Kwiatkowski-Phillips-Schmidt-Shin in combination for more confident decisions)
    ----------
    Parameters:
        df : pandas.core.frame.DataFrame
            The data
    ----------
    Returns:
        dict with the stationarity test results:
        {'column_name': 
            {'ADF': Boolean, 'KPSS': Boolean},
        ...
        }
    """

    stationarity_dict= {} # create an empty dictionary for the test results
    # Iterate over all columns except column one (date)
    for column in df.iloc[:,1:]:
        # Do not consider data which not vary over time, so skip the column which only consists of one value
        if (df[column].nunique() == 1):
            __own_logger.info("Skip column: %s because it not vary over time", column)
            continue
        # Check for stationarity
        # Augmented Dickey-Fuller Test
        adf_decision_stationary = False
        try:
            adf_output = adfuller(df[column])
            # Decision based on pval
            adf_pval = adf_output[1]
            if adf_pval < 0.05: 
                adf_decision_stationary = True
        except Exception as error:
            __own_logger.error("Error during ADF Test", exc_info=error)
        # Kwiatkowski-Phillips-Schmidt-Shin Test
        kpss_decision_stationary = False
        try:
            kpss_output = kpss(df[column])
            # Decision based on pval
            kpss_pval = kpss_output[1]
            if kpss_pval >= 0.05: 
                kpss_decision_stationary = True
        except Exception as error:
            __own_logger.error("Error during KPSS Test", exc_info=error)
        # Add the test results to the dict
        stationarity_dict[column] = {"ADF": adf_decision_stationary, "KPSS": kpss_decision_stationary}
    __own_logger.info("Stationarity: %s", stationarity_dict)

    return stationarity_dict

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
    # Iterate over all data where selected as training data (tagged in metadata column 'usage' with 'train')
    for video_idx in metadata.index[metadata.usage == 'train']:
        # Only use videos with "good" trial, so skip videos with detected amplitude lack
        performance_labels = [val.split('-') for val in metadata.manual_amplitude_lack][video_idx]
        skip = False
        for label in performance_labels:
            if label == 'true':
                skip=True
        if skip:
            continue
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

    # Concatenate the data in one frame by simply chain together the time series rows, but ignore the index of the rows to add so that we generate a continuous increasing index
    data_training = pd.concat(data_training_arr, ignore_index=True)
    # Remove the timestamp column
    data_training.drop('timestamp', axis=1, inplace=True)
    log_overview_data_frame(__own_logger, data_training)

    # Handling missing data (frames with no detected landmarks)
    __own_logger.info("Training Data: Detected missing data: %s", data_training.isna().sum())
    # Backward filling (take the next observation and fill bachward) for rows which where initially labeled as missing-data
    data_training = data_training.mask(data_training.missing_data == True, data_training.fillna(method='bfill'))

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

    # Time Series Stationarity
    data_stationarity = True
    # Copy the data for stationary data
    df_stationary_data = data_training.copy()
    # Test the columns for stationarity
    stationarity_results = stationarity_test(df_stationary_data)
    # Are the columns strict stationary?
    for column in stationarity_results:
        __own_logger.info("Data Analysis: Stationarity: Column %s is stationary: %s", column, stationarity_results[column])
        for value in stationarity_results[column].values():
            if value == False:
                data_stationarity = False
                break
        if not data_stationarity:
            break

    #Train the model using the training set
    model = pm.auto_arima(data_training.right_wrist_y_pos.to_numpy(), data_training.drop(['right_wrist_y_pos', 'left_wrist_y_pos', 'missing_data'], axis=1).to_numpy(), 
                          seasonal=False,   # TODO
                          stationary=data_stationarity, 
                          test='kpss', stepwise=True, 
                          trace=True)

    # Print the best params
    __own_logger.info("Training Data: Best model: ARIMA%s%s", model.order, model.seasonal_order)

    # #Testing out the CV scores is not enough to ensure the accuracy of the model. One could still run into the problem of high bias (underfitting) or high variances (overfitting). To see if this is the case, one can plot the learning curve:
    # # Train size as fraction of the maximum size of the training set
    # train_sizes_as_fraction = np.linspace(0.1, 1.0, 10)
    # train_sizes, train_scores, valid_scores = learning_curve(model, data_training.drop(['right_wrist_y_pos', 'left_wrist_y_pos', 'missing_data'], axis=1).to_numpy(),data_training.right_wrist_y_pos.to_numpy(),train_sizes=train_sizes_as_fraction,cv=5, scoring='neg_mean_absolute_percentage_error')
    # mean_train_scores=np.mean(train_scores,axis=1)
    # mean_valid_scores=np.mean(valid_scores,axis=1)
    # __own_logger.info("Mean train scores: %s", mean_train_scores)
    # __own_logger.info("Mean valid scores: %s", mean_valid_scores)
    # # Visualize the learning courve
    # # Create dict for visualization data
    # dict_visualization_data = {
    #     "label": ["Training score", "Cross-validation score"],
    #     "value": [mean_train_scores, mean_valid_scores],
    #     "x_data": train_sizes_as_fraction
    # }
    # # Create a Line-Circle Chart
    # figure_learning_courve = figure_time_series_data_as_layers(__own_logger, "Lernkurve", "Score", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Training Size")
    # # Append the figure to the plot
    # plot.appendFigure(figure_learning_courve.getFigure())
        
    # Show the plot in responsive layout, but only stretch the width
    plot.showPlotResponsive('stretch_width')

    # Save the Data to CSV
    # Training Data
    save_data(data_training, data_modeling_path, "data_train.csv")

    # Save the trained model with joblib
    file_path = os.path.join(data_modeling_path, 'baseline-svm-model.joblib')
    dump(model, file_path) 
