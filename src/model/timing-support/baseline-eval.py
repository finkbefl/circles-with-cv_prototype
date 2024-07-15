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
# For evaluation metrics
from sklearn import metrics
# Numpy
import numpy as np
# Serialization of the trained model
from joblib import load

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
__own_logger = OwnLogging("timing-support_" + Path(__file__).stem).logger

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
    plot = PlotMultipleFigures(os.path.join("output/timing-support",file_name), file_title)

    # Join the filepaths for the data
    data_raw_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw")
    data_modeling_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "modeling", "timing-support")

    __own_logger.info("Path of the raw input data: %s", data_raw_path)
    __own_logger.info("Path of the modeling input data: %s", data_modeling_path)

    # Get csv file, which was created during data collection and adapted during data analysis as DataFrame
    metadata = load_data(data_raw_path, 'training_videos_with_metadata.csv')

    # Get the Test Data
    # Some variable initializations
    data_test_arr = []
    # Iterate over all data where selected as test data (tagged in metadata column 'usage' with 'test')
    for video_idx in metadata.index[metadata.usage == 'test']:
        # The filename of the video contains also a number, but starting from 1
        video_name_num = video_idx + 1
        # Get all seperated test data (features) per raw video
        regex = re.compile('features_{}_.\.csv'.format(video_name_num))
        for dirpath, dirnames, filenames in os.walk(data_modeling_path):
            for test_data_file_name in filenames:
                if regex.match(test_data_file_name):
                    __own_logger.info("Testing data detected: %s", test_data_file_name)
                    # Get the data related to the specific video
                    try:
                            data_specific_video = load_data(data_modeling_path, test_data_file_name)
                    except FileNotFoundError as error:
                        __own_logger.error("########## Error when trying to access training data ##########", exc_info=error)
                    # Merge all data in one array
                    data_test_arr.append(data_specific_video)

    # Concatenate the data in one frame by simply chain together the time series rows, but ignore the index of the rows to add so that we generate a continuous increasing index
    data_test = pd.concat(data_test_arr, ignore_index=True)
    log_overview_data_frame(__own_logger, data_test)
    # Remove the timestamp column
    data_test.drop('timestamp', axis=1, inplace=True)
    log_overview_data_frame(__own_logger, data_test)

    # Handling missing data (frames with no detected landmarks)
    __own_logger.info("Detected missing data: %s", data_test.isna().sum())
    # Backward filling (take the next observation and fill backward) for rows which where initially labeled as missing-data
    data_test = data_test.mask(data_test.missing_data == True, data_test.fillna(method='bfill'))

    # Visualize the test data
    # Create dict for visualization data
    dict_visualization_data = {
        "label": data_test.columns.values, # Take all columns for visualization in dataframe
        "value": [data_test[data_test.columns.values][col] for col in data_test[data_test.columns.values]],
        # As x_data generate a consecutive number: a frame number for the whole merged time series, so the index + 1 can be used
        "x_data": data_test.index + 1
    }
    # Create a Line-Circle Chart
    figure_test_data = figure_time_series_data_as_layers(__own_logger, "Testdaten", "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Frame")
    # Append the figure to the plot
    plot.appendFigure(figure_test_data.getFigure())

    # Get the trained predictor via joblib
    file_path = os.path.join(data_modeling_path, 'baseline-svm-model.joblib')
    model = load(file_path)

    # Predict the target value for the whole test data
    y_pred, conf_interval = model.predict(len(data_test), exogenous=data_test.drop(['right_wrist_y_pos', 'left_wrist_y_pos', 'missing_data'], axis=1).to_numpy(), return_conf_int=True)
    data_test['prediction'] = y_pred

    # Visualize the prediction for the test data input
    # Create dict for visualization data
    dict_visualization_data = {
        "label": data_test.columns.values, # Take all columns for visualization in dataframe
        "value": [data_test[data_test.columns.values][col] for col in data_test[data_test.columns.values]],
        # As x_data generate a consecutive number: a frame number for the whole merged time series, so the index + 1 can be used
        "x_data": data_test.index + 1
    }
    # Create a Line-Circle Chart
    figure_test_data = figure_time_series_data_as_layers(__own_logger, "Testdaten: Vorhersage von 'right_wrist_y_pos'", "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Frame")
    # Append the figure to the plot
    plot.appendFigure(figure_test_data.getFigure())

    # Iterate over the single videos and using the single testdata for video-specific evalution
    # Iterate over all data where selected as test data (tagged in metadata column 'usage' with 'test')
    for video_idx in metadata.index[metadata.usage == 'test']:
        # The filename of the video contains also a number, but starting from 1
        video_name_num = video_idx + 1
        # Get all seperated test data (features) per raw video
        regex = re.compile('features_{}_.\.csv'.format(video_name_num))
        # Some variable initializations
        data_test_single_arr = []
        for dirpath, dirnames, filenames in os.walk(data_modeling_path):
            for test_data_file_name in filenames:
                if regex.match(test_data_file_name):
                    __own_logger.info("Testing data detected: %s", test_data_file_name)
                    # Get the data related to the specific video
                    try:
                            data_specific_video = load_data(data_modeling_path, test_data_file_name)
                    except FileNotFoundError as error:
                        __own_logger.error("########## Error when trying to access training data ##########", exc_info=error)
                    # Merge all data of one whole video in one array
                    data_test_single_arr.append(data_specific_video)

        # Concatenate the data of one whole video in one frame by simply chain together the time series rows, but ignore the index of the rows to add so that we generate a continuous increasing index
        data_test_single = pd.concat(data_test_single_arr, ignore_index=True)
        # Remove the timestamp column
        data_test_single.drop('timestamp', axis=1, inplace=True)
        # Handling missing data (frames with no detected landmarks): Backward filling (take the next observation and fill backward) for rows which where initially labeled as missing-data
        data_test_single = data_test_single.mask(data_test_single.missing_data == True, data_test_single.fillna(method='bfill'))
        # For missing data at the end, the bfill mechanism not work, so do now a ffill
        data_test_single = data_test_single.mask(data_test_single.missing_data == True, data_test_single.fillna(method='ffill'))
        # Predict the target value for the whole test data
        y_pred_single, conf_interval = model.predict(len(data_test_single), exogenous=data_test_single.drop(['right_wrist_y_pos', 'left_wrist_y_pos', 'missing_data'], axis=1).to_numpy(), return_conf_int=True)
        # Add prediciton to test data
        data_test_single['prediction'] = y_pred_single
        # Save to CSV
        save_data(data_test_single, data_modeling_path, "{0}_{1}.csv".format('data_test',video_name_num))
        # Create dict for visualization data
        dict_visualization_data = {
            "label": data_test_single.columns.values, # Take all columns for visualization in dataframe
            "value": [data_test_single[data_test_single.columns.values][col] for col in data_test_single[data_test_single.columns.values]],
            "x_data": list(range(1, y_pred_single.size + 1))
        }
        # Create a Line-Circle Chart
        figure_test_data_single = figure_time_series_data_as_layers(__own_logger, "Testdaten Video {}: Vorhersage von 'right_wrist_y_pos'".format(video_name_num), "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Frame")
        # Append the figure to the plot
        plot.appendFigure(figure_test_data_single.getFigure())

    # Save the testing Data to csv
    save_data(data_test, data_modeling_path, "data_test.csv")

    # Overall evaluation

    # MSE
    MSE = metrics.mean_squared_error(data_test.right_wrist_y_pos, y_pred)
    __own_logger.info("MSE: %s",MSE)
    # MAE
    MAE = metrics.mean_absolute_error(data_test.right_wrist_y_pos, y_pred)
    __own_logger.info("MAE: %s",MAE)
    # RMSE
    RMSE = np.sqrt(metrics.mean_squared_error(data_test.right_wrist_y_pos, y_pred))
    __own_logger.info("RMSE: %s",RMSE)
    # MAPE
    MAPE = metrics.mean_absolute_percentage_error(data_test.right_wrist_y_pos, y_pred)
    __own_logger.info("MAPE: %s",MAPE)

    # Visualize the metrics
    # Create dict for visualization data
    dict_visualization_data = {
        "label": ['MSE', 'MAE', 'RMSE', 'MAPE'],
        "value": [MSE, MAE, RMSE, MAPE]
    }
    # Create a bar chart
    figure_evaluation = figure_vbar(__own_logger, "Evaluierung", "Wert der Metrik", dict_visualization_data.get('label'), dict_visualization_data.get('value'), set_x_range=True, color_sequencing=False)
    # Append the figure to the plot
    plot.appendFigure(figure_evaluation.getFigure())

    # Evaluation per video
    # Get csv file, which was created during data collection and adapted during data analysis as DataFrame
    metadata = load_data(data_raw_path, 'training_videos_with_metadata.csv')
    # Iterate over all single evaluation data (get video index via tag in metadata column 'usage' with 'test')
    for video_idx in metadata.index[metadata.usage == 'test']:
        # The filename contains also a number, but starting from 1
        video_name_num = video_idx + 1
        eval_data_file_name = "data_test_{}.csv".format(video_name_num)
        __own_logger.info("Single evaluation data detected: %s", eval_data_file_name)
        # Get the data related to the specific video
        data_specific_video = load_data(data_modeling_path, eval_data_file_name)
        # Calc the metrics
        # MSE
        MSE = metrics.mean_squared_error(data_specific_video.right_wrist_y_pos, data_specific_video.prediction)
        __own_logger.info("Testdaten Video %d: MSE: %s",video_name_num, MSE)
        # MAE
        MAE = metrics.mean_absolute_error(data_specific_video.right_wrist_y_pos, data_specific_video.prediction)
        __own_logger.info("Testdaten Video %d: MAE: %s",video_name_num, MAE)
        # RMSE
        RMSE = np.sqrt(metrics.mean_squared_error(data_specific_video.right_wrist_y_pos, data_specific_video.prediction))
        __own_logger.info("Testdaten Video %d: RMSE: %s",video_name_num, RMSE)
        # MAPE
        MAPE = metrics.mean_absolute_percentage_error(data_specific_video.right_wrist_y_pos, data_specific_video.prediction)
        __own_logger.info("Testdaten Video %d: MAPE: %s",video_name_num, MAPE)

        # Create a bar chart
        figure_evaluation_single = figure_vbar(__own_logger, "Testdaten Video {}: Evaluierung".format(video_name_num), "Wert der Metrik", dict_visualization_data.get('label'), [MSE, MAE, RMSE, MAPE], set_x_range=True, color_sequencing=False)
        # Append the figure to the plot
        plot.appendFigure(figure_evaluation_single.getFigure())

    # Show the plot in responsive layout, but only stretch the width
    plot.showPlotResponsive('stretch_width')


