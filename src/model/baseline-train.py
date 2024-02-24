# Script for the Training of a Baseline Model

# Import the external packages
# Operating system functionalities
import sys
import os
from pathlib import Path
# To handle pandas data frames
import pandas as pd
# sklearn: SVM Model
from sklearn import svm
from sklearn.model_selection import GridSearchCV

# Import internal packages/ classes
# Import the src-path to sys path that the internal modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
# To handle the Logging for all modules in the same way
from utils.own_logging import OwnLogging, log_overview_data_frame
# To handle csv files
from utils.csv_operations import load_data,  save_data
# To plot data with bokeh
from utils.plot_data import PlotMultipleLayers, PlotMultipleFigures, figure_vbar, figure_hist, figure_hist_as_layers, figure_time_series_data_as_layers

#########################################################

# Initialize the logger
__own_logger = OwnLogging(Path(__file__).stem).logger

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
    plot = PlotMultipleFigures(os.path.join("output/circles-detection",file_name), file_title)

    # Join the filepaths for the data
    data_raw_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
    data_modeling_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "modeling")

    __own_logger.info("Path of the raw input data: %s", data_raw_path)
    __own_logger.info("Path of the modeling input data: %s", data_modeling_path)

    # Get csv file, which was created during data collection and adapted during data analysis as DataFrame
    metadata = load_data(data_raw_path, 'training_videos_with_metadata.csv')

    # Get the Training Data
    # Some variable initializations
    data_training_arr = []
    # Iterate over all data where selected as training data (tagged in metadata column 'usage' with 'train')
    for video_idx in metadata.index[metadata.usage == 'train']:
       # The filename of the video contains also a number, but starting from 1
       video_name_num = video_idx + 1
       train_data_file_name = "features_{}.csv".format(video_name_num)
       __own_logger.info("Training data detected: %s", train_data_file_name)
       # Get the data related to the specific video
       data_specific_video = load_data(data_modeling_path, train_data_file_name)
       # Merge all data in one array
       data_training_arr.append(data_specific_video)

    # Concatenate the data in one frame by simply chain together the time series rows, but ignore the index of the rows to add so that we generate a continuous increasing index
    data_training = pd.concat(data_training_arr, ignore_index=True)
    log_overview_data_frame(__own_logger, data_training)

    # Handling missing data (frames with no detected landmarks): Backward filling (take the next observation and fill bachward)
    __own_logger.info("Detected missing data: %s", data_training.isna().sum())
    data_training = data_training.fillna(method='bfill')

    # Visualize the training data
    # Create dict for visualization data
    dict_visualization_data = {
        "label": data_training.columns.values, # Take all columns for visualization in dataframe
        "value": [data_training[data_training.columns.values][col] for col in data_training[data_training.columns.values]],
        # As x_data generate a consecutive number: a frame number for the whole merged time series, so the index + 1 can be used
        "x_data": data_training.index + 1
    }
    # Create a Line-Circle Chart
    figure_training_data = figure_time_series_data_as_layers(__own_logger, "Trainingsdaten: Positionen der Füße", "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Frame")
    # Append the figure to the plot
    plot.appendFigure(figure_training_data.getFigure())

    # # Hyperparameter Tuning with Grid Search
    # param_grid = {  'kernel':('linear', 'poly', 'rbf', 'sigmoid'),
    #                 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    #                 'gamma':[0.0001, 0.001, 0.01, 1, 10, 100, 1000]}
    # grid_search = GridSearchCV(svm.SVC(), 
    #                        param_grid, 
    #                        cv=3, 
    #                        n_jobs=-1)
    # grid_search.fit(data_training.drop('circles_running', axis=1), data_training.circles_running)
    # __own_logger.info("Best parameters are {} \nScore : {}%".format(grid_search.best_params_, grid_search.best_score_*100))
    # # Create a svm Classifier
    # clf = svm.SVC(kernel=grid_search.best_params_['kernel'], C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'])

    # If the optimal hyperparameter are already known: Create a svm Classifier
    clf = svm.SVC(kernel='rbf', C=1000, gamma=1)

    #Train the model using the training sets
    clf.fit(data_training.drop('circles_running', axis=1), data_training.circles_running)

    # Get the Test Data
    # Some variable initializations
    data_test_arr = []
    # Iterate over all data where selected as test data (tagged in metadata column 'usage' with 'test')
    for video_idx in metadata.index[metadata.usage == 'test']:
       # The filename of the video contains also a number, but starting from 1
       video_name_num = video_idx + 1
       test_data_file_name = "features_{}.csv".format(video_name_num)
       __own_logger.info("Testing data detected: %s", test_data_file_name)
       # Get the data related to the specific video
       data_specific_video = load_data(data_modeling_path, test_data_file_name)
       # Merge all data in one array
       data_test_arr.append(data_specific_video)

    # Concatenate the data in one frame by simply chain together the time series rows, but ignore the index of the rows to add so that we generate a continuous increasing index
    data_test = pd.concat(data_test_arr, ignore_index=True)
    log_overview_data_frame(__own_logger, data_test)

    # Handling missing data (frames with no detected landmarks): Backward filling (take the next observation and fill bachward)
    __own_logger.info("Detected missing data: %s", data_test.isna().sum())
    data_test = data_test.fillna(method='bfill')

    # Visualize the test data
    # Create dict for visualization data
    dict_visualization_data = {
        "label": data_test.columns.values, # Take all columns for visualization in dataframe
        "value": [data_test[data_test.columns.values][col] for col in data_test[data_test.columns.values]],
        # As x_data generate a consecutive number: a frame number for the whole merged time series, so the index + 1 can be used
        "x_data": data_test.index + 1
    }
    # Create a Line-Circle Chart
    figure_test_data = figure_time_series_data_as_layers(__own_logger, "Testdaten: Positionen der Füße", "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Frame")
    # Append the figure to the plot
    plot.appendFigure(figure_test_data.getFigure())

    # Predict the target value for the whole test data
    y_pred = clf.predict(data_test.drop('circles_running', axis=1))
    data_test['prediction'] = y_pred

    # Visualize the prediction for the test data input
    # Create dict for visualization data
    dict_visualization_data = {
        "label": [data_test.circles_running.name, data_test.prediction.name],
        "value": [data_test.circles_running.values, data_test.prediction.values],
        # As x_data generate a consecutive number: a frame number for the whole merged time series, so the index + 1 can be used
        "x_data": data_test.index + 1
    }
    # Create a Line-Circle Chart
    figure_test_data = figure_time_series_data_as_layers(__own_logger, "Testdaten: Vorhersage der Kreisflanken", "Kreisflanken detektiert", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Frame")
    # Append the figure to the plot
    plot.appendFigure(figure_test_data.getFigure())

    # Iterate over the single videos and using the single testdata for video-specific evalution
    for idx, data_test_single_arr in enumerate(data_test_arr):
        data_test_single = pd.DataFrame(data_test_single_arr)
        # Handling missing data (frames with no detected landmarks): Backward filling (take the next observation and fill bachward)
        data_test_single = data_test_single.fillna(method='bfill')
        # For missing data at the end, the bfill mechanism not work, so do now a ffill
        data_test_single = data_test_single.fillna(method='ffill')
        # Predict the target value for the whole test data
        y_pred_single = clf.predict(data_test_single.drop('circles_running', axis=1))
        # Add prediciton to test data
        data_test_single['prediction'] = y_pred_single
        # Get video num
        video_num = metadata.index[metadata.usage == 'test'][idx] + 1
        # Save to CSV
        save_data(data_test_single, data_modeling_path, "{0}_{1}.csv".format('data_test',video_num))
        # Create a Line-Circle Chart
        figure_test_data_single = figure_time_series_data_as_layers(__own_logger, "Testdaten Video {}: Vorhersage der Kreisflanken".format(video_num), "Kreisflanken detektiert", list(range(1, y_pred_single.size + 1)), dict_visualization_data.get('label'), [data_test_single.circles_running, y_pred_single], "Frame")
        # Append the figure to the plot
        plot.appendFigure(figure_test_data_single.getFigure())
        
    # Show the plot in responsive layout, but only stretch the width
    plot.showPlotResponsive('stretch_width')

    # Save the Data to CSV
    # Training Data
    save_data(data_training, data_modeling_path, "data_train.csv")
    # Testing Data
    save_data(data_test, data_modeling_path, "data_test.csv")
