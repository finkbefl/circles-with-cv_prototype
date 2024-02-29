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
from sklearn.model_selection import GridSearchCV, learning_curve
# Numpy
import numpy as np
# Serialization of the trained model
from joblib import dump

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
       try:
            data_specific_video = load_data(data_modeling_path, train_data_file_name)
       except FileNotFoundError as error:
           __own_logger.error("########## Error when trying to access training data ##########", exc_info=error)
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

    # # Hyperparameter Tuning with Grid Search: But restrict to rbf kernel
    # # For value of gamma: For 'scale' it uses 1 / (n_features * X.var()), for 'auto' 1 / n_features
    # tuning_parameters = [{'kernel': ['rbf'], 'gamma': ['scale','auto',1e-2,1e-3, 1e-4], 'C': [1, 10, 100,1000]}]
    # #Define the model to be svm.SVC, specify the parameters space and scoring method
    # clf_gridsearch=GridSearchCV(svm.SVC(),tuning_parameters,scoring='accuracy')
    # # Use the training data to find the best params
    # clf_gridsearch.fit(data_training.drop('circles_running', axis=1), data_training.circles_running)
    # # Print out the mean scores for the different set of parameters
    # means = clf_gridsearch.cv_results_['mean_test_score']
    # __own_logger.info("Mean scores for different set of parameters: %s", means)
    # # Print the best params
    # __own_logger.info("Best parameters are: {} \n With score: {}%".format(clf_gridsearch.best_params_, clf_gridsearch.best_score_))

    # # Create a svm Classifier with the best params
    # clf_best=svm.SVC(kernel='rbf',C=clf_gridsearch.best_params_['C'],gamma=clf_gridsearch.best_params_['gamma'])

    # Create a svm Classifier with the best params: To speed up the training if the best params are already known
    clf_best=svm.SVC(kernel='rbf',C=1000,gamma='auto')

    #Testing out the CV scores is not enough to ensure the accuracy of the model. One could still run into the problem of high bias (underfitting) or high variances (overfitting). To see if this is the case, one can plot the learning curve:
    # Train size as fraction of the maximum size of the training set
    train_sizes_as_fraction = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, valid_scores = learning_curve(clf_best,data_training.drop('circles_running', axis=1),data_training.circles_running,train_sizes=train_sizes_as_fraction,cv=5)
    mean_train_scores=np.mean(train_scores,axis=1)
    mean_valid_scores=np.mean(valid_scores,axis=1)
    __own_logger.info("Mean train scores: %s", mean_train_scores)
    __own_logger.info("Mean valid scores: %s", mean_valid_scores)
    # Visualize the learning courve
    # Create dict for visualization data
    dict_visualization_data = {
        "label": ["Training score", "Cross-validation score"],
        "value": [mean_train_scores, mean_valid_scores],
        "x_data": train_sizes_as_fraction
    }
    # Create a Line-Circle Chart
    figure_learning_courve = figure_time_series_data_as_layers(__own_logger, "Lernkurve", "Score", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Training Size")
    # Append the figure to the plot
    plot.appendFigure(figure_learning_courve.getFigure())

    #Train the model using the training sets
    clf_best.fit(data_training.drop('circles_running', axis=1), data_training.circles_running)
        
    # Show the plot in responsive layout, but only stretch the width
    plot.showPlotResponsive('stretch_width')

    # Save the Data to CSV
    # Training Data
    save_data(data_training, data_modeling_path, "data_train.csv")

    # Save the trained model with joblib
    file_path = os.path.join(data_modeling_path, 'baseline-svm-model.joblib')
    dump(clf_best, file_path) 
