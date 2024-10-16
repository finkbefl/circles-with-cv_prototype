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
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
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
__own_logger = OwnLogging("circles-performance_" + Path(__file__).stem).logger

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
    plot = PlotMultipleFigures(os.path.join("output/circles-performance",file_name), file_title)

    # Join the filepaths for the data
    data_raw_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw")
    data_modeling_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "modeling", "circles-performance")

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

    # Concatenate the data in one frame by simply chain together the time series rows, but ignore the index of the rows to add so that we generate a continuous increasing index
    data_training = pd.concat(data_training_arr, ignore_index=True)
    log_overview_data_frame(__own_logger, data_training)

    # Handling missing data (frames with no detected landmarks)
    __own_logger.info("Detected missing data: %s", data_training.isna().sum())
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
    figure_training_data = figure_time_series_data_as_layers(__own_logger, "Trainingsdaten", "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Frame")
    # Append the figure to the plot
    plot.appendFigure(figure_training_data.getFigure())

    # Hyperparameter Tuning with Grid Search: But restrict to rbf kernel
    # Using a pipeline: Scale the data and then propagate the data to the model (when segmentation is needed, then we need the sklearn pipeline extension from seglearn, only scaler and model would also be possible wit sklearn pipeline)
    pipeline = sgl.Pype([
        #("seg", sgl.Segment()), 
        #("features", sgl.FeatureRep()), 
        ("scaler", StandardScaler()), 
        ("svc", svm.SVC())])
    # For value of gamma: For 'scale' it uses 1 / (n_features * X.var()), for 'auto' 1 / n_features
    tuning_parameters = [{'svc__kernel': ['rbf'], 'svc__gamma': ['scale','auto',1e-2,1e-3, 1e-4], 'svc__C': [1, 10, 100,1000]}]
    #tuning_parameters = [{'svc__kernel': ['rbf'], 'svc__gamma': ['scale','auto',1e-2,1e-3, 1e-4], 'svc__C': [1, 10, 100,1000], 'seg__width': [5,10,15], 'seg__overlap': [0.2, 0.4, 0.6, 0.8]}]
    #Define the model to be svm.SVC, specify the parameters space and scoring method
    clf_gridsearch=GridSearchCV(pipeline,tuning_parameters,scoring='f1')
    # Use the training data to find the best params
    clf_gridsearch.fit(data_training.drop(['amplitude_lack', 'missing_data'], axis=1).to_numpy(), data_training.amplitude_lack.to_numpy())
    # Print out the mean scores for the different set of parameters
    means = clf_gridsearch.cv_results_['mean_test_score']
    __own_logger.info("Mean scores for different set of parameters: %s", means)
    # Print the best params
    __own_logger.info("Best parameters are: {} \n With score: {}%".format(clf_gridsearch.best_params_, clf_gridsearch.best_score_))

    # Using the best params for segmentation
    #segment_best = sgl.Segment(clf_gridsearch.best_params_['seg__width'], clf_gridsearch.best_params_['seg__overlap'])
    # Create a svm Classifier with the best params
    clf_best=svm.SVC(kernel=clf_gridsearch.best_params_['svc__kernel'],C=clf_gridsearch.best_params_['svc__C'],gamma=clf_gridsearch.best_params_['svc__gamma'])

    # Create a svm Classifier with the best params: To speed up the training if the best params are already known
    # clf_best=svm.SVC(kernel='rbf',C=1,gamma='scale')

    # Using the pipeline, which includes data scaling
    scaler = StandardScaler()
    scaler.fit(data_training.drop(['amplitude_lack', 'missing_data'], axis=1).to_numpy())
    pipeline = sgl.Pype([
        #("seg", segment_best),
        #("features", sgl.FeatureRep()), 
        ("scaler", scaler), 
        ("svc", clf_best)])

    #Testing out the CV scores is not enough to ensure the accuracy of the model. One could still run into the problem of high bias (underfitting) or high variances (overfitting). To see if this is the case, one can plot the learning curve:
    # Train size as fraction of the maximum size of the training set
    train_sizes_as_fraction = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, valid_scores = learning_curve(pipeline,data_training.drop(['amplitude_lack', 'missing_data'], axis=1).to_numpy(),data_training.amplitude_lack.to_numpy(),train_sizes=train_sizes_as_fraction,cv=5, scoring='accuracy')
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
    figure_learning_courve = figure_time_series_data_as_layers(__own_logger, "Lernkurve", "Score (Accuracy)", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Training Size", legend_location='bottom_right')
    # Append the figure to the plot
    plot.appendFigure(figure_learning_courve.getFigure())

    #Train the model using the training sets
    pipeline.fit(data_training.drop(['amplitude_lack', 'missing_data'], axis=1).to_numpy(), data_training.amplitude_lack.to_numpy())

    # Calculate the gamma value 'scale' based on the scales data
    data_training_scaled = scaler.transform(data_training.drop(['amplitude_lack', 'missing_data'], axis=1).to_numpy())
    n_features = data_training_scaled.shape[1]
    variance = data_training_scaled.var()
    gamma = 1 / (n_features * variance)
    __own_logger.info("gamma value for 'scale' calculated: n_features = %s; variance = %s; gamma = %s", n_features, variance, gamma)
    __own_logger.info("gamma value for 'scale' retrieved from trained model: gamma = %s", clf_best._gamma)

    # Visualize the normalized training data via standardscaler to see which data is piped to the model
    # Only on one column
    column_to_analyze = 'right_foot_x_pos'
    data_training_scaled_df = pd.DataFrame(data=data_training_scaled, columns=data_training.drop(['amplitude_lack', 'missing_data'], axis=1).columns.values)
    # Create dict for visualization data
    dict_visualization_data = {
        "label": [column_to_analyze],
        "value": data_training_scaled.T,
        # As x_data generate a consecutive number: a frame number for the whole merged time series, so the index + 1 can be used
        "x_data": data_training_scaled_df.index + 1
    }
    # Create a Line-Circle Chart
    figure_training_normalized_data = figure_time_series_data_as_layers(__own_logger, "{}-Trainingsdaten normalisiert auf Mittelwert 0 und Einheitsvarianz".format(column_to_analyze), "{} (normalisiert)".format(column_to_analyze), dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Frame")
    # Append the figure to the plot
    plot.appendFigure(figure_training_normalized_data.getFigure())
    # Visualize the Probability Density Function as histogram
    hist, bin_edges = np.histogram(data_training_scaled_df[column_to_analyze], density=True, bins=int((data_training_scaled_df[column_to_analyze].max()-data_training_scaled_df[column_to_analyze].min())*10))
    figure_analyze_data_distribution = figure_hist(__own_logger, "Wahrscheinlichkeitsdichte der {}-Trainingsdaten normalisiert auf Mittelwert 0 \nund Einheitsvarianz als Histogramm".format(column_to_analyze), "{} (normalisiert)".format(column_to_analyze), "Wahrscheinlichkeit [%]", bin_edges, hist)
    # Calc destcriptive statistics
    column_statistics = data_training_scaled_df[column_to_analyze].describe()
    column_std = column_statistics.loc['std']
    column_mean = column_statistics.loc['mean']
    # Visualize mean and std
    figure_analyze_data_distribution.add_vertical_line(column_mean, hist.max()*0.9)
    figure_analyze_data_distribution.add_vertical_line(column_mean - column_std, hist.max()*0.9)
    figure_analyze_data_distribution.add_vertical_line(column_mean + column_std, hist.max()*0.9)
    figure_analyze_data_distribution.add_annotation(column_mean, hist.max()*0.95, "\u03BC = {0:.2f}".format(column_mean))
    figure_analyze_data_distribution.add_annotation(column_mean - column_std, hist.max()*0.95, "-\u03C3 = {0:.2f}".format(-column_std))
    figure_analyze_data_distribution.add_annotation(column_mean + column_std, hist.max()*0.95, "\u03C3 = {0:.2f}".format(column_std))

    plot.appendFigure(figure_analyze_data_distribution.getFigure())
        
    # Show the plot in responsive layout, but only stretch the width
    plot.showPlotResponsive('stretch_width')

    # Save the Data to CSV
    # Training Data
    save_data(data_training, data_modeling_path, "data_train.csv")

    # Save the trained model with joblib
    file_path = os.path.join(data_modeling_path, 'baseline-svm-model.joblib')
    dump(pipeline, file_path) 
