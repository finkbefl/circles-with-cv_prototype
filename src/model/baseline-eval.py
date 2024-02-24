# Script for the Evaluation of the Baseline Model

# Import the external packages
# Operating system functionalities
import sys
import os
from pathlib import Path
# To handle pandas data frames
import pandas as pd
# sklearn: Metrics
from sklearn import metrics

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
    file_name = "baseline-eval.html"
    file_title = "Evaluation of the the baseline model"
    __own_logger.info("Plot %s as multiple figures to file %s", file_title, file_name)
    plot = PlotMultipleFigures(os.path.join("output/circles-detection",file_name), file_title)

    # Join the filepaths for the data
    data_raw_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
    data_modeling_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "modeling")

    __own_logger.info("Path of the raw input data: %s", data_raw_path)
    __own_logger.info("Path of the modeling input data: %s", data_modeling_path)

    # Overall evaluation

    # Get the test data
    data_test = load_data(data_modeling_path, "data_test.csv")

    # Evaluation
    # Some variable initializations
    evaluation = []
    # Model Accuracy
    accuracy = metrics.accuracy_score(data_test.circles_running, data_test.prediction)
    __own_logger.info("Accuracy: %s",accuracy)

    # Model Precision
    precision = metrics.precision_score(data_test.circles_running, data_test.prediction)
    __own_logger.info("Precision: %s",precision)

    # Model Recall
    recall = metrics.recall_score(data_test.circles_running, data_test.prediction)
    __own_logger.info("Recall: %s",recall)

    # Visualize the metrics
    # Create dict for visualization data
    dict_visualization_data = {
        "label": ["accuracy", "precision", "recall"],
        "value": [accuracy, precision, recall]
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
        accuracy = metrics.accuracy_score(data_specific_video.circles_running, data_specific_video.prediction)
        __own_logger.info("Testdaten Video %d: Accuracy: %s",video_name_num, accuracy)
        precision = metrics.precision_score(data_specific_video.circles_running, data_specific_video.prediction)
        __own_logger.info("Testdaten Video %d: Precision: %s",video_name_num, precision)
        recall = metrics.recall_score(data_specific_video.circles_running, data_specific_video.prediction)
        __own_logger.info("Testdaten Video %d: Recall: %s",video_name_num, recall)
        # Create a bar chart
        figure_evaluation_single = figure_vbar(__own_logger, "Testdaten Video {}: Evaluierung".format(video_name_num), "Wert der Metrik", dict_visualization_data.get('label'), [accuracy, precision, recall], set_x_range=True, color_sequencing=False)
        # Append the figure to the plot
        plot.appendFigure(figure_evaluation_single.getFigure())



    # Show the plot in responsive layout, but only stretch the width
    plot.showPlotResponsive('stretch_width')


