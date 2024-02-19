# Script for the Training of a Baseline Model

# Import the external packages
# Operating system functionalities
import sys
import os
from pathlib import Path
# To handle pandas data frames
import pandas as pd
# sklearn
from sklearn import svm

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

    # Join the filepaths for the data
    data_raw_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
    data_modeling_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "modeling")

    __own_logger.info("Path of the raw input data: %s", data_raw_path)
    __own_logger.info("Path of the modeling input data: %s", data_modeling_path)

    # Get csv file, which was created during data collection and adapted during data analysis as DataFrame
    metadata = load_data(data_raw_path, 'training_videos_with_metadata.csv')

    # Get the Training Data
    # Some variable initializations
    input_features = []
    output_target = []
    # Iterate over all data where selected as training data (tagged in metadata column 'usage' with 'train')
    for video_idx in metadata.index[metadata.usage == 'train']:
       # The filename of the video contains also a number, but starting from 1
       video_name_num = video_idx + 1
       train_data_file_name = "features_{}.csv".format(video_name_num)
       __own_logger.info("Training data detected: %s", train_data_file_name)
       # Get the data related to the specific video
       data_specific_video = load_data(data_modeling_path, train_data_file_name)
       # Merge all data in on variable
       # TODO