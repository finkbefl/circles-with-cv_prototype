# Script for the Data Featurization

# Import the external packages
# Operating system functionalities
import sys
import os
from pathlib import Path
# To handle pandas data frames
import pandas as pd
# OpenCV
import cv2
# Mediapipe
import mediapipe as mp
# Regular Expressions
import re
# Multimedia: Manipulate and get infos
import ffmpeg
# Numpy
import numpy as np
# Fractions
from fractions import Fraction

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
    data_processed_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
    data_modeling_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "modeling")

    __own_logger.info("Path of the raw input data: %s", data_raw_path)
    __own_logger.info("Path of the processed input data: %s", data_processed_path)
    __own_logger.info("Path of the modeling output data: %s", data_modeling_path)

    # Get csv file, which was created during data collection and adapted during data analysis as DataFrame
    metadata = load_data(data_raw_path, 'training_videos_with_metadata.csv')

    # Create a plot for multiple figures
    file_name = "data-featurization.html"
    file_title = "Feature Engineering"
    __own_logger.info("Plot %s as multiple figures to file %s", file_title, file_name)
    plot = PlotMultipleFigures(os.path.join("output/circles-detection",file_name), file_title)

    # Iterate over all videos in the directory (sorted by number correctly)
    for file in sorted(os.listdir(data_processed_path), key=lambda s: [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]):
        # Decode the specified filename from the filesystem encoding
        filename = os.fsdecode(file)
        # Get the full file path
        file_input_path = os.path.join(data_processed_path, filename)
        # skip the item in the directory if it is not a file (subfolders)
        if not os.path.isfile(file_input_path):
            continue
        # Have a look on the file suffix
        file_path_pure = Path(file_input_path)
        # Rename videos with suffix MOV to mov otherwise opencv can not handle it
        if file_path_pure.suffix == '.MOV':
            file_path_pure.rename(file_path_pure.with_suffix('.mov'))
            file_path = file_path_pure.as_posix()
        # If the file does not have the suffix mov or mp4, then ignore it
        elif not (file_path_pure.suffix == '.mov' or file_path_pure.suffix == '.mp4'):
            continue

        __own_logger.info("Detected input video file: %s", filename)

        # Initialize MediaPipe Pose and Drawing utilities
        __own_logger.info("########## Init mediapipe ##########")
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose()

        # Open the video file
        __own_logger.info("########## Open the video file ##########")
        cap = cv2.VideoCapture(file_input_path)
 
        # Iterate over every single frame
        __own_logger.info("########## Iterate over every single frame ##########")
        # Some variable initializations
        features = pd.DataFrame()
        # foot positions as time series
        right_foot_x_pos = []
        right_foot_y_pos = []
        left_foot_x_pos = []
        left_foot_y_pos = []
        timestamp = []
        while cap.isOpened():
            # Read a frame
            ret, frame = cap.read()

            # If the frame was not read successfully, break the loop
            if not ret:
                break

            # Process the frame with MediaPipe Pose
            result = pose.process(frame)

            # Draw the pose landmarks on the frame
            if result.pose_landmarks:
                #__own_logger.info("Landmarks detected")
                #mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Extract foot positions (normalized to the width and the heigth of the image) and add it to list
                right_foot_x_pos.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x)
                right_foot_y_pos.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y)
                left_foot_x_pos.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x)
                left_foot_y_pos.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y)
                # Save timestamp of frame (in milli seconds)
                timestamp.append(cap.get(cv2.CAP_PROP_POS_MSEC))

            # # Display the frame
            # cv2.imshow(f'Data Featurization', frame)

            # # Press 'q' to quit
            # key = cv2.waitKey(0 ) & 0xFF
            # # if the `q` key was pressed, break from the loop
            # if key == ord('q'):
            #     break

        # Add the collected features to the DataFrame
        features['right_foot_x_pos'] = right_foot_x_pos
        features['right_foot_y_pos'] = right_foot_y_pos
        features['left_foot_x_pos'] = left_foot_x_pos
        features['left_foot_y_pos'] = left_foot_y_pos

        # Release everything if job is finished
        cap.release()
        cv2.destroyAllWindows()

        # Add the timestamps to to the features DataFrame
        features['timestamp'] = timestamp

        # Create the target variable: Set it to 1 when the circle is running, otherwise 0 
        # Add a column with zeros as preinitialization
        features['circles_running'] = 0
        # Extract the video index from the video file name (video with name 1 is row 0 of metadata)
        video_idx = int(str(Path(filename).with_suffix(''))) - 1
        # Extract the list of manual detected start- and end-points of circles
        start_list = [val.split('-') for val in metadata.manual_circle_start_ms][video_idx]
        end_list = [val.split('-') for val in metadata.manual_circle_end_ms][video_idx]
        # Iterate over the start-points
        for time_idx, start_time in enumerate(start_list):
            # get the correspon dig end-point
            end_time = end_list[time_idx]
            # set the target value to 1 when circles are manually detected
            features['circles_running'] = np.where((features['timestamp'] > float(start_time)) & (features['timestamp'] < float(end_time)), 1, features['circles_running'])

        # Visualize the features
        # Create dict for visualization data
        dict_visualization_data = {
            "label": features.columns.drop('timestamp').values, # Take all columns for visualization in dataframe, except timestamp
            "value": [features[features.columns.drop('timestamp').values][col] for col in features[features.columns.drop('timestamp').values]],
            # As x_range take the indizes of the dataframe rows, but starting with 1
            "x_range": [str(i) for i in features.index.values + 1]
        }
        # Create a Line-Circle Chart
        figure_features = figure_time_series_data_as_layers(__own_logger, "{}: Positionen der Füße".format(filename), "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_range'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Nummer des Frames")
        # Append the figure to the plot
        plot.appendFigure(figure_features.getFigure())

        # Save the Data to CSV: Add the filename of the video without suffix to the csv filename
        filename_as_path = Path(filename)
        save_data(features, data_modeling_path, "{0}_{1}{2}".format('features',filename_as_path.with_suffix(''), '.csv'))

        # For testing purposes: Break the loop after processing the first video
        #break
    
    # Show the plot in responsive layout, but only stretch the width
    plot.showPlotResponsive('stretch_width')

