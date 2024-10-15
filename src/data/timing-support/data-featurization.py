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
# Numpy
import numpy as np

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

# A flag to add additional features (wrist x-pos, shoulders, nose, knee, hip positions)
# TODO: Exclude some features in visualizations because of perceptibility and maximum number of colors in the bokeh diagrams 
ADD_ADDITIONAL_FEATURES = False

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

    # Join the filepaths for the data
    data_raw_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw")
    data_processed_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "processed", "timing-support")
    data_modeling_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "modeling", "timing-support")

    __own_logger.info("Path of the raw input data: %s", data_raw_path)
    __own_logger.info("Path of the processed input data: %s", data_processed_path)
    __own_logger.info("Path of the modeling output data: %s", data_modeling_path)

    # Get csv file, which was created during data collection and adapted during data analysis as DataFrame
    metadata = load_data(data_raw_path, 'training_videos_with_metadata.csv')

    # Create a plot for multiple figures
    file_name = "data-featurization.html"
    file_title = "Feature Engineering"
    __own_logger.info("Plot %s as multiple figures to file %s", file_title, file_name)
    plot = PlotMultipleFigures(os.path.join("output/timing-support",file_name), file_title)

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

        # # Possibility to skip videos with dedicated number for testing reasons: E.g. skip all videos with number 7-9 due to bad perspective for landmark detection
        # # TODO: Must be adapted to filenames of preprocessed videos
        # if int(filename.split('.')[0]) >= 7 and int(filename.split('.')[0]) <= 9:
        #         continue

        __own_logger.info("Detected input video file: %s", filename)

        # Initialize MediaPipe Pose and Drawing utilities
        __own_logger.info("########## Init mediapipe ##########")
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose()

        # Open the video file
        __own_logger.info("########## Open the video file ##########")
        cap = cv2.VideoCapture(file_input_path)

        # Get the fps of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
 
        # Iterate over every single frame
        __own_logger.info("########## Iterate over every single frame ##########")
        # Some variable initializations
        features = pd.DataFrame()
        # foot positions as time series
        right_foot_x_pos = []
        right_foot_y_pos = []
        left_foot_x_pos = []
        left_foot_y_pos = []
        # nose position as time series
        nose_x_pos = []
        nose_y_pos = []
        # shoulder positions as time series
        right_shoulder_x_pos = []
        right_shoulder_y_pos = []
        left_shoulder_x_pos = []
        left_shoulder_y_pos = []
        # knee positions as time series
        right_knee_x_pos = []
        right_knee_y_pos = []
        left_knee_x_pos = []
        left_knee_y_pos = []
        # hip positions as time series
        right_hip_x_pos = []
        right_hip_y_pos = []
        left_hip_x_pos = []
        left_hip_y_pos = []
        # wrist positions as time series
        right_wrist_x_pos = []
        right_wrist_y_pos = []
        left_wrist_x_pos = []
        left_wrist_y_pos = []
        timestamp = []
        missing_data = []
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
                # Extract nose position
                nose_x_pos.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x)
                nose_y_pos.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y)
                # Extract shoulder positions
                right_shoulder_x_pos.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
                right_shoulder_y_pos.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
                left_shoulder_x_pos.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x)
                left_shoulder_y_pos.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
                # Extract knee positions
                right_knee_x_pos.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x)
                right_knee_y_pos.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y)
                left_knee_x_pos.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x)
                left_knee_y_pos.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y)
                # Extract hip positions
                right_hip_x_pos.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x)
                right_hip_y_pos.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y)
                left_hip_x_pos.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x)
                left_hip_y_pos.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y)
                # Extract wrist positions
                right_wrist_x_pos.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x)
                right_wrist_y_pos.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y)
                left_wrist_x_pos.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x)
                left_wrist_y_pos.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y)
                # Save timestamp of frame (in milli seconds)
                timestamp.append(cap.get(cv2.CAP_PROP_POS_MSEC))
                # Label the data row with "no missing data"
                missing_data.append(False)
            else:
                # If no landmarks are detected: Set positions to Not a Number
                right_foot_x_pos.append(np.NaN)
                right_foot_y_pos.append(np.NaN)
                left_foot_x_pos.append(np.NaN)
                left_foot_y_pos.append(np.NaN)
                nose_x_pos.append(np.NaN)
                nose_y_pos.append(np.NaN)
                right_shoulder_x_pos.append(np.NaN)
                right_shoulder_y_pos.append(np.NaN)
                left_shoulder_x_pos.append(np.NaN)
                left_shoulder_y_pos.append(np.NaN)
                right_knee_x_pos.append(np.NaN)
                right_knee_y_pos.append(np.NaN)
                left_knee_x_pos.append(np.NaN)
                left_knee_y_pos.append(np.NaN)
                right_hip_x_pos.append(np.NaN)
                right_hip_y_pos.append(np.NaN)
                left_hip_x_pos.append(np.NaN)
                left_hip_y_pos.append(np.NaN)
                right_wrist_x_pos.append(np.NaN)
                right_wrist_y_pos.append(np.NaN)
                left_wrist_x_pos.append(np.NaN)
                left_wrist_y_pos.append(np.NaN)
                # But the timestamp can be set correctly
                timestamp.append(cap.get(cv2.CAP_PROP_POS_MSEC))
                # Label the data row with "missing data"
                missing_data.append(True)

            # # Naming a window
            # window_name = "Data Featurization"
            # # cv2.WINDOW_NORMAL makes the output window resizealbe
            # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            # # Show it in fullscreen
            # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # # Display the frame
            # cv2.imshow(window_name, frame)

            # # Press 'q' to quit
            # key = cv2.waitKey(0 ) & 0xFF
            # # if the `q` key was pressed, break from the loop
            # if key == ord('q'):
            #     break

        # Add the collected features to the DataFrame
        features['missing_data'] = missing_data
        features['right_wrist_y_pos'] = right_wrist_y_pos
        features['left_wrist_y_pos'] = left_wrist_y_pos
        features['right_foot_x_pos'] = right_foot_x_pos
        features['right_foot_y_pos'] = right_foot_y_pos
        features['left_foot_x_pos'] = left_foot_x_pos
        features['left_foot_y_pos'] = left_foot_y_pos
        if ADD_ADDITIONAL_FEATURES:
            features['right_wrist_x_pos'] = right_wrist_x_pos
            features['left_wrist_x_pos'] = left_wrist_x_pos
            features['right_shoulder_x_pos'] = right_shoulder_x_pos
            features['right_shoulder_y_pos'] = right_shoulder_y_pos
            features['left_shoulder_x_pos'] = left_shoulder_x_pos
            features['left_shoulder_y_pos'] = left_shoulder_y_pos
            features['nose_x_pos'] = nose_x_pos
            features['nose_y_pos'] = nose_y_pos
            features['right_knee_x_pos'] = right_knee_x_pos
            features['right_knee_y_pos'] = right_knee_y_pos
            features['left_knee_x_pos'] = left_knee_x_pos
            features['left_knee_y_pos'] = left_knee_y_pos
            features['right_hip_x_pos'] = right_hip_x_pos
            features['right_hip_y_pos'] = right_hip_y_pos
            features['left_hip_x_pos'] = left_hip_x_pos
            features['left_hip_y_pos'] = left_hip_y_pos

        # Add the timestamps to to the features DataFrame
        features['timestamp'] = timestamp
        # Convert the timestamps (in ms) into DateTime (raise an exception when parsing is invalid) and set it as index
        features = features.set_index(convert_series_into_date(timestamp, unit='ms'))

        # Release everything if job is finished
        cap.release()
        cv2.destroyAllWindows()

        # Visualize the features
        # Create dict for visualization data
        dict_visualization_data = {
            "label": features.drop('timestamp', axis=1).columns.values, # Take all columns for visualization in dataframe, except timestamp
            "value": [features[features.drop('timestamp', axis=1).columns.values][col] for col in features[features.drop('timestamp', axis=1).columns.values]],
            # As x_data take the timestamp (formatted as datetime), as the data frame rows are no longer representing the frame num (if no landmarks are detected in frames, then there are no entries as rows for these frames)
            #"x_data": [str(i) for i in features.index.values + 1]
            "x_data": features.index
        }
        # Create a Line-Circle Chart
        figure_features = figure_time_series_data_as_layers(__own_logger, "{}: Zeitreihendaten".format(filename), "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Laufzeit des Videos", x_axis_type='datetime')
        # Append the figure to the plot
        plot.appendFigure(figure_features.getFigure())

        # Save the Data to CSV: Add the filename of the video without suffix to the csv filename
        filename_as_path = Path(filename)
        # Save the data frame as features
        save_data(features, data_modeling_path, "{0}_{1}{2}".format('features',filename_as_path.with_suffix(''), '.csv'))

        # For testing purposes: Break the loop after processing the first video
        #break
    
    # Show the plot in responsive layout, but only stretch the width
    plot.showPlotResponsive('stretch_width')

