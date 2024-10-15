# A script to compare the pose estimation with different resolutions

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
# Time
import time
# Numpy
import numpy as np
# For evaluation metrics
from sklearn import metrics

import numpy

# Import internal packages/ classes
# Import the src-path to sys path that the internal modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
# To handle the Logging for all modules in the same way
from utils.own_logging import OwnLogging, log_overview_data_frame
# To handle csv files
from utils.csv_operations import save_data, convert_series_into_date
# To plot data with bokeh
from utils.plot_data import PlotMultipleLayers, PlotMultipleFigures, figure_vbar, figure_hist, figure_time_series_data_as_layers

#########################################################

# Initialize the logger
__own_logger = OwnLogging(Path(__file__).stem).logger

# Do you want to compare an other res with the origninal res or only analyze one res in detail?
mode_compare_with_orig_res = True

#########################################################
#########################################################
#########################################################

# When this script is called directly...
if __name__ == "__main__":
    # ...then calling the functions

    __own_logger.info("########## START ##########")

    # Join the filepath of the raw data file
    video_filename = "circle_mushroom_fromSide.mp4"
    #video_filename = "circle_mushroom_fromTop.mp4"
    video_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "PoC", video_filename)

    __own_logger.info("Input video: %s", video_path)

    # Initialize MediaPipe Pose and Drawing utilities
    __own_logger.info("########## Init mediapipe ##########")
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    # Open the video file
    __own_logger.info("########## Open the video file ##########")
    cap = cv2.VideoCapture(video_path)

    # Some variable initializations
    features = pd.DataFrame()
    frame_number = 1
    missing_values_num = 0
    csv_data = []
    framerate = 0
    nose_x = []
    nose_y = []
    left_foot_x = []
    left_foot_y = []
    right_foot_x = []
    right_foot_y = []
    nose_x_orig = []
    nose_y_orig = []
    left_foot_x_orig = []
    left_foot_y_orig = []
    right_foot_x_orig = []
    right_foot_y_orig = []
    timestamp = []
    frame_rate = []
    pose_duration = []

    __own_logger.info("########## Iterate over every single frame ##########")
    while cap.isOpened():
        start_time = time.time() # start time of the loop

        # Read a frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Save the frame in the orig res
        frame_orig = frame

        # Rezize the frame to the resolution width of 300 px, keep the aspect ratio
        frame = cv2.resize(frame, (300, 169))
        #frame = cv2.resize(frame, (150, 85))

        start_time_pose = time.time()
        # Process the frame with MediaPipe Pose
        result = pose.process(frame)
        # Calc the time needed for pose estimation
        pose_duration.append(time.time() - start_time_pose)

        if mode_compare_with_orig_res:
            result_orig = pose.process(frame_orig)

        # Draw the pose landmarks on the frame
        if result.pose_landmarks:
            __own_logger.info("Landmarks detected")
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Get the landpark pisitions (the points are normalized to the width and the heigth of the image)
            nose_x.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x)
            nose_y.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y)
            left_foot_x.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x)
            left_foot_y.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y)
            right_foot_x.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x)
            right_foot_y.append(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y)
        else:
            # If no landmarks are detected: Set positions to Not a Number
            nose_x.append(np.NaN)
            nose_y.append(np.NaN)
            left_foot_x.append(np.NaN)
            left_foot_y.append(np.NaN)
            right_foot_x.append(np.NaN)
            right_foot_y.append(np.NaN)
            missing_values_num += 1

        if mode_compare_with_orig_res:
            if result_orig.pose_landmarks:
                # Get the landpark pisitions (the points are normalized to the width and the heigth of the image)
                nose_x_orig.append(result_orig.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x)
                nose_y_orig.append(result_orig.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y)
                left_foot_x_orig.append(result_orig.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x)
                left_foot_y_orig.append(result_orig.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y)
                right_foot_x_orig.append(result_orig.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x)
                right_foot_y_orig.append(result_orig.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y)
            else:
                # If no landmarks are detected: Set positions to Not a Number
                nose_x_orig.append(np.NaN)
                nose_y_orig.append(np.NaN)
                left_foot_x_orig.append(np.NaN)
                left_foot_y_orig.append(np.NaN)
                right_foot_x_orig.append(np.NaN)
                right_foot_y_orig.append(np.NaN)

        # Save timestamp of frame (in milli seconds)
        timestamp.append(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Get the frame rate of the source video
        fps =  cap.get(cv2.CAP_PROP_FPS)
        #cv2.putText(frame,f'Framerate (source): {fps:.1f} FPS',(0,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
        # Put calculated frame rate for the generated video as text in image
        #cv2.putText(frame,f'Framerate (generated): {framerate:.1f} FPS',(0,40),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)   

        # Display the frame
        cv2.imshow(f'Pose Detection with MediaPipe', frame)

        # Define, how long the image should be shown or how long to wait until processing is continued
        # To get the video in original speed, this depends on the fps of the original video
        # For 25 fps the delay should approximately 1/25fps = 40 ms
        # But only in theory: Since the OS has a minimum time between switching threads, the function will not wait exactly delay ms, it will wait at least delay ms, depending on what else is running on your computer at that time.
        # The processing time, i.e. the runtime of a loop pass, also plays a role
        # the delay must therefore be selected smaller in order to achieve the original fps
        #cv2.waitKey(5)
        # We can define a delay of zero, then the program will wait for keyboard input (for analyzing frame to frame, long press is possible  )
        #cv2.waitKey(0)

         # Press 'q' to quit
        #key = cv2.waitKey(0 ) & 0xFF
        key = cv2.waitKey(1)
        # if the `q` key was pressed, break from the loop
        if key == ord('q'):
            break

        # Calculate the frame rate of the generated video via showing the frames
        framerate = 1.0 / (time.time() - start_time)
        __own_logger.info("Calculated frame rate for the generated video: %s", framerate)
        frame_rate.append(framerate)

        # Go to the next frame
        frame_number += 1

    if mode_compare_with_orig_res:
        # Add the collected features to the DataFrame
        features['right_foot_x_pos_300x169'] = right_foot_x
        #features['right_foot_y_pos'] = right_foot_y
        #features['left_foot_x_pos'] = left_foot_x
        #features['left_foot_y_pos'] = left_foot_y
        features['nose_x_pos_300x169'] = nose_x
        #features['nose_y_pos'] = nose_y
        features['right_foot_x_pos_1920x1080'] = right_foot_x_orig
        #features['right_foot_y_po_origs'] = right_foot_y_orig
        #features['left_foot_x_pos_orig'] = left_foot_x_orig
        #features['left_foot_y_pos_orig'] = left_foot_y_orig
        features['nose_x_pos_1920x1080'] = nose_x_orig
    else:
        # Add the collected features to the DataFrame
        features['right_foot_x_pos'] = right_foot_x
        #features['right_foot_y_pos'] = right_foot_y
        #features['left_foot_x_pos'] = left_foot_x
        #features['left_foot_y_pos'] = left_foot_y
        features['nose_x_pos'] = nose_x
        #features['nose_y_pos'] = nose_y
    # Add the timestamps to to the features DataFrame
    features['timestamp'] = timestamp

    # Visualize the features
    # Create dict for visualization data
    dict_visualization_data = {
        "label": features.columns.drop('timestamp').values, # Take all columns for visualization in dataframe, except timestamp
        "value": [features[features.columns.drop('timestamp').values][col] for col in features[features.columns.drop('timestamp').values]],
        # As x_data take the timestamp (formatted as datetime), as the data frame rows are no longer representing the frame num (if no landmarks are detected in frames, then there are no entries as rows for these frames)
        #"x_data": [str(i) for i in features.index.values + 1]
        "x_data": convert_series_into_date(features['timestamp'], unit='ms')
    }
    # Create a Line-Circle Chart
    if mode_compare_with_orig_res:
        figure_landmarks_time_series = figure_time_series_data_as_layers(__own_logger, "Vergleich der Posenschätzung mit unterschiedlichen Auflösungen", "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Laufzeit des Videos", x_axis_type='datetime')
    else:
        figure_landmarks_time_series = figure_time_series_data_as_layers(__own_logger, "Positionen ausgewählter Körpermerkmale", "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Laufzeit des Videos", x_axis_type='datetime')

    # Create the plot with the created figures
    file_name = "pose-estimation.html"
    file_title = "Spike: Pose Detection with MediaPipe"
    __own_logger.info("Plot bar chart with title %s as multiple figures to file %s", file_title, file_name)
    plot = PlotMultipleFigures(os.path.join("output/spike",file_name), file_title)
    plot.appendFigure(figure_landmarks_time_series.getFigure())
    # Show the plot in responsive layout, but only stretch the width
    plot.showPlotResponsive('stretch_width')

    # Print the average framerate of the source video
    print("Source video: Average framerate = " + str(fps) + "fps")
    __own_logger.info("Source video: Average framerate = %s fps", fps)
    # Print the average framerate of the resulting video
    print("Generated video: Average framerate = " + str(np.mean(frame_rate)) + "fps")
    __own_logger.info("Generated video: Average framerate = %s fps", np.mean(frame_rate))

    # Print average time needed for pose estimation
    print("Average time needed for pose estimation = " + str(np.mean(pose_duration)) + "s")
    __own_logger.info("Average time needed for pose estimation = %s s", np.mean(pose_duration))

    # Pint number of frames with missing values in generated time serie data
    print("Number of frames with no landmarks detected = " + str(missing_values_num))
    __own_logger.info("Number of frames with no landmarks detected = %s", missing_values_num)

    if mode_compare_with_orig_res:
        # Calc the metrics
        # but at first, remove frames with NaN values
        compare_features = pd.DataFrame()
        compare_features['right_foot_x_pos_1920x1080'] = features['right_foot_x_pos_1920x1080']
        compare_features['nose_x_pos_1920x1080'] = features['nose_x_pos_1920x1080']
        compare_features['right_foot_x_pos_300x169'] = features['right_foot_x_pos_300x169']
        compare_features['nose_x_pos_300x169'] = features['nose_x_pos_300x169']
        compare_features_wo_nan = compare_features.dropna()
        # MSE
        MSE = metrics.mean_squared_error(compare_features_wo_nan['right_foot_x_pos_300x169'], compare_features_wo_nan['right_foot_x_pos_1920x1080'])
        __own_logger.info("right_foot_x_pos MSE: %s",MSE)
        MSE = metrics.mean_squared_error(compare_features_wo_nan['nose_x_pos_300x169'], compare_features_wo_nan['nose_x_pos_1920x1080'])
        __own_logger.info("nose_x_pos MSE: %s",MSE)
        # MAE
        MAE = metrics.mean_absolute_error(compare_features_wo_nan['right_foot_x_pos_300x169'], compare_features_wo_nan['right_foot_x_pos_1920x1080'])
        __own_logger.info("right_foot_x_pos MAE: %s", MAE)
        MAE = metrics.mean_absolute_error(compare_features_wo_nan['nose_x_pos_300x169'], compare_features_wo_nan['nose_x_pos_1920x1080'])
        __own_logger.info("nose_x_pos MAE: %s", MAE)
        # RMSE
        RMSE = np.sqrt(metrics.mean_squared_error(compare_features_wo_nan['right_foot_x_pos_300x169'], compare_features_wo_nan['right_foot_x_pos_1920x1080']))
        __own_logger.info("right_foot_x_pos RMSE: %s", RMSE)
        RMSE = np.sqrt(metrics.mean_squared_error(compare_features_wo_nan['nose_x_pos_300x169'], compare_features_wo_nan['nose_x_pos_1920x1080']))
        __own_logger.info("nose_x_pos RMSE: %s", RMSE)
        # MAPE
        MAPE = metrics.mean_absolute_percentage_error(compare_features_wo_nan['right_foot_x_pos_300x169'], compare_features_wo_nan['right_foot_x_pos_1920x1080'])
        __own_logger.info("right_foot_x_pos MAPE: %s", MAPE)
        MAPE = metrics.mean_absolute_percentage_error(compare_features_wo_nan['nose_x_pos_300x169'], compare_features_wo_nan['nose_x_pos_1920x1080'])
        __own_logger.info("nose_x_pos MAPE: %s", MAPE)

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()