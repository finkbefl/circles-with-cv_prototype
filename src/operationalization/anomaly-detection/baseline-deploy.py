# Script for the Deployment of the Baseline Model

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
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# Serialization of the trained model
from joblib import load
# Time
import time
# Numpy
import numpy as np

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
__own_logger = OwnLogging("anomaly-detection_" + Path(__file__).stem).logger

#########################################################
#########################################################
#########################################################

# When this script is called directly...
if __name__ == "__main__":
    # ...then calling the functions

    __own_logger.info("########## START ##########")

    # Create a plot for multiple figures
    file_name = "baseline-deploy.html"
    file_title = "Deployment of the the baseline model"
    __own_logger.info("Plot %s as multiple figures to file %s", file_title, file_name)
    plot = PlotMultipleFigures(os.path.join("output/anomaly-detection",file_name), file_title)

    # Join the filepaths for the data
    data_modeling_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "modeling", "anomaly-detection")
    deployment_video_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw", "deployment")

    __own_logger.info("Path of the modeling input data: %s", data_modeling_path)

    # Get the trained svm classifier via joblib
    file_path = os.path.join(data_modeling_path, 'baseline-svm-model.joblib')
    clf = load(file_path)

    # Join the filepath of the raw data file
    file_input_path = os.path.join(deployment_video_path, "deployment_fromSide.mov")
    #file_input_path = os.path.join(deployment_video_path, "deployment_fromTop.mp4")
    #file_input_path = os.path.join(deployment_video_path, "deployment_fromSide_otherDirection.MOV")
    __own_logger.info("Input video: %s", file_input_path)

    # Initialize MediaPipe Pose and Drawing utilities
    __own_logger.info("########## Init mediapipe ##########")
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    # Open the video file
    __own_logger.info("########## Open the video file ##########")
    cap = cv2.VideoCapture(file_input_path)

    # Resizing the video for faster processing
    # We are resizing to a given width while maintaining the aspect ratio
    width_requested = 300
    # Get original resolution
    width_orig  = cap.get(3)
    height_orig = cap.get(4)
    # Calculate the ratio of the width and construct the dimensions/ requested resolution
    width_ration = width_requested / float(width_orig)
    resolution_requested = (width_requested, int(height_orig * width_ration))

    # Iterate over every single frame
    # Some variable initializations
    framerate = -1
    __own_logger.info("########## Iterate over every single frame ##########")
    print("Analyzing video...")
    # Some variable initializations
    data = pd.DataFrame()
    # foot positions as time series
    right_foot_x_pos_arr = []
    right_foot_y_pos_arr = []
    left_foot_x_pos_arr = []
    left_foot_y_pos_arr = []
    nose_x_pos_arr = []
    nose_y_pos_arr = []
    y_pred_single_arr = []
    missing_data_arr = []
    # Start time of video processing 
    start_time_video_processing = time.time()
    while cap.isOpened():
        start_time = time.time() # start time of the loop

        # Read a frame
        ret, frame = cap.read()

        # If the frame was not read successfully, break the loop
        if not ret:
            break
        
        # Resize the image (Interpolation to shrink an image: INTER_CUBIC is slow, INTER_LINEAR is fast)
        #frame = cv2.resize(frame, resolution_requested, interpolation=cv2.INTER_CUBIC)

        # Process the frame with MediaPipe Pose
        result = pose.process(frame)

        # If landmarks are detected
        if result.pose_landmarks:
            #__own_logger.info("Landmarks detected")

            # Draw the relevant pose landmarks on the frame
            # mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract footpositions (normalized to the width and the heigth of the image)
            right_foot_x_pos = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x
            right_foot_y_pos = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y
            left_foot_x_pos = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x
            left_foot_y_pos = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y
            # Extract nose position
            nose_x_pos = result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x
            nose_y_pos = result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y

            # Predict the target value: Returns -1 for outliers and 1 for inliers.
            y_pred_single = (clf.predict(pd.DataFrame({'right_foot_x_pos':right_foot_x_pos, 'right_foot_y_pos':right_foot_y_pos, 'left_foot_x_pos':left_foot_x_pos, 'left_foot_y_pos':left_foot_y_pos, 'nose_x_pos':nose_x_pos, 'nose_y_pos':nose_y_pos}, index=[0]).to_numpy())==-1)
            #print(y_pred_single)

            # Add the values to the lists
            right_foot_x_pos_arr.append(right_foot_x_pos)
            right_foot_y_pos_arr.append(right_foot_y_pos)
            left_foot_x_pos_arr.append(left_foot_x_pos)
            left_foot_y_pos_arr.append(left_foot_y_pos)
            nose_x_pos_arr.append(nose_x_pos)
            nose_y_pos_arr.append(nose_y_pos)
            y_pred_single_arr.append(y_pred_single)
            # Label the data row with "no missing data"
            missing_data_arr.append(False)

            # # Bigger size of the image for better visualization
            # frame = cv2.resize(frame, (1920,1080), interpolation=cv2.INTER_CUBIC)

            # # Signaling the prediction
            # if y_pred_single[0]:
            #     # Green dot when the frame is predicted as anomaly
            #     frame = cv2.circle(frame, (1820,80), radius=50, color=(0, 255, 0), thickness=-1)
            # else:
            #     # red if not
            #     frame = cv2.circle(frame, (1820,80), radius=50, color=(0, 0, 255), thickness=-1)
        else:
            # # If no landmarks detected
            # # Bigger size of the image for better visualization
            # frame = cv2.resize(frame, (1920,1080), interpolation=cv2.INTER_CUBIC)
            # # Dont change the Signaling for the detection of circles, so if no landmarks are detected, no change of the last status will be "predicted"

            # If no landmarks are detected: Set positions to Not a Number
            right_foot_x_pos_arr.append(np.NaN)
            right_foot_y_pos_arr.append(np.NaN)
            left_foot_x_pos_arr.append(np.NaN)
            left_foot_y_pos_arr.append(np.NaN)
            nose_x_pos_arr.append(np.NaN)
            nose_y_pos_arr.append(np.NaN)
            # Label the data row with "missing data"
            missing_data_arr.append(True)
            # And set the target variable to "no anomaly detected"
            y_pred_single_arr.append(False)

        # # Naming a window
        # window_name = "Data Featurization"
        # # cv2.WINDOW_NORMAL makes the output window resizealbe
        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # # Show it in fullscreen
        # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # # Add some text
        # cv2.putText(frame,"Press 'q' to quit",(0,40),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
        # cv2.putText(frame,"Anomaly:",(1500,80),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

        # # Get the frame rate of the source video
        # fps =  cap.get(cv2.CAP_PROP_FPS)
        # cv2.putText(frame,f'Framerate (source): {fps:.1f} FPS',(0,1000),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),1)
        # # Put calculated frame rate for the generated video as text in image
        # cv2.putText(frame,f'Framerate (generated): {framerate:.1f} FPS',(0,1040),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),1)   

        # # Display the frame
        # cv2.imshow(window_name, frame)

        # # Press 'q' to quit
        # key = cv2.waitKey(0 ) & 0xFF
        # # if the `q` key was pressed, break from the loop
        # if key == ord('q'):
        #     break

        # # Calculate the frame rate of the generated video via showing the frames
        # framerate = 1.0 / (time.time() - start_time)
            
    # Print the information how long the processing tooks
    fps = cap.get(cv2.CAP_PROP_FPS)
    totalNoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    durationInSeconds = totalNoFrames / fps
    processingInSeconds = (time.time() - start_time_video_processing)
    __own_logger.info("Video with length of %f seconds tooks %f seconds for processing", durationInSeconds, processingInSeconds)

    # Add the collected data to the DataFrame
    data['missing_data'] = missing_data_arr
    data['right_foot_x_pos'] = right_foot_x_pos_arr
    data['right_foot_y_pos'] = right_foot_y_pos_arr
    data['left_foot_x_pos'] = left_foot_x_pos_arr
    data['left_foot_y_pos'] = left_foot_y_pos_arr
    data['nose_x_pos'] = nose_x_pos_arr
    data['nose_y_pos'] = nose_y_pos_arr
    data['y_pred'] = y_pred_single_arr

    # # Anomaly detection with the whole data at once: Same result
    # # Handling missing data (frames with no detected landmarks): Backward filling (take the next observation and fill backward) for rows which where initially labeled as missing-data
    # data = data.mask(data.missing_data == True, data.fillna(method='bfill'))
    # # For missing data at the end, the bfill mechanism not work, so do now a ffill
    # data = data.mask(data.missing_data == True, data.fillna(method='ffill'))
    # # Predict the anomalies: Returns -1 for outliers and 1 for inliers.
    # anomalies_pred = (clf.predict(data.drop(['missing_data'], axis=1).to_numpy()) == -1)
    # # Add prediciton to data
    # data['y_pred'] = anomalies_pred
    # # add False (no anomaly) for rows with missing data
    # __own_logger.info("Number of missing_data, for which the anomaly is set to false: %d", data.missing_data.sum())
    # data['y_pred'] = np.where(data.missing_data, False, anomalies_pred)

    # Visualize the data
    # Create dict for visualization data
    dict_visualization_data = {
        "label": data.columns.values, # Take all columns for visualization in dataframe
        "value": [data[data.columns.values][col] for col in data[data.columns.values]],
        # As x_data generate a consecutive number: a frame number for the whole merged time series, so the index + 1 can be used
        "x_data": data.index + 1
    }
    # Create a Line-Circle Chart
    figure_test_data = figure_time_series_data_as_layers(__own_logger, "Daten: Positionen der Füße", "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Frame")
    # Append the figure to the plot
    plot.appendFigure(figure_test_data.getFigure())

    # Show the plot in responsive layout, but only stretch the width
    plot.showPlotResponsive('stretch_width')

    # Extracting the video parts with detected anomalies
    # Get the indizes for which frames a anomaly is detected
    #anomaly_indizes_diff = data.index[data['y_pred'] == True].to_series().diff()
    anomaly_indizes_list = data.index[data['y_pred'] == True].to_list()
    start_index_arr = []
    end_index_arr = []
    start_index = None
    end_index = None
    incrementer = 0
    # Sort the indizes in start- and end-indizes
    for index in anomaly_indizes_list:
        if start_index is None:
            start_index = index
            incrementer += 1
        elif index == (start_index + incrementer):
            end_index = index
            incrementer += 1
        else:
            start_index_arr.append(start_index)
            end_index_arr.append(end_index)
            start_index = index
            end_index = None
            incrementer = 1
    if start_index is not None:
        start_index_arr.append(start_index)
    if end_index is not None:
        end_index_arr.append(end_index)
    # Iterate over the start-indizes
    for idx, start_index in enumerate(start_index_arr):
        # If stop_index is None, then only a single frame was affected, which will be ignored
        # TODO: Ignore video-parts with e.g. less then 5 frames?
        if end_index_arr[idx] is not None:
            # Define the start- and end-frame
            # TODO: Use some additional preceding and following frames for visualization?
            start_frame = start_index
            end_frame = end_index_arr[idx]
            # Set the video's start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            # Read and write frames within the specified timestamps
            for frame_num in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break

                # Naming a window
                window_name = "Detected Anomaly"
                # cv2.WINDOW_NORMAL makes the output window resizealbe
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                # Show it in fullscreen
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                # Add some text
                cv2.putText(frame,"Press 'q' to quit",(0,40),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1) 

                # Display the frame
                cv2.imshow(window_name, frame)

                # Press 'q' to quit
                key = cv2.waitKey(0 ) & 0xFF
                # if the `q` key was pressed, break from the loop
                if key == ord('q'):
                    break

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()
