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
    file_name = "baseline-deploy.html"
    file_title = "Deployment of the the baseline model"
    __own_logger.info("Plot %s as multiple figures to file %s", file_title, file_name)
    plot = PlotMultipleFigures(os.path.join("output/circles-detection",file_name), file_title)

    # Join the filepaths for the data
    data_modeling_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "modeling")
    deployment_video_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "deployment")

    __own_logger.info("Path of the modeling input data: %s", data_modeling_path)

    # Get the trained svm classifier via joblib
    file_path = os.path.join(data_modeling_path, 'baseline-svm-model.joblib')
    clf = load(file_path)

    # Join the filepath of the raw data file
    file_input_path = os.path.join(deployment_video_path, "20240112_Dunningen_IphoneSE_KevinKieninger_FlorianFinkbeiner.mov")
    #file_input_path = os.path.join(deployment_video_path, "20240112_Dunningen_Xiaomi11Lite5GNE_FlorianFinkbeiner_KevinKieninger.mp4")
    #file_input_path = os.path.join(deployment_video_path, "20240308_Dunningen_IphoneSE_AndreasAbleitner.MOV")
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

            # Predict the target value
            y_pred_single = clf.predict(pd.DataFrame({'right_foot_x_pos':right_foot_x_pos, 'right_foot_y_pos':right_foot_y_pos, 'left_foot_x_pos':left_foot_x_pos, 'left_foot_y_pos':left_foot_y_pos, 'nose_x_pos':nose_x_pos, 'nose_y_pos':nose_y_pos}, index=[0]))
            print(y_pred_single)

            # Bigger size of the image for better visualization
            frame = cv2.resize(frame, (1920,1080), interpolation=cv2.INTER_CUBIC)

            # Signaling the detection of circles
            if y_pred_single[0]:
                # Green dot when circle is detected
                frame = cv2.circle(frame, (1820,80), radius=50, color=(0, 255, 0), thickness=-1)
            else:
                # red if not
                frame = cv2.circle(frame, (1820,80), radius=50, color=(0, 0, 255), thickness=-1)
        else:
            # If no landmarks detected
            # Bigger size of the image for better visualization
            frame = cv2.resize(frame, (1920,1080), interpolation=cv2.INTER_CUBIC)
            # Dont change the Signaling for the detection of circles, so if no landmarks are detected, no change of the last status will be "predictedSSSSSSSSSS"
            

        # Naming a window
        window_name = "Data Featurization"
        # cv2.WINDOW_NORMAL makes the output window resizealbe
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Show it in fullscreen
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Add some text
        cv2.putText(frame,"Press 'q' to quit",(0,40),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
        cv2.putText(frame,"Circle running:",(1500,80),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

        # Get the frame rate of the source video
        fps =  cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame,f'Framerate (source): {fps:.1f} FPS',(0,1000),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),1)
        # Put calculated frame rate for the generated video as text in image
        cv2.putText(frame,f'Framerate (generated): {framerate:.1f} FPS',(0,1040),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),1)   

        # Display the frame
        cv2.imshow(window_name, frame)

        # Press 'q' to quit
        key = cv2.waitKey(0 ) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord('q'):
            break

        # Calculate the frame rate of the generated video via showing the frames
        framerate = 1.0 / (time.time() - start_time)
            

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()
