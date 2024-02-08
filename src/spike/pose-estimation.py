# A script for a first spike to estimate the poses with mediapipe during circles on the mushroom
# Uses opencv for video and image processing

# Import the external packages
# Operating system functionalities
import sys
import os
from pathlib import Path
# Stream handling
import io
# To handle pandas data frames
import pandas as pd
# OpenCV
import cv2
# Mediapipe
import mediapipe as mp
# Time
import time
# Math
import math
# Numpy
import numpy as np

# Import internal packages/ classes
# Import the src-path to sys path that the internal modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
# To handle the Logging for all modules in the same way
from utils.own_logging import OwnLogging
# To handle csv files
from utils.csv_operations import save_data
# To plot data with bokeh
from utils.plot_data import PlotMultipleLayers, PlotMultipleFigures

#########################################################

# Initialize the logger
__own_logger = OwnLogging(Path(__file__).stem).logger

#########################################################

def log_overview_data_frame(df):
    """
    Logging some information about the data frame
    ----------
    Parameters:
        df : pandas.core.frame.DataFrame
            The data
    ----------
    Returns:
        no returns
    """

    # Print the first 5 rows
    __own_logger.info("Data Structure (first 5 rows): %s", df.head(5))
    # Print some information (pipe output of DataFrame.info to buffer instead of sys.stdout for correct logging)
    buffer = io.StringIO()
    buffer.write("Data Info: ")
    df.info(buf=buffer)
    __own_logger.info(buffer.getvalue())

#########################################################

def append_landmarks_to_list(landmarks, frame_number, data):
    """
    Append the detected body parts for every frame into a list
    ----------
    Parameters:
        landmarks : google.protobuf.pyext._message.RepeatedCompositeContainer
            The landmark of the pose detected from mediapipe
        frame_number : int
            The number of the frame
        data : list
            The list to write in the data
    ----------
    Returns:
        None
    """
    __own_logger.info("Landmark coordinates for frame %d", frame_number)
    # Iterate over all detected landmarks within this frame
    for idx, landmark in enumerate(landmarks):
        __own_logger.info("%s: (x: %f, y: %f, z: %f)", mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z)
        # Append the landmark to the list
        data.append([frame_number, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z])

#########################################################

def figure_vbar(figure_title, y_label, x_data, y_data, set_x_range=True, color_sequencing=True):
    """
    Function to create a vbar chart figure
    ----------
    Parameters:
        figure_title : str
            The title of the figure
        y_label : str
            The label of the y axis
        x_data : numbers.Real
            The x data to plot
        y_data : numbers.Real
            The y data to plot
        set_x_range : boolean
            set the x_data as range of the x-axis (for categorical data)
        color_sequencing : boolean
            A flag, whether every bar sould be drawn in another color with the known color sequence
    ----------
    Returns:
        The bokeh class
    """

    try:
        __own_logger.info("Figure for vbar chart: %s", figure_title)
        # Set the x_data as x_range (for categorical data)?
        if set_x_range:
            figure = PlotMultipleLayers(figure_title, None, y_label, x_range=x_data)
        # Dont set the x_range
        else:
            figure = PlotMultipleLayers(figure_title, None, y_label, x_range=None)
        figure.addVBarLayer(x_data, y_data, color_sequencing=color_sequencing)
        return figure
    except TypeError as error:
        __own_logger.error("########## Error when trying to create figure ##########", exc_info=error)
        sys.exit('A parameter does not match the given type')

#########################################################

def figure_hist(figure_title, x_label, y_label, edges, hist):
    """
    Function to create a histogram chart figure
    ----------
    Parameters:
        figure_title : str
            The title of the figure
        x_label : str
            The label of the x axis
        y_label : str
            The label of the y axis
        edges : numbers.Real
            The bins edges data to plot
        hist : numbers.Real
            The histogram data to plot
    ----------
    Returns:
        The bokeh class
    """

    try:
        __own_logger.info("Figure for hist chart: %s", figure_title)
        figure = PlotMultipleLayers(figure_title, x_axis_type=None, x_label=x_label, y_label=y_label)
        figure.addHist(edges, hist)
        return figure
    except TypeError as error:
        __own_logger.error("########## Error when trying to create figure ##########", exc_info=error)
        sys.exit('A parameter does not match the given type')

#########################################################

def figure_time_series_data_as_layers(figure_title, y_label, x_data, y_layers, y_datas):
    """
    Function to create a figure for time series data as multiple layers
    ----------
    Parameters:
        figure_title : str
            The title of the figure
        y_label : str
            The label of the y axis
        x_data : Series
                The x data to plot
        y_layers : array
            The names of the layers
        y_datas : DataFrame
            The y data to plot
    ----------
    Returns:
        The figure
    """

    try:
        __own_logger.info("Figure for times series data as multiple layers with title %s", figure_title)
        figure = PlotMultipleLayers(figure_title, "frame num", y_label, x_range=x_data)
        for (index, layer) in enumerate(y_layers):
            __own_logger.info("Add Layer for %s", layer)
            figure.addLineCircleLayer(layer, x_data, y_datas[index])
        return figure
    except TypeError as error:
        __own_logger.error("########## Error when trying to create figure ##########", exc_info=error)
        sys.exit('A parameter does not match the given type')

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
    frame_number = 1
    csv_data = []
    framerate = 0
    dists_right = []
    dists_left = []

    __own_logger.info("########## Iterate over every single frame ##########")
    while cap.isOpened():
        start_time = time.time() # start time of the loop

        # Read a frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Rezize the frame to a specific size, that it fits to the screen size
        frame = cv2.resize(frame, (960, 540))

        # Process the frame with MediaPipe Pose
        result = pose.process(frame_rgb)

        # Draw the pose landmarks on the frame
        if result.pose_landmarks:
            __own_logger.info("Landmarks detected")
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Add the landmark coordinates to the list and print them
            append_landmarks_to_list(result.pose_landmarks.landmark, frame_number, csv_data)

            # Find the Euclidean distance between two dimensional points:
            #dist_right = math.dist([result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x,result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y], [result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x,result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y])
            #dist_left = math.dist([result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x,result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y], [result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x,result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y])
            # Try to calculate the horizontal component of the distance (only x) between the nose and the foot indizes (the points are normalized to the width and the heigth of the image)
            dist_right = result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x - result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x
            dist_left = result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x - result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x
            __own_logger.info("X distance betwenn NOSE and RIGHT_FOOT_INDEX: %f", dist_right)
            __own_logger.info("X distance betwenn NOSE and LEFT_FOOT_INDEX: %f", dist_left)
            dists_right.append(dist_right)
            dists_left.append(dist_left)

            # get image resolution
            height, width, channels = frame.shape
            # Define 2D Points of interest: HEAD - RIGHT_FOOD_INDEX
            x1, y1 = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * width), int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * height)
            x2, y2 = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * width), int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * height)
            # Write Line between 2 points of interest
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
            # Define 2D Points of interest: HEAD - LEFT_FOOT_INDEX
            x1, y1 = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * width), int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * height)
            x2, y2 = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * width), int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * height)
            # Write Line between 2 points of interest
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)

        # Get the frame rate of the source video
        fps =  cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame,f'Framerate (source): {fps:.1f} FPS',(0,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
        # Put calculated frame rate for the generated video as text in image
        cv2.putText(frame,f'Framerate (generated): {framerate:.1f} FPS',(0,40),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)   

        # Display the frame
        cv2.imshow(f'Spike: Pose Detection with MediaPipe', frame)

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
        key = cv2.waitKey(0 ) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord('q'):
            break

        # Calculate the frame rate of the generated video via showing the frames
        framerate = 1.0 / (time.time() - start_time)
        __own_logger.info("Calculated frame rate for the generated video: %s", framerate)

        # Go to the next frame
        frame_number += 1

    # Write the landmarks data into the csv output file
    __own_logger.info("########## Save the landmarks in csv output file ##########")
    csv_data_frame = pd.DataFrame(csv_data, columns=['frame_num', 'landmark_name', 'landmark_x', 'landmark_x', 'landmark_z'])
    log_overview_data_frame(csv_data_frame)
    save_data(csv_data_frame, "output/spike", "pose-estimation.csv")

    # Visualize the horizontal distance average
    # Create dict for visualization data
    dict_visualization_data = {
        "label": ['|NOSE-RIGHT_FOOT_INDEX|', '|NOSE-LEFT_FOOT_INDEX|'],
        # Calc the mean of the absolute values (no negative values) in the list
        "value": [np.mean(np.absolute(dists_right)), np.mean(np.absolute(dists_left))],
    }
    # Show Bar-Chart
    figure_horizontal_distance = figure_vbar("Mittelwert der horizontalen Distanz (normierter Betrag)", "Distanz (normierter Betrag)", dict_visualization_data.get('label'), dict_visualization_data.get('value'), set_x_range=True)

    # Visualize the horizontal distance histogramm
    # Create values for visualization data (get the count of values via histogram)
    hist_dist_right, bin_edges_dist_right = np.histogram(dists_right, density=False, bins=15)
    hist_dist_left, bin_edges_dist_left = np.histogram(dists_left, density=False, bins=15)
    # Create dict for visualization data: right
    dict_visualization_data = {
        # Bins edges must be in form of string elements in list
        "label": ["%.5f" % number for number in bin_edges_dist_right],
        "value": hist_dist_right
    }
    figure_horizontal_distance_right_hist = figure_hist("Histogram der normierten horizontalen Distanz NOSE-RIGHT_FOOT_INDEX", "normierten horizontalen Distanz NOSE-RIGHT_FOOT_INDEX", "Anzahl Distanzmessungen", dict_visualization_data.get('label'), dict_visualization_data.get('value'))
    # Create dict for visualization data: left
    dict_visualization_data = {
        # Bins edges must be in form of string elements in list
        "label": ["%.5f" % number for number in bin_edges_dist_left],
        "value": hist_dist_left
    }
    figure_horizontal_distance_left_hist = figure_hist("Histogram der normierten horizontalen Distanz NOSE-LEFT_FOOT_INDEX", "normierten horizontalen Distanz NOSE-LEFT_FOOT_INDEX", "Anzahl Distanzmessungen", dict_visualization_data.get('label'), dict_visualization_data.get('value'))

    # Visualize the horizontal distance as time series data over all frames
    # Create dict for visualization data
    dict_visualization_data = {
        "label": ['NOSE-RIGHT_FOOT_INDEX', 'NOSE-LEFT_FOOT_INDEX'],
        "value": [dists_right, dists_left],
        # As x_range take the frame numbers as list, but strings as elements
        "x_range": [str(number) for number in list(range(1,len(dists_left)+1))]
    }
    # Show Line-Circle Chart
    figure_horizontal_distance_time_series = figure_time_series_data_as_layers("normierte horizontale Distanz über die Zeit/ Frames", "horizontale Distanz", dict_visualization_data.get('x_range'), dict_visualization_data.get('label'), dict_visualization_data.get('value'))

    # Create the plot with the created figures
    file_name = "pose-estimation.html"
    file_title = "Spike: Pose Detection with MediaPipe"
    __own_logger.info("Plot bar chart with title %s as multiple figures to file %s", file_title, file_name)
    plot = PlotMultipleFigures(os.path.join("output/spike",file_name), file_title)
    plot.appendFigure(figure_horizontal_distance.getFigure())
    plot.appendFigure(figure_horizontal_distance_right_hist.getFigure())
    plot.appendFigure(figure_horizontal_distance_left_hist.getFigure())
    plot.appendFigure(figure_horizontal_distance_time_series.getFigure())
    # Show the plot in responsive layout, but only stretch the width
    plot.showPlotResponsive('stretch_width')
        
    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()