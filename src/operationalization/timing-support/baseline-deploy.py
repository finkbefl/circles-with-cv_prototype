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
# Multimedia: Manipulate and get infos
import ffmpeg
# Fractions
from fractions import Fraction
# Using scipy for calculating the spectrum
from scipy import signal
# Using statsmodel for detecting stationarity
from statsmodels.tsa.stattools import adfuller, kpss
# Detection of relative minima of data
from scipy.signal import argrelmin
# For playing sound
import simpleaudio as sa

# Import internal packages/ classes
# Import the src-path to sys path that the internal modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))
# To handle the Logging for all modules in the same way
from utils.own_logging import OwnLogging, log_overview_data_frame
# To handle csv files
from utils.csv_operations import load_data, save_data, convert_series_into_date
# To plot data with bokeh
from utils.plot_data import PlotMultipleLayers, PlotMultipleFigures, figure_vbar, figure_hist, figure_hist_as_layers, figure_time_series_data_as_layers, figure_vbar_as_layers

#########################################################

# Initialize the logger
__own_logger = OwnLogging("timing-support_" + Path(__file__).stem).logger

#########################################################

def stationarity_test(df):
    """
    Function to test the data columns for stationarity (Augmented Dickey-Fuller and Kwiatkowski-Phillips-Schmidt-Shin in combination for more confident decisions)
    ----------
    Parameters:
        df : pandas.core.frame.DataFrame
            The data
    ----------
    Returns:
        dict with the stationarity test results:
        {'column_name': 
            {'ADF': Boolean, 'KPSS': Boolean},
        ...
        }
    """

    stationarity_dict= {} # create an empty dictionary for the test results
    # Iterate over all columns except column one (date)
    for column in df.iloc[:,1:]:
        # Do not consider data which not vary over time, so skip the column which only consists of one value
        if (df[column].nunique() == 1):
            __own_logger.info("Skip column: %s because it not vary over time", column)
            continue
        # Check for stationarity
        # Augmented Dickey-Fuller Test
        adf_decision_stationary = False
        try:
            adf_output = adfuller(df[column])
            # Decision based on pval
            adf_pval = adf_output[1]
            if adf_pval < 0.05: 
                adf_decision_stationary = True
        except Exception as error:
            __own_logger.error("Error during ADF Test", exc_info=error)
        # Kwiatkowski-Phillips-Schmidt-Shin Test
        kpss_decision_stationary = False
        try:
            kpss_output = kpss(df[column])
            # Decision based on pval
            kpss_pval = kpss_output[1]
            if kpss_pval >= 0.05: 
                kpss_decision_stationary = True
        except Exception as error:
            __own_logger.error("Error during KPSS Test", exc_info=error)
        # Add the test results to the dict
        stationarity_dict[column] = {"ADF": adf_decision_stationary, "KPSS": kpss_decision_stationary}
    __own_logger.info("Stationarity: %s", stationarity_dict)

    return stationarity_dict

#########################################################

def get_spectrum(input_signal, sampling_frequency):
    """
    Get a pandas Series with the fourier power spectum for a given signal.
    """
    input_signal_copy = input_signal.copy()
    input_signal_copy = np.asarray(input_signal_copy.values, dtype='float64')
    
    # Remove the mean  
    input_signal_copy -= input_signal_copy.mean()  
    
    # Estimate power spectral density using a periodogram.
    frequencies , power_spectrum = signal.periodogram(input_signal_copy, sampling_frequency, scaling='spectrum', nfft=512)

    return pd.Series(power_spectrum), frequencies

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
    plot = PlotMultipleFigures(os.path.join("output/timing-support",file_name), file_title)

    # Join the filepaths for the data
    data_modeling_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "modeling", "timing-support")
    deployment_video_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw", "deployment")

    __own_logger.info("Path of the modeling input data: %s", data_modeling_path)

    # Join the filepath of the raw data file
    #file_input_path = os.path.join(deployment_video_path, "deployment_fromSide.mov")
    file_input_path = os.path.join(deployment_video_path, "deployment_fromSide_Trial1.mp4")
    #file_input_path = os.path.join(deployment_video_path, "deployment_fromSide_Trial2.mp4")
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
    right_wrist_x_pos_arr = []
    right_wrist_y_pos_arr = []
    left_wrist_x_pos_arr = []
    left_wrist_y_pos_arr = []
    missing_data_arr = []
    timestamp = []
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
            #mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract footpositions (normalized to the width and the heigth of the image)
            right_foot_x_pos = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x
            right_foot_y_pos = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y
            left_foot_x_pos = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x
            left_foot_y_pos = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y
            # Extract wrist positions 
            right_wrist_x_pos = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x
            right_wrist_y_pos = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
            left_wrist_x_pos = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x
            left_wrist_y_pos = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y

            # Add the values to the lists
            right_foot_x_pos_arr.append(right_foot_x_pos)
            right_foot_y_pos_arr.append(right_foot_y_pos)
            left_foot_x_pos_arr.append(left_foot_x_pos)
            left_foot_y_pos_arr.append(left_foot_y_pos)
            right_wrist_x_pos_arr.append(right_wrist_x_pos)
            right_wrist_y_pos_arr.append(right_wrist_y_pos)
            left_wrist_x_pos_arr.append(left_wrist_x_pos)
            left_wrist_y_pos_arr.append(left_wrist_y_pos)
            # Label the data row with "no missing data"
            missing_data_arr.append(False)
            # Save timestamp of frame (in milli seconds)
            timestamp.append(cap.get(cv2.CAP_PROP_POS_MSEC))

            # # Bigger size of the image for better visualization
            # frame = cv2.resize(frame, (1920,1080), interpolation=cv2.INTER_CUBIC)

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
            right_wrist_x_pos_arr.append(np.NaN)
            right_wrist_y_pos_arr.append(np.NaN)
            left_wrist_x_pos_arr.append(np.NaN)
            left_wrist_y_pos_arr.append(np.NaN)
            # Label the data row with "missing data"
            missing_data_arr.append(True)
            # But the timestamp can be set correctly
            timestamp.append(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Naming a window
        window_name = "Timing Support"
        # cv2.WINDOW_NORMAL makes the output window resizealbe
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Show it in fullscreen
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Add some text
        cv2.putText(frame,"Press 'q' to quit",(0,40),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)

        # Get the frame rate of the source video
        fps =  cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame,f'Framerate (source): {fps:.1f} FPS',(0,1000),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),1)
        # Put calculated frame rate for the generated video as text in image
        cv2.putText(frame,f'Framerate (generated): {framerate:.1f} FPS',(0,1040),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),1)   

        # Display the frame
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1)
        # Press 'q' to quit
        # key = cv2.waitKey(0 ) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord('q'):
            break

        # Calculate the frame rate of the generated video via showing the frames
        framerate = 1.0 / (time.time() - start_time)
            
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
    data['right_wrist_x_pos'] = right_wrist_x_pos_arr
    data['right_wrist_y_pos'] = right_wrist_y_pos_arr
    data['left_wrist_x_pos'] = left_wrist_x_pos_arr
    data['left_wrist_y_pos'] = left_wrist_y_pos_arr

    # Convert the timestamps (in ms) into DateTime (raise an exception when parsing is invalid) and set it as index
    data = data.set_index(convert_series_into_date(timestamp, unit='ms'))

    # Handling missing data (frames with no detected landmarks)
    __own_logger.info("Data of Pose Detection: Detected missing data: %s", data.isna().sum())
    # Backward filling (take the next observation and fill backward) for rows which where initially labeled as missing-data
    data = data.mask(data.missing_data == True, data.fillna(method='bfill'))
    # For missing data at the end, the bfill mechanism not work, so do now a ffill
    data = data.mask(data.missing_data == True, data.fillna(method='ffill'))

    # Visualize the data
    # Create dict for visualization data
    dict_visualization_data = {
        "label": data.columns.values, # Take all columns for visualization in dataframe
        "value": [data[data.columns.values][col] for col in data[data.columns.values]],
        # As x_data generate a consecutive number: a frame number for the whole merged time series, so the index + 1 can be used
        "x_data": data.index
    }
    # Create a Line-Circle Chart
    figure_test_data = figure_time_series_data_as_layers(__own_logger, "Daten: Positionen der Füße", "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Laufzeit des Videos", x_axis_type='datetime')
    # Append the figure to the plot
    plot.appendFigure(figure_test_data.getFigure())

    # Time Series Stationarity
    # Copy the data for stationary data
    df_stationary_data = data.copy()
    # Test the columns for stationarity
    stationarity_results = stationarity_test(df_stationary_data)
    # Are the columns strict stationary?
    for column in stationarity_results:
        __own_logger.info("Data Analysis: Stationarity: Column %s is stationary: %s", column, stationarity_results[column])
        for value in stationarity_results[column].values():
            if value == False:
                #sys.exit('The data {} is not strict stationary! Fix it!'.format(column))
                __own_logger.info("Data Analysis: Column %s is not stationary!", column)


    # Get the frequency of the data: Calculate Spectrum (squared magnitude spectrum via fft)
    # At first get the sampling frequency of the video, with the help of the frame rate from video metadata
    metadata_streams = pd.DataFrame.from_dict(ffmpeg.probe(file_input_path)['streams'])
    metadata_video = metadata_streams[metadata_streams.codec_type == 'video']
    avg_frame_rate = metadata_video.avg_frame_rate.values[0]
    sampling_frequency = float(Fraction(avg_frame_rate))
    __own_logger.info("Video with frame rate/ sampling frequency of %s", sampling_frequency)
    # select the video stream
    metadata_video = metadata_streams[metadata_streams.codec_type == 'video']

    # Get the frequency of the data: Calculate Spectrum (squared magnitude spectrum via fft)
    power_spectrum_arr = []
    frequencies_arr = []
    for column in data.drop('missing_data', axis=1).columns:  
        power_spectrum, frequencies = get_spectrum(data[column], sampling_frequency)
        power_spectrum_arr.append(power_spectrum)
        frequencies_arr.append(frequencies)
    # Check for same frequencies in all spectrum of the different signals
    if not all([np.array_equal(a, b) for a, b in zip(frequencies_arr, frequencies_arr[1:])]):
        raise ValueError("The freqeuncies of the spectrums are not equal!")
    # Visualize all the spectrums
    # Create dict for visualization data
    dict_visualization_data = {
        "layer": data.drop('missing_data', axis=1).columns,
        "label": frequencies_arr,
        "value": power_spectrum_arr,
    }
    # Create a histogram: Frequency domain
    #figure_analyze_data_spectrum = figure_hist_as_layers(__own_logger, "Amplitudenspektrum", "Frequenz [Hz]", "Betrag im Quadrat des 2D-Fourier-Spektrums", dict_visualization_data.get('layer'), dict_visualization_data.get('label'), dict_visualization_data.get('value'))
    # Workaround: Realize it with vbar chart, as histogram is working with edges, but frequency are direct values, no edges
    figure_analyze_data_spectrum_all = figure_vbar_as_layers(__own_logger, "Datenanalyse des Videos: Amplitudenspektrum", "Betrag im Quadrat des 2D-Fourier-Spektrums", dict_visualization_data.get('layer'), dict_visualization_data.get('label')*10, dict_visualization_data.get('value'), set_x_range=False, width=0.05, x_label="Frequenz [Hz]")
    # Append the figure to the plot
    plot.appendFigure(figure_analyze_data_spectrum_all.getFigure())

    # Analyze the time series which are stationary, iterate over the available sprectrums
    max_freq_arr = []
    max_ampl_arr = []
    max_freq_foot_arr = []
    max_ampl_foot_arr = []
    max_name_foot_array = []
    max_freq_wrist_arr = []
    max_ampl_wrist_arr = []
    max_name_wrist_array = []
    for index, spectrum in enumerate(power_spectrum_arr):
        time_serie_to_analyze = data.drop('missing_data', axis=1).columns[index]
        # Visualize the distribution
        # Get distribution of values
        hist, bin_edges = np.histogram(data[time_serie_to_analyze], density=False, bins=100, range=(0,1))
        # Create the histogram
        figure_analyze_data_distribution = figure_hist(__own_logger, "Datenanalyse des Videos: Häufigkeitsverteilung von {}".format(time_serie_to_analyze), "Position normiert auf die Breite bzw. Höhe des Bildes", "Anzahl Frames", bin_edges, hist)
        # Append the figure to the plot
        plot.appendFigure(figure_analyze_data_distribution.getFigure())
        # Visualize the spectrum
        spectrum_to_analyze = spectrum
        frequencies_to_analyze = frequencies_arr[index]
        # Create a histogram: Frequency domain
        # Workaround: Realize it with vbar chart, as histogram is working with edges, but frequency are direct values, no edges
        figure_analyze_data_spectrum = figure_vbar(__own_logger, "Datenanalyse des Videos: Amplitudenspektrum von {}".format(time_serie_to_analyze), "Betrag im Quadrat des 2D-Fourier-Spektrums", frequencies_to_analyze, spectrum_to_analyze, set_x_range=False, color_sequencing=False, width=0.05, x_label="Frequenz [Hz]")
        # Get index with max value of spectrum amplitude
        idxmax = spectrum_to_analyze.idxmax()
        max_freq = frequencies_to_analyze[idxmax]
        max_freq_arr.append(max_freq)
        # Get the max value for this index 
        max = spectrum_to_analyze[idxmax]
        max_ampl_arr.append(max)
        if 'foot' in time_serie_to_analyze:
            # The time series is from foot
            max_name_foot_array.append(time_serie_to_analyze)
            max_ampl_foot_arr.append(max)
            max_freq_foot_arr.append(max_freq)
        elif 'wrist' in time_serie_to_analyze:
            # The time series is from wrist
            max_name_wrist_array.append(time_serie_to_analyze)
            max_ampl_wrist_arr.append(max)
            max_freq_wrist_arr.append(max_freq)
        # Get the frequency with the max spectrum amplitude
        __own_logger.info("Data Analysis %s: Max spectrum amplitude %s at frequency of the time series: %s Hz", time_serie_to_analyze, max, max_freq)
        # Add line to visualize the freq
        figure_analyze_data_spectrum.add_vertical_line(max_freq, max*1.05)
        figure_analyze_data_spectrum.add_annotation(max_freq, max *1.05, '{:.4f} Hz'.format(max_freq))
        # Append the figure to the plot
        plot.appendFigure(figure_analyze_data_spectrum.getFigure())
        # At first calc the period of the periodic part in number of frames
        period_s = 1/max_freq
        period_num = period_s * sampling_frequency
        __own_logger.info("Data Analysis %s: Period of the time series: %s s ; %s number of frames", time_serie_to_analyze, period_s, period_num)

    # Get the freq with the max amplitude from the spectrum
    idxmax = np.argmax(max_ampl_arr)
    max_freq = max_freq_arr[idxmax]
    max_ampl = max_ampl_arr[idxmax]
    __own_logger.info("Data Analysis: Max spectrum amplitude %s at frequency of the time series: %s Hz", max_ampl, max_freq)
    # Get the freq with the max amplitude from the spectrum, sperated by foot and wrist
    idxmax_foot = np.argmax(max_ampl_foot_arr)
    max_freq_foot = max_freq_foot_arr[idxmax_foot]
    max_ampl_foot = max_ampl_foot_arr[idxmax_foot]
    idxmax_wrist = np.argmax(max_ampl_wrist_arr)
    max_freq_wrist = max_freq_wrist_arr[idxmax_wrist]
    max_ampl_wrist= max_ampl_wrist_arr[idxmax_wrist]
    __own_logger.info("Data Analysis: Max spectrum amplitude for foots %s at frequency of the time series: %s Hz", max_ampl_foot, max_freq_foot)
    __own_logger.info("Data Analysis: Max spectrum amplitude for wrists %s at frequency of the time series: %s Hz", max_ampl_wrist, max_freq_wrist)
    # Create dict for visualization data
    dict_visualization_data = {
        "layer": data.drop('missing_data', axis=1).columns,
        "label": max_freq_arr,
        "value": max_ampl_arr,
    }
    # Create a bar chart
    figure_analyze_frequencies = figure_vbar_as_layers(__own_logger, "Datenanalyse des Videos: Maximale Amplituden der Spektren", "Betrag im Quadrat des 2D-Fourier-Spektrums", dict_visualization_data.get('layer'), dict_visualization_data.get('label')*10, dict_visualization_data.get('value'), set_x_range=False, width=0.05, x_label="Frequenz [Hz]")
    # Add line to visualize the max freqs seperated by foots and wrists
    figure_analyze_frequencies.add_vertical_line(max_freq_foot, max_ampl_foot*1.05)
    figure_analyze_frequencies.add_annotation(max_freq_foot, max_ampl_foot *1.05, 'Max Ampl. Füße')
    figure_analyze_frequencies.add_vertical_line(max_freq_wrist, max_ampl_wrist*1.05)
    figure_analyze_frequencies.add_annotation(max_freq_wrist, max_ampl_wrist *1.05, 'Max Ampl. Hände')
    # Append the figure to the plot
    plot.appendFigure(figure_analyze_frequencies.getFigure())

    # Extract timing information with detecting local minima: Analyze foot and wrist pos with the max spectrum amplitude
    time_serie_to_analyze = [max_name_foot_array[idxmax_foot], max_name_wrist_array[idxmax_wrist]]
    for columnname in time_serie_to_analyze:
        # Get the period number extracted with the spectrum
        index = data.drop('missing_data', axis=1).columns.get_loc(columnname)
        idxmax = power_spectrum_arr[index].idxmax()
        max_freq = frequencies_arr[index][idxmax]
        period_s = 1/max_freq
        period_num = period_s * sampling_frequency
        # Calc the indices of the local minima
        local_min_indices = argrelmin(data[columnname].values, order=int(period_num/2))
        # Create a time series which represents the local minima: Add a column with False values as preinitialization
        data[columnname + '_local_minima'] = False
        # Iterate over the detected local minima and set the colunm to True
        for local_min_index in local_min_indices[0]:
            data[columnname + '_local_minima'] = np.where((data.index == data.index[local_min_index]), True, data[columnname + '_local_minima'])
    # Visualize the local minima
    # Create dict for visualization data
    dict_visualization_data = {
        "label": [time_serie_to_analyze[0], time_serie_to_analyze[0] + '_local_minima', time_serie_to_analyze[1], time_serie_to_analyze[1] + '_local_minima'],
        "value": [data[time_serie_to_analyze[0]], data[time_serie_to_analyze[0] + '_local_minima'], data[time_serie_to_analyze[1]], data[time_serie_to_analyze[1] + '_local_minima']],
        "x_data": data.index
    }
    # Create a Line-Circle Chart
    figure_analyze_data_local_minima = figure_time_series_data_as_layers(__own_logger, "Datenanalyse des Videos: Zeitpunkte der Lokalen Minima von {} und {}".format(time_serie_to_analyze[0], time_serie_to_analyze[1]), "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Laufzeit des Videos", x_axis_type='datetime')
    # Append the figure to the plot
    plot.appendFigure(figure_analyze_data_local_minima.getFigure())

    # Show the plot in responsive layout, but only stretch the width
    plot.showPlotResponsive('stretch_width')

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()

    # Play beep in the half frequency of the foot as timing support for the hands!
    sound_frequency = 440  # The sound will be a 440 Hz beep
    sound_fs = 44100  # 44100 samples per second
    sound_duration_seconds = 0.1  # Duration of 100ms
    # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
    t = np.linspace(0, sound_duration_seconds, int(sound_duration_seconds * sound_fs), False)
    # Generate a 440 Hz sine wave
    note = np.sin(sound_frequency * t * 2 * np.pi)
    # Ensure that highest value is in 16-bit range
    audio = note * (2**15 - 1) / np.max(np.abs(note))
    # Convert to 16-bit data
    audio = audio.astype(np.int16)
    try:
        while True:
            start_time = time.time()
            print("Playing beep with frequency of {}/2 Hz as (both) hands timing support. Press 'Ctrl+C' to quit...".format(max_freq_foot))
            # Start playback
            play_obj = sa.play_buffer(audio, 1, 2, sound_fs)
            # Wait for playback to finish before exiting
            play_obj.wait_done()
            # Sleep the time of the rest of the period (but hald of the frequency of the foot because we habe 2 hands!)
            time.sleep((1/max_freq_foot) / 2 - (time.time()-start_time) )
    except KeyboardInterrupt:
        print('\n Stopped the beep by the user')