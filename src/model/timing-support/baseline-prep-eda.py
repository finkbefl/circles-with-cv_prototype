# Script for the Exploratory Data Analysis as Preparation for the Baseline Model

# Import the external packages
# Operating system functionalities
import sys
import os
# Regular expressions
import re
from pathlib import Path
# To handle pandas data frames
import pandas as pd
# For standardized features
from sklearn.preprocessing import StandardScaler
# Numpy
import numpy as np
# Serialization of the trained model
from joblib import dump
# Time series transformation with seglearn
import seglearn as sgl
# Using statsmodel for detecting stationarity
from statsmodels.tsa.stattools import adfuller, kpss
# Using statsmodel for decomposition analysis
from statsmodels.tsa.seasonal import seasonal_decompose
# Detection of relative minima of data
from scipy.signal import argrelmin
# Using scipy for calculating the spectrum
from scipy import signal
# Fractions
from fractions import Fraction

# Import internal packages/ classes
# Import the src-path to sys path that the internal modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))
# To handle the Logging for all modules in the same way
from utils.own_logging import OwnLogging, log_overview_data_frame
# To handle csv files
from utils.csv_operations import load_data,  save_data, convert_series_into_date
# To plot data with bokeh
from utils.plot_data import PlotMultipleLayers, PlotMultipleFigures, figure_vbar, figure_vbar_as_layers, figure_hist, figure_hist_as_layers, figure_time_series_data_as_layers

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
    frequencies , power_spectrum = signal.periodogram(input_signal_copy, sampling_frequency, scaling='spectrum')

    return pd.Series(power_spectrum), frequencies

#########################################################
#########################################################
#########################################################

# When this script is called directly...
if __name__ == "__main__":
    # ...then calling the functions

    __own_logger.info("########## START ##########")

    # Create a plot for multiple figures
    file_name = "baseline-preparation-eda.html"
    file_title = "Exploratory Data Analysis as preparation for the baseline model"
    __own_logger.info("Plot %s as multiple figures to file %s", file_title, file_name)
    plot = PlotMultipleFigures(os.path.join("output/timing-support",file_name), file_title)

    # Join the filepaths for the data
    data_raw_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw")
    data_modeling_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "modeling", "timing-support")

    __own_logger.info("Path of the raw input data: %s", data_raw_path)
    __own_logger.info("Path of the modeling input data: %s", data_modeling_path)

    # Get csv file, which was created during data collection and adapted during data analysis as DataFrame
    metadata = load_data(data_raw_path, 'training_videos_with_metadata.csv')

    # # Get the Training Data
    # # Some variable initializations
    # data_training_arr = []
    # # Iterate over all data where selected as training data (tagged in metadata column 'usage' with 'train')
    # for video_idx in metadata.index[metadata.usage == 'train']:
    #     # The filename of the video contains also a number, but starting from 1
    #     video_name_num = video_idx + 1
    #     # Get all seperated training data (features) per raw video
    #     regex = re.compile('features_{}_.\.csv'.format(video_name_num))
    #     for dirpath, dirnames, filenames in os.walk(data_modeling_path):
    #         for train_data_file_name in filenames:
    #             if regex.match(train_data_file_name):
    #                 __own_logger.info("Training data detected: %s", train_data_file_name)
    #                 # Get the data related to the specific video
    #                 try:
    #                         data_specific_video = load_data(data_modeling_path, train_data_file_name)
    #                 except FileNotFoundError as error:
    #                     __own_logger.error("########## Error when trying to access training data ##########", exc_info=error)
    #                 # Merge all data in one array
    #                 data_training_arr.append(data_specific_video)

    # __own_logger.info("Find %d Videos for Training", len(data_training_arr))

    # Get the data for the specific video which should be analyzed in detail
    data_video_to_analyze = load_data(data_modeling_path, 'features_2_0.csv')

    # Convert the timestamps (in ms) into DateTime (raise an exception when parsing is invalid) and set it as index
    data_video_to_analyze = data_video_to_analyze.set_index(convert_series_into_date(data_video_to_analyze.timestamp, unit='ms'))
    # Remove the timestamp column
    data_video_to_analyze.drop('timestamp', axis=1, inplace=True)
    log_overview_data_frame(__own_logger, data_video_to_analyze)

    # # Concatenate the data in one frame by simply chain together the time series rows, but ignore the index of the rows to add so that we generate a continuous increasing index
    # data_training = pd.concat(data_training_arr, ignore_index=True)
    # # Remove the timestamp column
    # data_training.drop('timestamp', axis=1, inplace=True)
    # log_overview_data_frame(__own_logger, data_training)

    # Handling missing data (frames with no detected landmarks)
    # __own_logger.info("Training Data: Detected missing data: %s", data_training.isna().sum())
    # # Backward filling (take the next observation and fill bachward) for rows which where initially labeled as missing-data
    # data_training = data_training.mask(data_training.missing_data == True, data_training.fillna(method='bfill'))
    __own_logger.info("Data to Analyze: Detected missing data: %s", data_video_to_analyze.isna().sum())
    # Backward filling (take the next observation and fill bachward) for rows which where initially labeled as missing-data
    data_video_to_analyze = data_video_to_analyze.mask(data_video_to_analyze.missing_data == True, data_video_to_analyze.fillna(method='bfill'))

    # # Visualize the training data
    # # Create dict for visualization data
    # dict_visualization_data = {
    #     "label": data_training.columns.values, # Take all columns for visualization in dataframe
    #     "value": [data_training[data_training.columns.values][col] for col in data_training[data_training.columns.values]],
    #     # As x_data generate a consecutive number: a frame number for the whole merged time series, so the index + 1 can be used
    #     "x_data": data_training.index + 1
    # }
    # # Create a Line-Circle Chart
    # figure_training_data = figure_time_series_data_as_layers(__own_logger, "Trainingsdaten: Positionen der Füße und Handgelenke", "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Frame")
    # # Append the figure to the plot
    # plot.appendFigure(figure_training_data.getFigure())

    # Visualize the data to analyze in detail
    # Create dict for visualization data
    dict_visualization_data = {
        "label": data_video_to_analyze.columns.values, # Take all columns for visualization in dataframe
        "value": [data_video_to_analyze[data_video_to_analyze.columns.values][col] for col in data_video_to_analyze[data_video_to_analyze.columns.values]],
        # As x_data generate a consecutive number: a frame number for the whole merged time series, so the index + 1 can be used
        "x_data": data_video_to_analyze.index
    }
    # Create a Line-Circle Chart
    figure_analyze_data = figure_time_series_data_as_layers(__own_logger, "Datenanalyse des Videos 2_0: Positionen der Füße und Handgelenke", "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Laufzeit des Videos", x_axis_type='datetime')
    # Append the figure to the plot
    plot.appendFigure(figure_analyze_data.getFigure())

    # Analyze the specific video (the time series data) in detail
    # Descriptive Statistics
    __own_logger.info("Descriptive Statistics: DataFrame describe: %s", data_video_to_analyze.describe())
    # Data Analysis
    # Correlation
    __own_logger.info("Data Analysis: DataFrame correlation: %s", data_video_to_analyze.corr())     # TODO: Heatmap
    # Skewness
    __own_logger.info("Data Analysis: DataFrame skewness: %s", data_video_to_analyze.skew(axis='index'))
    # Time Series Stationarity
    # Copy the data for stationary data
    df_stationary_data = data_video_to_analyze.copy()
    # Test the columns for stationarity
    stationarity_results = stationarity_test(df_stationary_data)
    for column in stationarity_results:
        __own_logger.info("Data Analysis: DataFrame stationarity: Column %s is stationary: %s", column, stationarity_results[column])

    # Get the frequency of the data: Calculate Spectrum (squared magnitude spectrum via fft)
    # At first get the sampling frequency of the video 2 (but index of rows starting with 0, so it is index 2-1): The frame rate (Calc float numbers from fractions)
    sampling_frequency = float(Fraction(metadata.avg_frame_rate[2-1]))
    _power_spectrum_1, frequencies_1 = get_spectrum(data_video_to_analyze.right_wrist_y_pos, sampling_frequency)
    _power_spectrum_2, frequencies_2 = get_spectrum(data_video_to_analyze.left_wrist_y_pos, sampling_frequency)
    _power_spectrum_3, frequencies_3 = get_spectrum(data_video_to_analyze.right_foot_x_pos, sampling_frequency)
    _power_spectrum_4, frequencies_4 = get_spectrum(data_video_to_analyze.right_foot_y_pos, sampling_frequency)
    _power_spectrum_5, frequencies_5 = get_spectrum(data_video_to_analyze.left_foot_x_pos, sampling_frequency)
    _power_spectrum_6, frequencies_6 = get_spectrum(data_video_to_analyze.left_foot_y_pos, sampling_frequency)
    # Check for same frequencies in all spectrum of the different signals
    frequencies = [frequencies_1, frequencies_2, frequencies_3, frequencies_4, frequencies_5, frequencies_6]
    if not all([np.array_equal(a, b) for a, b in zip(frequencies, frequencies[1:])]):
        raise ValueError("The freqeuncies of the spectrums are not equal!")
    # Visualize all the spectrums
    # Create dict for visualization data
    dict_visualization_data = {
        "layer": ['right_wrist_y_pos', 'left_wrist_y_pos', 'right_foot_x_pos', 'right_foot_y_pos', 'left_foot_x_pos', 'left_foot_y_pos'],
        #"label": [_power_spectrum_1.index, _power_spectrum_2.index, _power_spectrum_3.index, _power_spectrum_4.index, _power_spectrum_5.index, _power_spectrum_6.index],
        "label": [frequencies_1, frequencies_2, frequencies_3, frequencies_4, frequencies_5, frequencies_6],
        "value": [_power_spectrum_1, _power_spectrum_2, _power_spectrum_3, _power_spectrum_4, _power_spectrum_5, _power_spectrum_6],
    }
    # Create a histogram: Frequency domain
    #figure_analyze_data_spectrum = figure_hist_as_layers(__own_logger, "Amplitudenspektrum", "Frequenz [Hz]", "Betrag im Quadrat des 2D-Fourier-Spektrums", dict_visualization_data.get('layer'), dict_visualization_data.get('label'), dict_visualization_data.get('value'))
    # Workaround: Realize it with vbar chart, as histogram is working with edges, but frequency are direct values, no edges
    figure_analyze_data_spectrum_all = figure_vbar_as_layers(__own_logger, "Datenanalyse des Videos 2_0: Amplitudenspektrum", "Betrag im Quadrat des 2D-Fourier-Spektrums", dict_visualization_data.get('layer'), dict_visualization_data.get('label')*10, dict_visualization_data.get('value'), set_x_range=False, width=0.05)
    # Append the figure to the plot
    plot.appendFigure(figure_analyze_data_spectrum_all.getFigure())

    # Analyze right_foot_x_pos
    time_serie_to_analyze = 'right_foot_x_pos'
    # Visualize the distribution
    # Get distribution of values
    hist, bin_edges = np.histogram(data_video_to_analyze[time_serie_to_analyze], density=False, bins=100, range=(0,1))
    # Create the histogram
    figure_analyze_data_distribution = figure_hist(__own_logger, "Datenanalyse des Videos 2_0: Häufigkeitsverteilung von {}".format(time_serie_to_analyze), "Position normiert auf die Breite bzw. Höhe des Bildes", "Anzahl Frames", bin_edges, hist)
    # Append the figure to the plot
    plot.appendFigure(figure_analyze_data_distribution.getFigure())
    # Visualize the spectrum
    spectrum_to_analyze = _power_spectrum_3.copy()
    frequencies_to_analyze = frequencies_3.copy()
    # Create a histogram: Frequency domain
    # Workaround: Realize it with vbar chart, as histogram is working with edges, but frequency are direct values, no edges
    figure_analyze_data_spectrum = figure_vbar(__own_logger, "Datenanalyse des Videos 2_0: Amplitudenspektrum von {}".format(time_serie_to_analyze), "Betrag im Quadrat des 2D-Fourier-Spektrums", frequencies_to_analyze, spectrum_to_analyze, set_x_range=False, color_sequencing=False, width=0.05)
    # Get the max value for this index 
    max = np.max(spectrum_to_analyze)
    # Get index with max value of spectrum amplitude
    idxmax = spectrum_to_analyze.idxmax()
    max_freq = frequencies_3[idxmax]
    # Get the frequency with the max spaectrum amplitude
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
    # Decompose the data columns
    res = seasonal_decompose(data_video_to_analyze[time_serie_to_analyze], model='additive', period=int(period_num))
    # Visualize the decomposition
    # Create dict for visualization data
    dict_visualization_data = {
        "label": ['Beobachtet', 'Trend', 'Saisonalität', 'Rest'],
        "value": [res.observed, res.trend, res.seasonal, res.resid],
        "x_data": data_video_to_analyze.index
    }
    # Create a Line-Circle Chart
    figure_analyze_data = figure_time_series_data_as_layers(__own_logger, "Datenanalyse des Videos 2_0: Dekomposition von {}".format(time_serie_to_analyze), "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Laufzeit des Videos", x_axis_type='datetime')
    # Append the figure to the plot
    plot.appendFigure(figure_analyze_data.getFigure())

    # Analyze left_foot_x_pos
    time_serie_to_analyze = 'left_foot_x_pos'
    spectrum_to_analyze = _power_spectrum_5.copy()
    frequencies_to_analyze = frequencies_5.copy()
    # Visualize the spectrum
    # Create a histogram: Frequency domain
    # Workaround: Realize it with vbar chart, as histogram is working with edges, but frequency are direct values, no edges
    figure_analyze_data_spectrum = figure_vbar(__own_logger, "Datenanalyse des Videos 2_0: Amplitudenspektrum von {}".format(time_serie_to_analyze), "Betrag im Quadrat des 2D-Fourier-Spektrums", frequencies_to_analyze, spectrum_to_analyze, set_x_range=False, color_sequencing=False, width=0.05)
    # Get the max value for this index 
    max = np.max(spectrum_to_analyze)
    # Get index with max value of spectrum amplitude
    idxmax = spectrum_to_analyze.idxmax()
    max_freq = frequencies_3[idxmax]
    # Get the frequency with the max spaectrum amplitude
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
    # Decompose the data columns
    res = seasonal_decompose(data_video_to_analyze[time_serie_to_analyze], model='additive', period=int(period_num))
    # Visualize the decomposition
    # Create dict for visualization data
    dict_visualization_data = {
        "label": ['Beobachtet', 'Trend', 'Saisonalität', 'Rest'],
        "value": [res.observed, res.trend, res.seasonal, res.resid],
        "x_data": data_video_to_analyze.index
    }
    # Create a Line-Circle Chart
    figure_analyze_data = figure_time_series_data_as_layers(__own_logger, "Datenanalyse des Videos 2_0: Dekomposition von {}".format(time_serie_to_analyze), "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Laufzeit des Videos", x_axis_type='datetime')
    # Append the figure to the plot
    plot.appendFigure(figure_analyze_data.getFigure())

    # Analyze right_wrist_y_pos
    time_serie_to_analyze = 'right_wrist_y_pos'
    # Visualize the spectrum
    spectrum_to_analyze = _power_spectrum_1.copy()
    frequencies_to_analyze = frequencies_1.copy()
    # Create a histogram: Frequency domain
    # Workaround: Realize it with vbar chart, as histogram is working with edges, but frequency are direct values, no edges
    figure_analyze_data_spectrum = figure_vbar(__own_logger, "Datenanalyse des Videos 2_0: Amplitudenspektrum von {}".format(time_serie_to_analyze), "Betrag im Quadrat des 2D-Fourier-Spektrums", frequencies_to_analyze, spectrum_to_analyze, set_x_range=False, color_sequencing=False, width=0.05)
    # Get the max value for this index 
    max = np.max(spectrum_to_analyze)
    # Get index with max value of spectrum amplitude
    idxmax = spectrum_to_analyze.idxmax()
    max_freq = frequencies_3[idxmax]
    # Get the frequency with the max spaectrum amplitude
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
    # Decompose the data columns
    res = seasonal_decompose(data_video_to_analyze[time_serie_to_analyze], model='additive', period=int(period_num))
    # Visualize the decomposition
    # Create dict for visualization data
    dict_visualization_data = {
        "label": ['Beobachtet', 'Trend', 'Saisonalität', 'Rest'],
        "value": [res.observed, res.trend, res.seasonal, res.resid],
        "x_data": data_video_to_analyze.index
    }
    # Create a Line-Circle Chart
    figure_analyze_data = figure_time_series_data_as_layers(__own_logger, "Datenanalyse des Videos 2_0: Dekomposition von {}".format(time_serie_to_analyze), "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Laufzeit des Videos", x_axis_type='datetime')
    # Append the figure to the plot
    plot.appendFigure(figure_analyze_data.getFigure())
    # Extract timing of put the hand back to the gymnastic mushroom
    # Calc the indices of the local minima
    local_min_indices_right_wrist = argrelmin(data_video_to_analyze[time_serie_to_analyze].values, order=int(period_num/2))
    # Create a time series which represents the local minima: Add a column with False values as preinitialization
    data_video_to_analyze[time_serie_to_analyze + '_local_minima'] = False
    # Iterate over the detected local minima and set the colunm to True
    for local_min_index in local_min_indices_right_wrist[0]:
        data_video_to_analyze[time_serie_to_analyze + '_local_minima'] = np.where((data_video_to_analyze.index == data_video_to_analyze.index[local_min_index]), True, data_video_to_analyze[time_serie_to_analyze + '_local_minima'])
    # Visualize the local minima
    # Create dict for visualization data
    dict_visualization_data = {
        "label": [time_serie_to_analyze, time_serie_to_analyze + '_local_minima'],
        "value": [data_video_to_analyze[time_serie_to_analyze], data_video_to_analyze[time_serie_to_analyze + '_local_minima']],
        "x_data": data_video_to_analyze.index
    }
    # Create a Line-Circle Chart
    figure_analyze_data_local_minima = figure_time_series_data_as_layers(__own_logger, "Datenanalyse des Videos 2_0: Zeitpunkte des Kontakts zum Turnpilz (Lokale Minima von {})".format(time_serie_to_analyze), "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Laufzeit des Videos", x_axis_type='datetime')
    # Append the figure to the plot
    plot.appendFigure(figure_analyze_data_local_minima.getFigure())

    # Analyze left_wrist_y_pos
    time_serie_to_analyze = 'left_wrist_y_pos'
    # Visualize the spectrum
    spectrum_to_analyze = _power_spectrum_2.copy()
    frequencies_to_analyze = frequencies_2.copy()
    # Create a histogram: Frequency domain
    # Workaround: Realize it with vbar chart, as histogram is working with edges, but frequency are direct values, no edges
    figure_analyze_data_spectrum = figure_vbar(__own_logger, "Datenanalyse des Videos 2_0: Amplitudenspektrum von {}".format(time_serie_to_analyze), "Betrag im Quadrat des 2D-Fourier-Spektrums", frequencies_to_analyze, spectrum_to_analyze, set_x_range=False, color_sequencing=False, width=0.05)
    # Get the max value for this index 
    max = np.max(spectrum_to_analyze)
    # Get index with max value of spectrum amplitude
    idxmax = spectrum_to_analyze.idxmax()
    max_freq = frequencies_3[idxmax]
    # Get the frequency with the max spaectrum amplitude
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
    # Decompose the data columns
    res = seasonal_decompose(data_video_to_analyze[time_serie_to_analyze], model='additive', period=int(period_num))
    # Visualize the decomposition
    # Create dict for visualization data
    dict_visualization_data = {
        "label": ['Beobachtet', 'Trend', 'Saisonalität', 'Rest'],
        "value": [res.observed, res.trend, res.seasonal, res.resid],
        "x_data": data_video_to_analyze.index
    }
    # Create a Line-Circle Chart
    figure_analyze_data = figure_time_series_data_as_layers(__own_logger, "Datenanalyse des Videos 2_0: Dekomposition von {}".format(time_serie_to_analyze), "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Laufzeit des Videos", x_axis_type='datetime')
    # Append the figure to the plot
    plot.appendFigure(figure_analyze_data.getFigure())
        
    # Show the plot in responsive layout, but only stretch the width
    plot.showPlotResponsive('stretch_width')
