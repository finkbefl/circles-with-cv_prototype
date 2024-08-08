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
# Boxplot
import matplotlib.pyplot as plt
# Heatmap
import seaborn as sns

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
    file_name = "baseline-preparation-eda.html"
    file_title = "Exploratory Data Analysis as preparation for the baseline model"
    __own_logger.info("Plot %s as multiple figures to file %s", file_title, file_name)
    plot = PlotMultipleFigures(os.path.join("output/timing-support",file_name), file_title)

    # Join the filepaths for the data
    data_raw_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw")
    data_modeling_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "modeling", "timing-support")

    __own_logger.info("Path of the raw input data: %s", data_raw_path)
    __own_logger.info("Path of the modeling input data: %s", data_modeling_path)

    # Analyze one good trail in Detail

    # Get csv file, which was created during data collection and adapted during data analysis as DataFrame
    metadata = load_data(data_raw_path, 'training_videos_with_metadata.csv')

    # Get the data for the specific video which should be analyzed in detail
    data_video_to_analyze = load_data(data_modeling_path, 'features_2_0.csv')
    data_video_to_analyze = load_data(data_modeling_path, 'features_12_4.csv')
    data_video_to_analyze = load_data(data_modeling_path, 'features_12_4_cut.csv')

    # Convert the timestamps (in ms) into DateTime (raise an exception when parsing is invalid) and set it as index
    data_video_to_analyze = data_video_to_analyze.set_index(convert_series_into_date(data_video_to_analyze.timestamp, unit='ms'))
    # Remove the timestamp column
    data_video_to_analyze.drop('timestamp', axis=1, inplace=True)
    log_overview_data_frame(__own_logger, data_video_to_analyze)

    # Handling missing data (frames with no detected landmarks)
    __own_logger.info("Data to Analyze: Detected missing data: %s", data_video_to_analyze.isna().sum())
    # Backward filling (take the next observation and fill bachward) for rows which where initially labeled as missing-data
    data_video_to_analyze = data_video_to_analyze.mask(data_video_to_analyze.missing_data == True, data_video_to_analyze.fillna(method='bfill'))
    # For missing data at the end, the bfill mechanism not work, so do now a ffill
    data_video_to_analyze = data_video_to_analyze.mask(data_video_to_analyze.missing_data == True, data_video_to_analyze.fillna(method='ffill'))

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
    # Visualize the data to analyze in detail without the missing data info
    # Create dict for visualization data
    dict_visualization_data = {
        "label": data_video_to_analyze.drop('missing_data', axis=1).columns.values, # Take all columns for visualization in dataframe
        "value": [data_video_to_analyze.drop('missing_data', axis=1)[data_video_to_analyze.drop('missing_data', axis=1).columns.values][col] for col in data_video_to_analyze.drop('missing_data', axis=1)[data_video_to_analyze.drop('missing_data', axis=1).columns.values]],
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
    # As boxplot
    boxprops=dict(linestyle='-', linewidth=0.5, color='#008080', facecolor = "#008080")
    flierprops=dict(markerfacecolor = '#7f7f7f', markeredgecolor = '#7f7f7f')
    medianprops=dict(linestyle='-', linewidth=2, color='#ff8000')
    whiskerprops=dict(linestyle='-', linewidth=2, color='#800080')
    capprops=dict(linestyle='-', linewidth=2, color='#800080')
    bplt = data_video_to_analyze.plot.box(patch_artist = True, figsize=(10, 5), showmeans=True, grid=True, boxprops=boxprops, flierprops=flierprops, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops, return_type='dict')
    [item.set_markerfacecolor('#ff8000') for item in bplt['means']]
    [item.set_markeredgecolor('#ff8000') for item in bplt['means']]
    plt.ylabel("Position normiert auf die Breite bzw. Höhe des Bildes")
    plt.show()

    # Data Analysis
    # Correlation
    __own_logger.info("Data Analysis: DataFrame correlation: %s", data_video_to_analyze.corr())     # TODO: Heatmap
    # plot the heatmap
    #sns.heatmap(data_video_to_analyze.corr())
    #plt.show()
    # Skewness
    __own_logger.info("Data Analysis: DataFrame skewness: %s", data_video_to_analyze.skew(axis='index'))
    # # Time Series Stationarity
    # # Copy the data for stationary data
    # df_stationary_data = data_video_to_analyze.copy()
    # # Test the columns for stationarity
    # stationarity_results = stationarity_test(df_stationary_data)
    # # Are the columns strict stationary?
    # for column in stationarity_results:
    #     __own_logger.info("Data Analysis: Stationarity: Column %s is stationary: %s", column, stationarity_results[column])
    #     for value in stationarity_results[column].values():
    #         if value == False:
    #             #sys.exit('The data {} is not strict stationary! Fix it!'.format(column))
    #             __own_logger.info("Data Analysis: Column %s is not stationary", column)

    # Get the frequency of the data: Calculate Spectrum (squared magnitude spectrum via fft)
    # At first get the sampling frequency of the video 2 (but index of rows starting with 0, so it is index 2-1): The frame rate (Calc float numbers from fractions)
    sampling_frequency = float(Fraction(metadata.avg_frame_rate[2-1]))
    # Get the frequency of the data: Calculate Spectrum (squared magnitude spectrum via fft)
    power_spectrum_arr = []
    frequencies_arr = []
    for column in data_video_to_analyze.drop('missing_data', axis=1).columns:  
        power_spectrum, frequencies = get_spectrum(data_video_to_analyze[column], sampling_frequency)
        power_spectrum_arr.append(power_spectrum)
        frequencies_arr.append(frequencies)
    # Check for same frequencies in all spectrum of the different signals
    if not all([np.array_equal(a, b) for a, b in zip(frequencies_arr, frequencies_arr[1:])]):
        raise ValueError("The freqeuncies of the spectrums are not equal!")
    # Visualize all the spectrums
    # Create dict for visualization data
    dict_visualization_data = {
        "layer": data_video_to_analyze.drop('missing_data', axis=1).columns,
        "label": frequencies_arr,
        "value": power_spectrum_arr,
    }
    # Create a histogram: Frequency domain
    #figure_analyze_data_spectrum = figure_hist_as_layers(__own_logger, "Amplitudenspektrum", "Frequenz [Hz]", "Betrag im Quadrat des 2D-Fourier-Spektrums", dict_visualization_data.get('layer'), dict_visualization_data.get('label'), dict_visualization_data.get('value'))
    # Workaround: Realize it with vbar chart, as histogram is working with edges, but frequency are direct values, no edges
    figure_analyze_data_spectrum_all = figure_vbar_as_layers(__own_logger, "Datenanalyse des Videos 2_0: Leistungsspektren", "Betrag im Quadrat des 2D-Fourier-Amplitudenspektrums", dict_visualization_data.get('layer'), dict_visualization_data.get('label')*10, dict_visualization_data.get('value'), set_x_range=False, width=0.05, x_label="Frequenz [Hz]")
    # Append the figure to the plot
    plot.appendFigure(figure_analyze_data_spectrum_all.getFigure())

    # Analyze the time series iterate over the available sprectrums
    max_freq_arr = []
    max_ampl_arr = []
    max_freq_foot_arr = []
    max_ampl_foot_arr = []
    max_name_foot_array = []
    max_freq_wrist_arr = []
    max_ampl_wrist_arr = []
    max_name_wrist_array = []
    for index, spectrum in enumerate(power_spectrum_arr):
        time_serie_to_analyze = data_video_to_analyze.drop('missing_data', axis=1).columns[index]
        # Visualize the distribution
        # Get distribution of values
        hist, bin_edges = np.histogram(data_video_to_analyze[time_serie_to_analyze], density=False, bins=100, range=(0,1))
        # Create the histogram
        figure_analyze_data_distribution = figure_hist(__own_logger, "Datenanalyse des Videos 2_0: Häufigkeitsverteilung von {}".format(time_serie_to_analyze), "Position normiert auf die Breite bzw. Höhe des Bildes", "Anzahl Frames", bin_edges, hist)
        # Append the figure to the plot
        plot.appendFigure(figure_analyze_data_distribution.getFigure())
        # Visualize the spectrum
        spectrum_to_analyze = spectrum
        frequencies_to_analyze = frequencies_arr[index]
        # Create a histogram: Frequency domain
        # Workaround: Realize it with vbar chart, as histogram is working with edges, but frequency are direct values, no edges
        figure_analyze_data_spectrum = figure_vbar(__own_logger, "Datenanalyse des Videos 2_0: Leistungsspektrum von {}".format(time_serie_to_analyze), "Betrag im Quadrat des 2D-Fourier-Amplitudenspektrums", frequencies_to_analyze, spectrum_to_analyze, set_x_range=False, color_sequencing=False, width=0.05, x_label="Frequenz [Hz]")
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
        figure_analyze_data_spectrum.add_annotation(max_freq, max *1.0, '  {:.4f} Hz'.format(max_freq), text_align='left')
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
        "layer": data_video_to_analyze.drop('missing_data', axis=1).columns,
        "label": max_freq_arr,
        "value": max_ampl_arr,
    }
    # Create a bar chart
    figure_analyze_frequencies = figure_vbar_as_layers(__own_logger, "Datenanalyse des Videos 2_0: Maximale Amplituden der Spektren", "Betrag im Quadrat des 2D-Fourier-Amplitudenspektrums", dict_visualization_data.get('layer'), dict_visualization_data.get('label')*10, dict_visualization_data.get('value'), set_x_range=False, width=0.05, x_label="Frequenz [Hz]")
    # Add line to visualize the max freqs seperated by foots and wrists
    figure_analyze_frequencies.add_vertical_line(max_freq_foot, max_ampl_foot*1.05)
    figure_analyze_frequencies.add_annotation(max_freq_foot, max_ampl_foot *1.0, '    Max Ampl. Füße', text_align='left')
    figure_analyze_frequencies.add_vertical_line(max_freq_wrist, max_ampl_wrist*1.05)
    figure_analyze_frequencies.add_annotation(max_freq_wrist, max_ampl_wrist *1.0, '    Max Ampl. Hände', text_align='left')
    # Append the figure to the plot
    plot.appendFigure(figure_analyze_frequencies.getFigure())

    # Extract timing information with detecting local minima: Analyze right foot x pos and right wrist y pos
    time_serie_to_analyze = ['right_foot_x_pos', 'right_wrist_y_pos']
    for columnname in time_serie_to_analyze:
        # Calc the indices of the local minima
        local_min_indices = argrelmin(data_video_to_analyze[columnname].values, order=int(period_num/2))
        # Create a time series which represents the local minima: Add a column with False values as preinitialization
        data_video_to_analyze[columnname + '_local_minima'] = False
        # Iterate over the detected local minima and set the colunm to True
        for local_min_index in local_min_indices[0]:
            data_video_to_analyze[columnname + '_local_minima'] = np.where((data_video_to_analyze.index == data_video_to_analyze.index[local_min_index]), True, data_video_to_analyze[columnname + '_local_minima'])

    # Visualize the local minima
    # Create dict for visualization data
    dict_visualization_data = {
        "label": [time_serie_to_analyze[0], time_serie_to_analyze[0] + '_local_minima', time_serie_to_analyze[1], time_serie_to_analyze[1] + '_local_minima'],
        "value": [data_video_to_analyze[time_serie_to_analyze[0]], data_video_to_analyze[time_serie_to_analyze[0] + '_local_minima'], data_video_to_analyze[time_serie_to_analyze[1]], data_video_to_analyze[time_serie_to_analyze[1] + '_local_minima']],
        "x_data": data_video_to_analyze.index
    }
    # Create a Line-Circle Chart
    figure_analyze_data_local_minima = figure_time_series_data_as_layers(__own_logger, "Datenanalyse des Videos 2_0: Zeitpunkte der Lokalen Minima", "Position normiert auf die Breite bzw. Höhe des Bildes", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Laufzeit des Videos", x_axis_type='datetime')
    # Append the figure to the plot
    plot.appendFigure(figure_analyze_data_local_minima.getFigure())

    # Now, analyze all videos, but with less visualization
    video_num_arr = []
    frequencies_by_hand_arr = []
    frequencies_by_foot_arr = []
    # Iterate over all videos
    for video_idx in metadata.index:
        # Get the performace label (amplitude_lack)
        performance_labels_amplitude_lack = [val.split('-') for val in metadata.manual_amplitude_lack][video_idx]
        # The filename of the video contains also a number, but starting from 1
        video_name_num = video_idx + 1
        # Get all seperated data (features) per raw video
        regex = re.compile('features_{}_.\.csv'.format(video_name_num))
        for dirpath, dirnames, filenames in os.walk(data_modeling_path):
            # Iterate over the seperated trials (sorted by number correctly)
            for train_data_file_name in sorted(filenames, key=lambda s: [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]):
                if regex.match(train_data_file_name):
                    __own_logger.info("Data to analyze detected: %s", train_data_file_name)
                    # Get the data related to the specific video
                    try:
                            data_video_to_analyze = load_data(data_modeling_path, train_data_file_name)
                    except FileNotFoundError as error:
                        __own_logger.error("########## Error when trying to access training data ##########", exc_info=error)

                    # Convert the timestamps (in ms) into DateTime (raise an exception when parsing is invalid) and set it as index
                    data_video_to_analyze = data_video_to_analyze.set_index(convert_series_into_date(data_video_to_analyze.timestamp, unit='ms'))
                    # Remove the timestamp column
                    data_video_to_analyze.drop('timestamp', axis=1, inplace=True)
                    log_overview_data_frame(__own_logger, data_video_to_analyze)

                    # Handling missing data (frames with no detected landmarks)
                    __own_logger.info("Data to Analyze: Detected missing data: %s", data_video_to_analyze.isna().sum())
                    # Backward filling (take the next observation and fill bachward) for rows which where initially labeled as missing-data
                    data_video_to_analyze = data_video_to_analyze.mask(data_video_to_analyze.missing_data == True, data_video_to_analyze.fillna(method='bfill'))
                    # For missing data at the end, the bfill mechanism not work, so do now a ffill
                    data_video_to_analyze = data_video_to_analyze.mask(data_video_to_analyze.missing_data == True, data_video_to_analyze.fillna(method='ffill'))

                    # Analyze the specific video (the time series data) in detail
                    # Descriptive Statistics
                    __own_logger.info("Descriptive Statistics: DataFrame describe: %s", data_video_to_analyze.describe())
                    # Data Analysis
                    # Correlation
                    __own_logger.info("Data Analysis: DataFrame correlation: %s", data_video_to_analyze.corr())     # TODO: Heatmap
                    # Skewness
                    __own_logger.info("Data Analysis: DataFrame skewness: %s", data_video_to_analyze.skew(axis='index'))
                    # # Time Series Stationarity
                    # # Copy the data for stationary data
                    # df_stationary_data = data_video_to_analyze.copy()
                    # # Test the columns for stationarity
                    # stationarity_results = stationarity_test(df_stationary_data)
                    # # Are the columns strict stationary?
                    # for column in stationarity_results:
                    #     __own_logger.info("Data Analysis: Stationarity: Column %s is stationary: %s", column, stationarity_results[column])
                    #     for value in stationarity_results[column].values():
                    #         if value == False:
                    #             #sys.exit('The data {} is not strict stationary! Fix it!'.format(column))
                    #             __own_logger.info("Data Analysis: Column %s is not stationary.", column)

                    # Get the frequency of the data: Calculate Spectrum (squared magnitude spectrum via fft)
                    # At first get the sampling frequency of the video 2 (but index of rows starting with 0, so it is index 2-1): The frame rate (Calc float numbers from fractions)
                    sampling_frequency = float(Fraction(metadata.avg_frame_rate[2-1]))
                    # Get the frequency of the data: Calculate Spectrum (squared magnitude spectrum via fft)
                    power_spectrum_arr = []
                    frequencies_arr = []
                    for column in data_video_to_analyze.drop('missing_data', axis=1).columns:  
                        power_spectrum, frequencies = get_spectrum(data_video_to_analyze[column], sampling_frequency)
                        power_spectrum_arr.append(power_spectrum)
                        frequencies_arr.append(frequencies)
                    # Check for same frequencies in all spectrum of the different signals
                    if not all([np.array_equal(a, b) for a, b in zip(frequencies_arr, frequencies_arr[1:])]):
                        raise ValueError("The freqeuncies of the spectrums are not equal!")

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
                        time_serie_to_analyze = data_video_to_analyze.drop('missing_data', axis=1).columns[index]
                        # Set the data to analyze
                        spectrum_to_analyze = spectrum
                        frequencies_to_analyze = frequencies_arr[index]
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
                        # At first calc the period of the periodic part in number of frames
                        period_s = 1/max_freq
                        period_num = period_s * sampling_frequency
                        __own_logger.info("Data Analysis %s: Period of the time series: %s s ; %s number of frames", time_serie_to_analyze, period_s, period_num)

                    # Check, if at least one column 'wrist" and one column "foot" are available/stationary, if not, then this video can not be analyzed
                    if len(max_ampl_foot_arr) <= 0 or len(max_ampl_wrist_arr) <= 0:
                        __own_logger.info("Data Analysis %s: Video can not be analzed due to missing stationary data (at least one foot- and one wrist-column)", time_serie_to_analyze)
                        break

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
                    # Save in array
                    video_num_arr.append(train_data_file_name.replace('features_', '').replace('.csv', ''))
                    frequencies_by_hand_arr.append(max_freq_wrist)
                    frequencies_by_foot_arr.append(max_freq_foot)
                    # Create dict for visualization data
                    dict_visualization_data = {
                        "layer": data_video_to_analyze.drop('missing_data', axis=1).columns,
                        "label": max_freq_arr,
                        "value": max_ampl_arr,
                    }
                    # Create a bar chart
                    figure_analyze_frequencies = figure_vbar_as_layers(__own_logger, "Datenanalyse des Videos {} (mangelnde Amplitude: {}): Maximale Amplituden der Spektren".format(train_data_file_name.replace('features_', '').replace('.csv', ''), performance_labels_amplitude_lack[int(train_data_file_name.replace('features_', '').replace('.csv', '').split('_')[1])]), "Betrag im Quadrat des 2D-Fourier-Amplitudenspektrums", dict_visualization_data.get('layer'), dict_visualization_data.get('label')*10, dict_visualization_data.get('value'), set_x_range=False, width=0.05, x_label="Frequenz [Hz]")
                    # Add line to visualize the max freqs seperated by foots and wrists
                    figure_analyze_frequencies.add_vertical_line(max_freq_foot, max_ampl_foot*1.05)
                    figure_analyze_frequencies.add_annotation(max_freq_foot, max_ampl_foot *1.0, '    Max Ampl. Füße', text_align='left')
                    figure_analyze_frequencies.add_vertical_line(max_freq_wrist, max_ampl_wrist*1.05)
                    figure_analyze_frequencies.add_annotation(max_freq_wrist, max_ampl_wrist *1.0, '    Max Ampl. Hände', text_align='left')
                    # Append the figure to the plot
                    plot.appendFigure(figure_analyze_frequencies.getFigure())

    # Create a dataframe with the frequency results
    circles_frequencies_detected = pd.DataFrame({'circles_num':video_num_arr, 'freq_by_hand': frequencies_by_hand_arr, 'freq_by_foot': frequencies_by_foot_arr})
    log_overview_data_frame(__own_logger, circles_frequencies_detected)
    # Descriptive Statistics
    __own_logger.info("Descriptive Statistics: DataFrame about the detected frequencies of the circles describe: %s", circles_frequencies_detected.drop('circles_num', axis=1).describe())
    # As boxplot
    boxprops=dict(linestyle='-', linewidth=0.5, color='#008080', facecolor = "#008080")
    flierprops=dict(markerfacecolor = '#7f7f7f', markeredgecolor = '#7f7f7f')
    medianprops=dict(linestyle='-', linewidth=2, color='#ff8000')
    whiskerprops=dict(linestyle='-', linewidth=2, color='#800080')
    capprops=dict(linestyle='-', linewidth=2, color='#800080')
    bplt = circles_frequencies_detected.drop('circles_num', axis=1).plot.box(patch_artist = True, figsize=(10, 5), showmeans=True, grid=True, boxprops=boxprops, flierprops=flierprops, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops, return_type='dict')
    [item.set_markerfacecolor('#ff8000') for item in bplt['means']]
    [item.set_markeredgecolor('#ff8000') for item in bplt['means']]
    plt.ylabel("Kreisflankenfrequenz [Hz]")
    plt.show()

    figure_results = figure_vbar_as_layers(__own_logger, "Detektierte Kreisflankenfrequenzen", "Kreisflankenfrequenz [Hz]", circles_frequencies_detected.drop('circles_num', axis=1).columns.values, ["{}".format(str(i)) for i in video_num_arr], [frequencies_by_hand_arr, frequencies_by_foot_arr], set_x_range=True, width=0.3, x_label="Video Nr.", single_x_range=True, x_offset=0.3)
    plot.appendFigure(figure_results.getFigure())
        
    # Show the plot in responsive layout, but only stretch the width
    plot.showPlotResponsive('stretch_width')
