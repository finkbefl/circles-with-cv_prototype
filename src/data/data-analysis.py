# Script for the Exploratory Data Analysis (EDA)

# Import the external packages
# Operating system functionalities
import sys
import os
from pathlib import Path
# To handle pandas data frames
import pandas as pd
# OpenCV
import cv2
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

    # Join the filepath of the raw data files
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

    __own_logger.info("Path of the raw input data: %s", data_path)

    # Get csv file, which was created during data collection as DataFrame
    data_collection = load_data(data_path, 'training_videos.csv')

    # Some variable initializations for matadata
    codec_name = []
    width = []
    height = []
    coded_width = []
    coded_height = []
    pix_fmt = []
    r_frame_rate = []
    avg_frame_rate = []
    duration = []
    bit_rate = []
    nb_frames = []

    # Iterate over all videos in the directory (sorted by number correctly)
    for file in sorted(os.listdir(data_path), key=lambda s: [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]):
        # Decode the specified filename from the filesystem encoding
        filename = os.fsdecode(file)
        # Get the full file path
        file_path = os.path.join(data_path, filename)
        # skip the item in the directory if it is not a file (subfolders)
        if not os.path.isfile(file_path):
            continue
        # Have a look on the file suffix
        file_path_pure = Path(file_path)
        # Rename videos with suffix MOV to mov otherwise opencv can not handle it
        if file_path_pure.suffix == '.MOV':
            file_path_pure.rename(file_path_pure.with_suffix('.mov'))
            file_path = file_path_pure.as_posix()
        # If the file does not have the suffix mov or mp4, then ignore it
        elif not (file_path_pure.suffix == '.mov' or file_path_pure.suffix == '.mp4'):
            continue

        __own_logger.info("Detected input video file: %s", filename)

        # Get metadata as DataFrame
        # Select the streams
        metadata_streams = pd.DataFrame.from_dict(ffmpeg.probe(file_path)['streams'])
        # select the video stream
        metadata_video = metadata_streams[metadata_streams.codec_type == 'video']
        # Save some important metadata in corresponding arrays
        codec_name.append(metadata_video.codec_name.values[0])
        width.append(metadata_video.width.values[0])
        height.append(metadata_video.height.values[0])
        coded_width.append(metadata_video.coded_width.values[0])
        coded_height.append(metadata_video.coded_height.values[0])
        pix_fmt.append(metadata_video.pix_fmt.values[0])
        r_frame_rate.append(metadata_video.r_frame_rate.values[0])
        avg_frame_rate.append(metadata_video.avg_frame_rate.values[0])
        duration.append(metadata_video.duration.values[0])
        bit_rate.append(metadata_video.bit_rate.values[0])
        nb_frames.append(metadata_video.nb_frames.values[0])

        metadata_format = pd.DataFrame.from_dict(ffmpeg.probe(file_path)['format'])


    
        # For testing purposes: Break the loop after processing the first video
        #break

    # Add the collected metadata to the existing DataFrame
    data_collection['codec_name'] = codec_name
    data_collection['width'] = width
    data_collection['height'] = height
    data_collection['coded_width'] = coded_width
    data_collection['coded_height'] = coded_height
    data_collection['pix_fmt'] = pix_fmt
    data_collection['r_frame_rate'] = r_frame_rate
    data_collection['avg_frame_rate'] = avg_frame_rate
    data_collection['duration'] = duration
    data_collection['bit_rate'] = bit_rate
    data_collection['nb_frames'] = nb_frames

    # Visualize the metadata
    # location_group
    # Create values for visualization data (get the count of values via histogram)
    hist, bin_edges = np.histogram(data_collection.location_group, density=False, bins=data_collection.location_group.max())
    # Create labels
    labels = list("Gruppe {}".format(str(i)) for i in range(data_collection.location_group.min(), data_collection.location_group.max() + 1))
    # Create dict for visualization data
    dict_visualization_data = {
        "label": labels,
        "value": hist
    }
    # Print the histogram as vbar as the bin-edges here are not actually the limits of the bins, but the corresponding value. The x-axis would therefore be shifted.
    figure_loc_hist = figure_vbar(__own_logger, "Häufigkeitsverteilung der Gruppe des Ortes", "Anzahl Videos", dict_visualization_data.get('label'), dict_visualization_data.get('value'), set_x_range=True, color_sequencing=False)
    # device_name
    # Create values for visualization data (get the count of values)
    labels = data_collection.device_name.drop_duplicates().values
    values = list(len(data_collection[data_collection.device_name==device]) for device in data_collection.device_name.drop_duplicates().values)
    # Create dict for visualization data
    dict_visualization_data = {
        "label": labels,
        "value": values
    }
    # Print the histogram as vbar as the bins are no numbers
    figure_device_hist = figure_vbar(__own_logger, "Häufigkeitsverteilung des Aufnahmegeräts", "Anzahl Videos", dict_visualization_data.get('label'), dict_visualization_data.get('value'), set_x_range=True, color_sequencing=False)
    # manual_preprocessing
    # Create values for visualization data (get the count of values)
    labels = data_collection.manual_preprocessing.drop_duplicates().values
    values = list(len(data_collection[data_collection.manual_preprocessing==answer]) for answer in data_collection.manual_preprocessing.drop_duplicates().values)
    # Create dict for visualization data
    dict_visualization_data = {
        "label": labels,
        "value": values
    }
    # Print the histogram as vbar as the bins are no numbers
    figure_manual_preproc_hist = figure_vbar(__own_logger, "Häufigkeitsverteilung der manuellen Vorverarbeitung", "Anzahl Videos", dict_visualization_data.get('label'), dict_visualization_data.get('value'), set_x_range=True, color_sequencing=False)
    # codec_name
    # Create values for visualization data (get the count of values)
    labels = data_collection.codec_name.drop_duplicates().values
    values = list(len(data_collection[data_collection.codec_name==value]) for value in data_collection.codec_name.drop_duplicates().values)
    # Create dict for visualization data
    dict_visualization_data = {
        "label": labels,
        "value": values
    }
    # Print the histogram as vbar as the bins are no numbers
    figure_codec_hist = figure_vbar(__own_logger, "Häufigkeitsverteilung des Codecs", "Anzahl Videos", dict_visualization_data.get('label'), dict_visualization_data.get('value'), set_x_range=True, color_sequencing=False)
    # width
    # Create values for visualization data (get the count of values via histogram)
    hist, bin_edges = np.histogram(data_collection.width, density=False, bins=15)
    hist_1, bin_edges_1 = np.histogram(data_collection.coded_width, density=False, bins=15)
    # Bins must be equal to get correct visualization! Check it!
    if (bin_edges != bin_edges_1).any():
        __own_logger.error("########## Error when trying to visualize multiple histogramms ##########")
        sys.exit('Bins are not equal for visualize multiple histogramms')
    # Create dict for visualization data
    dict_visualization_data = {
        # Bins edges must be in form of string elements in list
        "layer": ["width", "coded_width"],
        "label": [["%.5f" % number for number in bin_edges], ["%.5f" % number for number in bin_edges_1]],
        "value": [hist, hist_1]
    }
    figure_width_hist = figure_hist_as_layers(__own_logger, "Häufigkeitsverteilung der Auflösungsbreite", "Pixel", "Anzahl Videos", dict_visualization_data.get('layer'), dict_visualization_data.get('label'), dict_visualization_data.get('value'))
    # height
    # Create values for visualization data (get the count of values via histogram)
    hist, bin_edges = np.histogram(data_collection.height, density=False, bins=15)
    hist_1, bin_edges_1 = np.histogram(data_collection.coded_height, density=False, bins=15)
    # Bins must be equal to get correct visualization! Equalize if necessary
    edge_min = min([bin_edges.min(), bin_edges_1.min()])
    edge_max = max([bin_edges.max(), bin_edges_1.max()])
    hist, bin_edges = np.histogram(data_collection.height, density=False, bins=15, range=(edge_min,edge_max))
    hist_1, bin_edges_1 = np.histogram(data_collection.coded_height, density=False, bins=15, range=(edge_min,edge_max))
    # Bins must be equal to get correct visualization! Check it!
    if (bin_edges != bin_edges_1).any():
        __own_logger.error("########## Error when trying to visualize multiple histogramms ##########")
        sys.exit('Bins are not equal for visualize multiple histogramms')
    # Create dict for visualization data
    dict_visualization_data = {
        # Bins edges must be in form of string elements in list
        "layer": ["height", "coded_height"],
        "label": [["%.5f" % number for number in bin_edges], ["%.5f" % number for number in bin_edges_1]],
        "value": [hist, hist_1]
    }
    figure_height_hist = figure_hist_as_layers(__own_logger, "Häufigkeitsverteilung der Auflösungshöhe", "Pixel", "Anzahl Videos", dict_visualization_data.get('layer'), dict_visualization_data.get('label'), dict_visualization_data.get('value'))
    # pix_fmt
    # Create values for visualization data (get the count of values)
    labels = data_collection.pix_fmt.drop_duplicates().values
    values = list(len(data_collection[data_collection.pix_fmt==value]) for value in data_collection.pix_fmt.drop_duplicates().values)
    # Create dict for visualization data
    dict_visualization_data = {
        "label": labels,
        "value": values
    }
    # Print the histogram as vbar as the bins are no numbers
    figure_pix_fmt_hist = figure_vbar(__own_logger, "Häufigkeitsverteilung des Pixelformats", "Anzahl Videos", dict_visualization_data.get('label'), dict_visualization_data.get('value'), set_x_range=True, color_sequencing=False)
    # frame_rate
    # Calc float numbers from fractions
    r_frame_rate = list(float(Fraction(num)) for num in data_collection.r_frame_rate)
    avg_frame_rate = list(float(Fraction(num)) for num in data_collection.avg_frame_rate)
    # Create values for visualization data (get the count of values via histogram)
    hist, bin_edges = np.histogram(r_frame_rate, density=False, bins=15)
    hist_1, bin_edges_1 = np.histogram(avg_frame_rate, density=False, bins=15)
    # Bins must be equal to get correct visualization! Equalize if necessary
    if bin_edges.min() < bin_edges_1.min() or bin_edges.max() > bin_edges_1.max():
        # Range of bin_edges is bigger, use it also for hist_1
        hist_1, bin_edges_1 = np.histogram(avg_frame_rate, density=False, bins=bin_edges)
    elif bin_edges_1.min() < bin_edges.min() or bin_edges_1.max() > bin_edges.max():
        # Range of bin_edges_1 is bigger, use it also for hist
        hist, bin_edges = np.histogram(r_frame_rate, density=False, bins=bin_edges_1)
    # Bins must be equal to get correct visualization! Check it!
    if (bin_edges != bin_edges_1).any():
        __own_logger.error("########## Error when trying to visualize multiple histogramms ##########")
        sys.exit('Bins are not equal for visualize multiple histogramms')
    # Create dict for visualization data
    dict_visualization_data = {
        # Bins edges must be in form of string elements in list
        "layer": ["r_frame_rate", "avg_frame_rate"],
        "label": [["%.5f" % number for number in bin_edges], ["%.5f" % number for number in bin_edges_1]],
        "value": [hist, hist_1]
    }
    figure_frame_rate_hist = figure_hist_as_layers(__own_logger, "Häufigkeitsverteilung der Bildwiederholrate", "Pixel", "Anzahl Videos", dict_visualization_data.get('layer'), dict_visualization_data.get('label'), dict_visualization_data.get('value'))
    # duration
    # Create dict for visualization data
    dict_visualization_data = {
        "label": ['duration'],
        "value": [data_collection.duration.astype(float)],
        # As x_data take the number of the videos
        "x_data": ["#{}".format(str(i)) for i in data_collection.video_number]
    }
    # Show Line-Circle Chart
    figure_duration_time_series = figure_time_series_data_as_layers(__own_logger, "Länge der Videos", "Dauer in s", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Nummer des Videos", set_x_range=True)
    # Additional, visualize it as histogramm
    # Create values for visualization data (get the count of values via histogram)
    hist, bin_edges = np.histogram(dict_visualization_data.get('value'), density=False, bins=10)
    # Print the histogram
    figure_duration_hist = figure_hist(__own_logger, "Häufigkeitsverteilung der Länge der Videos", "Dauer in s", "Anzahl Videos", bin_edges, hist)
    #bit_rate
    # Create dict for visualization data
    dict_visualization_data = {
        "label": ['bit_rate'],
        "value": [data_collection.bit_rate.astype(float)],
        # As x_data take the number of the videos
        "x_data": ["#{}".format(str(i)) for i in data_collection.video_number]
    }
    # Show Line-Circle Chart
    figure_bit_rate_time_series = figure_time_series_data_as_layers(__own_logger, "Bitrate", "bit/s", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Nummer des Videos", set_x_range=True)
    # Additional, visualize it as histogramm
    # Create values for visualization data (get the count of values via histogram)
    hist, bin_edges = np.histogram(dict_visualization_data.get('value'), density=False, bins=10)
    # Print the histogram
    figure_bit_rate_hist = figure_hist(__own_logger, "Häufigkeitsverteilung der Bitrate", "bit/s", "Anzahl Videos", bin_edges, hist)
    # nb_frames
    # Create dict for visualization data
    dict_visualization_data = {
        "label": ['nb_frames'],
        "value": [data_collection.nb_frames.astype(float)],
        # As x_data take the number of the videos
        "x_data": ["#{}".format(str(i)) for i in data_collection.video_number]
    }
    # Show Line-Circle Chart
    figure_nb_frames_time_series = figure_time_series_data_as_layers(__own_logger, "Anzahl Frames", "Anzahl Frames", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Nummer des Videos", set_x_range=True)
    # Additional, visualize it as histogramm
    # Create values for visualization data (get the count of values via histogram)
    hist, bin_edges = np.histogram(dict_visualization_data.get('value'), density=False, bins=10)
    # Print the histogram
    figure_nb_frames_hist = figure_hist(__own_logger, "Häufigkeitsverteilung der Anzahl Frames", "Anzahl Frames", "Anzahl Videos", bin_edges, hist)
    # manual_circle_detection_num_attempts
    # Count number of digits (detected time point in videos) which are seperated by '-' (The "timestamp" of kdenlive consists of the timestamp in seconds, and the number of the frame for this second, seperated by ':')
    start_count = [(sum(inner.split(':')[0].isdigit() for inner in val.split('-'))) for val in data_collection.manual_circle_start_kdenlive]
    end_count = [(sum(inner.split(':')[0].isdigit() for inner in val.split('-'))) for val in data_collection.manual_circle_end_kdenlive]
    # Create dict for visualization data
    dict_visualization_data = {
        "label": ['manual_circle_start_detection_count_attempts', 'manual_circle_stop_detection_count_attempts'],
        "value": [start_count, end_count],
        # As x_data take the number of the videos
        "x_data": ["#{}".format(str(i)) for i in data_collection.video_number]
    }
    # Show Line-Circle Chart
    figure_manual_circle_detect_time_series = figure_time_series_data_as_layers(__own_logger, "Anzahl manuell detektierter Kreisflankenversuche", "Anzahl detektierter Zeitpunkte", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Nummer des Videos", set_x_range=True)
    # manual_anomalie_detection_num_validation
    # Count number of digits (detected time point in videos) which are seperated by '-' (The "timestamp" of kdenlive consists of the timestamp in seconds, and the number of the frame for this second, seperated by ':')
    start_count = [(sum(inner.split(':')[0].isdigit() for inner in val.split('-'))) for val in data_collection.manual_anomalie_start_kdenlive]
    end_count = [(sum(inner.split(':')[0].isdigit() for inner in val.split('-'))) for val in data_collection.manual_anomalie_end_kdenlive]
    # Create dict for visualization data
    dict_visualization_data = {
        "label": ['manual_anomalie_start_detection_count', 'manual_anomalie_stop_detection_count'],
        "value": [start_count, end_count],
        # As x_data take the number of the videos
        "x_data": ["#{}".format(str(i)) for i in data_collection.video_number]
    }
    # Show Line-Circle Chart
    figure_manual_anomalie_detect_time_series = figure_time_series_data_as_layers(__own_logger, "Anzahl manuell detektierter Anomalien (nur Validation- und Test-Set)", "Anzahl detektierter Zeitpunkte", dict_visualization_data.get('x_data'), dict_visualization_data.get('label'), dict_visualization_data.get('value'), "Nummer des Videos", set_x_range=True)


    # Create the plot with the created figures
    file_name = "data-analysis.html"
    file_title = "Explorative Datenanalyse"
    __own_logger.info("Plot %s as multiple figures to file %s", file_title, file_name)
    plot = PlotMultipleFigures(os.path.join("output",file_name), file_title)
    plot.appendFigure(figure_loc_hist.getFigure())
    plot.appendFigure(figure_device_hist.getFigure())
    plot.appendFigure(figure_manual_preproc_hist.getFigure())
    plot.appendFigure(figure_codec_hist.getFigure())
    plot.appendFigure(figure_width_hist.getFigure())
    plot.appendFigure(figure_height_hist.getFigure())
    plot.appendFigure(figure_pix_fmt_hist.getFigure())
    plot.appendFigure(figure_frame_rate_hist.getFigure())
    plot.appendFigure(figure_duration_time_series.getFigure())
    plot.appendFigure(figure_duration_hist.getFigure())
    plot.appendFigure(figure_nb_frames_time_series.getFigure())
    plot.appendFigure(figure_nb_frames_hist.getFigure())
    plot.appendFigure(figure_bit_rate_time_series.getFigure())
    plot.appendFigure(figure_bit_rate_hist.getFigure())
    plot.appendFigure(figure_manual_circle_detect_time_series.getFigure())
    plot.appendFigure(figure_manual_anomalie_detect_time_series.getFigure())
    # Show the plot in responsive layout, but only stretch the width
    #plot.showPlotResponsive('stretch_width')
    # Show the plot in fixed layout
    plot.showPlotResponsive('fixed')



    # Save the DataFrame to CSV
    save_data(data_collection, data_path, 'training_videos_with_metadata.csv')

