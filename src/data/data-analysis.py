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

# Import internal packages/ classes
# Import the src-path to sys path that the internal modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
# To handle the Logging for all modules in the same way
from utils.own_logging import OwnLogging, log_overview_data_frame
# To handle csv files
from utils.csv_operations import load_data,  save_data
# To plot data with bokeh
from utils.plot_data import PlotMultipleLayers, PlotMultipleFigures, figure_vbar, figure_hist, figure_time_series_data_as_layers

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
    # Save the DataFrame to CSV
    save_data(data_collection, data_path, 'training_videos_with_metadata.csv')

