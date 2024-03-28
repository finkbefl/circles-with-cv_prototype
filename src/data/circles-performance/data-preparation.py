# Script for the Data Preparation

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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))
# To handle the Logging for all modules in the same way
from utils.own_logging import OwnLogging, log_overview_data_frame
# To handle csv files
from utils.csv_operations import load_data,  save_data
# To plot data with bokeh
from utils.plot_data import PlotMultipleLayers, PlotMultipleFigures, figure_vbar, figure_hist, figure_hist_as_layers, figure_time_series_data_as_layers

#########################################################

# Initialize the logger
__own_logger = OwnLogging("circles-performance_" + Path(__file__).stem).logger

#########################################################
#########################################################
#########################################################

# When this script is called directly...
if __name__ == "__main__":
    # ...then calling the functions

    __own_logger.info("########## START ##########")

    # Join the filepaths for the data
    data_raw_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw")
    data_processed_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "processed", "circles-performance")

    __own_logger.info("Path of the raw input data: %s", data_raw_path)
    __own_logger.info("Path of the processed output data: %s", data_processed_path)

    # Get csv file, which was created during data collection and adapted during data analysis as DataFrame
    metadata = load_data(data_raw_path, 'training_videos_with_metadata.csv')

    # Iterate over all videos in the directory (sorted by number correctly)
    for file in sorted(os.listdir(data_raw_path), key=lambda s: [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]):
        # Decode the specified filename from the filesystem encoding
        filename = os.fsdecode(file)
        # Get the full file path
        file_input_path = os.path.join(data_raw_path, filename)
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

        __own_logger.info("Detected input video file: %s", filename)

        # Open the video file
        __own_logger.info("########## Open the video file ##########")
        cap = cv2.VideoCapture(file_input_path)
        # Get the fps of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        __own_logger.info("Input video file with fps: %s", fps)
        # Set target fps: Only for playing the video faster or slower, not for resampling the video!
        #output_fps = fps * 0.7
        output_fps = fps * 1
        __own_logger.info("Output video file with fps: %s", output_fps)

        # Resizing the output video
        # We are resizing to a given width while maintaining the aspect ratio
        width_requested = 300
        # Get original resolution
        width_orig  = cap.get(3)
        height_orig = cap.get(4)
        # Calculate the ratio of the width and construct the dimensions/ requested resolution
        width_ration = width_requested / float(width_orig)
        resolution_requested = (width_requested, int(height_orig * width_ration))

        # Get the start- and end-points for running-circles
        # Extract the video index from the video file name (video with name 1 is row 0 of metadata)
        video_idx = int(str(Path(filename).with_suffix(''))) - 1
        # Extract the list of manual detected start- and end-points of circles
        start_list = [val.split('-') for val in metadata.manual_circle_start_kdenlive][video_idx]
        end_list = [val.split('-') for val in metadata.manual_circle_end_kdenlive][video_idx]

        # Get the total number of frames in the video
        frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # Iterate over the start-points of manual detected circles
        for time_idx, start_stamp in enumerate(start_list):
            # extract the second and the number of frame, seperated by ':'
            start_stamp = start_stamp.split(':')
            # get the corresponding end-point
            end_stamp = end_list[time_idx].split(':')

            # Calculate the start and end frame indices
            start_frame = int(float(start_stamp[0]) * fps) + int(start_stamp[1])
            end_frame = int(float(end_stamp[0]) * fps) + int(end_stamp[1]) + 1
            # calculate the timestamps out of the second and the frame number within this second
            start_time = (float(start_stamp[0]) + ((1/fps) * int(start_stamp[1]))) * 1000
            end_time = (float(end_stamp[0]) + ((1/fps) * int(end_stamp[1]))) * 1000

            # Set the full file path for the output
            file_output_path = os.path.join(data_processed_path, "{0}_{1}{2}".format(Path(filename).with_suffix(''), time_idx, Path(filename).suffix))

            # Set the video's start frame
            #cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time)

            # Create a video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(file_output_path, fourcc, output_fps, resolution_requested)
            # Iterate over every single frame
            __own_logger.info("########## Iterate over every single frame ##########")
            # Some variable initializations
            # For background substraction
            alpha=0.999
            isFirstTime=True

            # Read and write frames within the specified timestamps
            for frame_num in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize the image (Interpolation to shrink an image: INTER_CUBIC is slow, INTER_LINEAR is fast)
                frame = cv2.resize(frame, resolution_requested, interpolation=cv2.INTER_CUBIC)

                # # equalize the histogram of color image to enhance the contrast
                # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)  # convert to HSV
                # # OpenCV equalizeHist() function is working only for grayscale image
                # frame[:,:,2] = cv2.equalizeHist(frame[:,:,2])     # equalize the histogram of the V channel
                # frame= cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)   # convert the HSV image back to RGB format

                # # Color transformation
                # # Convert to gray
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # # Background substraction
                # # create background
                # if isFirstTime==True:
                #     bg_img=frame
                #     isFirstTime=False
                # else:
                #     bg_img = dst = cv2.addWeighted(frame,(1-alpha),bg_img,alpha,0)
                # # create foreground
                # fg_img = cv2.absdiff(frame,bg_img) 
                # # Display the resulting image
                # cv2.imshow('Video Capture',frame)
                # cv2.imshow('Background',bg_img)
                # cv2.imshow('Foreground',fg_img)

                # # Display the frame
                # cv2.imshow(f'Spike: Pose Detection with MediaPipe', frame)
                # # Press 'q' to quit
                # key = cv2.waitKey(0 ) & 0xFF
                # # if the `q` key was pressed, break from the loop
                # if key == ord('q'):
                #     break
                # print(cap.get(cv2.CAP_PROP_POS_MSEC))

                # Save the frame into the output video file
                out.write(frame)

            # For testing purposes: Break the loop after processing the first video
            #break

        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

            # # Resampling with ffmpeg (not in-built functionality in opencv available)
            # stream = ffmpeg.input(file_output_path)
            # stream = ffmpeg.filter(stream, 'fps', fps=5)
            # # Addition in filename
            # p = Path(file_output_path)
            # file_output_path = "{0}_{1}{2}".format(Path.joinpath(p.parent, p.stem), "resampl", p.suffix)
            # stream = ffmpeg.output(stream, file_output_path)
            # ffmpeg.run(stream)

        # For testing purposes: Break the loop after processing the first video
        #break

