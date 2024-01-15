# Import the libraries
# Operating system functionalities
import os
# OpenCV
import cv2
# Mediapipe
import mediapipe as mp

import time

def write_landmarks_to_csv(landmarks, frame_number, csv_data):
    print(f"Landmark coordinates for frame {frame_number}:")
    for idx, landmark in enumerate(landmarks):
        print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
        csv_data.append([frame_number, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z])
    print("\n")

# Join the filepath of the raw data file
video_filename = "Spike_Pose-Estimation.mp4"
video_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", video_filename)

# Join the filepath of the output csv file
csv_filename = "spike-pose-estimation.csv"
output_csv = os.path.join(os.path.dirname(__file__), "..", "..", "output", csv_filename)

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_number = 0
# TODO: Pandas data frame?
csv_data = []

framerate = 0

while cap.isOpened():
    start_time = time.time() # start time of the loop

    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    result = pose.process(frame_rgb)

    # Draw the pose landmarks on the frame
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Add the landmark coordinates to the list and print them
        write_landmarks_to_csv(result.pose_landmarks.landmark, frame_number, csv_data)

    # Rezize the image to a specific size
    image = cv2.resize(frame, (960, 540))

    
    cv2.putText(image,f'{framerate:.1f} FPS',(0,30),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)

    # Display the frame
    cv2.imshow('MediaPipe Pose', image)

    # Define, how long the image should be shown or how long to wait until processing is continued
    # To get the video in original speed, this depends on the fps of the original video
    # For 25 fps the delay should approximately 1/25fps = 40 ms
    # But only in theory: Since the OS has a minimum time between switching threads, the function will not wait exactly delay ms, it will wait at least delay ms, depending on what else is running on your computer at that time.
    # The processing time, i.e. the runtime of a loop pass, also plays a role
    # the delay must therefore be selected smaller in order to achieve the original fps
    cv2.waitKey(5)
    # We can define a delay of zero, then the program will wait for keyboard input (for analyzing frame to frame)
    #cv2.waitKey(0)

    # Exit if 'q' keypyt

    framerate = 1.0 / (time.time() - start_time)
    print("FPS: ", framerate)

# TODO: Write output to csv file
    
# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()