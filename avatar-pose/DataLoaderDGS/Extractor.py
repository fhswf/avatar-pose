"""
Extractor.py

Authors: Carolin Gottschalk, Jonas D. Stephan
License: Apache License 2.0

Description:
This script extracts pose estimation data from videos using the HumanSkeletonExtractor.
It processes all MP4 videos from a dataset, extracts skeleton data frame by frame,
and saves the results as JSON files.
"""

from SkeletonRecognitionUtils import HumanSkeletonExtrator
import os
import json
import cv2
from tqdm import tqdm

# Define dataset path
dataset_path="../../../../sign-language-avatar-gloss-dgs"

# Initialize the skeleton extractor with the "holistic" model and confidence threshold
extractor=HumanSkeletonExtrator("holistic",0.5)

# Path to the folder containing cut videos
cut_videos_path=dataset_path+"/videos-cut/"


def extract_pose_estimation_data(video_path):
    """
    Extracts pose estimation data from a video file and saves it as a JSON file.

    :param video_path: Path to the video file.
    :type video_path: str
    """
    # Define the output path for extracted data
    pev_path = video_path.replace('.mp4', '.pev')
    pev_path = pev_path.replace('videos-cut', 'extracted')

    # Skip processing if the file already exists
    if os.path.exists(pev_path):
        return

    print(f"Processing file: {video_path}")

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    frame_count = 0
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, desc="frames done", unit="frames")
    skeleton_data_objects = []

    while True:
        success, frame = video_capture.read()

        if not success:
            break

        # Extract skeleton data from the frame
        frame_skeletons = extractor.extract_skeleton_from_frame(frame, frame_count)

        if frame_skeletons is not None:
            data_dict = json.loads(frame_skeletons)

            if data_dict.get("skeletonpoints") is not None:
                skeleton_data_objects.append(frame_skeletons)
        # else:
        #     print("Frame " + str(frame_count) + " is None")

        frame_count += 1
        progress_bar.update(1)

    # Release the video file
    video_capture.release()

    # Prepare JSON output data
    data = {"frames": [json.loads(obj) for obj in skeleton_data_objects]}

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(pev_path), exist_ok=True)

    # Save extracted data to JSON file
    with open(pev_path, 'w') as file:
        json.dump(data, file)

# Iterate through all files and subdirectories in the dataset
for root, dirs, files in os.walk(cut_videos_path):
    for file in files:
        # Process only MP4 video files
        if file.endswith(".mp4"):
            full_path = os.path.join(root, file)
            extract_pose_estimation_data(full_path)