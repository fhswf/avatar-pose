"""
Extractor.py

Authors: Carolin Gottschalk, Jonas D. Stephan, Manuela Wittmann
License: Apache License 2.0

Description:
This script extracts pose estimation data from videos using the HumanSkeletonExtractor.
It processes all MP4 videos from a dataset, extracts skeleton data frame by frame,
and saves the results as JSON files.
"""

import os
import json
import argparse
from pose_estimation_recognition_utils_rtmlib import RTMPoseEstimator3D, video3d_result_to_video_skeleton_data_with_confidence


def extract_pose_estimation_data(video_path, extractor):
    """
    Extracts pose estimation data from a video file and saves it as a JSON file.

    :param video_path: Path to the video file.
    :type video_path: str
    :param extractor: Initialized RTMPoseEstimator3D instance.
    """
    # Define the output path for extracted data
    pev_path = video_path.replace('.mp4', '.pev')
    pev_path = pev_path.replace('videos-cut', 'extracted')

    # Skip processing if the file already exists
    if os.path.exists(pev_path):
        return

    print(f"Processing file: {video_path}")

    result = extractor.process_video(video_path)
    
    # Prepare JSON output data
    data = video3d_result_to_video_skeleton_data_with_confidence(result)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(pev_path), exist_ok=True)

    # Save extracted data to JSON file
    with open(pev_path, "w", encoding="utf-8") as f:
        json_list = [json.loads(item.to_json()) for item in data]
        json.dump({"frames": json_list}, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Extract pose estimation data from videos.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,  # Jetzt muss der Pfad beim Aufruf zwingend übergeben werden
        help="Path to the root dataset folder (e.g., C:\\Users\\wittm\\sign-language-avatar-gloss-dgs)"
    )
    # Optional: Falls du auch die Modell-Links flexibel halten möchtest, kannst du sie hier ebenfalls als Parameter definieren.
    # Ich lasse sie vorerst als feste Defaults im Code, da du ja explizit nur den dataset_path meintest.
    args = parser.parse_args()

    dataset_path = args.dataset_path
    cut_videos_path = os.path.join(dataset_path, "videos-cut")

    # Initialize the skeleton extractor with the "holistic" model and confidence threshold
    extractor = RTMPoseEstimator3D(
        det_model_path='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
        pose_model_path='https://cobtras.com/data/model.onnx',
        device='cuda',
        mode='individual',
        special_model='fhswf/rtm133lifting-finetuning-signs-fingerspelling-dgs/rtm133lifting-finetuning-sings-fingerspelling-dgs-best.onnx'
    )

    # Iterate through all files and subdirectories in the dataset
    for root, dirs, files in os.walk(cut_videos_path):
        for file in files:
            # Process only MP4 video files
            if file.endswith(".mp4"):
                full_path = os.path.join(root, file)
                extract_pose_estimation_data(full_path, extractor)


if __name__ == "__main__":
    main()