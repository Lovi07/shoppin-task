import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import tempfile
import os

# Function to extract VGG16 features from a frame
def extract_vgg_features(frame):
    frame = cv2.resize(frame, (224, 224))  # Resize frame to 224x224 (required by VGG16)
    img_array = image.img_to_array(frame)  # Convert frame to a NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess input for VGG16
    features = VGG16(weights="imagenet", include_top=False, pooling="avg").predict(img_array)  # Extract features
    return features.flatten()  # Flatten features to 1D array

# Function to compute histogram difference
def histogram_difference(frame1, frame2):
    hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])  # Compute histogram for frame1
    hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])  # Compute histogram for frame2
    hist1 = cv2.normalize(hist1, hist1).flatten()  # Normalize and flatten histogram
    hist2 = cv2.normalize(hist2, hist2).flatten()  # Normalize and flatten histogram
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)  # Compare histograms

# Function to detect scene changes using histogram comparison
def detect_scene_changes(video_path, threshold=0.2):
    cap = cv2.VideoCapture(video_path)  # Open the video file
    prev_frame = None
    scene_change_frames = []

    while True:
        ret, frame = cap.read()  # Read the next frame
        if not ret:
            break

        if prev_frame is not None:
            diff = histogram_difference(prev_frame, frame)  # Compute histogram difference
            if diff > threshold:  # If difference exceeds threshold, consider it a scene change
                scene_change_frames.append(frame)

        prev_frame = frame  # Update previous frame

    cap.release()  # Release the video capture object
    return scene_change_frames[:5]  # Limit to 5 frames

# Function to select frames based on motion
def motion_based_selection(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)  # Open the video file
    prev_frame = None
    motion_scores = []

    while True:
        ret, frame = cap.read()  # Read the next frame
        if not ret:
            break

        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, frame)  # Compute absolute difference between frames
            motion_score = np.mean(diff)  # Compute mean difference as motion score
            motion_scores.append((frame, motion_score))  # Save frame and motion score

        prev_frame = frame  # Update previous frame

    cap.release()  # Release the video capture object

    # Sort frames by motion score and select top frames
    motion_scores.sort(key=lambda x: x[1], reverse=True)
    selected_frames = [x[0] for x in motion_scores[:num_frames]]
    return selected_frames

# Function to cluster frames using VGG16 features
def cluster_frames(video_path, num_clusters=5):
    cap = cv2.VideoCapture(video_path)  # Open the video file
    frames = []
    features = []

    while True:
        ret, frame = cap.read()  # Read the next frame
        if not ret:
            break

        frames.append(frame)  # Save the frame
        feature = extract_vgg_features(frame)  # Extract features using VGG16
        features.append(feature)  # Save the features

    cap.release()  # Release the video capture object

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)  # Cluster the frames

    # Select one frame from each cluster
    selected_frames = []
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]  # Find frames in the cluster
        centroid_index = cluster_indices[0]  # Select the first frame in the cluster
        selected_frames.append(frames[centroid_index])  # Save the frame

    return selected_frames

# Function to convert video to 15 FPS
def convert_to_15fps(video_path, output_path):
    cap = cv2.VideoCapture(video_path)  # Open the video file
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get the original FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get the frame width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get the frame height

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use MP4 codec
    out = cv2.VideoWriter(output_path, fourcc, 15, (width, height))  # Set output FPS to 15

    while True:
        ret, frame = cap.read()  # Read the next frame
        if not ret:
            break

        # Write the frame to the output video
        out.write(frame)

        # Skip frames to achieve 15 FPS
        for _ in range(int(fps / 15) - 1):
            cap.read()

    cap.release()  # Release the video capture object
    out.release()  # Release the video writer object

# Streamlit app
def main():
    st.title("Video Frame Selection App")
    st.write("Upload a 60-second video to extract the best 5 frames using three methods.")

    # Upload video
    uploaded_file = st.file_uploader("Upload a 60-second video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_video_path = temp_file.name

        # Convert the video to 15 FPS
        output_video_path = "temp_15fps_video.mp4"
        convert_to_15fps(temp_video_path, output_video_path)

        # Motion-based selection
        st.header("Motion-Based Frames")
        motion_frames = motion_based_selection(output_video_path, num_frames=5)
        for i, frame in enumerate(motion_frames):
            st.image(frame, caption=f"Motion Frame {i + 1}", use_column_width=True)

        # Scene change detection
        st.header("Scene Change-Based Frames")
        scene_change_frames = detect_scene_changes(output_video_path, threshold=0.2)
        for i, frame in enumerate(scene_change_frames):
            st.image(frame, caption=f"Scene Change Frame {i + 1}", use_column_width=True)

        # Clustering-based selection
        st.header("Clustering-Based Frames")
        clustered_frames = cluster_frames(output_video_path, num_clusters=5)
        for i, frame in enumerate(clustered_frames):
            st.image(frame, caption=f"Clustered Frame {i + 1}", use_column_width=True)

        # Clean up temporary files
        os.unlink(temp_video_path)
        os.unlink(output_video_path)

# Run the app
if __name__ == "__main__":
    main()
