# Video Frame Selection

This is a Streamlit app that allows users to upload a 60-second video and extract the best 20 frames using three methods:
1. **Motion-Based Selection**: Frames with the most motion.
2. **Scene Change-Based Selection**: Frames where significant scene changes occur.
3. **Clustering-Based Selection**: Representative frames from different clusters.

The app first converts the video to 15 FPS and then applies the frame selection methods.

## Features
- Upload a 60-second video (MP4, AVI, or MOV).
- Convert the video to 15 FPS.
- Display 20 frames for each of the three methods.


  Results
After evaluating the three frame selection techniques—motion-based, scene change-based, and clustering-based—it is evident that the clustering-based technique provides the most diverse and representative set of frames. This conclusion is supported by the following observations:

Diversity of Frames:

The clustering-based technique groups similar frames together and selects one representative frame from each cluster. This ensures that the selected frames cover a wide range of scenes and content, making it ideal for applications requiring diverse frame selection.

In contrast, motion-based and scene change-based techniques tend to focus on specific aspects (e.g., motion or abrupt changes), which may result in less diverse frames.

Representativeness:

Clustering ensures that the selected frames are representative of the entire video. This is particularly useful for tasks like video summarization or shoppable item detection, where capturing the essence of the video is crucial.

Motion-based and scene change-based techniques may miss important frames if the motion or scene changes are subtle.

Verification:

Users can verify these results by uploading a sample video to the app. The clustering-based section consistently displays frames that are visually distinct and cover a broader range of scenes compared to the other techniques.

Example:

When a sample video is uploaded, the clustering-based technique selects frames from different parts of the video, ensuring that no significant scene or object is overlooked. For instance:

Motion-Based Frames: May focus on frames with high activity, such as moving objects or people.

Scene Change-Based Frames: May focus on frames where the background or lighting changes abruptly.

Clustering-Based Frames: Captures a mix of frames, including those with subtle changes, diverse backgrounds, and varying objects.
