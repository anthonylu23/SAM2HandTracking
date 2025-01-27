# SAM2HandTracking
Hand segmentation pipeline using OpenCV, MediaPipe, and SAM2. The script locates wrist landmarks and uses those points as prompts for the SAM2 model, which then propagates hand segmentation masks across all frames in the video.

# Usage
Inside the cloned SAM2 repo, clone this repository. 
Inside demo.py, set the paths to your input video and your output video. Run demo.py from the SAM2 repo.
