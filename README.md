# SAM2HandTracking
Hand segmentation pipeline using OpenCV, MediaPipe, and SAM2. The script locates wrist landmarks and uses those points as prompts for the SAM2 model, which then propagates hand segmentation masks across all frames in the video.

# Usage
Inside the cloned SAM2 repo, clone this repository. 
Inside demo.py, set the paths to your input video and your output video. Run demo.py from the SAM2 repo.

# Notes
Including only wrist landmarks were chosen over including all or other hand landmarks to improve performance, although the pipeline could be modified to accomodate other landmarks for prompts. The checkpoints for SAM2 can also be changed for improved accuracy at the cost of performance.

Possible issue: The SAM2 model uses Python library decord, which may not run if you are running python 3.11+. A virtual environment of a past version of Python may be needed. 
