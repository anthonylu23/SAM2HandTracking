# SAM2HandTracking
Hand segmentation pipeline using OpenCV, MediaPipe, and SAM2. The script locates wrist landmarks and uses those points as prompts for the SAM2 model, which then propagates hand segmentation masks across all frames in the video.

# Getting Started
Requirements:
Python 3.7+
[OpenCV]([url](https://pypi.org/project/opencv-python/))
[MediaPipe]([url](https://pypi.org/project/mediapipe/))
[PyTorch]([url](https://pytorch.org/))
[Supervision]([url](https://github.com/roboflow/supervision))

# Installation
Install SAM2 [here]([url](https://github.com/facebookresearch/sam2/tree/main?tab=readme-ov-file#model-description)).

Install dependencies

Inside the cloned SAM2 repo, clone this repository. 

# Usage
Inside demo.py, set the paths to your input video and your output video. Run demo.py from the SAM2 repo.
