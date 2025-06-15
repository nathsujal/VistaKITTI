# Depth Estimation

This module implements a depth estimation pipeline using stereo image pairs and KITTI calibration data.

## 🔍 Problem Overview

Humans perceive depth using binocular vision; machines can mimic this using stereo cameras. By analyzing the disparity (pixel difference) between corresponding points in the left and right images, we estimate depth. This is the fundamental principle behind stereo vision.

## 📐 Mathematics of Stereo Vision

1. Disparity (d) is defined as:

```math
d = x_L - x_R
```

2. Depth (Z) can be computed using:

```math
Z = {f.B \over d}
```

Where:

 f = focal length of the camera

 B = baseline (distance between two cameras)

 d = disparity

## 🛠 Implementation Details

- Stereo Method: SGBM (Semi-Global Block Matching)

- Disparity Range: Tuned using <code>numDisparities</code>, must be divisible by 16.

- Depth Map Normalization: Values clipped between <code>0.5m</code> and <code>80m</code> to avoid noise.

- Calibration: Uses <code>P_rect_02</code> and <code>P_rect_03</code> from KITTI’s <code>calib_cam_to_cam.txt</code>.

## 📁 Files

- <code>depth-analysis.ipynb</code>: Jupyter notebook to visualize disparity and depth.

- <code>stereo_depth.py</code>: Class-based module with:

    - <code>get_disparity_maps()</code>

    - <code>get_depth_maps()</code>

    - Automatic calibration and Q-matrix construction

### 🧪 Example Usage
```python
from stereo_depth import StereoDepthProcessor
import glob

left_imgs = sorted(glob.glob("datasets/data_scene_flow/training/image_2/*_10.png"))
right_imgs = sorted(glob.glob("datasets/data_scene_flow/training/image_3/*_10.png"))
calibs = sorted(glob.glob("datasets/data_scene_flow_calib/training/calib_cam_to_cam/*.txt"))

processor = StereoDepthProcessor(left_imgs[:5], right_imgs[:5], calibs[:5])
disparities = processor.get_disparity_maps()
depths = processor.get_depth_maps()
```