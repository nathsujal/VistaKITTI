import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from typing import List, Tuple

class StereoDepthProcessor:
    def __init__(
            self,
            left_images: List[str],
            right_images: List[str],
            calib_files: List[str],
            num_disparities: int=96,
            block_size: int=10
    ):
        assert len(left_images) == len(right_images) == len(calib_files), "Mismatch in number of inputs"
        assert num_disparities%16 == 0, "num_disparities should be divisible by 16"
        
        self.left_images = left_images
        self.right_images = right_images
        self.calib_files = calib_files
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        self.fx_baseline_q = [self.read_calib_files(path) for path in self.calib_files]
        self.disparities = []
        self.depths = []

    def read_calib_files(
            self,
            filepath: str
    ) -> Tuple[np.float64, np.float64, np.ndarray]:
        with open(filepath, "r") as f:
            lines = f.readlines()
        data = {}
        for line in lines:
            if line.startswith("P_rect_02:") or line.startswith("P_rect_03:"):
                key, val = line.split(":", 1)
                data[key.strip()] = np.array([float(x) for x in val.strip().split()]).reshape(3, 4)

        P2 = data["P_rect_02"]
        P3 = data["P_rect_03"]

        fx = P2[0, 0]
        baseline = abs(P2[0, 3] - P3[0, 3]) / fx

        Q = np.zeros((4, 4), dtype=np.float64)
        Q[0, 0] = 1.0
        Q[1, 1] = 1.0
        Q[2, 3] = fx
        Q[3, 2] = 1.0 / (P3[0, 3] / fx)
        Q[0, 3] = -P2[0, 2]
        Q[1, 3] = -P2[1, 2]

        return fx, baseline, Q
    
    def compute_disparity(
            self,
            imgL: str,
            imgR: str
    ) -> np.ndarray:
        disp = self.stereo.compute(imgL, imgR).astype(np.float32) / 16.0
        disp[disp < 0] = 0

        return disp
    
    def disparity_to_depth(
            self,
            disparity: np.ndarray,
            fx: np.float64,
            baseline: np.float64,
            min_depth: int = 0,
            max_depth: int = 80.0
    ) -> np.ndarray:
        depth_map = np.zeros_like(disparity, dtype=np.float32)
        valid = disparity > 0
        depth_map[valid] = (fx * baseline) / disparity[valid]
        depth_map[depth_map > max_depth] = max_depth
        depth_map[depth_map < min_depth] = 0

        return depth_map
    
    def get_disparity_maps(self) -> List[np.ndarray]:
        if not self.disparities:
            for i in range(len(self.left_images)):
                imgL = cv2.imread(self.left_images[i], cv2.IMREAD_GRAYSCALE)
                imgR = cv2.imread(self.right_images[i], cv2.IMREAD_GRAYSCALE)
                disp = self.compute_disparity(imgL, imgR)
                self.disparities.append(disp)
        return self.disparities
    
    def get_depth_maps(self) -> List[np.ndarray]:
        if not self.depths:
            disparities = self.get_disparity_maps()
            for i, disp in enumerate(disparities):
                fx, baseline, _ = self.fx_baseline_q[i]
                depth = self.disparity_to_depth(disp, fx, baseline)
                self.depths.append(depth)
        return self.depths
    
if __name__ == "__main__":
    left_imgs = sorted(glob.glob("datasets/data_scene_flow/training/image_2/*_10.png"))
    right_imgs = sorted(glob.glob("datasets/data_scene_flow/training/image_3/*_10.png"))
    calib_files = sorted(glob.glob("datasets/data_scene_flow_calib/training/calib_cam_to_cam/*.txt"))

    # Process first 5 stereo pairs
    processor = StereoDepthProcessor(left_imgs[:5], right_imgs[:5], calib_files[:5])

    disparities = processor.get_disparity_maps()
    depths = processor.get_depth_maps()

    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.title("Disparity Map")
    plt.imshow(disparities[0], cmap='plasma')
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Depth Map")
    plt.imshow(depths[0], cmap='inferno')
    plt.axis("off")
    plt.show()
