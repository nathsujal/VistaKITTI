import os
import sys
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from I_depth_estimation.stereo_depth import StereoDepthProcessor
from II_semantic_segmentation.semantic_segmenter import SemanticSegmenter


def visualize_all_maps(
    left_img: np.ndarray,
    right_img: np.ndarray,
    disp: np.ndarray,
    depth: np.ndarray,
    semantic_seg: np.ndarray,
    figsize=(12, 10),
    index: int = None
):
    """
    Display images in the following layout:

    left_image   | right_image
    --------------------------
         disparity map
    --------------------------
           depth map
    --------------------------
       semantic segmented image
    """
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=figsize)

    # 1. Top row: left and right images side by side
    ax_top = axes[0]
    combined_lr = np.concatenate((left_img, right_img), axis=1)
    ax_top.imshow(combined_lr[..., ::-1])  # Convert BGR to RGB
    title = f"Left Image | Right Image"
    if index is not None:
        title += f" (Sample {index + 1})"
    ax_top.set_title(title)
    ax_top.axis("off")

    # 2. Disparity map
    axes[1].imshow(disp, cmap='gray')
    axes[1].set_title("Disparity Map")
    axes[1].axis("off")

    # 3. Depth map
    axes[2].imshow(depth, cmap='inferno')
    axes[2].set_title("Depth Map")
    axes[2].axis("off")

    # 4. Semantic segmented image
    axes[3].imshow(semantic_seg[..., ::-1])  # BGR to RGB
    axes[3].set_title("Semantic Segmentation")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    # Load image paths
    left_imgs = sorted(glob.glob("datasets/data_scene_flow/training/image_2/*_10.png"))
    right_imgs = sorted(glob.glob("datasets/data_scene_flow/training/image_3/*_10.png"))
    calib_files = sorted(glob.glob("datasets/data_scene_flow_calib/training/calib_cam_to_cam/*.txt"))

    # Limit to first N samples (or adjust as needed)
    N = 5
    left_imgs = left_imgs[:N]
    right_imgs = right_imgs[:N]
    calib_files = calib_files[:N]

    # Initialize processors
    segmenter = SemanticSegmenter(model_path="models/best.pt")
    processor = StereoDepthProcessor(left_imgs, right_imgs, calib_files)

    disparities = processor.get_disparity_maps()
    depths = processor.get_depth_maps()

    # Process and display each stereo pair
    for i in range(len(left_imgs)):
        left_img = cv2.imread(left_imgs[i])
        right_img = cv2.imread(right_imgs[i])
        disparity = disparities[i]
        depth = depths[i]

        semantic_seg = segmenter.segment_images([left_img], alpha=0.9, return_labels=True)[0]

        visualize_all_maps(left_img, right_img, disparity, depth, semantic_seg, index=i)


if __name__ == "__main__":
    main()
