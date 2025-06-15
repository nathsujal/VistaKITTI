# Semantic Segmentation

This module provides a YOLOv8-based pipeline for semantic segmentation using the KITTI semantic dataset.

## 🧠 Goal

Segment and classify scene elements like roads, vehicles, buildings, poles, and pedestrians from camera images.

## 🛠 Implementation Details

- Model: <code>YOLOv8n-seg</code> trained on KITTI semantic masks.

- Dataset Format: Converted KITTI labels into YOLO-style format (images + masks).

- Augmentations: Leveraged built-in YOLOv8 transformations.

## 📁 Files

- <code>KITTI_semantic_to_YOLOv8_masks.ipynb</code>: Parses raw KITTI labels, applies class mapping, and splits dataset.

- <code>train_yolo_seg.ipynb</code>: Generates <code>dataset.yaml</code> and trains YOLOv8.

- <code>semantic_segmenter.py</code>: Loads trained model, segments images with overlayed masks and optional class labels.

### 🧪 Example Usage
```python
from semantic_segmenter import SemanticSegmenter
segmenter = SemanticSegmenter("models/best.pt")
images = ["datasets/data_semantics/testing/image_2/000024_10.png"]
results = segmenter.segment_images(images, alpha=0.6, return_labels=True)
```