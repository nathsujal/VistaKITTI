import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import List, Union

class SemanticSegmenter:
    def __init__(
            self,
            model_path: Union[str, Path] = "models/best.pt"
    ):
        """
        Initialize the YOLOv8 segmenter with a trained model.

        :param model_path: Path to YOLOv8 segmentation model (e.g., best.pt)
        """
        self.model = YOLO(str(model_path))
        self.class_names = self.model.names

    def segment_images(
        self,
        image_list: List[Union[str, np.ndarray]],
        alpha: float = 0.5,
        return_labels: bool = False
    ) -> List[np.ndarray]:
        """
        Perform segmentation on a list of images.

        :param image_list: List of image paths or np.ndarray images
        :param alpha: Transparency factor for mask overlay (0 to 1)
        :param return_labels: Whether to return label names and probabilities
        :param visualize: Whether to show the images using OpenCV window
        :param save_dir: Directory path to save the output images
        :return: List of segmented images or (image, label info) tuples
        """
        results_list = []

        for idx, image_input in enumerate(image_list):
            # Load image
            image = self._read_image(image_input)
            orig_image = image.copy()

            # Run YOLOv8 segmentation
            results = self.model.predict(image, verbose=False, conf=0.25)[0]

            # Initialize overlay
            overlay = np.zeros_like(image)
            label_info = []

            for mask, cls_id, conf in zip(results.masks.data, results.boxes.cls, results.boxes.conf):
                color = self._get_color(int(cls_id))
                
                # Resize mask
                binary_mask = (mask.cpu().numpy() * 255).astype(np.uint8)
                binary_mask_resized = cv2.resize(binary_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                binary_mask_3c = np.stack([binary_mask_resized] * 3, axis=-1)

                # Apply color to overlay where mask is present
                overlay = np.where(binary_mask_3c > 0, color, overlay)

                # Draw label and confidence
                if return_labels:
                    label = self.class_names[int(cls_id)]
                    prob = float(conf)
                    label_text = f"{label} {prob:.2f}"

                    # Get contour to locate region
                    contours, _ = cv2.findContours(binary_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # Pick largest region
                        c = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(c)

                        # Calculate maximum font scale that fits inside bounding box
                        for font_scale in np.linspace(2.0, 0.3, 50):
                            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                            if text_w <= w and text_h <= h:
                                break  # Found the largest font that fits

                        # Compute center position to align text inside mask bounding box
                        text_x = x + (w - text_w) // 2
                        text_y = y + (h + text_h) // 2

                        # Check if the text center lies within the actual mask, shift until it does
                        max_attempts = 10
                        while binary_mask_resized[text_y - text_h // 2, text_x] == 0 and max_attempts > 0:
                            text_y -= 1  # Move upward slightly if not inside mask
                            max_attempts -= 1

                        # Draw text
                        cv2.putText(
                            overlay, label_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.7,
                            (0, 0, 0), thickness=1, lineType=cv2.LINE_AA
                        )

                    label_info.append({
                        "class_id": int(cls_id),
                        "label": label,
                        "confidence": prob
                    })

            # Blend with original image
            blended = cv2.addWeighted(orig_image, 1 - alpha, overlay, alpha, 0)

            # results_list.append((blended, label_info) if return_labels else blended)
            results_list.append(blended)

        return results_list

    def _read_image(self, img: Union[str, np.ndarray]) -> np.ndarray:
        """
        Load image from path or pass-through if already ndarray.
        """
        if isinstance(img, str):
            img = cv2.imread(img)
            if img is None:
                raise FileNotFoundError(f"Image not found at: {img}")
        elif isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
        else:
            raise ValueError("Unsupported image input type.")
        return img

    def _get_color(self, idx: int) -> np.ndarray:
        """
        Generate a deterministic color for a given class index.
        """
        np.random.seed(idx)
        return np.random.randint(0, 255, size=3, dtype=np.uint8)

if __name__ == "__main__":
    image_paths = [
        'datasets/data_semantics/testing/image_2/000024_10.png',
        'datasets/data_semantics/testing/image_2/000087_10.png'
        ]
    
    segmenter = SemanticSegmenter(model_path="models/best.pt")

    results = segmenter.segment_images(
        image_paths,
        alpha=0.7,
        return_labels=True
    )

    for img in results:
        cv2.imshow('Semantic Segmentation', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()