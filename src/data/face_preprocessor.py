import os
from pathlib import Path
from typing import Tuple, Dict, Any, Union, Optional
import numpy as np
import cv2
from PIL import Image

class YuNetFacePreprocessor:
    """
    Standardized, aspect-ratio-preserving face detection and preprocessing pipeline
    using OpenCV YuNet (yunet.onnx).
    
    Guarantees:
    1. Detection of human faces with configurable confidence threshold.
    2. Exact 1:1 square crop around the face with margin (default: 25% context padding).
    3. Proper boundary reflection padding when face is near image border.
    4. Deterministic, aspect-ratio-preserving fallback (center-square crop) when no face is detected.
    5. Complete transparency via metadata recording (detection status, bounding box, dimensions).
    6. Absolutely ZERO anisotropic stretching.
    """
    def __init__(
        self,
        model_path: str = "yunet.onnx",
        target_size: Tuple[int, int] = (224, 224),
        margin: float = 0.25,
        score_threshold: float = 0.60,
        nms_threshold: float = 0.30,
        version: str = "phase7_v1"
    ):
        self.model_path = Path(model_path)
        self.target_size = target_size
        self.margin = margin
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.version = version

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"[YuNetFacePreprocessor] Model weights not found at: {self.model_path.resolve()}"
            )

    def _get_detector(self, width: int, height: int):
        detector = cv2.FaceDetectorYN.create(
            model=str(self.model_path),
            config="",
            input_size=(width, height),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=5
        )
        detector.setInputSize((width, height))
        return detector

    def process_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Processes a single image:
        - Detects primary face with YuNet.
        - Crops square bounding box with context margin.
        - Falls back to center-square crop if detection fails.
        - Resizes to target_size (default: 224x224).
        
        Returns:
            processed_pil_img: PIL Image of size target_size.
            meta: Dictionary containing preprocessing telemetry.
        """
        # Load image into RGB numpy array
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            pil_orig = Image.open(img_path).convert("RGB")
            orig_path_str = str(img_path.resolve())
        elif isinstance(image_input, Image.Image):
            pil_orig = image_input.convert("RGB")
            orig_path_str = "in_memory_pil"
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                image_input = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                image_input = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            pil_orig = Image.fromarray(image_input)
            orig_path_str = "in_memory_numpy"
        else:
            raise TypeError(f"Unsupported image_input type: {type(image_input)}")

        orig_w, orig_h = pil_orig.size
        bgr = cv2.cvtColor(np.array(pil_orig), cv2.COLOR_RGB2BGR)

        meta = {
            "original_path": orig_path_str,
            "original_width": orig_w,
            "original_height": orig_h,
            "original_aspect_ratio": round(orig_w / float(orig_h), 4),
            "target_size": list(self.target_size),
            "preprocessing_version": self.version,
            "face_detection_status": "none",
            "fallback_used": "none",
            "confidence_score": 0.0,
            "bbox_raw": None,
            "crop_coords": None
        }

        # Step 1: Detect face using OpenCV YuNet
        faces = None
        try:
            detector = self._get_detector(orig_w, orig_h)
            _, faces = detector.detect(bgr)
        except Exception as e:
            meta["detection_error"] = str(e)

        # Step 2: Handle detected face vs fallback
        if faces is not None and len(faces) > 0:
            # Select the largest face by area
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            fx, fy, fw, fh = float(largest_face[0]), float(largest_face[1]), float(largest_face[2]), float(largest_face[3])
            conf = float(largest_face[-1])

            meta["face_detection_status"] = "yunet_detected"
            meta["confidence_score"] = round(conf, 4)
            meta["bbox_raw"] = [round(fx, 2), round(fy, 2), round(fw, 2), round(fh, 2)]

            # Compute face center
            cx = fx + fw / 2.0
            cy = fy + fh / 2.0

            # Compute square crop dimension with context margin
            side = int(round(max(fw, fh) * (1.0 + self.margin)))
            side = max(side, 32) # Minimum crop size

            x1 = int(round(cx - side / 2.0))
            y1 = int(round(cy - side / 2.0))
            x2 = x1 + side
            y2 = y1 + side

            meta["crop_coords"] = [x1, y1, x2, y2]

            # Compute boundary padding if crop extends outside original image
            pad_left = max(0, -x1)
            pad_top = max(0, -y1)
            pad_right = max(0, x2 - orig_w)
            pad_bottom = max(0, y2 - orig_h)

            if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                bgr_padded = cv2.copyMakeBorder(
                    bgr, pad_top, pad_bottom, pad_left, pad_right,
                    borderType=cv2.BORDER_REFLECT_101
                )
            else:
                bgr_padded = bgr

            crop_y1 = y1 + pad_top
            crop_y2 = y2 + pad_top
            crop_x1 = x1 + pad_left
            crop_x2 = x2 + pad_left

            cropped_bgr = bgr_padded[crop_y1:crop_y2, crop_x1:crop_x2]

        else:
            # Deterministic Fallback: Never stretch!
            meta["confidence_score"] = 0.0

            if orig_w == orig_h:
                # Already a 1:1 square image
                cropped_bgr = bgr
                meta["face_detection_status"] = "no_face_detected"
                meta["fallback_used"] = "square_direct"
                meta["crop_coords"] = [0, 0, orig_w, orig_h]
            else:
                # Center-square crop preserving aspect ratio
                min_dim = min(orig_w, orig_h)
                x1 = (orig_w - min_dim) // 2
                y1 = (orig_h - min_dim) // 2
                x2 = x1 + min_dim
                y2 = y1 + min_dim

                cropped_bgr = bgr[y1:y2, x1:x2]
                meta["face_detection_status"] = "no_face_detected"
                meta["fallback_used"] = "center_square_crop"
                meta["crop_coords"] = [x1, y1, x2, y2]

        # Step 3: Resize square crop to exact target size (224x224) using bilinear interpolation
        # Since input is strictly square (H == W), aspect ratio is exactly 1:1 preserved!
        resized_bgr = cv2.resize(
            cropped_bgr,
            self.target_size,
            interpolation=cv2.INTER_LINEAR
        )
        resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        final_pil = Image.fromarray(resized_rgb)

        return final_pil, meta
