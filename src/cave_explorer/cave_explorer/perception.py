"""YOLO-based artefact detection and image annotation.

Owner: A.
Tasks: Perception 2.

The detector receives an RGB/BGR camera image and returns detections using
the shared DetectionBatch interface. Bounding boxes remain in the original
camera image coordinates so they can be used by Perception 3.
"""

from pathlib import Path
import time

import cv2
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory

from .interfaces import Detection2D, DetectionBatch


class Detector:
    """Detect Martian cave artefacts using the trained YOLOv8 model."""

    def __init__(
        self,
        model_path: str | None = None,
        confidence_threshold: float = 0.15,
    ):
        if model_path is None:
            # Retrained on simulator-labelled images (datasets/autolabel); the
            # earlier hand-collected model is models/artifact_yolov8n.pt.
            package_share = get_package_share_directory('cave_explorer')
            model_path = (
                Path(package_share)
                / 'models'
                / 'artifact_yolov8n_autolabel.pt'
            )

        self.model_path = Path(model_path).expanduser().resolve()
        self.confidence_threshold = confidence_threshold

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f'YOLO model not found: {self.model_path}'
            )

        self.model = YOLO(str(self.model_path))

        # Last inference time, useful for evaluation/reporting.
        self.last_inference_ms = None

    def detect(
        self,
        rgb_image,
        stamp_ns: int,
        source_frame_id: str,
    ) -> DetectionBatch:
        """Run YOLO and return detections in original image coordinates."""

        if rgb_image is None:
            raise ValueError('rgb_image must not be None')

        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError(
                'rgb_image must be a three-channel image'
            )

        image_height, image_width = rgb_image.shape[:2]

        start_time = time.perf_counter()

        results = self.model(
            rgb_image,
            conf=self.confidence_threshold,
            verbose=False,
        )

        self.last_inference_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        detections = []

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = self.model.names[class_id]

            # YOLO xyxy is already expressed in coordinates of the
            # original input image.
            x_min, y_min, x_max, y_max = (
                float(value)
                for value in box.xyxy[0].tolist()
            )

            # Keep coordinates inside the original image boundaries.
            x_min = max(0.0, min(x_min, float(image_width)))
            y_min = max(0.0, min(y_min, float(image_height)))
            x_max = max(0.0, min(x_max, float(image_width)))
            y_max = max(0.0, min(y_max, float(image_height)))

            # Ignore invalid boxes rather than passing invalid geometry
            # to the shared Detection2D interface.
            if x_max <= x_min or y_max <= y_min:
                continue

            detection = Detection2D(
                class_name=class_name,
                confidence=confidence,
                bbox_xyxy=(
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                ),
            )

            detections.append(detection)

        return DetectionBatch(
            stamp_ns=int(stamp_ns),
            source_frame_id=source_frame_id,
            image_width=image_width,
            image_height=image_height,
            detections=tuple(detections),
        )

    def annotate(self, rgb_image, batch: DetectionBatch):
        """Return a copy of the image with detection boxes and labels."""

        annotated = rgb_image.copy()

        for detection in batch.detections:
            x_min, y_min, x_max, y_max = detection.bbox_xyxy

            x1 = int(round(x_min))
            y1 = int(round(y_min))
            x2 = int(round(x_max))
            y2 = int(round(y_max))

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = (
                f'{detection.class_name} '
                f'{detection.confidence:.2f}'
            )

            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        return annotated