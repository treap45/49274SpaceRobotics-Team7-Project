"""Selective image collection for Perception 1.

Owner: A.
Task: Perception 1.

This module stores selected RGB camera images for dataset creation.
Images are saved manually rather than saving every camera frame, allowing
the operator to collect useful views, different distances and angles,
and negative examples.
"""

from pathlib import Path

import cv2


class DatasetCollector:
    """Store selected camera images for the Perception 1 dataset."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()

        # Create the dataset directory automatically if required.
        self.dataset_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Continue numbering from images already in the directory.
        self.image_count = self._next_image_index()

        # Keep the latest raw camera image.
        self.latest_image = None

    def _next_image_index(self) -> int:
        """Find the next available image number."""

        highest_index = -1

        for image_path in self.dataset_dir.glob('image_*.jpg'):
            try:
                index = int(image_path.stem.split('_')[-1])
                highest_index = max(highest_index, index)
            except ValueError:
                # Ignore files that do not follow image_XXXX.jpg.
                continue

        return highest_index + 1

    def update_image(self, image):
        """Store the latest raw RGB/BGR camera image."""

        if image is None:
            return

        self.latest_image = image.copy()

    def save_latest_image(self):
        """Save one selected image from the latest camera frame.

        Returns:
            tuple[bool, str]:
                success flag and a human-readable result message.
        """

        if self.latest_image is None:
            return False, 'No camera image received yet.'

        filename = f'image_{self.image_count:04d}.jpg'
        filepath = self.dataset_dir / filename

        success = cv2.imwrite(
            str(filepath),
            self.latest_image,
        )

        if not success:
            return False, f'Failed to save {filename}'

        self.image_count += 1

        return True, f'Saved {filename}'

    @property
    def saved_image_count(self) -> int:
        """Return the next image index used by the collector."""

        return self.image_count
