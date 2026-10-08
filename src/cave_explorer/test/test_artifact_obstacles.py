"""Tests for artefact obstacle voxels built from depth inside detection boxes."""

import numpy as np
import pytest

from cave_explorer.artifact_obstacles import ArtifactObstacles
from cave_explorer.artifact_tracker import SIM_CAMERA_MODEL
from cave_explorer.interfaces import Detection2D, DetectionBatch, GREEN_CRYSTALS

WIDTH, HEIGHT = 720, 480
# Camera at (0, 0, 0.5) in map looking along +x (as in test_artifact_tracker).
OPTICAL_IN_MAP = {'translation': (0.0, 0.0, 0.5), 'rotation': (-0.5, 0.5, -0.5, 0.5)}


def batch(*boxes):
    return DetectionBatch(1, 'camera_link', WIDTH, HEIGHT,
                          tuple(Detection2D(GREEN_CRYSTALS, 0.9, box) for box in boxes))


def wall(depth_m):
    return np.full((HEIGHT, WIDTH), depth_m, dtype=np.float32)


def test_box_pixels_become_voxels_at_the_object():
    obstacles = ArtifactObstacles()
    added = obstacles.add(batch((320, 200, 400, 280)), wall(3.0), SIM_CAMERA_MODEL,
                          OPTICAL_IN_MAP)
    points = obstacles.points()
    assert added == len(points) > 0
    assert points[:, 0] == pytest.approx(3.0, abs=0.15)
    assert points[:, 2].min() >= 0.15 - 1e-6 and points[:, 2].max() <= 2.0 + 0.15
    # Only the box's width: 80 px at 3 m is about 1.15 m.
    assert np.ptp(points[:, 1]) < 1.4


def test_pixels_outside_boxes_are_ignored():
    obstacles = ArtifactObstacles()
    obstacles.add(batch(), wall(3.0), SIM_CAMERA_MODEL, OPTICAL_IN_MAP)
    assert len(obstacles) == 0


def test_floor_below_min_height_is_dropped():
    obstacles = ArtifactObstacles(min_height_m=0.15)
    # Bottom rows of the image look down at the floor: z well below 0.15 m.
    obstacles.add(batch((320, 440, 400, 480)), wall(1.5), SIM_CAMERA_MODEL, OPTICAL_IN_MAP)
    assert len(obstacles) == 0


@pytest.mark.parametrize('depth_m', [0.2, 7.0, np.nan, np.inf])
def test_invalid_or_far_depth_adds_nothing(depth_m):
    obstacles = ArtifactObstacles(max_range_m=6.0)
    obstacles.add(batch((320, 200, 400, 280)), wall(depth_m), SIM_CAMERA_MODEL, OPTICAL_IN_MAP)
    assert len(obstacles) == 0


def test_voxels_accumulate_without_duplicates():
    obstacles = ArtifactObstacles()
    box = (320, 200, 400, 280)
    first = obstacles.add(batch(box), wall(3.0), SIM_CAMERA_MODEL, OPTICAL_IN_MAP)
    assert obstacles.add(batch(box), wall(3.0), SIM_CAMERA_MODEL, OPTICAL_IN_MAP) == 0
    obstacles.add(batch((100, 200, 180, 280)), wall(3.0), SIM_CAMERA_MODEL, OPTICAL_IN_MAP)
    assert len(obstacles) > first
    obstacles.reset()
    assert len(obstacles) == 0
