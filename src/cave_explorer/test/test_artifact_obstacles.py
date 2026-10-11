"""Tests for artefact obstacle voxels built from depth inside detection boxes."""

import numpy as np
import pytest

from cave_explorer.artifact_obstacles import ArtifactObstacles, keep_out_points
from cave_explorer.artifact_tracker import SIM_CAMERA_MODEL
from cave_explorer.interfaces import (
    ArtifactTrack,
    Detection2D,
    DetectionBatch,
    GREEN_CRYSTALS,
    WHITE_SPHERE,
)

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


def artefact(class_name, x, y, artifact_id=1):
    return ArtifactTrack(artifact_id, class_name, (x, y, 0.3), 0, 3, True)


def test_keep_out_disk_is_half_the_measured_width_for_any_type():
    sizes = {1: 5.4, 2: 1.6}
    points = keep_out_points([artefact(GREEN_CRYSTALS, 10.0, 5.0),
                              artefact(WHITE_SPHERE, 20.0, 5.0, artifact_id=2)],
                             sizes.get, (0.0, 0.0))
    near_crystal = np.hypot(points[:, 0] - 10.0, points[:, 1] - 5.0)
    near_sphere = np.hypot(points[:, 0] - 20.0, points[:, 1] - 5.0)
    assert near_crystal[near_crystal < 5].max() <= 2.7 + 1e-5
    assert near_crystal.min() < 0.15                            # filled, not a ring
    assert 0.65 < near_sphere[near_sphere < 5].max() <= 0.8 + 1e-5
    assert np.all(points[:, 2] == pytest.approx(0.5))          # inside Nav2's 0.1-2.5 m band


def test_keep_out_disk_needs_a_measured_width_and_is_capped():
    track = [artefact(GREEN_CRYSTALS, 10.0, 5.0)]
    assert len(keep_out_points(track, lambda _id: None, (0.0, 0.0))) == 0
    points = keep_out_points(track, lambda _id: 20.0, (0.0, 0.0))
    assert np.hypot(points[:, 0] - 10.0, points[:, 1] - 5.0).max() <= 3.0 + 1e-5


def test_keep_out_disk_never_covers_the_robot():
    crystals = [artefact(GREEN_CRYSTALS, 0.0, 0.0)]
    size = lambda _id: 5.4      # noqa: E731
    points = keep_out_points(crystals, size, (2.0, 0.0))
    assert np.hypot(points[:, 0] - 2.0, points[:, 1]).min() >= 1.0 - 1e-5
    assert len(keep_out_points(crystals, size, (0.5, 0.0))) == 0
    assert len(keep_out_points([], size, (0.0, 0.0))) == 0


def test_contact_patch_is_ahead_of_the_robot_and_clear_of_it_after_backing_off():
    obstacles = ArtifactObstacles()
    assert obstacles.mark_contact(5.0, 5.0, 0.0) > 0
    points = obstacles.points()
    assert points[:, 0].min() >= 5.0 + 0.3 - 0.15 and points[:, 0].max() <= 5.0 + 1.0 + 0.15
    assert np.abs(points[:, 1] - 5.0).max() <= 0.6 + 0.15
    # After backing off 0.8 m the patch is ahead of the 0.5 m chassis front.
    assert points[:, 0].min() > 5.0 - 0.8 + 0.5
    assert obstacles.mark_contact(5.0, 5.0, 0.0) == 0          # same place: nothing new
    obstacles.reset()
    assert len(obstacles) == 0


def test_contact_patch_follows_the_heading():
    obstacles = ArtifactObstacles()
    obstacles.mark_contact(0.0, 0.0, np.pi / 2)
    points = obstacles.points()
    assert points[:, 1].min() > 0.1 and np.abs(points[:, 0]).max() <= 0.75


def test_keep_out_disks_only_for_the_given_classes():
    tracks = [artefact(GREEN_CRYSTALS, 10.0, 5.0), artefact(WHITE_SPHERE, 20.0, 5.0, artifact_id=2)]
    points = keep_out_points(tracks, lambda _id: 2.0, (0.0, 0.0), classes={WHITE_SPHERE})
    assert len(points) > 0
    assert np.hypot(points[:, 0] - 20.0, points[:, 1] - 5.0).max() <= 1.0 + 1e-5
