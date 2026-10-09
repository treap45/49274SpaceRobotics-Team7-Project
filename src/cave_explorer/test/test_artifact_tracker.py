"""Tests for artefact localisation, association and id assignment (Workplan 4.3)."""

import math

from geometry_msgs.msg import TransformStamped
import numpy as np
import pytest

from cave_explorer.artifact_tracker import ArtifactTracker, SIM_CAMERA_MODEL
from cave_explorer.interfaces import (
    BLUE_MUSHROOMS,
    Detection2D,
    DetectionBatch,
    GREEN_ALIEN,
    ICE_FORMATION,
)

SECOND = 10**9
WIDTH, HEIGHT = 720, 480
# Optical frame (z forward, x right, y down) of a camera at (0, 0, 0.5) in map
# looking along +x: optical z -> map x, optical x -> map -y, optical y -> map -z.
OPTICAL_IN_MAP = {
    'translation': (0.0, 0.0, 0.5),
    'rotation': (-0.5, 0.5, -0.5, 0.5),
}


def depth_image(value=np.inf):
    return np.full((HEIGHT, WIDTH), value, dtype=np.float32)


def batch(stamp_s, *detections):
    return DetectionBatch(int(stamp_s * SECOND), 'camera_link', WIDTH, HEIGHT,
                          tuple(detections))


def centred_box(class_name=GREEN_ALIEN, half=40, u=360.0, v=240.0):
    return Detection2D(class_name, 0.9, (u - half, v - half, u + half, v + half))


def object_at(depth_m, half=40, u=360, v=240):
    depth = depth_image()
    depth[v - half:v + half, u - half:u + half] = depth_m
    return depth


def see(tracker, stamp_s, depth_m, class_name=GREEN_ALIEN):
    return tracker.update(batch(stamp_s, centred_box(class_name)), object_at(depth_m),
                          SIM_CAMERA_MODEL, OPTICAL_IN_MAP)


def test_box_centre_projects_straight_ahead_into_map():
    tracker = ArtifactTracker(confirm_count=1)
    (track,) = see(tracker, 1, 3.0)
    assert track.position_map == pytest.approx((3.0, 0.0, 0.5), abs=1e-6)


def test_off_centre_pixel_lands_to_the_right():
    tracker = ArtifactTracker(confirm_count=1)
    u = 360 + SIM_CAMERA_MODEL['fx']        # one focal length right: 45 degrees
    depth = object_at(2.0, u=int(round(u)))
    (track,) = tracker.update(batch(1, centred_box(u=u)), depth, SIM_CAMERA_MODEL,
                              OPTICAL_IN_MAP)
    assert track.position_map[0] == pytest.approx(2.0, abs=0.02)
    assert track.position_map[1] == pytest.approx(-2.0, abs=0.02)


def test_repeated_observation_keeps_one_track_and_does_not_crash():
    """Regression: ArtifactTrack is frozen, so updates must not assign to it."""
    tracker = ArtifactTracker(confirm_count=1)
    see(tracker, 1, 3.0)
    (track,) = see(tracker, 2, 3.2)
    assert track.observation_count == 2
    assert track.position_map[0] == pytest.approx(3.1, abs=1e-6)
    assert track.last_seen_ns == 2 * SECOND


def test_id_assigned_only_once_confirmed():
    tracker = ArtifactTracker(confirm_count=3)
    assert see(tracker, 1, 3.0) == []
    assert see(tracker, 2, 3.0) == []
    (track,) = see(tracker, 3, 3.0)
    assert track.artifact_id == 1 and track.confirmed


def test_unconfirmed_noise_does_not_consume_ids():
    tracker = ArtifactTracker(confirm_count=2)
    see(tracker, 1, 6.0)                    # one-off bad observation
    see(tracker, 2, 3.0)
    (track,) = see(tracker, 3, 3.0)
    assert track.artifact_id == 1


def test_same_class_far_apart_stays_separate_and_classes_never_merge():
    tracker = ArtifactTracker(confirm_count=1)
    see(tracker, 1, 3.0)
    see(tracker, 2, 6.0)
    tracks = see(tracker, 3, 3.0, BLUE_MUSHROOMS)
    assert len(tracks) == 3
    assert len({t.artifact_id for t in tracks}) == 3


@pytest.mark.parametrize('depth_m', [0.2, 10.0, 12.0, np.nan, np.inf, 0.0])
def test_depth_outside_clip_is_rejected_not_clamped(depth_m):
    tracker = ArtifactTracker(confirm_count=1)
    assert see(tracker, 1, depth_m) == []
    assert 'valid depth pixels' in tracker.last_rejections[0]


def test_background_around_the_object_does_not_pull_the_depth():
    tracker = ArtifactTracker(confirm_count=1)
    depth = object_at(8.0, half=40)              # far wall fills the whole box
    depth[240 - 20:240 + 20, 360 - 20:360 + 20] = 3.0    # object: central half
    (track,) = tracker.update(batch(1, centred_box(half=40)), depth, SIM_CAMERA_MODEL,
                              OPTICAL_IN_MAP)
    assert track.position_map[0] == pytest.approx(3.0, abs=1e-6)


def test_transform_stamped_to_optical_frame_is_accepted():
    msg = TransformStamped()
    msg.header.frame_id = 'map'
    msg.child_frame_id = 'camera_rgb_optical_frame'
    msg.transform.translation.z = 0.5
    (msg.transform.rotation.x, msg.transform.rotation.y,
     msg.transform.rotation.z, msg.transform.rotation.w) = OPTICAL_IN_MAP['rotation']
    tracker = ArtifactTracker(confirm_count=1)
    (track,) = tracker.update(batch(1, centred_box()), object_at(3.0), SIM_CAMERA_MODEL, msg)
    assert track.position_map == pytest.approx((3.0, 0.0, 0.5), abs=1e-6)


def test_transform_to_camera_link_is_refused():
    msg = TransformStamped()
    msg.child_frame_id = 'camera_link'
    msg.transform.rotation.w = 1.0
    with pytest.raises(ValueError, match='optical'):
        ArtifactTracker().update(batch(1, centred_box()), object_at(3.0),
                                 SIM_CAMERA_MODEL, msg)


def test_time_going_backwards_resets_tracks():
    tracker = ArtifactTracker(confirm_count=1)
    see(tracker, 10, 3.0)
    (track,) = see(tracker, 1, 5.0)
    assert track.artifact_id == 1
    assert track.position_map[0] == pytest.approx(5.0, abs=1e-6)


def test_stale_provisional_tracks_are_dropped():
    tracker = ArtifactTracker(confirm_count=2, provisional_timeout_sec=10.0)
    see(tracker, 1, 3.0)
    tracker.update(batch(20), depth_image(), SIM_CAMERA_MODEL, OPTICAL_IN_MAP)
    assert see(tracker, 21, 3.0) == []      # starts again from one observation


def test_intrinsics_match_workplan():
    assert SIM_CAMERA_MODEL['fx'] == pytest.approx(207.84, abs=0.02)
    assert math.isclose(SIM_CAMERA_MODEL['cx'], 360.0)


def test_one_large_artefact_seen_from_different_sides_stays_one_track():
    """Regression: estimates of one green_alien spread over 1.7 m in simulation."""
    tracker = ArtifactTracker(confirm_count=1)
    for stamp, depth in enumerate((3.0, 4.2, 4.7), start=1):
        tracks = see(tracker, stamp, depth)
    assert len(tracks) == 1
    assert tracks[0].observation_count == 3


@pytest.mark.parametrize('target_width', [4, 12, 40])
def test_foreground_depth_wins_over_far_background(target_width):
    tracker = ArtifactTracker(confirm_count=1)
    depth = depth_image(8.0)
    # Even 10% foreground defeats a wall covering the other 90%.
    start = 360 - target_width // 2
    depth[220:260, start:start + target_width] = 3.0
    (track,) = tracker.update(batch(1, centred_box(ICE_FORMATION)), depth,
                              SIM_CAMERA_MODEL, OPTICAL_IN_MAP)
    assert track.class_name == ICE_FORMATION
    assert track.position_map == pytest.approx((3.0, 0.0, 0.5))


def test_off_centre_foreground_supplies_depth_and_projection_pixel():
    tracker = ArtifactTracker(confirm_count=1)
    depth = depth_image(8.0)
    depth[244:256, 370:378] = 3.0
    (track,) = tracker.update(batch(1, centred_box(ICE_FORMATION)), depth,
                              SIM_CAMERA_MODEL, OPTICAL_IN_MAP)
    # Selected foreground pixel centre is (374, 250), not bbox centre (360, 240).
    assert track.position_map == pytest.approx(
        (3.0, -14 * 3.0 / SIM_CAMERA_MODEL['fx'],
         0.5 - 10 * 3.0 / SIM_CAMERA_MODEL['fy']))


def test_small_near_outliers_do_not_win_over_supported_foreground():
    tracker = ArtifactTracker(confirm_count=1)
    depth = depth_image(8.0)
    depth[220:260, 358:362] = 3.0
    depth[220:222, 340:342] = 0.5
    (track,) = tracker.update(batch(1, centred_box(ICE_FORMATION)), depth,
                              SIM_CAMERA_MODEL, OPTICAL_IN_MAP)
    assert track.position_map == pytest.approx((3.0, 0.0, 0.5))


def test_noisy_foreground_band_uses_robust_depth_and_ignores_invalid_pixels():
    tracker = ArtifactTracker(confirm_count=1)
    depth = depth_image(8.0)
    depth[220:260, 358:362] = np.tile([2.95, 3.0, 3.0, 3.05], (40, 1))
    depth[230:234, 340:344] = np.nan
    (track,) = tracker.update(batch(1, centred_box(ICE_FORMATION)), depth,
                              SIM_CAMERA_MODEL, OPTICAL_IN_MAP)
    assert track.position_map == pytest.approx((3.0, 0.0, 0.5))


def test_sparse_depth_is_rejected():
    tracker = ArtifactTracker(confirm_count=1)
    depth = depth_image()
    depth[240:244, 360:364] = 3.0
    assert tracker.update(batch(1, centred_box(ICE_FORMATION)), depth,
                          SIM_CAMERA_MODEL, OPTICAL_IN_MAP) == []
    assert 'valid depth pixels' in tracker.last_rejections[0]


def test_unsupported_depth_bands_are_rejected():
    tracker = ArtifactTracker(confirm_count=1)
    depth = depth_image()
    # Enough valid pixels overall, but each band has fewer than 80 pixels.
    for row in range(20):
        depth[220 + row, 340:380] = 0.5 + row * 0.45
    assert tracker.update(batch(1, centred_box(ICE_FORMATION)), depth,
                          SIM_CAMERA_MODEL, OPTICAL_IN_MAP) == []
    assert 'foreground support' in tracker.last_rejections[0]


def test_clipped_sampling_window_uses_actual_image_pixels():
    tracker = ArtifactTracker(confirm_count=1)
    sample, reason = tracker._depth_in_bbox(depth_image(3.0), (-40, -40, 40, 40))
    assert reason is None
    assert sample == pytest.approx((3.0, 10.0, 10.0))


def test_box_outside_depth_image_is_rejected():
    sample, reason = ArtifactTracker()._depth_in_bbox(depth_image(3.0),
                                                     (800, 0, 880, 80))
    assert sample is None
    assert 'no pixels' in reason


def test_sparse_intermediate_depths_do_not_bridge_target_to_wall():
    tracker = ArtifactTracker(confirm_count=1)
    depth = depth_image(8.0)
    depth[220:260, 358:362] = 3.0
    # Mixed edge readings form a chain of small gaps, but are not a surface.
    depth[220, 340:359] = np.linspace(3.25, 7.75, 19)
    (track,) = tracker.update(batch(1, centred_box(ICE_FORMATION)), depth,
                              SIM_CAMERA_MODEL, OPTICAL_IN_MAP)
    assert track.position_map[0] == pytest.approx(3.0)


@pytest.mark.parametrize('target_width', [2, 3])
def test_small_supported_foreground_is_not_swamped_by_wall(target_width):
    tracker = ArtifactTracker(confirm_count=1)
    depth = depth_image(8.0)
    depth[220:260, 368:368 + target_width] = 3.0
    (track,) = tracker.update(batch(1, centred_box(ICE_FORMATION)), depth,
                              SIM_CAMERA_MODEL, OPTICAL_IN_MAP)
    u = 368 + target_width / 2
    assert track.position_map == pytest.approx(
        (3.0, -(u - 360) * 3.0 / SIM_CAMERA_MODEL['fx'], 0.5))


@pytest.mark.parametrize('u, v', [(20, 20), (700, 460)])
def test_image_boundary_box_projects_selected_pixels(u, v):
    tracker = ArtifactTracker(confirm_count=1)
    # Boxes lie exactly against the image boundary; selected pixels stay inside.
    detection = centred_box(ICE_FORMATION, half=20, u=u, v=v)
    (track,) = tracker.update(batch(1, detection), depth_image(3.0),
                              SIM_CAMERA_MODEL, OPTICAL_IN_MAP)
    assert track.position_map == pytest.approx(
        (3.0, -(u - 360) * 3.0 / SIM_CAMERA_MODEL['fx'],
         0.5 - (v - 240) * 3.0 / SIM_CAMERA_MODEL['fy']))


def test_repeated_viewpoints_confirm_ice_in_a_distinct_map_frame():
    tracker = ArtifactTracker()  # Production confirmation count, not one-shot.
    target_world = (33.7, 10.6, 0.6)
    world_from_map_translation = (12.0, -7.0, 0.0)
    for stamp, (camera_x, camera_y) in enumerate(
            [(28.0, 9.0), (29.0, 10.6), (28.0, 12.0)], start=1):
        z = target_world[0] - camera_x
        u = round(360 + SIM_CAMERA_MODEL['fx'] * (camera_y - target_world[1]) / z)
        v = round(240 + SIM_CAMERA_MODEL['fy'] * (0.5 - target_world[2]) / z)
        depth = depth_image(9.0)
        depth[v - 20:v + 20, u - 20:u + 20] = z
        optical_in_map = dict(OPTICAL_IN_MAP)
        optical_in_map['translation'] = (
            camera_x - world_from_map_translation[0],
            camera_y - world_from_map_translation[1], 0.5)
        tracks = tracker.update(batch(stamp, centred_box(ICE_FORMATION, half=160)),
                                depth, SIM_CAMERA_MODEL, optical_in_map)
        if stamp < 3:
            assert tracks == []
    (track,) = tracks
    assert track.observation_count == 3
    estimated_world = tuple(p + t for p, t in zip(track.position_map,
                                                  world_from_map_translation))
    assert estimated_world == pytest.approx(target_world, abs=0.02)


@pytest.mark.parametrize('kwargs', [
    {'depth_band_width_m': 0}, {'depth_band_width_m': np.nan},
    {'depth_band_width_m': np.inf}, {'min_foreground_fraction': 0},
    {'min_foreground_fraction': 1.1}, {'min_foreground_fraction': np.nan},
])
def test_invalid_foreground_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ArtifactTracker(**kwargs)
