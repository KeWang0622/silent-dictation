import numpy as np

from silentdict.roi import CROP, MouthCropper, interpolate_points, resample_indices, stable_reference


def test_resample_to_25_fps_by_time():
    ts = np.arange(0, 2.0, 1 / 30)  # a 30 fps camera
    idx = resample_indices(ts)
    assert len(idx) == 50
    assert np.all(np.diff(idx) >= 0)


def test_resample_handles_uneven_frame_rate():
    ts = np.concatenate([np.arange(0, 1, 1 / 30), 1 + np.arange(0, 1, 1 / 15)])  # low light halves the rate
    idx = resample_indices(ts)
    assert len(idx) == round((ts[-1] - ts[0]) * 25) + 1  # one frame per 40 ms of real time
    assert idx[-1] == len(ts) - 1
    assert np.all(np.diff(idx[30:]) <= 1)  # the slow half repeats frames instead of skipping time


def test_interpolate_points_fills_gaps():
    a = np.zeros((4, 2), np.float32)
    b = np.full((4, 2), 10, np.float32)
    out = interpolate_points([None, a, None, b, None])
    assert np.allclose(out[0], a) and np.allclose(out[2], 5) and np.allclose(out[4], b)
    assert interpolate_points([None, None]) is None


def test_cropper_returns_96px_mouth_clip():
    ref = stable_reference()
    frames = [np.random.default_rng(i).integers(0, 255, (256, 256), dtype=np.uint8) for i in range(12)]
    roi = MouthCropper()(frames, [ref.copy() for _ in frames])
    assert roi.shape == (12, CROP, CROP) and roi.dtype == np.uint8
