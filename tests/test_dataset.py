import numpy as np

from silentdict import dataset, paths


def test_every_fifth_clip_is_held_out():
    roi = np.zeros((30, 96, 96), np.uint8)
    for i in range(10):
        dataset.add_item(roi, f"sentence number {i}", "calibration")
    assert len(dataset.load_items("val")) == 2
    assert len(dataset.load_items("train")) == 8
    first = dataset.load_items()[0]
    dataset.delete_item(first.id)
    assert first.id not in {it.id for it in dataset.load_items()}


def test_prompts_start_with_custom_phrases_and_skip_recorded():
    paths.PERSONAL.mkdir(parents=True, exist_ok=True)
    (paths.PERSONAL / "custom_prompts.txt").write_text("Ship the build to Priya\n")
    prompts = dataset.load_prompts(20)
    assert prompts[0] == "Ship the build to Priya"
    assert len(prompts) == 20 and len(set(prompts)) == 20
    dataset.add_item(np.zeros((30, 96, 96), np.uint8), prompts[0], "calibration")
    assert "Ship the build to Priya" not in dataset.load_prompts(20)
