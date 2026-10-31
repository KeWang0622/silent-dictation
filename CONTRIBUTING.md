# Contributing

Thanks for helping make silent dictation better. Bug reports, accuracy reports and pull requests are all welcome.

## Development setup

```bash
./setup.sh                      # Python environment and model files
PYTHONPATH=. .venv/bin/python -m silentdict.mac   # run the app from the checkout
bin/build-app                   # build, install and open Silent Dictation.app
.venv/bin/python -m pytest -q tests
```

When you run the app from a terminal, macOS asks for permissions on behalf of that terminal. The built app asks for its own.

Useful environment variables while developing:

| Variable | Effect |
| --- | --- |
| `SILENTDICT_DATA=/tmp/sd` | use a throwaway data folder instead of `~/Library/Application Support/Silent Dictation/data` |
| `SILENTDICT_FAKE_CAMERA=clip.mp4` | replay a video file instead of opening the camera |
| `SILENTDICT_FAKE_REWIND=1` | restart that video whenever recording starts |
| `SILENTDICT_NO_PASTE=1` | never paste into other apps |
| `SILENTDICT_AUTOMATION=steps.json` | scripted UI run: pages, simulated key presses, screenshots (see `silentdict/mac/automation.py`) |
| `SILENTDICT_TRAIN_EPOCHS=2` | shorter personal training, for testing the training flow |
| `SILENTDICT_DEVTOOLS=1` | enable the Web Inspector for the main window |

## Layout

| Path | What it is |
| --- | --- |
| `silentdict/controller.py` | camera, push-to-talk gestures, recognition, calibration and training, independent of any UI |
| `silentdict/mac/` | the native app: AppKit window, menu-bar item, floating indicator and the web UI in `web/` |
| `silentdict/flare/` | the FLARE lip reader ported to plain PyTorch, its tokenizer and the checkpoint converter |
| `silentdict/vsr.py`, `pipeline.py` | the Auto-AVSR lip reader, and the pipeline that picks a model and rescores its guesses |
| `silentdict/rewrite.py` | on-device language-model rescoring (MLX) |
| `silentdict/face.py`, `roi.py`, `activity.py` | face tracking, mouth crops and lip-movement detection |
| `silentdict/train.py` | personal fine-tuning for both models and model selection on held-out sentences |
| `benchmarks/` | word error rate on 120 VidTIMIT sentences, through the app's own pipeline |
| `macos/` | the native launcher, Info.plist and icon |
| `third_party/espnet` | the subset of ESPnet that Auto-AVSR needs, patched for Python 3.12 and Apple GPUs |

## Guidelines

- Keep everything on-device. No network calls at run time.
- Keep the UI in English and the copy short and plain.
- Please don't attach recordings of anyone's face to issues or pull requests.
- Run the tests and `ruff check silentdict tests benchmarks --select F,E9` before opening a pull request.
- If a change can affect accuracy, run `bin/benchmark` before and after and put both numbers in the pull request.
