<p align="center">
  <img src="docs/assets/icon.png" width="128" alt="Silent Dictation icon">
</p>

<h1 align="center">Silent Dictation</h1>

<p align="center">
  <b>Type by mouthing words. No sound, no microphone, no wearable.</b><br>
  Hold <kbd>fn</kbd>, silently mouth a sentence at your Mac's camera, release, and it's typed at your cursor.<br>
  Lip reading runs entirely on your Mac.
</p>

<p align="center">
  <a href="https://github.com/KeWang0622/silent-dictation/actions/workflows/ci.yml"><img src="https://github.com/KeWang0622/silent-dictation/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/macOS-13%2B-black?logo=apple" alt="macOS 13+">
  <img src="https://img.shields.io/badge/Apple%20Silicon-M1%E2%80%93M5-6246ea" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/runs-100%25%20on--device-2fd8b0" alt="On-device">
  <a href="LICENSE"><img src="https://img.shields.io/badge/code-Apache--2.0-blue" alt="Apache-2.0"></a>
</p>

<p align="center">
  <img src="docs/assets/screenshot-home.png" width="860" alt="Silent Dictation main window">
</p>

Voice dictation apps like Willow Voice and Wispr Flow are fast, but they need you to talk out loud, which doesn't work in a library, an open office or on a train. Silent Dictation works the same way (hold a key, speak, release) except you don't make a sound: it reads your lips through the camera you already have and types what you mouthed into any app.

> **Inspiration.** This project started after seeing [Aarush Yadav's Subvocal announcement](https://x.com/AarushYadav47/status/2116198463847408066) (YC F26): glasses that read silent speech with non-contact sensing. I wanted to see how far a plain Mac camera and open lip-reading models can go today, with nothing to wear. Silent Dictation is an independent open-source project and is **not affiliated with Subvocal**.

## Highlights

- **Feels like Willow.** Hold <kbd>fn</kbd> to dictate, double-tap for hands-free, press again to stop. <kbd>fn</kbd> plus another key cancels, so your normal shortcuts keep working. Prefer another key? Pick Right ⌥, Right ⌘ or Right ⌃.
- **A floating indicator** shows a live view of your mouth and lip activity while it reads, then the result.
- **Learns your lips.** A five-minute calibration has you mouth short sentences while it shows, word by word, what the model read. Bad takes (too short, lips barely moved, face lost) are rejected on the spot. Then it fine-tunes a personal model on your Mac and keeps it only if it beats the base model on sentences it never trained on.
- **Gets better as you use it.** Correct a dictation in History with one click and it becomes training data for the next round.
- **Private by design.** Video is processed in memory and never uploaded. History keeps only small grey mouth crops so you can review and retrain, and you can turn that off.
- **Quick.** Beam search with cached attention runs on the Apple GPU: about a second for a seven-second sentence on an M5 Max.
- **Optional AI clean-up.** A small on-device language model (Qwen3-4B via MLX) can fix look-alike words, such as "fine" versus "five". It is limited to choosing among the lip reader's candidates or changing a single word, and in Auto mode it turns on only if calibration shows it helps you.

<p align="center">
  <img src="docs/assets/screenshot-calibrate.png" width="860" alt="Calibration with word-by-word feedback">
</p>

<p align="center">
  <img src="docs/assets/overlay.png" width="182" alt="Floating indicator while reading lips">
  &nbsp;&nbsp;
  <img src="docs/assets/overlay-result.png" width="410" alt="Floating indicator with the result">
</p>

<table>
  <tr>
    <td><img src="docs/assets/screenshot-setup.png" alt="Setup guide"></td>
    <td><img src="docs/assets/screenshot-dark.png" alt="Calibrate in dark mode"></td>
  </tr>
  <tr>
    <td align="center">A short setup guide for permissions, camera and key</td>
    <td align="center">Light and dark mode</td>
  </tr>
</table>

## How it works

```mermaid
flowchart LR
  A[Camera<br/>25 fps] --> B[MediaPipe<br/>face landmarks]
  B --> C[Mouth crop 96×96<br/>silence trimmed]
  C --> D[Auto-AVSR<br/>visual encoder]
  D --> E[Beam search +<br/>language model]
  E --> F{AI clean-up?}
  F -- optional --> G[Qwen3-4B<br/>on-device]
  F --> H[Typed at<br/>your cursor]
  G --> H
```

1. While you hold the key, the camera tracks 478 face landmarks per frame with MediaPipe and cuts a stabilised 96×96 grey crop of your mouth, resampled to 25 fps by real timestamps.
2. Stretches at the start and end where your lips don't move are trimmed, so the model doesn't invent words for them.
3. The [Auto-AVSR](https://github.com/mpc001/auto_avsr) visual speech recogniser (a Conformer encoder with a Transformer decoder, 19.1% word error rate on the LRS3 benchmark) decodes the clip with beam search and a subword language model.
4. The best transcript (optionally refined by the local LLM) is pasted into the app you're using. If Accessibility access is off, it goes to the clipboard instead.

Calibration fine-tunes the model on your own clips with two strategies: a light one (visual front-end, normalisation layers and biases) and a full one. Every fifth sentence is held out permanently, and the app keeps whichever of the base model, your previous personal model and the new one reads those held-out sentences best.

## Install

You need an Apple Silicon Mac with macOS 13 or later, about 3 GB of free disk space and [uv](https://docs.astral.sh/uv/) (`brew install uv`).

```bash
git clone https://github.com/KeWang0622/silent-dictation.git
cd silent-dictation
./setup.sh        # Python environment + checksum-verified models (~1 GB)
bin/build-app     # installs "Silent Dictation.app" in /Applications and opens it
```

Add `--with-llm` to `./setup.sh` for the optional AI clean-up model (another 2 GB). `bin/build-app` keeps the code, models and Python environment in `~/Library/Application Support/Silent Dictation/runtime` (as APFS clones, so they take no extra space), which means you can move or delete the source folder afterwards (keep uv's Python, which the app links to). To update, run `git pull` and `bin/build-app` again; your permissions and calibration stay.

On first launch a short setup guide walks you through:

1. **Permissions.** Camera to see your lips, Input Monitoring to notice the key, and Accessibility to type into other apps.
2. **Camera check.** A live preview with a box on your mouth. An iPhone works great as the camera through Continuity Camera.
3. **Your key.** <kbd>fn</kbd> by default. If <kbd>fn</kbd> opens the emoji picker, set *System Settings › Keyboard › Press 🌐 key to › Do Nothing*.
4. **Calibration** (recommended). About five minutes of silent sentences, then a few minutes of training in the background.

## Using it

| Action | What happens |
| --- | --- |
| Hold <kbd>fn</kbd>, mouth a sentence, release | The text is typed where your cursor is |
| Double-tap <kbd>fn</kbd> | Hands-free: keeps reading until you press <kbd>fn</kbd> again |
| <kbd>fn</kbd> + another key | Cancels, so shortcuts like <kbd>fn</kbd> + arrow still work |
| Dictate while the Silent Dictation window is in front | The result shows in *Try it here* instead of being pasted |

Tips for good results:

- Face the camera from about an arm's length away with light on your face. Front light matters more than anything else.
- Mouth clearly, the way you'd want a friend across the room to read your lips, but without sound.
- Short everyday sentences work best. Pause briefly before and after.
- If you also run Willow Voice, Wispr Flow or Superwhisper on <kbd>fn</kbd>, both start at once. Silent Dictation warns you and lets you switch to Right ⌥.

## Accuracy and limitations

Be realistic: lip reading is much harder than speech recognition. Many sounds look identical on the lips (p/b/m, f/v, t/d/n), and the base model was trained on TED talks, where people speak aloud to a good camera. Silent mouthing by a new person in front of a laptop camera is harder still, so expect noticeably more mistakes than voice dictation, especially before calibration. On the test clip used during development the base model reads 19 of 22 words correctly. Calibration is the single biggest improvement, and the Calibrate page shows your before-and-after accuracy on held-out sentences.

Current limitations:

- English only.
- Apple Silicon only. Intel Macs would run on the CPU and be slow.
- Beards, masks, strong side light and turning your head reduce accuracy.
- macOS treats each rebuild of the launcher as a new app, so permissions have to be granted again after `bin/build-app --force`.

## Privacy

Everything runs locally and there are no network calls at run time. Camera frames live only in memory. Your data stays in `~/Library/Application Support/Silent Dictation/data`: settings, usage counts, calibration clips and the personal model, and (if *Keep clips for review* is on) small 96×96 grey mouth crops of recent dictations. Delete that folder to erase everything.

## Command line

The app covers everyday use. For scripting and experiments there is `bin/silentdict`:

```bash
bin/silentdict doctor --camera-test       # check models, permissions and the camera
bin/silentdict transcribe clip.mp4        # lip-read a video file
bin/silentdict train                      # retrain your personal model
bin/silentdict eval                       # base vs personal accuracy on held-out sentences
```

## Uninstall

Quit Silent Dictation, then delete `/Applications/Silent Dictation.app` and `~/Library/Application Support/Silent Dictation`. If you turned on *Open at login*, also delete `~/Library/LaunchAgents/io.github.kewang0622.silentdictation.plist`.

## Roadmap

- A newer visual speech model ([FLARE](https://arxiv.org/abs/2610.19559) reports 13.2% WER on LRS3) for a large accuracy gain.
- Parameter-efficient (LoRA) personal adaptation, so more of your data helps without overfitting.
- Streaming decoding while the key is held, for lower latency on long sentences.
- More languages, starting with Mandarin.
- A signed, notarised download that doesn't need a terminal.

See [docs/RESEARCH.md](docs/RESEARCH.md) for the model survey behind these choices, and [CONTRIBUTING.md](CONTRIBUTING.md) to help.

## Credits

- [Aarush Yadav's Subvocal](https://x.com/AarushYadav47/status/2116198463847408066) for the idea of typing by silent speech. Not affiliated.
- [Auto-AVSR](https://github.com/mpc001/auto_avsr) by Pingchuan Ma et al. for the lip-reading model, the language model and the mouth-cropping procedure.
- [Chaplin](https://github.com/amanvirparhar/chaplin) by Amanvir Parhar, which showed real-time lip reading with these weights on a Mac.
- [ESPnet](https://github.com/espnet/espnet) (vendored subset), [MediaPipe](https://ai.google.dev/edge/mediapipe), [PyTorch](https://pytorch.org), [MLX](https://github.com/ml-explore/mlx) and [Qwen3](https://github.com/QwenLM/Qwen3).
- The Harvard sentences (IEEE, 1969, public domain) used in calibration.

## License

The code is released under the [Apache-2.0 license](LICENSE); see [NOTICE](NOTICE) for third-party material. The default lip-reading weights are **not** part of this repository: `setup.sh` downloads them from their authors' releases, which are for **non-commercial research use**, and a personal model fine-tuned from them inherits those terms.
