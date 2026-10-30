# Research notes (October 2026)

Why a camera and software, and which models to build on.

## The problem

Voice dictation (Willow Voice, Wispr Flow, macOS dictation) is fast but needs sound, which does not
work in an open office, a library or on a train. Hardware answers exist: Subvocal, announced on
30 October 2026 by [Aarush Yadav](https://x.com/AarushYadav47/status/2116198463847408066), reads silent
mouthing with glasses; AlterEgo uses wearable sensing. Silent Dictation asks how far the camera already
built into a Mac can go, with no wearable at all.

## Visual speech recognition today

| System | LRS3 WER (visual only) | Weights | Notes |
|---|---|---|---|
| Auto-AVSR (Ma et al., ICASSP 2023) | 19.1% | public, research use | Conformer + transformer decoder; easy to run; **used here** |
| USR (Haliassos et al., 2024) | 22.3% (Large, high-resource) | public | one model for audio, video, audio-visual |
| VALLR (ICCV 2025) | 18.7% | not official | phonemes from video, then an LLM writes words |
| FLARE (arXiv 2610.19559, Oct 2026) | 13.2% (34.3% on WildVSR) | public, MIT code | distils Whisper features, joint face + lip streams |

LRS3 is TED talks: people speaking aloud, filmed well. Silent, slightly reduced mouthing by a new
person in front of a laptop camera is harder, so expect higher error rates before calibration.

## Why calibration matters

Lip shapes, facial hair, teeth, lighting and especially how someone mouths *without* voicing differ
from the training data. Speaker adaptation is effective with little data: *Personalized Lip Reading:
Adapting to Your Unique Lip Movements with Vision and Language* (Yeo et al., AAAI 2025) reports clear
gains from minutes of a target speaker's video using LoRA/prompt tuning. Silent Dictation does a
simple, robust version: it tries a light adaptation (visual front-end, normalisation layers and biases)
and a full fine-tune, both with augmentation, picks the epoch and strategy on a fixed held-out set, and
activates the result only when held-out WER beats both the base model and your previous personal model.

## Language model correction

Many sounds look identical on the lips (p/b/m, f/v, t/d/n...). A language model can pick the intended
words from context. Chaplin uses an LLM to correct the top transcript; Silent Dictation passes the
5-best list, recent context and a personal vocabulary to a local Qwen3-4B model (MLX) and keeps it only
when it lowers your held-out WER.

## Next steps

1. FLARE encoder backend (largest expected accuracy gain; needs a fairseq-free port and a face-stream crop).
2. Parameter-efficient adaptation (LoRA) so more data can be used without forgetting.
3. Mandarin: CMLR-trained models exist (Ma et al. 2022), but conversational accuracy is untested.
4. Streaming decoding while the key is held, to cut latency on long sentences.

## Sources

- Chaplin: https://github.com/amanvirparhar/chaplin
- Auto-AVSR: https://github.com/mpc001/auto_avsr
- Visual Speech Recognition for Multiple Languages: https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages
- USR: https://arxiv.org/abs/2411.02256 and https://github.com/ahaliassos/usr
- VALLR: https://arxiv.org/abs/2503.21408
- FLARE: https://arxiv.org/abs/2610.19559 and https://github.com/jxzhanggg/FLARE
- Personalized Lip Reading (AAAI 2025): https://ojs.aaai.org/index.php/AAAI/article/view/33026
- Subvocal announcement: https://x.com/AarushYadav47/status/2116198463847408066
