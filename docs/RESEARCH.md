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
| Auto-AVSR (Ma et al., ICASSP 2023) | 19.1% | public, research use | Conformer + transformer decoder, lips only; **second model here** |
| USR (Haliassos et al., 2024) | 22.3% (Large, high-resource) | public | one model for audio, video, audio-visual |
| VALLR (ICCV 2025) | 18.7% | not official | phonemes from video, then an LLM writes words |
| FLARE (arXiv 2610.19559, NeurIPS 2026) | 15.9% (Large), 13.2% (FLARE-LLM, 7B decoder) | CC BY-NC 4.0, MIT code | distils Whisper features, joint face + lip streams; **default here** (Large) |

LRS3 is TED talks: people speaking aloud, filmed well. Silent, slightly reduced mouthing by a new
person in front of a laptop camera is harder, so expect higher error rates before calibration.

## Benchmark (VidTIMIT, 120 sentences)

To compare changes fairly, every candidate is run on the same 120 clips: 12 VidTIMIT speakers (6 women,
6 men) reading 10 sentences each, with reference transcripts from Whisper large-v3-turbo on the audio.
VidTIMIT is not among either model's training sets. `bin/benchmark` reproduces the app's numbers.

| Setup | WER | Notes |
|---|---|---|
| Auto-AVSR, beam 12 (0.2 default) | 47.5% | 15.8% of sentences exactly right |
| Auto-AVSR, beam 20 | 46.1% | the correct sentence is in the 20-best list often enough for 33.2% oracle WER |
| Auto-AVSR + Qwen3-4B rescoring | ~46% | only about 3% relative |
| FLARE Large, lips only, beam 10 | 36.8% | |
| FLARE Large, lips + face (crop 2.0× face size), beam 10 | 34.0% | |
| **FLARE Large, lips + face (crop 1.6×), beam 10** | **32.7%** | 23.3% exactly right; beam 5: 33.5%, beam 20: 32.8% |
| FLARE + Qwen3-4B rescoring, beam 10, λ = 0.3 | 30.7% | 30.8% exactly right (Balanced mode) |
| FLARE + Qwen3-4B rescoring, beam 20, λ = 0.3 | 30.3% | 31.7% exactly right (Accurate mode); 29.7% with λ tuned per held-out speaker |
| FLARE + Auto-AVSR as a second scorer | 31.2% | not worth loading a second model |
| FLARE + both | 30.2% | |

The FLARE port in `silentdict/flare` was checked against the original fairseq code: encoder outputs agree
to 7e-6 and the benchmark transcripts are identical. With the encoder on the Apple GPU and beam search on
the CPU, a sentence takes 0.2–0.4 s on an M5 Max.

**Phrase mode.** When the sentence is one of a known list, scoring each listed phrase against the video
(forced decoding) is much more reliable than free reading. With Auto-AVSR, 87.8% of sentences were
picked exactly from a list of 84 phrases and 95.8% from a list of 10. Falling back to free decoding when
the best phrase scores worse than −1.0 per token kept 76% of in-list sentences right and 4% wrong, while
12% of out-of-list sentences were wrongly replaced by a phrase; on a mix the WER fell from 46.1% to 30.4%.
FLARE can score phrases the same way (`FlareReader.score_texts`), so this is a strong candidate for a
"my phrases" feature.

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
words from context. Chaplin asks an LLM to correct the top transcript, but a generative rewrite can also
invent words. Silent Dictation instead scores each of the lip reader's ten best transcripts with a local
Qwen3-4B model (MLX) as "something a person typed", optionally after a line of your own vocabulary, and
picks the best sum of lip-reading log-probability and 0.3 × language log-probability. It can only choose
among sentences the lip reader proposed. Calibration measures it on your held-out sentences, and Auto
mode turns it off if it doesn't help you.

## Next steps

1. Phrase mode with FLARE forced scoring (see above), plus a way to add your own phrases.
2. Parameter-efficient adaptation (LoRA) so more data can be used without forgetting.
3. A FLARE-LLM style decoder if a small LLM can run it at interactive speed.
4. A silent-mouthing test set: VidTIMIT is voiced speech, and real silent mouthing differs.
5. Mandarin: CMLR-trained models exist (Ma et al. 2022), but conversational accuracy is untested.
6. Streaming decoding while the key is held, to cut latency on long sentences.

## Sources

- Chaplin: https://github.com/amanvirparhar/chaplin
- Auto-AVSR: https://github.com/mpc001/auto_avsr
- Visual Speech Recognition for Multiple Languages: https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages
- USR: https://arxiv.org/abs/2411.02256 and https://github.com/ahaliassos/usr
- VALLR: https://arxiv.org/abs/2503.21408
- FLARE: https://arxiv.org/abs/2610.19559 and https://github.com/jxzhanggg/FLARE
- FLARE weights: https://huggingface.co/Anony94/FLARE-checkpoints
- VidTIMIT: https://conradsanderson.id.au/vidtimit/
- Personalized Lip Reading (AAAI 2025): https://ojs.aaai.org/index.php/AAAI/article/view/33026
- Subvocal announcement: https://x.com/AarushYadav47/status/2116198463847408066
