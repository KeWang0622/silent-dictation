# Benchmark

`bin/benchmark` measures word error rate on 120 sentences from the [VidTIMIT](https://conradsanderson.id.au/vidtimit/) dataset: 12 speakers (6 women, 6 men) reading 10 sentences each. Every clip goes through the same code as a dictation in the app: MediaPipe face tracking, mouth and face crops, the lip reader and, with `--llm`, language-model rescoring. Base models only; your personal model is never used.

## Get the videos

Download the video zips for these speakers from the VidTIMIT page and unzip them into one folder:

```
fadg0 faks0 fcft0 fcmh0 fedw0 fjas0 mabw0 mbdg0 mccs0 mdab0 mjsw0 mrgg0
```

You should end up with folders like `vidtimit/fadg0/video/sa1/001`. The audio isn't needed.

## Run

```bash
bin/benchmark ~/data/vidtimit                          # FLARE, Balanced mode (beam 10)
bin/benchmark ~/data/vidtimit --llm                    # with AI clean-up (needs ./setup.sh --with-llm)
bin/benchmark ~/data/vidtimit --beam 20 --llm          # Accurate mode
bin/benchmark ~/data/vidtimit --model auto_avsr        # the older model
bin/benchmark ~/data/vidtimit -v --out results.json    # print and save every sentence
```

A run takes two to four minutes on an M5 Max. Results on that machine:

| Setup | Word error rate | Sentences fully right |
| --- | --- | --- |
| Auto-AVSR, beam 12 | 47.5% | 15.8% |
| FLARE, beam 10 | 32.7% | 23.3% |
| FLARE, beam 10, `--llm` | 30.7% | 30.8% |
| FLARE, beam 20, `--llm` | 30.3% | 31.7% |

Reference transcripts are in [vidtimit_refs.json](vidtimit_refs.json). They were made with Whisper large-v3-turbo from the VidTIMIT audio, and "aluminium" and "aluminum" count as the same word. VidTIMIT is voiced speech filmed in a studio, so real silent mouthing in front of a laptop will score worse; use the numbers to compare versions and settings, not as a promise for your own lips.
