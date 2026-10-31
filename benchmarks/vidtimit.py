"""Word error rate of Silent Dictation on 120 VidTIMIT sentences (12 speakers x 10 sentences).

    bin/benchmark /path/to/vidtimit                     # FLARE, Balanced mode (beam 10)
    bin/benchmark /path/to/vidtimit --llm               # plus language-model rescoring
    bin/benchmark /path/to/vidtimit --model auto_avsr --beam 12

The folder holds one directory per speaker as extracted from the VidTIMIT zips
(https://conradsanderson.id.au/vidtimit/), e.g. fadg0/video/sa1/001. Every clip goes through the app's own
pipeline: MediaPipe face tracking, mouth and face crops, the lip reader and, with --llm, rescoring.
Reference transcripts are in vidtimit_refs.json. The base models are used (never your personal model).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from silentdict.capture import make_frame
from silentdict.face import FaceTracker
from silentdict.pipeline import Recognizer, resolve_model
from silentdict.text import corpus_wer, normalize

HERE = Path(__file__).resolve().parent
SPELLING = {"ALUMINIUM": "ALUMINUM"}  # count British and American spellings as the same word


def norm(text: str) -> str:
    return " ".join(SPELLING.get(w, w) for w in normalize(text).split())


def load_frames(folder: Path) -> list:
    files = sorted((p for p in folder.iterdir() if p.name.isdigit()), key=lambda p: int(p.name))
    tracker = FaceTracker(video_mode=True)
    try:
        return [make_frame(cv2.imread(str(f)), i / 25.0, tracker) for i, f in enumerate(files)]
    finally:
        tracker.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Word error rate on 120 VidTIMIT sentences.")
    ap.add_argument("data", type=Path, help="extracted VidTIMIT folder (one directory per speaker)")
    ap.add_argument("--model", default="auto", choices=["auto", "flare", "auto_avsr"])
    ap.add_argument("--beam", type=int, default=None, help="beam size (default: Balanced mode, 10 or 12)")
    ap.add_argument("--llm", action="store_true", help="rescore candidates with the on-device language model")
    ap.add_argument("--speakers", help="comma-separated subset, e.g. fadg0,mabw0")
    ap.add_argument("--out", type=Path, help="write per-sentence results as JSON")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every sentence")
    args = ap.parse_args(argv)

    refs = json.loads((HERE / "vidtimit_refs.json").read_text())
    if args.speakers:
        keep = set(args.speakers.split(","))
        refs = {k: v for k, v in refs.items() if k.split("_")[0] in keep}
    model = resolve_model(args.model)
    beam = args.beam or (10 if model == "flare" else 12)
    rescorer = None
    if args.llm:
        from silentdict.rewrite import Rescorer
        rescorer = Rescorer()
    rec = Recognizer(model=model, personal=False, decode={"beam_size": beam}, rescorer=rescorer)
    rec.warm_up()
    print(f"{model}, beam {beam}{', rescoring' if rescorer else ''}: {len(refs)} sentences", flush=True)

    pairs, rows, secs = [], {}, []
    for key, ref in refs.items():
        speaker, sentence = key.split("_", 1)
        folder = args.data / speaker / "video" / sentence
        if not folder.is_dir():
            print(f"missing {folder}", file=sys.stderr)
            return 1
        res = rec.recognize_frames(load_frames(folder))
        hyp = norm(res.text)
        pairs.append((norm(ref), hyp))
        secs.append(res.timings.get("vsr", 0.0) + res.timings.get("llm", 0.0))
        rows[key] = {"ref": ref, "hyp": res.text, "error": res.error, "seconds": round(secs[-1], 3)}
        if args.verbose:
            mark = "=" if hyp == norm(ref) else "x"
            print(f"{mark} {key:12s} {res.text}  [{ref}]", flush=True)

    exact = sum(r == h for r, h in pairs) / len(pairs)
    print(f"word error rate {corpus_wer(pairs):.1%}  ·  exact sentences {exact:.1%}  ·  "
          f"{sum(secs) / len(secs):.2f} s per sentence (cropping excluded)")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    t0 = time.time()
    code = main()
    print(f"done in {time.time() - t0:.0f} s", file=sys.stderr)
    raise SystemExit(code)
