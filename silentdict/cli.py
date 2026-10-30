"""Command line entry point: silentdict <command>."""
from __future__ import annotations

import argparse
import shutil
import sys

from . import CREDITS, __version__, paths

COMMANDS = """commands:
  doctor       check models, camera and macOS permissions
  calibrate    record your own silent sentences, then train a personal model
  run          everyday dictation: hold the key, mouth a sentence, release to type
  review       confirm or correct recent dictation to create more training data
  train        retrain the personal model from all calibration + reviewed clips
  eval         word error rate of the base vs personal model on your held-out clips
  transcribe   lip-read a video file (offline, for testing)
"""


def _common(p: argparse.ArgumentParser):
    p.add_argument("--device", default="auto", help="encoder device: auto, mps or cpu")
    p.add_argument("--camera", type=_camera, default=0,
                   help="camera index (0 = built-in) or a video file replayed as a fake camera")
    p.add_argument("--key", default="right_option",
                   help="push-to-talk key: right_option, right_command, right_control, right_shift, fn")


def _camera(value: str):
    return int(value) if value.isdigit() else value


def ensure_user_files():
    paths.PERSONAL.mkdir(parents=True, exist_ok=True)
    if not paths.PERSONAL_CONTEXT.exists():
        shutil.copy(paths.ROOT / "prompts" / "context.example.txt", paths.PERSONAL_CONTEXT)


def check_models() -> list[str]:
    need = [paths.VSR_DIR / "model.pth", paths.LM_DIR / "model.pth", paths.SPM_MODEL, paths.SPM_UNITS,
            paths.FACE_MODEL]
    return [str(p.relative_to(paths.ROOT)) for p in need if not p.exists()]


def doctor(args):
    import torch

    from .hotkey import input_monitoring_allowed
    from .output import accessibility_allowed
    from .permissions import STATUS, camera_status, ensure_camera_access
    from .rewrite import DEFAULT_LLM

    ok = True
    missing = check_models()
    print(f"silentdict {__version__}\n{CREDITS}\n")
    print("models        ", "OK" if not missing else "MISSING " + ", ".join(missing) + " (run ./setup.sh)")
    ok &= not missing
    print("local LLM     ", "OK" if DEFAULT_LLM.exists() else "not installed (optional: ./setup.sh --llm)")
    print("Apple GPU     ", "OK (MPS)" if torch.backends.mps.is_available() else "not available, using CPU")
    im = input_monitoring_allowed(prompt=args.prompt)
    ax = accessibility_allowed(prompt=args.prompt)
    print("Input Monitor ", "OK" if im else "NOT GRANTED (needed for the push-to-talk key)")
    print("Accessibility ", "OK" if ax else "NOT GRANTED (needed to paste into other apps)")
    cam = camera_status()
    print("Camera access ", "OK" if cam == 3 else STATUS.get(cam, "unknown").upper())
    ok &= im and ax
    if args.prompt and cam == 0:
        cam = 3 if ensure_camera_access() else 2
    if args.camera_test and cam != 3 and isinstance(args.camera, int):
        print("camera         skipped: camera access not granted")
        ok = False
    elif args.camera_test:
        import time

        from .capture import CameraEngine
        eng = CameraEngine(on_segment=lambda f: None, camera_index=args.camera)
        eng.start()
        eng.ready.wait(10)
        if eng.error:
            print("camera        ", eng.error)
            ok = False
        else:
            faces = frames = 0
            brightness = []
            t_end = time.time() + 3
            last = None
            while time.time() < t_end:
                f = eng.latest
                if f is not None and f is not last:
                    frames += 1
                    faces += f.obs is not None
                    brightness.append(float(f.gray.mean()))
                    last = f
                time.sleep(0.01)
            mean_b = sum(brightness[-15:]) / max(len(brightness[-15:]), 1)
            print("camera        ", f"{eng.measured_fps:.0f} fps, face found in {faces}/{frames} frames, "
                  f"brightness {mean_b:.0f}/255")
            if mean_b < 2:
                print("               the image is black: is the lens covered? Try another camera with --camera 1")
                ok = False
        eng.shutdown()
    if not (im and ax):
        print("\nGrant the permissions to the app you run this from (Terminal, iTerm, VS Code...) in\n"
              "System Settings > Privacy & Security, then restart that app. 'silentdict doctor --prompt' opens the requests.")
    return 0 if ok else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog="silentdict", description="Camera-only silent dictation for macOS.",
                                     epilog=COMMANDS, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"silentdict {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("doctor", help="check setup")
    _common(p)
    p.add_argument("--camera-test", action="store_true", help="open the camera for 3 seconds")
    p.add_argument("--prompt", action="store_true", help="ask macOS for missing permissions")

    p = sub.add_parser("run", help="everyday dictation")
    _common(p)
    p.add_argument("--llm", dest="llm", action="store_true", default=None, help="force local LLM rewrite on")
    p.add_argument("--no-llm", dest="llm", action="store_false", help="force local LLM rewrite off")
    p.add_argument("--base", action="store_true", help="ignore the personal model")
    p.add_argument("--no-paste", action="store_true", help="print results only")
    p.add_argument("--no-preview", action="store_true", help="no camera window")
    p.add_argument("--no-history", action="store_true", help="do not keep mouth clips for later review")

    p = sub.add_parser("calibrate", help="record personal calibration sentences")
    _common(p)
    p.add_argument("--count", type=int, default=60, help="sentences to record this session")
    p.add_argument("--no-train", action="store_true", help="only record, do not train afterwards")
    p.add_argument("--no-feedback", action="store_true", help="do not show the base model's reading")

    p = sub.add_parser("train", help="retrain the personal model")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--freeze-frontend", action="store_true")
    p.add_argument("--min-items", type=int, default=20)

    p = sub.add_parser("eval", help="evaluate on held-out personal clips")
    p.add_argument("--device", default="auto")

    sub.add_parser("review", help="confirm or correct recent dictation")

    p = sub.add_parser("transcribe", help="lip-read a video file")
    p.add_argument("video")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--roi", action="store_true", help="the video is already a 96x96 mouth crop")
    p.add_argument("--llm", action="store_true", help="also show the local LLM rewrite")
    p.add_argument("--base", action="store_true")
    p.add_argument("--device", default="auto")

    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 0
    ensure_user_files()
    if args.cmd != "doctor":
        missing = check_models()
        if missing:
            print("Missing model files: " + ", ".join(missing) + "\nRun ./setup.sh first.")
            return 1

    if args.cmd == "doctor":
        return doctor(args)
    if args.cmd == "run":
        from .app import DictationApp
        DictationApp(key=args.key, camera=args.camera, device=args.device, llm=args.llm, personal=not args.base,
                     paste=not args.no_paste, preview=not args.no_preview, history=not args.no_history).run()
        return 0
    if args.cmd == "calibrate":
        from .calibrate import Calibrator
        total = Calibrator(key=args.key, camera=args.camera, device=args.device, count=args.count,
                           feedback=not args.no_feedback).run()
        if not args.no_train:
            if total >= 20:
                from . import train
                train.run(device=args.device)
            else:
                print(f"Record at least 20 sentences before training (you have {total}).")
        return 0
    if args.cmd == "train":
        from . import train
        train.run(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, device=args.device,
                  freeze_frontend=args.freeze_frontend, min_items=args.min_items)
        return 0
    if args.cmd == "eval":
        from . import train
        train.evaluate(device=args.device)
        return 0
    if args.cmd == "review":
        from .review import run
        run()
        return 0
    if args.cmd == "transcribe":
        return transcribe(args)
    return 0


def transcribe(args):
    import time

    import cv2
    import numpy as np

    from .pipeline import Recognizer

    rewriter = None
    if args.llm:
        from .rewrite import Rewriter
        rewriter = Rewriter()
    rec = Recognizer(rewriter=rewriter, device=args.device, personal=not args.base)
    t0 = time.perf_counter()
    if args.roi:
        cap = cv2.VideoCapture(args.video)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(cv2.resize(f, (96, 96)), cv2.COLOR_BGR2GRAY))
        res = rec.recognize_roi(np.stack(frames), rewrite=rewriter is not None)
    else:
        from .capture import read_video_file
        res = rec.recognize_frames(read_video_file(args.video, args.start, args.end), rewrite=rewriter is not None)
    if res.error:
        print("error:", res.error)
        return 1
    for i, h in enumerate(res.hyps, 1):
        print(f"{i}. {h.text.lower()}  ({h.score:.1f})")
    print("->", res.text)
    print(f"({time.perf_counter() - t0:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
