"""Command line entry point: silentdict <command>.

The everyday experience is the Mac app (bin/build-app). The command line is for checking a setup,
retraining and evaluating the personal model, and lip-reading video files.
"""
from __future__ import annotations

import argparse
import shutil
import sys

from . import CREDITS, __version__, paths

COMMANDS = """commands:
  app          open the Mac app from this checkout (bin/build-app installs the real app)
  doctor       check models, camera and macOS permissions
  train        retrain the personal model from your calibration and reviewed clips
  eval         word error rate of the base vs personal model on your held-out clips
  transcribe   lip-read a video file
"""


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

    ok = True
    missing = check_models()
    print(f"silentdict {__version__}\n{CREDITS}\n")
    print("models        ", "OK" if not missing else "MISSING " + ", ".join(missing) + " (run ./setup.sh)")
    ok &= not missing
    print("local LLM     ", "OK" if paths.LLM_DIR.exists() else "not installed (optional: ./setup.sh --with-llm)")
    print("Apple GPU     ", "OK (MPS)" if torch.backends.mps.is_available() else "not available, using CPU")
    print("data folder   ", paths.DATA)
    im = input_monitoring_allowed(prompt=args.prompt)
    ax = accessibility_allowed(prompt=args.prompt)
    print("Input Monitor ", "OK" if im else "NOT GRANTED (needed for the dictation key)")
    print("Accessibility ", "OK" if ax else "NOT GRANTED (needed to paste into other apps)")
    cam = camera_status()
    print("Camera access ", "OK" if cam == 3 else STATUS.get(cam, "unknown").upper())
    if args.prompt and cam == 0:
        cam = 3 if ensure_camera_access() else 2
    if args.camera_test and cam != 3 and isinstance(args.camera, int):
        print("camera         skipped: camera access not granted")
        ok = False
    elif args.camera_test:
        ok &= _camera_test(args.camera)
    print("\nThese permissions belong to the app that runs this command (Terminal, iTerm, VS Code...).\n"
          "The Mac app asks for its own permissions the first time you open it.")
    return 0 if ok else 1


def _camera_test(index) -> bool:
    import time

    from .capture import CameraEngine

    eng = CameraEngine(on_segment=lambda f: None, camera_index=index)
    eng.start()
    eng.ready.wait(10)
    if eng.error:
        print("camera        ", eng.error)
        eng.shutdown()
        return False
    faces = frames = 0
    brightness = []
    last = None
    t_end = time.time() + 3
    while time.time() < t_end:
        f = eng.latest
        if f is not None and f is not last:
            frames += 1
            faces += f.obs is not None
            brightness.append(float(f.gray.mean()))
            last = f
        time.sleep(0.01)
    eng.shutdown()
    mean_b = sum(brightness[-15:]) / max(len(brightness[-15:]), 1)
    print("camera        ", f"{eng.measured_fps:.0f} fps, face found in {faces}/{frames} frames, "
          f"brightness {mean_b:.0f}/255")
    if mean_b < 2:
        print("               the image is black: is the lens covered? Try another camera with --camera 1")
        return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(prog="silentdict", description="Camera-only silent dictation for macOS.",
                                     epilog=COMMANDS, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"silentdict {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("app", help="open the Mac app")

    p = sub.add_parser("doctor", help="check setup")
    p.add_argument("--camera", type=_camera, default=0, help="camera index, or a video file")
    p.add_argument("--camera-test", action="store_true", help="open the camera for 3 seconds")
    p.add_argument("--prompt", action="store_true", help="ask macOS for missing permissions")

    p = sub.add_parser("train", help="retrain the personal model")
    p.add_argument("--device", default="auto", help="auto, mps or cpu")
    p.add_argument("--epochs", type=int, default=None, help="override the epochs of each strategy")
    p.add_argument("--lr", type=float, default=None, help="override the learning rate of each strategy")
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--strategy", choices=["light", "full", "both"], default="both")
    p.add_argument("--min-items", type=int, default=20)

    p = sub.add_parser("eval", help="evaluate on held-out personal clips")
    p.add_argument("--device", default="auto")

    p = sub.add_parser("transcribe", help="lip-read a video file")
    p.add_argument("video")
    p.add_argument("--start", type=float, default=0.0, help="seconds")
    p.add_argument("--end", type=float, default=None, help="seconds")
    p.add_argument("--roi", action="store_true", help="the video is already a 96x96 mouth crop")
    p.add_argument("--llm", action="store_true", help="also apply the optional local LLM clean-up")
    p.add_argument("--base", action="store_true", help="ignore the personal model")
    p.add_argument("--beam", type=int, default=None, help="beam size (default 20)")
    p.add_argument("--device", default="auto")

    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 0
    from .dataset import NotEnoughData

    ensure_user_files()
    if args.cmd != "doctor":
        missing = check_models()
        if missing:
            print("Missing model files: " + ", ".join(missing) + "\nRun ./setup.sh first.")
            return 1

    if args.cmd == "app":
        from .mac.main import main as app_main
        return app_main()
    if args.cmd == "doctor":
        return doctor(args)
    if args.cmd == "train":
        from . import train
        strategies = ("light", "full") if args.strategy == "both" else (args.strategy,)
        try:
            train.run(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, device=args.device,
                      min_items=args.min_items, strategies=strategies)
        except NotEnoughData as exc:
            print(exc)
            return 1
        return 0
    if args.cmd == "eval":
        from . import train
        try:
            train.evaluate(device=args.device)
        except NotEnoughData as exc:
            print(exc)
            return 1
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
    decode = {"beam_size": args.beam} if args.beam else None
    rec = Recognizer(rewriter=rewriter, device=args.device, personal=not args.base, decode=decode)
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
