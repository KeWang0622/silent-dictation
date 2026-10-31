"""Personal adaptation: fine-tune the lip-reading model on your own silent speech.

Every run starts again from the public base model and uses all personal clips collected so far
(calibration + reviewed everyday dictation). The held-out clips never change their split, so the
reported word error rate is an honest measure of progress over time. Works with both lip readers:
FLARE (lip + face crops) and Auto-AVSR (lip crops).
"""
from __future__ import annotations

import csv
import json
import math
import time

import numpy as np
import torch

from . import paths
from .dataset import Item, NotEnoughData, load_items
from .text import corpus_wer
from .vsr import pick_device

STRATEGIES = {
    # Adapts the visual front-end(s), normalisation layers and biases: few parameters, robust with little data.
    "light": {"lr": {"flare": 1e-4, "auto_avsr": 1e-4}, "epochs": 10},
    # Adapts every weight with a small learning rate: best once you have more sentences.
    "full": {"lr": {"flare": 2e-5, "auto_avsr": 3e-5}, "epochs": 10},
}


# ---------------------------------------------------------------------------- data
def _resample(n: int, rng: np.random.Generator) -> np.ndarray:
    speed = rng.uniform(0.9, 1.1)
    m = max(8, int(round(n / speed)))
    return np.clip(np.round(np.arange(m) * speed).astype(int), 0, n - 1)


def _augment_stream(x: np.ndarray, idx, offset, flip, gain, bias, drops, mean, std) -> torch.Tensor:
    oy, ox = offset
    t = torch.from_numpy(np.ascontiguousarray(x[idx][:, oy:oy + 88, ox:ox + 88])).float() / 255.0
    t = (t - mean) / std
    if flip:
        t = t.flip(-1)
    t = t * gain + bias
    for start, length in drops:
        t[start:start + length] = t.mean()
    return t


def augment_clip(clip, rng: np.random.Generator, mean: float, std: float):
    """Same random speed, crop, flip, brightness and frame drops for the lip and face streams."""
    idx = _resample(len(clip.lip), rng)
    offset = tuple(int(v) for v in rng.integers(0, 9, size=2))
    flip, gain, bias = rng.random() < 0.5, rng.uniform(0.85, 1.15), rng.uniform(-0.3, 0.3)
    drops = []
    for _ in range(max(1, len(idx) // 25)):
        length = int(rng.integers(0, 11))
        drops.append((int(rng.integers(0, max(1, len(idx) - length))), length))
    args = (idx, offset, flip, gain, bias, drops, mean, std)
    lip = _augment_stream(clip.lip, *args)
    face = _augment_stream(clip.face, *args) if clip.face is not None else None
    return lip, face


def collate(tensors: list[torch.Tensor | None]):
    real = [t for t in tensors if t is not None]
    lens = torch.tensor([len(t) if t is not None else 0 for t in tensors])
    out = torch.zeros(len(tensors), 1, int(max(len(t) for t in real)), 88, 88)
    for i, t in enumerate(tensors):
        if t is not None:
            out[i, 0, : len(t)] = t
    return out, lens


def decode_items(reader, items: list[Item], rescorer=None, on_item=None) -> list[tuple[str, str]]:
    pairs = []
    for i, it in enumerate(items):
        if on_item:
            on_item(i / max(1, len(items)))
        hyps = reader(it.clip(), nbest=10)
        hyp = hyps[0].text if hyps else ""
        if rescorer is not None and len(hyps) > 1:
            hyp = rescorer.best(hyps, reader.name)
        pairs.append((it.text, hyp))
    return pairs


# ---------------------------------------------------------------------------- backends
class AutoAvsrBackend:
    name = "auto_avsr"
    mean, std = 0.421, 0.165

    def reader(self, device, personal, decode=None):
        from .vsr import LipReader
        return LipReader(device=device, personal=personal, decode=decode)

    def quick_reader(self, device):
        return self.reader(device, False, {"beam_size": 5, "lm_weight": 0.0})

    def model(self, dev):
        from .vsr import load_vsr_model
        return load_vsr_model(dev)

    def trainable(self, name: str, scope: str) -> bool:
        return scope == "full" or name.startswith(("encoder.frontend", "encoder.embed")) or "norm" in name \
            or name.endswith(".bias")

    def labels(self, items):
        from .vsr import Tokenizer
        tok = Tokenizer()
        return {it.id: torch.tensor(tok.encode(it.text)) for it in items}

    def loss(self, model, batch, labels, dev, rng):
        from .vsr import batch_loss
        lips = [augment_clip(it.clip(), rng, self.mean, self.std)[0] for it in batch]
        xs, lens = collate(lips)
        return batch_loss(model, xs.to(dev), lens.to(dev), [labels[it.id] for it in batch])[0]

    def load_into_reader(self, reader, state):
        reader.model.load_state_dict(state)

    def save(self, state, lm_weight=None):
        torch.save(state, paths.PERSONAL_WEIGHTS)

    weights_path, decode_path = paths.PERSONAL_WEIGHTS, paths.PERSONAL_DECODE


class FlareBackend:
    name = "flare"
    mean, std = 0.421, 0.165

    def reader(self, device, personal, decode=None):
        from .flare.reader import FlareReader
        return FlareReader(device=device, personal=personal, decode=decode)

    def quick_reader(self, device):
        return self.reader(device, False, {"beam_size": 4})

    def model(self, dev):
        from .flare.reader import load_flare
        model = load_flare(dev)
        model.encoder.checkpoint = True
        return model

    def trainable(self, name: str, scope: str) -> bool:
        if scope == "full":
            return True
        return name.startswith(("encoder.face.", "encoder.lip.", "encoder.post_extract_proj", "encoder.layer_norm")) \
            or "norm" in name or name.endswith(".bias")

    def labels(self, items):
        from .flare.reader import Tokenizer
        tok = Tokenizer()
        return {it.id: tok.encode(it.text) for it in items}

    def loss(self, model, batch, labels, dev, rng):
        from .flare.reader import batch_loss
        streams = [augment_clip(it.clip(), rng, self.mean, self.std) for it in batch]
        lips, lens = collate([s[0] for s in streams])
        faces = None
        if all(s[1] is not None for s in streams):
            faces, _ = collate([s[1] for s in streams])
            faces = faces[:, :, : lips.size(2)].to(dev)
        return batch_loss(model, faces, lips.to(dev), lens.to(dev), [labels[it.id] for it in batch])

    def load_into_reader(self, reader, state):
        reader.model.load_state_dict(state, strict=False)

    weights_path, decode_path = paths.PERSONAL_FLARE_WEIGHTS, paths.PERSONAL_FLARE_DECODE


def backend_for(model: str | None = None):
    from .pipeline import resolve_model
    return FlareBackend() if resolve_model(model or "auto") == "flare" else AutoAvsrBackend()


def frame_budget() -> int:
    """Video frames per training batch, scaled to this Mac's memory (roughly 20 MB per frame with checkpointing)."""
    import os
    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        ram_gb = 16
    return int(min(600, max(150, (ram_gb - 8) * 25)))


def make_batches(items, order, max_frames: int, max_items: int):
    batches, cur, frames = [], [], 0
    for i in order:
        n = max(1, items[i].frames)
        if cur and (frames + n > max_frames or len(cur) >= max_items):
            batches.append(cur)
            cur, frames = [], 0
        cur.append(items[i])
        frames += n
    if cur:
        batches.append(cur)
    return batches


def freeze_batchnorm(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


# ---------------------------------------------------------------------------- training
def fine_tune(backend, train_items, val_items=None, epochs: int = 10, lr: float = 1e-4, batch_size: int = 4,
              device: str = "auto", seed: int = 0, log=print, quick_reader=None, scope: str = "full", on_epoch=None):
    """Returns (best weights on cpu: only the parameters that were trained, history, best score)."""
    dev = pick_device(device)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = backend.model(dev)
    labels = backend.labels(train_items)
    names, params = [], []
    for n, p in model.named_parameters():
        use = backend.trainable(n, scope)
        p.requires_grad_(use)
        if use:
            names.append(n)
            params.append(p)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    budget = frame_budget() if backend.name == "flare" else 10**9
    steps_per_epoch = len(make_batches(train_items, range(len(train_items)), budget, batch_size))
    total = epochs * steps_per_epoch
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    trained = set(names)
    best_state, best_score, history = None, float("inf"), []
    for epoch in range(1, epochs + 1):
        model.train()
        freeze_batchnorm(model)
        order = rng.permutation(len(train_items))
        losses, t0 = [], time.time()
        for batch in make_batches(train_items, order, budget, batch_size):
            loss = backend.loss(model, batch, labels, dev, rng)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sched.step()
            losses.append(loss.detach().item())
        model.eval()
        if dev.type == "mps":
            torch.mps.empty_cache()
        entry = {"epoch": epoch, "loss": float(np.mean(losses)), "seconds": round(time.time() - t0, 1)}
        state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()
                 if k in trained or backend.name == "auto_avsr"}
        if val_items and quick_reader is not None:
            backend.load_into_reader(quick_reader, state)
            entry["val_wer"] = corpus_wer(decode_items(quick_reader, val_items))
            score = entry["val_wer"]
        else:
            score = entry["loss"]
        history.append(entry)
        log(f"  epoch {epoch:2d}  loss {entry['loss']:.3f}" +
            (f"  val WER {entry['val_wer']:.1%}" if "val_wer" in entry else "") + f"  ({entry['seconds']}s)")
        if on_epoch:
            on_epoch(epoch, epochs, entry)
        if score < best_score:
            best_score, best_state = score, state
    return best_state, history, best_score


def run(epochs: int | None = None, lr: float | None = None, batch_size: int = 4, device: str = "auto",
        min_items: int = 20, use_llm: bool | None = None, log=print, progress=None,
        strategies=("light", "full"), model: str | None = None) -> dict:
    """progress(stage, fraction) receives: baseline, adapt, tune, llm, done."""
    progress = progress or (lambda stage, frac: None)
    backend = backend_for(model)
    train_items = load_items("train")
    val_items = load_items("val")
    if len(train_items) + len(val_items) < min_items:
        raise NotEnoughData(f"Only {len(train_items) + len(val_items)} personal sentences so far. Record at least "
                            f"{min_items} in Calibrate first.")
    log(f"Model: {backend.name}. Personal clips: {len(train_items)} train, {len(val_items)} held out")

    # 1) baseline on the held-out clips
    progress("baseline", 0.0)
    reader = backend.reader(device, False)
    base_pairs = decode_items(reader, val_items, on_item=lambda f: progress("baseline", 0.6 * f))
    base_wer = corpus_wer(base_pairs)
    log(f"Base model WER on your held-out clips: {base_wer:.1%}")
    current_wer = None
    if backend.weights_path.exists():
        current = backend.reader(device, True)
        current_wer = corpus_wer(decode_items(current, val_items, on_item=lambda f: progress("baseline", 0.6 + 0.4 * f)))
        del current
        log(f"Current personal model WER on the same clips: {current_wer:.1%}")

    # 2) adapt with each strategy, choosing epoch and strategy by held-out WER (fast decoding)
    quick = backend.quick_reader(device)
    t0 = time.time()
    best_run, hist, strategy = None, [], None
    for k, name in enumerate(strategies):
        cfg = STRATEGIES[name]
        n_epochs = epochs or cfg["epochs"]
        rate = lr or cfg["lr"][backend.name]
        log(f"Adapting ({name}: lr {rate}, {n_epochs} epochs)")

        def on_epoch(e, total, entry, k=k):
            progress("adapt", (k + e / total) / len(strategies))

        state, h, score = fine_tune(backend, train_items, val_items, epochs=n_epochs, lr=rate, batch_size=batch_size,
                                    device=device, log=log, quick_reader=quick, scope=name, on_epoch=on_epoch)
        log(f"  {name}: best quick held-out WER {score:.1%}")
        if best_run is None or score < best_run[0]:
            best_run, hist, strategy = (score, state), h, name
        backend.load_into_reader(quick, backend.model(torch.device("cpu")).state_dict())  # reset for the next strategy
    state = best_run[1]
    del quick
    log(f"Adaptation took {time.time() - t0:.0f}s; best strategy: {strategy}")

    # 3) full decoder with the personal weights (Auto-AVSR: also tune the language-model weight)
    progress("tune", 0.0)
    backend.load_into_reader(reader, state)
    decode = {}
    if backend.name == "auto_avsr":
        best = None
        weights = (0.0, 0.15, 0.3, 0.5)
        for i, lm_weight in enumerate(weights):
            reader.set_decode(lm_weight=lm_weight)
            w = corpus_wer(decode_items(reader, val_items, on_item=lambda f, i=i: progress("tune", (i + f) / len(weights))))
            log(f"  personal model, LM weight {lm_weight}: WER {w:.1%}")
            if best is None or w < best[0]:
                best = (w, lm_weight)
        personal_wer, decode["lm_weight"] = best
        reader.set_decode(lm_weight=decode["lm_weight"])
    else:
        personal_wer = corpus_wer(decode_items(reader, val_items, on_item=lambda f: progress("tune", f)))
        log(f"  personal model: WER {personal_wer:.1%}")

    # 4) does the local language model help *you*? Decide from data.
    llm_wer = None
    if use_llm is not False and paths.LLM_DIR.exists():
        progress("llm", 0.0)
        from .rewrite import Rescorer
        llm_wer = corpus_wer(decode_items(reader, val_items, rescorer=Rescorer(), on_item=lambda f: progress("llm", f)))
        log(f"  personal model + language-model rescoring: WER {llm_wer:.1%}")

    # 5) keep whichever model is best on the held-out clips: new personal, current personal or base
    options = {"base": base_wer, "new": personal_wer}
    if current_wer is not None:
        options["current"] = current_wer
    winner = min(options, key=lambda k: (options[k], {"base": 0, "current": 1, "new": 2}[k]))
    paths.PERSONAL.mkdir(parents=True, exist_ok=True)
    if winner == "current":
        log(f"The current personal model is still the best ({current_wer:.1%}); keeping it unchanged.")
    else:
        if winner == "new":
            if backend.name == "flare":  # half precision keeps a full fine-tune at ~800 MB
                state = {k: v.half() if v.is_floating_point() else v for k, v in state.items()}
            torch.save(state, backend.weights_path)
            log(f"Saved personal model: WER {base_wer:.1%} (base) -> {personal_wer:.1%}")
        else:
            if backend.weights_path.exists():
                backend.weights_path.replace(backend.weights_path.with_suffix(".previous.pth"))
            decode = {}
            log("The personal model did not beat the base model on held-out clips; using the base model. "
                "Record or review more sentences and train again.")
        decode["use_llm"] = bool(llm_wer is not None and llm_wer <= personal_wer) if use_llm is None else bool(use_llm)
        json.dump(decode, open(backend.decode_path, "w"), indent=2)

    report = {
        "time": time.strftime("%Y-%m-%d %H:%M"), "model": backend.name, "train_clips": len(train_items),
        "val_clips": len(val_items), "base_wer": base_wer, "previous_personal_wer": current_wer,
        "personal_wer": personal_wer, "llm_wer": llm_wer, "active": winner, "active_wer": options[winner],
        "strategy": strategy, "decode": decode, "epochs": hist,
        "examples": [{"ref": r, "base": b} for r, b in base_pairs[:10]],
    }
    paths.REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json.dump(report, open(paths.REPORTS / f"train-{stamp}.json", "w"), indent=2)
    progress_csv = paths.REPORTS / "progress.csv"
    new = not progress_csv.exists()
    with open(progress_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "train_clips", "val_clips", "base_wer", "personal_wer", "llm_wer", "active",
                        "active_wer", "model"])
        w.writerow([report["time"], len(train_items), len(val_items), f"{base_wer:.4f}", f"{personal_wer:.4f}",
                    "" if llm_wer is None else f"{llm_wer:.4f}",
                    {"new": "new personal", "current": "previous personal", "base": "base"}[winner],
                    f"{options[winner]:.4f}", backend.name])
    progress("done", 1.0)
    return report


def evaluate(device: str = "auto", log=print, model: str | None = None) -> dict:
    backend = backend_for(model)
    val_items = load_items("val")
    if not val_items:
        raise NotEnoughData("No held-out sentences yet. Calibrate in the app first.")
    out = {}
    for name, personal in (("base", False), ("personal", True)):
        if personal and not backend.weights_path.exists():
            continue
        out[name] = corpus_wer(decode_items(backend.reader(device, personal), val_items))
        log(f"{backend.name} {name:9s} WER {out[name]:.1%} on {len(val_items)} held-out clips")
    return out
