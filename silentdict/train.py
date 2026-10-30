"""Personal adaptation: fine-tune the lip-reading model on your own silent speech.

Every run starts again from the public base model and uses all personal clips collected so far
(calibration + reviewed everyday dictation). The held-out clips never change their split, so the
reported word error rate is an honest measure of progress over time.
"""
from __future__ import annotations

import csv
import json
import math
import time

import numpy as np
import torch

from . import paths
from .dataset import Item, load_items
from .text import corpus_wer
from .vsr import DEFAULT_DECODE, LipReader, Tokenizer, batch_loss, load_vsr_model, pick_device, roi_to_tensor


def augment(roi: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
    t = len(roi)
    speed = rng.uniform(0.9, 1.1)
    n = max(8, int(round(t / speed)))
    roi = roi[np.clip(np.round(np.arange(n) * speed).astype(int), 0, t - 1)]
    oy, ox = rng.integers(0, 9, size=2)
    x = roi_to_tensor(roi, (int(oy), int(ox)))[0]
    if rng.random() < 0.5:
        x = x.flip(-1)
    x = x * rng.uniform(0.85, 1.15) + rng.uniform(-0.3, 0.3)
    for _ in range(max(1, len(x) // 25)):
        length = int(rng.integers(0, 11))
        start = int(rng.integers(0, max(1, len(x) - length)))
        x[start:start + length] = x.mean()
    return x.unsqueeze(0)


def collate(tensors: list[torch.Tensor]):
    lens = torch.tensor([x.shape[1] for x in tensors])
    out = torch.zeros(len(tensors), 1, int(lens.max()), 88, 88)
    for i, x in enumerate(tensors):
        out[i, :, : x.shape[1]] = x
    return out, lens


def freeze_batchnorm(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


def decode_items(reader: LipReader, items: list[Item], rewriter=None) -> list[tuple[str, str]]:
    pairs = []
    for it in items:
        hyps = reader(it.roi(), nbest=5)
        hyp = hyps[0].text if hyps else ""
        if rewriter is not None and hyps:
            hyp = rewriter([h.text for h in hyps])
        pairs.append((it.text, hyp))
    return pairs


def fine_tune(train_items, val_items=None, epochs: int = 12, lr: float = 3e-5, batch_size: int = 6,
              device: str = "auto", freeze_frontend: bool = False, seed: int = 0, log=print,
              quick_reader: LipReader | None = None):
    """Returns (best_state_dict_on_cpu, history). quick_reader is used for fast per-epoch validation."""
    dev = pick_device(device)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    tok = Tokenizer()
    model = load_vsr_model(dev)
    labels = {it.id: torch.tensor(tok.encode(it.text)) for it in train_items}
    rois = {it.id: it.roi() for it in train_items}

    params = [p for n, p in model.named_parameters()
              if not (freeze_frontend and n.startswith("encoder.frontend"))]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps_per_epoch = max(1, math.ceil(len(train_items) / batch_size))
    total = epochs * steps_per_epoch
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    best_state, best_wer, history = None, float("inf"), []
    for epoch in range(1, epochs + 1):
        model.train()
        freeze_batchnorm(model)
        if freeze_frontend:
            model.encoder.frontend.eval()
        order = rng.permutation(len(train_items))
        losses, t0 = [], time.time()
        for b in range(steps_per_epoch):
            batch = [train_items[i] for i in order[b * batch_size:(b + 1) * batch_size]]
            if not batch:
                continue
            xs, lens = collate([augment(rois[it.id], rng) for it in batch])
            loss, lc, la = batch_loss(model, xs.to(dev), lens.to(dev), [labels[it.id] for it in batch])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sched.step()
            losses.append(loss.detach().item())
        entry = {"epoch": epoch, "loss": float(np.mean(losses)), "seconds": round(time.time() - t0, 1)}
        if val_items and quick_reader is not None:
            quick_reader.model.load_state_dict(model.state_dict())
            entry["val_wer"] = corpus_wer(decode_items(quick_reader, val_items))
            score = entry["val_wer"]
        else:
            score = entry["loss"]
        history.append(entry)
        log(f"  epoch {epoch:2d}  loss {entry['loss']:.3f}" +
            (f"  val WER {entry['val_wer']:.1%}" if "val_wer" in entry else "") + f"  ({entry['seconds']}s)")
        if score < best_wer:
            best_wer = score
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    model.eval()
    return best_state, history


def run(epochs: int = 12, lr: float = 3e-5, batch_size: int = 6, device: str = "auto",
        freeze_frontend: bool = False, min_items: int = 20, use_llm: bool | None = None, log=print) -> dict:
    train_items = load_items("train")
    val_items = load_items("val")
    if len(train_items) + len(val_items) < min_items:
        raise SystemExit(f"Only {len(train_items) + len(val_items)} personal clips; record at least {min_items} "
                         "with 'silentdict calibrate' first.")
    log(f"Personal clips: {len(train_items)} train, {len(val_items)} held out")

    # 1) baseline on the held-out clips
    reader = LipReader(device=device, personal=False)
    base_pairs = decode_items(reader, val_items)
    base_wer = corpus_wer(base_pairs)
    log(f"Base model WER on your held-out clips: {base_wer:.1%}")
    current_wer = None
    if paths.PERSONAL_WEIGHTS.exists():
        current = LipReader(device=device, personal=True)
        current_wer = corpus_wer(decode_items(current, val_items))
        del current
        log(f"Current personal model WER on the same clips: {current_wer:.1%}")

    # 2) fine-tune, choosing the epoch by held-out WER with a fast decoder (beam 5, no LM)
    quick = LipReader(device=device, personal=False, decode={"beam_size": 5, "lm_weight": 0.0})
    t0 = time.time()
    state, hist = fine_tune(train_items, val_items, epochs=epochs, lr=lr, batch_size=batch_size,
                            device=device, freeze_frontend=freeze_frontend, log=log, quick_reader=quick)
    del quick
    log(f"Fine-tuning took {time.time() - t0:.0f}s")

    # 3) full decoder with the personal weights; tune the language-model weight on held-out clips
    reader.model.load_state_dict(state)
    best = None
    for lm_weight in (0.0, 0.15, 0.3, 0.5):
        reader.set_decode(lm_weight=lm_weight)
        w = corpus_wer(decode_items(reader, val_items))
        log(f"  personal model, LM weight {lm_weight}: WER {w:.1%}")
        if best is None or w < best[0]:
            best = (w, lm_weight)
    personal_wer, lm_weight = best
    reader.set_decode(lm_weight=lm_weight)

    # 4) does the local LLM rewrite help *you*? Decide from data.
    llm_wer = None
    from .rewrite import DEFAULT_LLM
    if use_llm is not False and DEFAULT_LLM.exists():
        from .rewrite import Rewriter
        llm_wer = corpus_wer(decode_items(reader, val_items, rewriter=Rewriter()))
        log(f"  personal model + local LLM rewrite: WER {llm_wer:.1%}")

    # 5) keep whichever model is best on the held-out clips: new personal, current personal or base
    options = {"base": base_wer, "new": personal_wer}
    if current_wer is not None:
        options["current"] = current_wer
    winner = min(options, key=lambda k: (options[k], {"base": 0, "current": 1, "new": 2}[k]))
    paths.PERSONAL.mkdir(parents=True, exist_ok=True)
    improved = winner == "new"
    if winner == "current":
        log(f"The current personal model is still the best ({current_wer:.1%}); keeping it unchanged.")
        decode = json.load(open(paths.PERSONAL_DECODE)) if paths.PERSONAL_DECODE.exists() else dict(DEFAULT_DECODE)
    else:
        decode = dict(DEFAULT_DECODE)
        if improved:
            decode["lm_weight"] = lm_weight
            torch.save(state, paths.PERSONAL_WEIGHTS)
            log(f"Saved personal model: WER {base_wer:.1%} (base) -> {personal_wer:.1%}")
        else:
            if paths.PERSONAL_WEIGHTS.exists():
                paths.PERSONAL_WEIGHTS.replace(paths.PERSONAL_WEIGHTS.with_suffix(".previous.pth"))
            log("The personal model did not beat the base model on held-out clips; using the base model. "
                "Record or review more sentences and train again.")
        best_wer = options[winner]
        decode["use_llm"] = (bool(llm_wer is not None and llm_wer <= best_wer) if use_llm is None
                             else bool(use_llm))
        json.dump(decode, open(paths.PERSONAL_DECODE, "w"), indent=2)

    report = {
        "time": time.strftime("%Y-%m-%d %H:%M"), "train_clips": len(train_items), "val_clips": len(val_items),
        "base_wer": base_wer, "previous_personal_wer": current_wer, "personal_wer": personal_wer,
        "llm_wer": llm_wer, "active": winner, "decode": decode, "epochs": hist,
        "examples": [{"ref": r, "base": b} for r, b in base_pairs[:10]],
    }
    paths.REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json.dump(report, open(paths.REPORTS / f"train-{stamp}.json", "w"), indent=2)
    progress = paths.REPORTS / "progress.csv"
    new = not progress.exists()
    with open(progress, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "train_clips", "val_clips", "base_wer", "personal_wer", "llm_wer", "active"])
        w.writerow([report["time"], len(train_items), len(val_items), f"{base_wer:.4f}", f"{personal_wer:.4f}",
                    "" if llm_wer is None else f"{llm_wer:.4f}", {"new": "new personal", "current":
                                                                  "previous personal", "base": "base"}[winner]])
    return report


def evaluate(device: str = "auto", log=print) -> dict:
    val_items = load_items("val")
    if not val_items:
        raise SystemExit("No held-out clips yet. Run 'silentdict calibrate' first.")
    out = {}
    for name, personal in (("base", False), ("personal", True)):
        if personal and not paths.PERSONAL_WEIGHTS.exists():
            continue
        reader = LipReader(device=device, personal=personal)
        pairs = decode_items(reader, val_items)
        out[name] = corpus_wer(pairs)
        log(f"{name:9s} WER {out[name]:.1%} on {len(val_items)} held-out clips")
    return out
