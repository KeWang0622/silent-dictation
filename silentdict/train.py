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
from .dataset import Item, NotEnoughData, load_items
from .text import accept_rewrite, corpus_wer
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


def decode_items(reader: LipReader, items: list[Item], rewriter=None, on_item=None) -> list[tuple[str, str]]:
    pairs = []
    for i, it in enumerate(items):
        if on_item:
            on_item(i / max(1, len(items)))
        hyps = reader(it.roi(), nbest=5)
        hyp = hyps[0].text if hyps else ""
        if rewriter is not None and hyps:
            hyp = rewriter([h.text for h in hyps])
        pairs.append((it.text, hyp))
    return pairs


STRATEGIES = {
    # Adapts the visual front-end, normalisation layers and biases: few parameters, robust with little data.
    "light": {"lr": 1e-4, "epochs": 10},
    # Adapts every weight with a small learning rate: best once you have more sentences.
    "full": {"lr": 3e-5, "epochs": 10},
}


def _trainable(name: str, scope: str) -> bool:
    if scope == "full":
        return True
    return name.startswith(("encoder.frontend", "encoder.embed")) or "norm" in name or name.endswith(".bias")


def fine_tune(train_items, val_items=None, epochs: int = 12, lr: float = 3e-5, batch_size: int = 6,
              device: str = "auto", freeze_frontend: bool = False, seed: int = 0, log=print,
              quick_reader: LipReader | None = None, scope: str = "full", on_epoch=None):
    """Returns (best_state_dict_on_cpu, history, best_score). quick_reader gives fast per-epoch validation."""
    dev = pick_device(device)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    tok = Tokenizer()
    model = load_vsr_model(dev)
    labels = {it.id: torch.tensor(tok.encode(it.text)) for it in train_items}
    rois = {it.id: it.roi() for it in train_items}

    params = []
    for n, p in model.named_parameters():
        use = _trainable(n, scope) and not (freeze_frontend and n.startswith("encoder.frontend"))
        p.requires_grad_(use)
        if use:
            params.append(p)
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
        if on_epoch:
            on_epoch(epoch, epochs, entry)
        if score < best_wer:
            best_wer = score
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    model.eval()
    return best_state, history, best_wer


def run(epochs: int | None = None, lr: float | None = None, batch_size: int = 6, device: str = "auto",
        freeze_frontend: bool = False, min_items: int = 20, use_llm: bool | None = None, log=print,
        progress=None, strategies=("light", "full")) -> dict:
    """progress(stage, fraction) receives: baseline, adapt, tune, llm, done."""
    progress = progress or (lambda stage, frac: None)
    train_items = load_items("train")
    val_items = load_items("val")
    if len(train_items) + len(val_items) < min_items:
        raise NotEnoughData(f"Only {len(train_items) + len(val_items)} personal sentences so far. Record at least "
                            f"{min_items} in Calibrate first.")
    log(f"Personal clips: {len(train_items)} train, {len(val_items)} held out")

    # 1) baseline on the held-out clips
    progress("baseline", 0.0)
    reader = LipReader(device=device, personal=False)
    base_pairs = decode_items(reader, val_items, on_item=lambda f: progress("baseline", 0.6 * f))
    base_wer = corpus_wer(base_pairs)
    log(f"Base model WER on your held-out clips: {base_wer:.1%}")
    current_wer = None
    if paths.PERSONAL_WEIGHTS.exists():
        current = LipReader(device=device, personal=True)
        current_wer = corpus_wer(decode_items(current, val_items, on_item=lambda f: progress("baseline", 0.6 + 0.4 * f)))
        del current
        log(f"Current personal model WER on the same clips: {current_wer:.1%}")

    # 2) adapt with each strategy, choosing epoch and strategy by held-out WER (fast decoder: beam 5, no LM)
    quick = LipReader(device=device, personal=False, decode={"beam_size": 5, "lm_weight": 0.0})
    t0 = time.time()
    best_run, hist, strategy = None, [], None
    for k, name in enumerate(strategies):
        cfg = dict(STRATEGIES[name])
        cfg["epochs"] = epochs or cfg["epochs"]
        cfg["lr"] = lr or cfg["lr"]
        log(f"Adapting ({name}: lr {cfg['lr']}, {cfg['epochs']} epochs)")

        def on_epoch(e, total, entry, k=k):
            progress("adapt", (k + e / total) / len(strategies))

        state, h, score = fine_tune(train_items, val_items, epochs=cfg["epochs"], lr=cfg["lr"], batch_size=batch_size,
                                    device=device, freeze_frontend=freeze_frontend, log=log, quick_reader=quick,
                                    scope=name, on_epoch=on_epoch)
        log(f"  {name}: best quick held-out WER {score:.1%}")
        if best_run is None or score < best_run[0]:
            best_run, hist, strategy = (score, state), h, name
    state = best_run[1]
    del quick
    log(f"Adaptation took {time.time() - t0:.0f}s; best strategy: {strategy}")

    # 3) full decoder with the personal weights; tune the language-model weight on held-out clips
    progress("tune", 0.0)
    reader.model.load_state_dict(state)
    best = None
    weights = (0.0, 0.15, 0.3, 0.5)
    for i, lm_weight in enumerate(weights):
        reader.set_decode(lm_weight=lm_weight)
        w = corpus_wer(decode_items(reader, val_items, on_item=lambda f, i=i: progress("tune", (i + f) / len(weights))))
        log(f"  personal model, LM weight {lm_weight}: WER {w:.1%}")
        progress("tune", (i + 1) / len(weights))
        if best is None or w < best[0]:
            best = (w, lm_weight)
    personal_wer, lm_weight = best
    reader.set_decode(lm_weight=lm_weight)

    # 4) does the local LLM rewrite help *you*? Decide from data.
    llm_wer = None
    if use_llm is not False and paths.LLM_DIR.exists():
        progress("llm", 0.0)
        from .rewrite import Rewriter
        rw = Rewriter()
        pairs = []
        for it in val_items:
            hyps = reader(it.roi(), nbest=5)
            if not hyps:
                pairs.append((it.text, ""))
                continue
            offered = [h.text for h in hyps if h.score >= hyps[0].score - 2.5]
            out = rw(offered)
            pairs.append((it.text, out if accept_rewrite(out, offered) else hyps[0].text))
        llm_wer = corpus_wer(pairs)
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
        "llm_wer": llm_wer, "active": winner, "strategy": strategy, "decode": decode, "epochs": hist,
        "examples": [{"ref": r, "base": b} for r, b in base_pairs[:10]],
    }
    paths.REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json.dump(report, open(paths.REPORTS / f"train-{stamp}.json", "w"), indent=2)
    report["active_wer"] = options[winner]
    progress_csv = paths.REPORTS / "progress.csv"
    new = not progress_csv.exists()
    with open(progress_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "train_clips", "val_clips", "base_wer", "personal_wer", "llm_wer", "active",
                        "active_wer"])
        w.writerow([report["time"], len(train_items), len(val_items), f"{base_wer:.4f}", f"{personal_wer:.4f}",
                    "" if llm_wer is None else f"{llm_wer:.4f}",
                    {"new": "new personal", "current": "previous personal", "base": "base"}[winner],
                    f"{options[winner]:.4f}"])
    progress("done", 1.0)
    return report


def evaluate(device: str = "auto", log=print) -> dict:
    val_items = load_items("val")
    if not val_items:
        raise NotEnoughData("No held-out sentences yet. Calibrate in the app first.")
    out = {}
    for name, personal in (("base", False), ("personal", True)):
        if personal and not paths.PERSONAL_WEIGHTS.exists():
            continue
        reader = LipReader(device=device, personal=personal)
        pairs = decode_items(reader, val_items)
        out[name] = corpus_wer(pairs)
        log(f"{name:9s} WER {out[name]:.1%} on {len(val_items)} held-out clips")
    return out
