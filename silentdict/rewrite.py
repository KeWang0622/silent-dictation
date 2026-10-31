"""Optional on-device language model (MLX) that re-ranks the lip reader's candidates.

The language model only scores sentences the lip reader already proposed, so it can pick the more natural of
look-alike readings ("five" over "fine") but can never invent words. On 120 read sentences (VidTIMIT, see
bin/benchmark) this cut FLARE's word error rate from 32.7% to 30.7% at beam 10, and to 30.3% at beam 20.
"""
from __future__ import annotations

import threading

from . import paths
from .text import sentence_case

DEFAULT_LLM = paths.LLM_DIR
WEIGHTS = {"flare": 0.3, "auto_avsr": 0.2}  # language-model weight, chosen on held-out speakers


class Rescorer:
    def __init__(self, model_path=None):
        from mlx_lm import load

        self.model, self.tok = load(str(model_path or DEFAULT_LLM))
        self.lock = threading.Lock()
        vocab = self._load_vocab()
        self.prefix = ("Words this person often uses: " + ", ".join(vocab) + "\n" if vocab else "") + \
            "Here is something a person typed:\n"
        self.prefix_ids = self.tok.encode(self.prefix)
        self.cache: dict[str, float] = {}

    @staticmethod
    def _load_vocab() -> list[str]:
        if not paths.PERSONAL_CONTEXT.exists():
            return []
        words = [w.strip() for w in paths.PERSONAL_CONTEXT.read_text().splitlines()]
        return [w for w in words if w and not w.startswith("#")][:100]

    def logprob(self, text: str) -> float:
        """log P(text | prefix) under the language model, for an upper-case transcript."""
        if text in self.cache:
            return self.cache[text]
        import mlx.core as mx

        ids = self.tok.encode(self.prefix + sentence_case(text))
        n = len(self.prefix_ids)
        x = mx.array(ids)[None]
        with self.lock:
            logits = self.model(x[:, :-1]).astype(mx.float32)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            tok_lp = mx.take_along_axis(lp, x[:, 1:, None], axis=-1)[0, :, 0]
            value = float(tok_lp[n - 1:].sum())
        if len(self.cache) > 2000:
            self.cache.clear()
        self.cache[text] = value
        return value

    def best(self, hyps, model: str = "flare") -> str:
        """The candidate with the best combined lip-reading + language-model score."""
        lam = WEIGHTS.get(model, 0.2)
        return max(hyps, key=lambda h: (h.logprob if h.logprob is not None else h.score) + lam * self.logprob(h.text)).text
