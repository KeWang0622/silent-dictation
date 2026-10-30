"""Optional local LLM that turns lip-reading n-best lists into a clean sentence (MLX, on-device)."""
from __future__ import annotations

import re
import threading

from . import paths
from .text import normalize, sentence_case

DEFAULT_LLM = paths.MODELS / "llm" / "Qwen3-4B-Instruct-2507-4bit"

SYSTEM = """You fix the output of a lip-reading model used for silent dictation.
The person mouthed one short utterance without making a sound. A visual speech recogniser produced
several candidate transcripts (lower case, no punctuation), best first. Lip reading confuses words that
look alike on the lips: p/b/m, f/v, th, t/d/n/s/z, k/g/ng, ch/j/sh, w/r/oo, and many vowels.

Write the single most likely intended utterance:
- Start from candidate 1. Switch to another candidate, or replace a word with a look-alike word,
  only when that makes the sentence clearly more natural.
- Use the recent context and the user's vocabulary when they make a reading clearly more likely.
- Never add new ideas, greetings or explanations. Keep the length about the same.
- Write normal sentence case with punctuation, never all caps.
Reply with the utterance only."""

EXAMPLES = [
    ("1. i'll be there in fine minutes\n2. i'll be there in five minutes\n3. all be there in five minutes",
     "I'll be there in five minutes."),
    ("1. can we bush the release to next we can\n2. can we push the release to next weekend",
     "Can we push the release to next weekend?"),
]


class Rewriter:
    def __init__(self, model_path=None, max_tokens: int = 96):
        from mlx_lm import generate, load

        self._generate = generate
        self.model, self.tokenizer = load(str(model_path or DEFAULT_LLM))
        self.max_tokens = max_tokens
        self.lock = threading.Lock()
        self.vocab = self._load_vocab()

    @staticmethod
    def _load_vocab() -> list[str]:
        if not paths.PERSONAL_CONTEXT.exists():
            return []
        words = [w.strip() for w in paths.PERSONAL_CONTEXT.read_text().splitlines()]
        return [w for w in words if w and not w.startswith("#")][:200]

    def prompt(self, candidates: list[str], context: list[str] | None = None) -> str:
        lines = [f"{i + 1}. {c.lower()}" for i, c in enumerate(candidates)]
        user = "Candidates:\n" + "\n".join(lines)
        if context:
            user = "Recent context: " + " ".join(context) + "\n\n" + user
        if self.vocab:
            user = "User vocabulary: " + ", ".join(self.vocab) + "\n\n" + user
        messages = [{"role": "system", "content": SYSTEM}]
        for question, answer in EXAMPLES:
            messages += [{"role": "user", "content": "Candidates:\n" + question},
                         {"role": "assistant", "content": answer}]
        messages.append({"role": "user", "content": user})
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def __call__(self, candidates: list[str], context: list[str] | None = None) -> str:
        if not candidates:
            return ""
        with self.lock:
            out = self._generate(self.model, self.tokenizer, prompt=self.prompt(candidates, context),
                                 max_tokens=self.max_tokens, verbose=False)
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip().splitlines()
        text = out[0].strip().strip('"').strip() if out else ""
        # Guard against the model wandering off: fall back to the recogniser's best guess.
        n_best = len(normalize(candidates[0]).split())
        n_out = len(normalize(text).split())
        if not text or n_out > 2 * n_best + 3 or n_out < max(1, n_best // 2 - 1):
            return sentence_case(candidates[0])
        if text.isupper():
            text = sentence_case(text)
        text = text[0].upper() + text[1:]
        return text if text[-1] in ".?!\"')" else text + "."
