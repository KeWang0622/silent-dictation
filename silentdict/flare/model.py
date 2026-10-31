"""FLARE Large in plain PyTorch: dual face + lip front ends, Conformer-style encoder, Transformer decoder."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from .convnext import ConvEncoderT

DIM, HEADS, KV_HEADS, LAYERS, DEC_LAYERS, DEC_HEADS, FFN = 768, 12, 2, 18, 6, 4, 3072
ROPE_THETA = 1_000_000.0
PAD, BOS, EOS, UNK = 1, 0, 2, 3


# ---------------------------------------------------------------------------- encoder
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def rotary(t: int, device, dtype, dim: int = DIM // HEADS):
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    freqs = torch.outer(torch.arange(t, device=device, dtype=torch.float32), inv)
    emb = torch.cat((freqs, freqs), -1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, -1)
    return torch.cat((-x2, x1), -1)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = DIM // HEADS
        self.q_proj = nn.Linear(DIM, HEADS * self.head_dim)
        self.k_proj = nn.Linear(DIM, KV_HEADS * self.head_dim)
        self.v_proj = nn.Linear(DIM, KV_HEADS * self.head_dim)
        self.o_proj = nn.Linear(HEADS * self.head_dim, DIM, bias=False)

    def forward(self, x, cos, sin, keep=None):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, HEADS, -1).transpose(1, 2)
        k = self.k_proj(x).view(b, t, KV_HEADS, -1).transpose(1, 2)
        v = self.v_proj(x).view(b, t, KV_HEADS, -1).transpose(1, 2)
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        rep = HEADS // KV_HEADS
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
        return self.o_proj(out.transpose(1, 2).reshape(b, t, DIM))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(DIM, FFN, bias=False)
        self.up_proj = nn.Linear(DIM, FFN, bias=False)
        self.down_proj = nn.Linear(FFN, DIM, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


class GLU(nn.Module):
    def forward(self, x):
        out, gate = x.chunk(2, dim=1)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, dim: int, kernel: int = 15):
        super().__init__()
        self.pad = (kernel // 2, kernel // 2)
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad))


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class ConvModule(nn.Module):
    def __init__(self):
        super().__init__()
        # indices match the original nn.Sequential so checkpoint keys load unchanged
        self.net = nn.Sequential(RMSNorm(DIM), nn.Linear(DIM, 2 * DIM), Transpose(), GLU(), DepthWiseConv1d(DIM),
                                 Transpose(), RMSNorm(DIM), Swish(), nn.Linear(DIM, DIM), nn.Identity())

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()
        self.mlp1, self.mlp2 = MLP(), MLP()
        self.conv = ConvModule()
        self.input_layernorm1, self.input_layernorm2 = RMSNorm(DIM), RMSNorm(DIM)
        self.input_layernorm3, self.post_layernorm = RMSNorm(DIM), RMSNorm(DIM)

    def forward(self, x, cos, sin, keep=None):
        x = x + 0.5 * self.mlp1(self.input_layernorm1(x))
        x = x + self.self_attn(self.input_layernorm2(x), cos, sin, keep)
        x = x + self.conv(x)
        x = x + 0.5 * self.mlp2(self.input_layernorm3(x))
        return self.post_layernorm(x)


class Frontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ConvEncoderT()
        self.proj = nn.Linear(DIM, DIM)

    def forward(self, x):  # (B, 1, T, 88, 88) -> (B, DIM, T)
        return self.proj(self.resnet(x).transpose(1, 2)).transpose(1, 2)


class Encoder(nn.Module):
    checkpoint = False  # set for training: recompute activations in the backward pass to save memory

    def __init__(self):
        super().__init__()
        self.face = Frontend()
        self.lip = Frontend()
        self.layer_norm = nn.LayerNorm(3 * DIM)
        self.post_extract_proj = nn.Linear(3 * DIM, DIM)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(LAYERS)])

    def forward(self, face, lip, padding_mask=None):
        """face/lip: (B, 1, T, 88, 88) normalised grey crops, or None. Returns (B, T, DIM)."""
        ref = lip if lip is not None else face
        b, _, t = ref.shape[:3]
        zeros = ref.new_zeros(b, DIM, t)
        ckpt = self.checkpoint and self.training and torch.is_grad_enabled()
        run = (lambda fn, *a: torch.utils.checkpoint.checkpoint(fn, *a, use_reentrant=False)) if ckpt else \
            (lambda fn, *a: fn(*a))
        f = run(self.face, face) if face is not None else zeros
        l = run(self.lip, lip) if lip is not None else zeros
        x = torch.cat([zeros, f, l], 1).transpose(1, 2)  # audio stream is unused (zeros) for lip reading
        x = self.post_extract_proj(self.layer_norm(x))
        cos, sin = rotary(t, x.device, x.dtype)
        keep = None if padding_mask is None else ~padding_mask[:, None, None, :]
        for layer in self.layers:
            x = run(layer, x, cos, sin, keep)
        return x


# ---------------------------------------------------------------------------- decoder
def sinusoidal(n: int, dim: int = DIM) -> torch.Tensor:
    half = dim // 2
    scale = math.log(10000) / (half - 1)
    freqs = torch.exp(torch.arange(half, dtype=torch.float32) * -scale)
    emb = torch.arange(n, dtype=torch.float32)[:, None] * freqs[None]
    emb = torch.cat([emb.sin(), emb.cos()], 1)
    emb[PAD] = 0
    return emb


class MHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.k_proj = nn.Linear(DIM, DIM), nn.Linear(DIM, DIM)
        self.v_proj, self.out_proj = nn.Linear(DIM, DIM), nn.Linear(DIM, DIM)

    def heads(self, x):  # (B, T, DIM) -> (B, H, T, d)
        return x.view(x.size(0), x.size(1), DEC_HEADS, -1).transpose(1, 2)

    def forward(self, q, k, v, mask=None, causal=False):
        out = F.scaled_dot_product_attention(self.heads(q), k, v, attn_mask=mask, is_causal=causal)
        return self.out_proj(out.transpose(1, 2).reshape(q.size(0), q.size(1), DIM))


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn, self.encoder_attn = MHA(), MHA()
        self.self_attn_layer_norm, self.encoder_attn_layer_norm = nn.LayerNorm(DIM), nn.LayerNorm(DIM)
        self.fc1, self.fc2 = nn.Linear(DIM, FFN), nn.Linear(FFN, DIM)
        self.final_layer_norm = nn.LayerNorm(DIM)


class Decoder(nn.Module):
    def __init__(self, vocab: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, DIM, padding_idx=PAD)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(DEC_LAYERS)])
        self.layer_norm = nn.LayerNorm(DIM)
        self.register_buffer("positions", sinusoidal(2050), persistent=False)

    def cross_kv(self, enc):
        return [(l.encoder_attn.heads(l.encoder_attn.k_proj(enc)), l.encoder_attn.heads(l.encoder_attn.v_proj(enc)))
                for l in self.layers]

    def forward(self, tokens, enc, enc_keep=None):
        """Teacher forcing. tokens: (B, L) starting with EOS. enc: (B, T, DIM). Returns logits (B, L, V)."""
        L = tokens.size(1)
        x = self.embed_tokens(tokens) + self.positions[PAD + 1: PAD + 1 + L]
        cross = self.cross_kv(enc)
        for layer, (ck, cv) in zip(self.layers, cross):
            h = layer.self_attn_layer_norm(x)
            sa = layer.self_attn
            x = x + sa(sa.q_proj(h), sa.heads(sa.k_proj(h)), sa.heads(sa.v_proj(h)), causal=True)
            h = layer.encoder_attn_layer_norm(x)
            x = x + layer.encoder_attn(layer.encoder_attn.q_proj(h), ck, cv, mask=enc_keep)
            x = x + layer.fc2(F.relu(layer.fc1(layer.final_layer_norm(x))))
        return self.layer_norm(x) @ self.embed_tokens.weight.t()

    def step(self, tok, t, cache, cross):
        """One incremental step for a batch of beams. tok: (B,), t: position index. cache: per-layer [k, v]."""
        x = self.embed_tokens(tok)[:, None] + self.positions[PAD + 1 + t]
        for i, (layer, (ck, cv)) in enumerate(zip(self.layers, cross)):
            h = layer.self_attn_layer_norm(x)
            sa = layer.self_attn
            k, v = sa.heads(sa.k_proj(h)), sa.heads(sa.v_proj(h))
            if cache[i] is not None:
                k, v = torch.cat([cache[i][0], k], 2), torch.cat([cache[i][1], v], 2)
            cache[i] = [k, v]
            x = x + sa(sa.q_proj(h), k, v)
            h = layer.encoder_attn_layer_norm(x)
            x = x + layer.encoder_attn(layer.encoder_attn.q_proj(h), ck, cv)
            x = x + layer.fc2(F.relu(layer.fc1(layer.final_layer_norm(x))))
        return (self.layer_norm(x) @ self.embed_tokens.weight.t())[:, 0]


class FlareModel(nn.Module):
    def __init__(self, vocab: int = 1000):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(vocab)


# ---------------------------------------------------------------------------- search
@dataclass
class Hyp:
    tokens: list
    score: float      # length-normalised log-probability (what the beam ranks by)
    logprob: float    # total log-probability including </s>


@torch.inference_mode()
def beam_search(decoder: Decoder, enc: torch.Tensor, beam: int = 10, max_len: int | None = None,
                lenpen: float = 1.0, min_len: int = 1) -> list[Hyp]:
    """fairseq-style beam search for one utterance. enc: (1, T, DIM). Returns finished hypotheses, best first."""
    max_len = max_len or enc.size(1)
    cross = [(k.expand(beam, -1, -1, -1), v.expand(beam, -1, -1, -1)) for k, v in decoder.cross_kv(enc)]
    cache = [None] * len(decoder.layers)
    tokens = torch.full((beam, 1), EOS, dtype=torch.long, device=enc.device)
    scores = torch.zeros(beam, device=enc.device)
    finished: list[Hyp] = []
    for step in range(max_len + 1):
        lprobs = decoder.step(tokens[:, -1], step, cache, cross).float().log_softmax(-1)
        lprobs[:, PAD] = -math.inf
        if step >= max_len:
            keep_eos = lprobs[:, EOS].clone()
            lprobs.fill_(-math.inf)
            lprobs[:, EOS] = keep_eos
        if step < min_len:
            lprobs[:, EOS] = -math.inf
        cand = scores[:, None] + lprobs
        if step == 0:
            cand[1:] = -math.inf  # all beams start identical
        top, idx = cand.view(-1).topk(min(2 * beam, cand.numel()))
        src, tok = idx // lprobs.size(1), idx % lprobs.size(1)
        keep_src, keep_tok, keep_score = [], [], []
        for rank, (s, w, sc) in enumerate(zip(src.tolist(), tok.tolist(), top.tolist())):
            if sc == -math.inf:
                break
            if w == EOS:
                if rank < beam:
                    finished.append(Hyp(tokens[s, 1:].tolist(), sc / (step + 1) ** lenpen, sc))
            elif len(keep_src) < beam:
                keep_src.append(s)
                keep_tok.append(w)
                keep_score.append(sc)
        if len(finished) >= beam or not keep_src or step >= max_len:
            break
        while len(keep_src) < beam:  # pad with copies of the last live hypothesis
            keep_src.append(keep_src[-1])
            keep_tok.append(keep_tok[-1])
            keep_score.append(-math.inf)
        order = torch.tensor(keep_src, device=enc.device)
        tokens = torch.cat([tokens[order], torch.tensor(keep_tok, device=enc.device)[:, None]], 1)
        scores = torch.tensor(keep_score, device=enc.device)
        cache = [[k[order], v[order]] for k, v in cache]
    finished.sort(key=lambda h: -h.score)
    return finished[:beam]
