"""Prompting, tokenization, LoRA training loop, NEFTune and Yes/No scoring.

Design notes
* Prompts match scripts/eval_baseline.py (same system prompt + "Passage/Question/Answer").
* Batches are LEFT-padded with explicit position_ids, so in every row the answer
  tokens sit at the same trailing positions. This lets us project only the last few
  hidden states through the 152k-vocab LM head (``logits_to_keep``), which is a large
  saving on CPU.
* Evaluation is a single forward pass comparing the logits of the "Yes" and "No"
  answer tokens (no sampling, no unparseable outputs). p_yes is stored per item so
  paired statistics can be computed later.
"""

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class Encoder:
    def __init__(self, tokenizer, system_prompt: str, max_len: int):
        self.tok = tokenizer
        self.system_prompt = system_prompt
        self.max_len = max_len
        self.yes_id = self._single_id("Yes")
        self.no_id = self._single_id("No")
        self.end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        # token overhead of the prompt with an empty passage (+2 answer tokens, +4 slack)
        self.overhead = len(self._prompt_ids("", "x" * 1)) + 6
        self.n_truncated = 0

    def _single_id(self, word: str) -> int:
        ids = self.tok.encode(word, add_special_tokens=False)
        assert len(ids) == 1, f"{word!r} is not a single token: {ids}"
        return ids[0]

    def _prompt_ids(self, passage: str, question: str) -> list[int]:
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Passage: {passage}\nQuestion: {question}\nAnswer:"},
        ]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.tok.encode(text, add_special_tokens=False)

    def prompt(self, rec: dict) -> list[int]:
        """Prompt ids; the passage (never the question) is truncated to fit max_len."""
        q_len = len(self.tok.encode(rec["question"], add_special_tokens=False))
        budget = self.max_len - self.overhead - q_len
        p_ids = self.tok.encode(rec["passage"], add_special_tokens=False)
        passage = rec["passage"]
        if len(p_ids) > budget:
            passage = self.tok.decode(p_ids[:max(budget, 16)])
            self.n_truncated += 1
        return self._prompt_ids(passage, rec["question"])

    def train_example(self, rec: dict) -> list[int]:
        ans = self.yes_id if rec["label"] == 1 else self.no_id
        return self.prompt(rec) + [ans, self.end_id]


def left_pad(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, T - len(s):] = torch.tensor(s)
        mask[i, T - len(s):] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids, mask, pos


class NEFTune:
    """Uniform embedding noise, scale alpha / sqrt(L * d) (Jain et al. 2024), training only."""

    def __init__(self, model, alpha: float):
        self.alpha = alpha
        self.handle = model.get_input_embeddings().register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if not module.training:
            return output
        L, d = output.shape[1], output.shape[2]
        mag = self.alpha / math.sqrt(L * d)
        return output + torch.zeros_like(output).uniform_(-mag, mag)


def autocast_ctx(device: str):
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def score_yes_no(model, enc: Encoder, records: list[dict], device: str, batch_size: int = 16) -> list[float]:
    """Returns p(Yes) renormalised over {Yes, No} for each record, in input order."""
    model.eval()
    prompts = [enc.prompt(r) for r in records]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # length bucketing
    out = [0.0] * len(prompts)
    pad = enc.tok.pad_token_id
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        ids, mask, pos = left_pad([prompts[i] for i in idx], pad)
        with autocast_ctx(device):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                           position_ids=pos.to(device), logits_to_keep=1).logits[:, -1, :]
        two = logits[:, [enc.yes_id, enc.no_id]].float()
        p_yes = torch.softmax(two, dim=-1)[:, 0].cpu().tolist()
        for i, p in zip(idx, p_yes):
            out[i] = p
    return out


def accuracy(p_yes: list[float], records: list[dict]) -> float:
    return sum((p > 0.5) == (r["label"] == 1) for p, r in zip(p_yes, records)) / len(records)


def train_lora(model, enc: Encoder, records: list[dict], cfg: dict, seed: int, device: str,
               micro_bs: int, log_fh, max_steps: int | None = None, neftune: bool = False,
               on_epoch_end=None) -> dict:
    from transformers import get_cosine_schedule_with_warmup

    eff_bs = cfg["effective_batch_size"]
    assert eff_bs % micro_bs == 0, "effective_batch_size must be divisible by micro batch"
    accum = eff_bs // micro_bs
    examples = [enc.train_example(r) for r in records]
    steps_per_epoch = math.ceil(len(examples) / eff_bs)
    total_steps = steps_per_epoch * cfg["epochs"]
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
    sched = get_cosine_schedule_with_warmup(opt, math.ceil(cfg["optim"]["warmup_ratio"] * total_steps), total_steps)
    use_scaler = device == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    hook = NEFTune(model, cfg["neftune_alpha"]) if neftune else None

    rng = random.Random(seed)
    pad = enc.tok.pad_token_id
    step, t0 = 0, time.time()
    try:
        for epoch in range(cfg["epochs"]):
            perm = list(range(len(examples)))
            rng.shuffle(perm)
            for s in range(steps_per_epoch):
                if step >= total_steps:
                    break
                model.train()
                batch = perm[s * eff_bs:(s + 1) * eff_bs]
                n_tok = 2 * len(batch)
                loss_sum = 0.0
                for m in range(0, len(batch), micro_bs):
                    mb = [examples[i] for i in batch[m:m + micro_bs]]
                    ids, mask, pos = left_pad(mb, pad)
                    ids, mask, pos = ids.to(device), mask.to(device), pos.to(device)
                    with autocast_ctx(device):
                        # positions T-3, T-2 predict the answer token and <|im_end|>
                        logits = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                                       logits_to_keep=3).logits[:, :2, :].float()
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                           ids[:, -2:].reshape(-1), reduction="sum") / n_tok
                    (scaler.scale(loss) if scaler else loss).backward()
                    loss_sum += loss.item()
                if scaler:
                    scaler.unscale_(opt)
                gnorm = torch.nn.utils.clip_grad_norm_(params, cfg["optim"]["max_grad_norm"]).item()
                if scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                rec = {"step": step, "epoch": epoch, "loss": round(loss_sum, 5), "grad_norm": round(gnorm, 4),
                       "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
                log_fh.write(json.dumps(rec) + "\n")
                log_fh.flush()
                if step % 10 == 0 or step == 1:
                    eta = (time.time() - t0) / step * (total_steps - step)
                    print(f"  step {step}/{total_steps} loss={loss_sum:.4f} gnorm={gnorm:.2f} "
                          f"eta={eta/60:.1f}min", flush=True)
            if on_epoch_end:
                on_epoch_end(epoch, step)
            if step >= total_steps:
                break
    finally:
        if hook:
            hook.handle.remove()
    return {"steps": step, "total_steps_planned": steps_per_epoch * cfg["epochs"],
            "train_time_s": round(time.time() - t0, 1)}
