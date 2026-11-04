"""Train + evaluate one (condition, seed) run.

Usage:
    python scripts/train.py --condition fold_a/mixed3_f0.50 --seed 0
    python scripts/train.py --condition untuned            # base model, eval only
    python scripts/train.py --condition shared/clean --seed 0 --max-steps 3 --eval-limit 32 --tag smoke

Outputs (results/runs/<run_id>/):
    run_config.json   resolved config, condition, data sha256, git commit, environment
    train_log.jsonl   one line per optimizer step (loss, grad norm, lr, elapsed)
    metrics.json      val accuracy per epoch, test accuracy per eval set, timings
    predictions.json  p(Yes) for every val/test item, in file order (for paired stats)
Adapter weights go to results/checkpoints/<run_id>/ (git-ignored).
A summary row is appended to results/run_registry.csv. A run with metrics.json is
considered complete and is skipped unless --force is given.
"""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os  # noqa: E402

# Must be set before CUDA initialises; reduces fragmentation with variable-length batches.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from src.experiment.conditions import CONDITIONS  # noqa: E402
from src.experiment.modeling import Encoder, accuracy, load_jsonl, score_yes_no, train_lora  # noqa: E402

RUNS_DIR = Path("results/runs")
CKPT_DIR = Path("results/checkpoints")
REGISTRY = Path("results/run_registry.csv")
REGISTRY_FIELDS = ["run_id", "condition", "category", "fold", "seed", "status", "started_utc", "finished_utc",
                   "train_time_s", "eval_time_s", "steps", "final_loss", "device", "git_commit", "val_acc",
                   "clean", "typo", "formatting", "deletion", "shuffle", "distractor", "synonym"]


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "scripts", "configs"],
                                        text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def run_id_for(condition: str, seed: int | None, tag: str | None) -> str:
    base = condition.replace("/", "__") + ("" if seed is None else f"__s{seed}")
    return base + (f"__{tag}" if tag else "")


def append_registry(row: dict) -> None:
    new = not REGISTRY.exists()
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, help="condition name or 'untuned'")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", default="configs/train_config.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--micro-bs", type=int, default=None, help="micro batch; grads are accumulated to effective_batch_size (default 4: 16 OOMs a 15 GB T4)")
    ap.add_argument("--eval-bs", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=None, help="debug only: cap optimizer steps")
    ap.add_argument("--eval-limit", type=int, default=None, help="debug only: first N items per eval set")
    ap.add_argument("--tag", default=None, help="suffix for debug runs (kept out of analysis)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--no-save-adapter", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    untuned = args.condition == "untuned"
    if not untuned and args.seed is None:
        ap.error("--seed is required for training conditions")
    cond = None if untuned else CONDITIONS[args.condition]
    if (args.max_steps or args.eval_limit) and not args.tag:
        ap.error("--max-steps/--eval-limit are debug options and require --tag")

    run_id = run_id_for(args.condition, None if untuned else args.seed, args.tag)
    run_dir = RUNS_DIR / run_id
    if (run_dir / "metrics.json").exists() and not args.force:
        print(f"[skip] {run_id} already complete")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.threads:
        torch.set_num_threads(args.threads)
    micro_bs = args.micro_bs or 4
    seed = 0 if untuned else args.seed
    torch.manual_seed(seed)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"=== {run_id} on {args.device} (micro_bs={micro_bs}) ===", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=torch.float32).to(args.device)
    enc = Encoder(tok, cfg["system_prompt"], cfg["max_seq_len"])

    run_config = {
        "run_id": run_id, "condition": args.condition, "seed": None if untuned else args.seed, "tag": args.tag,
        "category": "untuned" if untuned else cond.category, "fold": None if untuned else cond.fold,
        "corruption_fraction": None if untuned else cond.corruption_fraction,
        "neftune": False if untuned else cond.neftune,
        "train_data": None if untuned else cond.data_path.as_posix(),
        "train_data_sha256_lf": None if untuned else hashlib.sha256(
            cond.data_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),  # EOL-agnostic, matches MANIFEST
        "config": cfg, "micro_batch": micro_bs, "max_steps_override": args.max_steps,
        "eval_limit": args.eval_limit, "git_commit": git_commit(), "started_utc": started,
        "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                "transformers": __import__("transformers").__version__, "peft": __import__("peft").__version__,
                "device": args.device, "threads": torch.get_num_threads(), "platform": platform.platform(),
                "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None},
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    val = load_jsonl(Path(cfg["val_set"]))[: args.eval_limit]
    metrics = {"val_acc_by_epoch": []}
    train_info = {"steps": 0, "train_time_s": 0.0}

    if not untuned:
        lcfg = cfg["lora"]
        model = get_peft_model(model, LoraConfig(r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
                                                 target_modules=lcfg["target_modules"], task_type="CAUSAL_LM"))
        trainable, total = model.get_nb_trainable_parameters()
        metrics["trainable_params"], metrics["total_params"] = trainable, total
        print(f"trainable params: {trainable:,} / {total:,}", flush=True)
        records = load_jsonl(cond.data_path)

        def on_epoch_end(epoch, step):
            acc = accuracy(score_yes_no(model, enc, val, args.device, args.eval_bs), val)
            metrics["val_acc_by_epoch"].append({"epoch": epoch, "step": step, "val_acc": acc})
            print(f"  [epoch {epoch}] val_acc={acc:.4f}", flush=True)

        with open(run_dir / "train_log.jsonl", "w", encoding="utf-8") as fh:
            train_info = train_lora(model, enc, records, cfg, args.seed, args.device, micro_bs, fh,
                                    max_steps=args.max_steps, neftune=cond.neftune, on_epoch_end=on_epoch_end)
        if not args.no_save_adapter:
            model.save_pretrained(CKPT_DIR / run_id)

    t_eval = time.time()
    preds = {}
    val_p = score_yes_no(model, enc, val, args.device, args.eval_bs)
    preds["val"] = [round(p, 5) for p in val_p]
    metrics["val_acc"] = accuracy(val_p, val)
    metrics["test_acc"] = {}
    for name, path in cfg["eval_sets"].items():
        recs = load_jsonl(Path(path))[: args.eval_limit]
        p = score_yes_no(model, enc, recs, args.device, args.eval_bs)
        preds[name] = [round(x, 5) for x in p]
        metrics["test_acc"][name] = accuracy(p, recs)
        print(f"  [test] {name:11s} acc={metrics['test_acc'][name]:.4f}", flush=True)
    eval_time = round(time.time() - t_eval, 1)

    log_path = run_dir / "train_log.jsonl"
    final_loss = None
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        tail = [json.loads(l)["loss"] for l in lines[-20:]]
        final_loss = round(sum(tail) / len(tail), 5) if tail else None
    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metrics.update({**train_info, "eval_time_s": eval_time, "final_loss_avg20": final_loss,
                    "n_passages_truncated": enc.n_truncated, "finished_utc": finished})

    (run_dir / "predictions.json").write_text(json.dumps(preds), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.tag:  # debug runs never enter the record of record
        print(f"=== done (debug, not registered) {run_id} ===", flush=True)
        return
    append_registry({
        "run_id": run_id, "condition": args.condition, "category": run_config["category"],
        "fold": run_config["fold"], "seed": run_config["seed"], "status": "complete",
        "started_utc": started, "finished_utc": finished, "train_time_s": train_info["train_time_s"],
        "eval_time_s": eval_time, "steps": train_info["steps"], "final_loss": final_loss, "device": args.device,
        "git_commit": run_config["git_commit"], "val_acc": round(metrics["val_acc"], 4),
        **{k: round(v, 4) for k, v in metrics["test_acc"].items()},
    })
    print(f"=== done {run_id}: train {train_info['train_time_s']}s, eval {eval_time}s ===", flush=True)


if __name__ == "__main__":
    main()
