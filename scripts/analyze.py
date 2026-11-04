"""Statistical analysis for protocol v1. Reads results/runs/*, writes results/analysis/.

Primary comparison (protocol §1), per fold f:
    effect_f = acc(Mixed-3@0.5) - mean_k acc(single_k)   on the fold's 3 UNSEEN families
averaged over seeds. Uncertainty: paired bootstrap over the 1,000 test QUESTIONS. All 7
test files are perturbations of the same 1,000 questions in the same order, so one
resample of question indices is applied jointly to every family, seed and condition.
Meaningful iff the 95% CI excludes 0 AND |effect| > cross-seed SD of the per-seed effect.
The pooled effect averages the two folds (each fold resampled with the same indices).

Secondary comparisons use the same machinery (Mixed-3 vs clean / NEFTune / augmentation /
best single family; seen-family and clean-test effects; the volume sweep).

    python scripts/analyze.py [--runs-dir results/runs] [--out results/analysis] [--boot 10000]
"""

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment.conditions import ALL_FAMILIES, CONDITIONS, FOLDS  # noqa: E402

EVAL_SETS = ["clean"] + ALL_FAMILIES


def load_labels(cfg: dict) -> dict[str, np.ndarray]:
    out = {}
    for name, path in cfg["eval_sets"].items():
        with open(path, encoding="utf-8") as f:
            out[name] = np.array([json.loads(l)["label"] for l in f], dtype=bool)
    return out


def load_runs(runs_dir: Path, labels: dict) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """-> {condition: {seed: {eval_set: correctness bool array}}}; debug-tagged runs excluded."""
    runs: dict = {}
    for d in sorted(runs_dir.iterdir()):
        if not (d / "metrics.json").exists():
            continue
        rc = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
        if rc.get("tag") or rc.get("eval_limit") or rc.get("max_steps_override"):
            continue
        preds = json.loads((d / "predictions.json").read_text(encoding="utf-8"))
        seed = -1 if rc["seed"] is None else rc["seed"]
        runs.setdefault(rc["condition"], {})[seed] = {
            k: (np.array(preds[k]) > 0.5) == labels[k] for k in EVAL_SETS}
    return runs


def per_item(runs, cond, families, seeds) -> np.ndarray:
    """(n_seeds, n_items) correctness averaged over the given families."""
    return np.stack([np.mean([runs[cond][s][f] for f in families], axis=0) for s in seeds]).astype(float)


def shared_seeds(runs, conds) -> list[int]:
    sets = [set(runs.get(c, {})) for c in conds]
    return sorted(set.intersection(*sets)) if sets and all(sets) else []


def contrast(runs, a: str, b: list[str], families: list[str], idx: np.ndarray) -> dict | None:
    """acc(a) - mean(acc(b_k)) on `families`, with paired bootstrap over question indices.

    Shared conditions (clean, neftune) have the same runs in both folds; seeds are
    matched by value where available."""
    seeds = shared_seeds(runs, [a] + b)
    if not seeds:
        return None
    A = per_item(runs, a, families, seeds)                              # (S, N)
    B = np.mean([per_item(runs, c, families, seeds) for c in b], axis=0)  # (S, N)
    D = A - B
    per_seed = D.mean(axis=1)
    item_diff = D.mean(axis=0)                                          # (N,)
    boot = item_diff[idx].mean(axis=1)
    return {"effect": float(item_diff.mean()), "ci95": [float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5))], "per_seed": per_seed.tolist(),
            "seed_sd": float(per_seed.std(ddof=1)) if len(seeds) > 1 else None,
            "acc_a": float(A.mean()), "acc_b": float(B.mean()), "seeds": seeds,
            "_item_diff": item_diff}


def verdict(c: dict) -> str:
    lo, hi = c["ci95"]
    excludes = lo > 0 or hi < 0
    beats_sd = c["seed_sd"] is not None and abs(c["effect"]) > c["seed_sd"]
    if excludes and beats_sd:
        return "meaningful (CI excludes 0 and |effect| > seed SD)"
    if excludes:
        return "CI excludes 0 but |effect| <= seed SD -> not meaningful by protocol"
    return "null (CI includes 0)"


def pooled(cs: list[dict], idx: np.ndarray) -> dict:
    d = np.mean([c["_item_diff"] for c in cs], axis=0)
    boot = d[idx].mean(axis=1)
    ps = np.mean([c["per_seed"] for c in cs], axis=0) if len({len(c["per_seed"]) for c in cs}) == 1 else None
    return {"effect": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "per_seed": None if ps is None else ps.tolist(),
            "seed_sd": float(ps.std(ddof=1)) if ps is not None and len(ps) > 1 else None}


def fmt(c: dict | None) -> str:
    if c is None:
        return "n/a (runs missing)"
    sd = "n/a" if c.get("seed_sd") is None else f"{100 * c['seed_sd']:.2f}"
    return (f"{100 * c['effect']:+.2f} pp  [95% CI {100 * c['ci95'][0]:+.2f}, {100 * c['ci95'][1]:+.2f}]"
            f"  seed-SD {sd}")


def acc_table(runs) -> list[dict]:
    rows = []
    for cond in ["untuned"] + list(CONDITIONS):
        if cond not in runs:
            continue
        seeds = sorted(runs[cond])
        row = {"condition": cond, "n_seeds": len([s for s in seeds if s >= 0]) or 1}
        for e in EVAL_SETS:
            v = [runs[cond][s][e].mean() for s in seeds]
            row[e] = (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
        fold = CONDITIONS[cond].fold if cond in CONDITIONS else None
        for part in ("seen", "unseen"):
            if fold:
                v = [np.mean([runs[cond][s][f].mean() for f in FOLDS[fold][part]]) for s in seeds]
                row[part] = (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
        rows.append(row)
    return rows


def strip(o):
    if isinstance(o, dict):
        return {str(k): strip(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="results/runs")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--config", default="configs/train_config.json")
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    labels = load_labels(cfg)
    runs = load_runs(Path(args.runs_dir), labels)
    n_items = len(labels["clean"])
    idx = np.random.default_rng(12345).integers(0, n_items, size=(args.boot, n_items))

    res = {"n_runs": sum(len(v) for v in runs.values()), "primary": {}, "secondary": {}, "sweep": {}}
    lines = ["# Analysis output (auto-generated by scripts/analyze.py)", "",
             f"Complete runs found: {res['n_runs']}  |  bootstrap resamples: {args.boot}", ""]

    # ---- primary
    lines += ["## Primary: Mixed-3@0.5 minus mean single-family, UNSEEN families", ""]
    prim = []
    for fold, spec in FOLDS.items():
        singles = [f"{fold}/single_{f}" for f in spec["seen"]]
        c = contrast(runs, f"{fold}/mixed3_f0.50", singles, spec["unseen"], idx)
        if c:
            c["verdict"] = verdict(c)
            prim.append(c)
        res["primary"][fold] = c
        lines.append(f"- **{fold}** (unseen: {', '.join(spec['unseen'])}): {fmt(c)}"
                     + (f" -> {c['verdict']}" if c else ""))
    if len(prim) == 2:
        p = pooled(prim, idx)
        p["verdict"] = verdict(p)
        res["primary"]["pooled"] = p
        lines.append(f"- **pooled (both folds)**: {fmt(p)} -> {p['verdict']}")
    lines.append("")

    # ---- secondary
    lines += ["## Secondary contrasts (Mixed-3@0.5 minus X)", "",
              "| fold | comparison | test families | effect |", "|---|---|---|---|"]
    for fold, spec in FOLDS.items():
        m = f"{fold}/mixed3_f0.50"
        singles = [f"{fold}/single_{f}" for f in spec["seen"]]
        unseen_singles = {}
        for s in singles:
            cs = contrast(runs, s, ["shared/clean"], spec["unseen"], idx)
            unseen_singles[s] = cs["acc_a"] if cs else None
        best = max((s for s in singles if unseen_singles[s] is not None), key=lambda s: unseen_singles[s],
                   default=None)
        comps = [("clean baseline", ["shared/clean"], spec["unseen"]),
                 ("NEFTune", ["shared/neftune"], spec["unseen"]),
                 ("EDA augmentation", [f"{fold}/augment"], spec["unseen"]),
                 (f"best single ({best.split('_', 1)[1] if best else '?'})", [best] if best else [], spec["unseen"]),
                 ("mean single (seen families)", singles, spec["seen"]),
                 ("mean single (clean test)", singles, ["clean"]),
                 ("clean baseline (clean test)", ["shared/clean"], ["clean"])]
        for name, b, fams in comps:
            c = contrast(runs, m, b, fams, idx) if b else None
            if c:
                c["verdict"] = verdict(c)
            res["secondary"][f"{fold}: mixed3 - {name}"] = c
            tf = "unseen" if fams == spec["unseen"] else ("seen" if fams == spec["seen"] else "clean")
            lines.append(f"| {fold} | {name} | {tf} | {fmt(c)} |")
    lines.append("")

    # ---- volume sweep (unseen families); fraction 0 = clean baseline
    lines += ["## Volume sweep: Mixed-3 accuracy on unseen families vs corruption fraction", "",
              "| fold | 0.00 (clean) | 0.10 | 0.25 | 0.50 | 1.00 |", "|---|---|---|---|---|---|"]
    for fold, spec in FOLDS.items():
        row, cells = {}, []
        for frac, cond in [(0.0, "shared/clean")] + [(f, f"{fold}/mixed3_f{f:.2f}") for f in (0.1, 0.25, 0.5, 1.0)]:
            if cond in runs:
                v = [np.mean([runs[cond][s][f].mean() for f in spec["unseen"]]) for s in sorted(runs[cond])]
                row[frac] = (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
                cells.append(f"{100 * row[frac][0]:.2f} ± {100 * row[frac][1]:.2f}")
            else:
                cells.append("n/a")
        res["sweep"][fold] = row
        lines.append(f"| {fold} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- accuracy table
    table = acc_table(runs)
    res["accuracy_table"] = table
    hdr = ["condition", "n"] + EVAL_SETS + ["seen", "unseen"]
    lines += ["## Accuracy (%) per condition, mean ± SD over seeds", "",
              "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in table:
        cells = [r["condition"], str(r["n_seeds"])]
        for e in EVAL_SETS + ["seen", "unseen"]:
            cells.append(f"{100 * r[e][0]:.1f} ± {100 * r[e][1]:.1f}" if e in r else "–")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # ---- transfer matrix: single-family train -> each test family, delta vs clean baseline
    if "shared/clean" in runs:
        base = {e: np.mean([runs["shared/clean"][s][e].mean() for s in runs["shared/clean"]]) for e in EVAL_SETS}
        lines += ["## Transfer matrix: single-family model accuracy minus clean baseline (pp)", "",
                  "| trained on \\ tested on | " + " | ".join(ALL_FAMILIES) + " |", "|---|" + "---|" * 6]
        res["transfer"] = {}
        for fold, spec in FOLDS.items():
            for fam in spec["seen"]:
                cond = f"{fold}/single_{fam}"
                if cond not in runs:
                    continue
                d = {e: float(np.mean([runs[cond][s][e].mean() for s in runs[cond]]) - base[e]) for e in ALL_FAMILIES}
                res["transfer"][fam] = d
                lines.append(f"| {fam} | " + " | ".join(
                    (f"**{100 * d[e]:+.1f}**" if e == fam else f"{100 * d[e]:+.1f}") for e in ALL_FAMILIES) + " |")
        lines.append("")

    (out / "summary.json").write_text(json.dumps(strip(res), indent=2), encoding="utf-8")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    plot(res, out)


def plot(res: dict, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skipping figures)")
        return
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for fold, row in res["sweep"].items():
        if row:
            xs = sorted(row)
            ax.errorbar(xs, [100 * row[x][0] for x in xs], yerr=[100 * row[x][1] for x in xs],
                        marker="o", capsize=3, label=fold)
    ax.set_xlabel("corruption fraction (Mixed-3)")
    ax.set_ylabel("unseen-family accuracy (%)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "volume_sweep.png", dpi=160)
    print(f"wrote {out / 'volume_sweep.png'}")


if __name__ == "__main__":
    main()
