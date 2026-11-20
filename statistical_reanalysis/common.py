"""Shared loaders and bootstrap machinery for the post-hoc reanalysis.

Everything here reads the ORIGINAL run records in results/runs/ and the frozen data in data/.
Nothing in results/ or data/ is ever written. Outputs go to statistical_reanalysis/outputs/.

Conventions (identical to scripts/analyze.py so that the original numbers are reproduced):
* correctness of a run on an eval set = (p_yes > 0.5) == gold label
* seeds are matched by value across conditions
* the per-item difference is averaged over families and seeds before resampling
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment.conditions import ALL_FAMILIES, CONDITIONS, FOLDS  # noqa: E402

OUT = ROOT / "statistical_reanalysis" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
EVAL_SETS = ["clean"] + ALL_FAMILIES
N_BOOT = 10_000
BOOT_SEED = 12345  # same generator seed as scripts/analyze.py


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_cfg() -> dict:
    return json.loads((ROOT / "configs" / "train_config.json").read_text(encoding="utf-8"))


def load_labels() -> np.ndarray:
    """Gold labels of the 1,000 test items (identical across the 7 eval files; checked in 01)."""
    return np.array([r["label"] for r in load_jsonl(ROOT / "data" / "splits" / "test.jsonl")], dtype=bool)


def load_probs() -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """{condition: {seed: {eval_set: p_yes array}}}. Debug-tagged runs are excluded (as in analyze.py).
    The untuned model is stored under seed -1."""
    runs: dict = {}
    for d in sorted((ROOT / "results" / "runs").iterdir()):
        if not (d / "metrics.json").exists():
            continue
        rc = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
        if rc.get("tag") or rc.get("eval_limit") or rc.get("max_steps_override"):
            continue
        preds = json.loads((d / "predictions.json").read_text(encoding="utf-8"))
        seed = -1 if rc["seed"] is None else rc["seed"]
        runs.setdefault(rc["condition"], {})[seed] = {k: np.asarray(preds[k], dtype=float) for k in EVAL_SETS}
    return runs


def correctness(probs, labels, ties: str = "no") -> dict:
    """Converts p_yes to per-item credit. ties: 'no' (original rule p>0.5), 'yes' (p>=0.5),
    'half' (exact 0.5 earns 0.5 credit, i.e. expected accuracy under a fair coin tie-break)."""
    out: dict = {}
    for cond, seeds in probs.items():
        out[cond] = {}
        for s, sets in seeds.items():
            out[cond][s] = {}
            for e, p in sets.items():
                if ties == "no":
                    c = ((p > 0.5) == labels).astype(float)
                elif ties == "yes":
                    c = ((p >= 0.5) == labels).astype(float)
                elif ties == "half":
                    c = ((p > 0.5) == labels).astype(float)
                    c[p == 0.5] = 0.5
                elif ties == "prob":  # probability assigned to the gold answer (threshold-free)
                    c = np.where(labels, p, 1 - p)
                else:
                    raise ValueError(ties)
                out[cond][s][e] = c
    return out


# --------------------------------------------------------------------------- clusters / overlap

_tok = re.compile(r"\w+")


def shingles(text: str, k: int = 5) -> set:
    t = _tok.findall(text.lower())
    return {tuple(t[i:i + k]) for i in range(max(1, len(t) - k + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def passage_flags(near_dup_threshold: float = 0.5) -> list[dict]:
    """Per test item: exact train-passage overlap, max 5-gram Jaccard with any train passage,
    and a cluster id that groups test items sharing an exact or near-duplicate passage."""
    train = load_jsonl(ROOT / "data" / "splits" / "train.jsonl")
    test = load_jsonl(ROOT / "data" / "splits" / "test.jsonl")
    tr_p = sorted({r["passage"] for r in train})
    tr_set = set(tr_p)
    tr_sh = [shingles(p) for p in tr_p]
    inv = defaultdict(set)
    for i, sh in enumerate(tr_sh):
        for g in sh:
            inv[g].add(i)

    te_p = sorted({r["passage"] for r in test})
    te_sh = {p: shingles(p) for p in te_p}
    best = {}
    for p in te_p:
        cnt = Counter()
        for g in te_sh[p]:
            for i in inv.get(g, ()):
                cnt[i] += 1
        # exact maximum over every train passage sharing at least one shingle (deterministic;
        # most_common(k) would break count ties in hash-seed-dependent order)
        j = max((jaccard(te_sh[p], tr_sh[i]) for i in cnt), default=0.0)
        best[p] = 1.0 if p in tr_set else j

    # union-find over distinct test passages: exact identity is implicit; merge near-duplicates
    parent = {p: p for p in te_p}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    inv_t = defaultdict(set)
    for i, p in enumerate(te_p):
        for g in te_sh[p]:
            inv_t[g].add(i)
    for i, p in enumerate(te_p):
        cand = Counter()
        for g in te_sh[p]:
            for k in inv_t[g]:
                if k != i:
                    cand[k] += 1
        for k in cand:
            if jaccard(te_sh[p], te_sh[te_p[k]]) >= near_dup_threshold:
                parent[find(p)] = find(te_p[k])
    roots = {}
    rows = []
    for idx, r in enumerate(test):
        root = find(r["passage"])
        cid = roots.setdefault(root, len(roots))
        rows.append({"item": idx, "label": r["label"], "cluster": cid,
                     "train_passage_exact": r["passage"] in tr_set,
                     "train_passage_max_jaccard5": round(best[r["passage"]], 4),
                     "train_passage_near": (r["passage"] in tr_set) or best[r["passage"]] >= near_dup_threshold})
    return rows


# --------------------------------------------------------------------------- bootstrap engine

class Resampler:
    """Holds a fixed set of bootstrap draws so that every contrast in one analysis uses the same
    resamples (paired across families, seeds, conditions and folds)."""

    def __init__(self, n_items: int, clusters: np.ndarray | None, mask: np.ndarray | None = None,
                 n_boot: int = N_BOOT, seed: int = BOOT_SEED):
        self.mask = np.ones(n_items, bool) if mask is None else mask.astype(bool)
        self.items = np.flatnonzero(self.mask)
        rng = np.random.default_rng(seed)
        if clusters is None:
            self.kind = "item"
            self.idx = rng.integers(0, len(self.items), size=(n_boot, len(self.items)))
        else:
            self.kind = "cluster"
            cl = clusters[self.items]
            uniq, inv = np.unique(cl, return_inverse=True)
            self.n_clusters = len(uniq)
            self.inv = inv
            self.sizes = np.bincount(inv, minlength=len(uniq)).astype(float)
            self.idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))

    def boot(self, item_diff: np.ndarray) -> np.ndarray:
        d = item_diff[self.items]
        if self.kind == "item":
            return d[self.idx].mean(axis=1)
        sums = np.bincount(self.inv, weights=d, minlength=len(self.sizes))
        return sums[self.idx].sum(axis=1) / self.sizes[self.idx].sum(axis=1)

    def point(self, item_diff: np.ndarray) -> float:
        return float(item_diff[self.items].mean())


def per_item(corr, cond, families, seeds) -> np.ndarray:
    return np.stack([np.mean([corr[cond][s][f] for f in families], axis=0) for s in seeds])


def shared_seeds(corr, conds) -> list[int]:
    sets = [set(corr.get(c, {})) for c in conds]
    return sorted(set.intersection(*sets)) if sets and all(sets) else []


def contrast_diff(corr, a: str, b: list[str], families: list[str]):
    """Returns (item_diff averaged over seeds, per-seed diff matrix (S,N), seeds)."""
    seeds = shared_seeds(corr, [a] + b)
    A = per_item(corr, a, families, seeds)
    B = np.mean([per_item(corr, c, families, seeds) for c in b], axis=0)
    D = A - B
    return D.mean(axis=0), D, seeds


T975 = {2: 4.302652729911275, 3: 3.182446305284263, 4: 2.776445105197799}


def summarize(item_diff, D, rs: Resampler) -> dict:
    boot = rs.boot(item_diff)
    eff = rs.point(item_diff)
    per_seed = D[:, rs.items].mean(axis=1)
    sd = float(per_seed.std(ddof=1)) if len(per_seed) > 1 else float("nan")
    se_item = float(boot.std(ddof=1))
    S = len(per_seed)
    se_seed = sd / np.sqrt(S)
    se_comb = float(np.sqrt(se_item ** 2 + se_seed ** 2))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "effect_pp": 100 * eff,
        "ci95_pp": [100 * float(lo), 100 * float(hi)],
        "se_item_pp": 100 * se_item,
        "per_seed_pp": [100 * float(x) for x in per_seed],
        "seed_sd_pp": 100 * sd,
        "seed_t_ci95_pp": [100 * (eff - T975[S - 1] * se_seed), 100 * (eff + T975[S - 1] * se_seed)] if S > 1 else None,
        "combined_se_pp": 100 * se_comb,
        "combined_ci95_pp": [100 * (eff - 1.96 * se_comb), 100 * (eff + 1.96 * se_comb)],
        "boot_p_two_sided": float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))),
        "protocol_verdict": verdict(100 * eff, 100 * lo, 100 * hi, 100 * sd),
        "n_items": int(len(rs.items)),
        **({"n_clusters": int(rs.n_clusters)} if rs.kind == "cluster" else {}),
    }


def verdict(eff, lo, hi, sd) -> str:
    excl = lo > 0 or hi < 0
    if excl and abs(eff) > sd:
        return "meaningful"
    if excl:
        return "CI excludes 0 but |effect| <= seed SD (not meaningful)"
    return "null"


def hierarchical_boot(corr, a, b, families, rs: Resampler, n_boot: int = N_BOOT, seed: int = 777) -> np.ndarray:
    """Two-stage bootstrap: independently for every condition, resample its seeds with replacement;
    jointly resample test clusters/items. Seed pairing across conditions is deliberately NOT kept,
    which is conservative. With only 3 seeds this still understates training variance
    (a bootstrap of n=3 has only 10 distinct multisets), so treat it as a rough cross-check."""
    rng = np.random.default_rng(seed)
    seeds = shared_seeds(corr, [a] + b)
    mats = {c: per_item(corr, c, families, seeds) for c in [a] + b}  # (S, N)
    S = len(seeds)
    out = np.empty(n_boot)
    base = rs.boot  # uses rs.idx row-by-row below
    # pre-draw seed choices
    choice = {c: rng.integers(0, S, size=(n_boot, S)) for c in mats}
    for i in range(n_boot):
        A = mats[a][choice[a][i]].mean(axis=0)
        B = np.mean([mats[c][choice[c][i]].mean(axis=0) for c in b], axis=0)
        d = (A - B)[rs.items]
        if rs.kind == "item":
            out[i] = d[rs.idx[i]].mean()
        else:
            sums = np.bincount(rs.inv, weights=d, minlength=len(rs.sizes))
            out[i] = sums[rs.idx[i]].sum() / rs.sizes[rs.idx[i]].sum()
    del base
    return out


def primary_specs():
    """[(name, a, b_list, families)] for the per-fold primary contrast."""
    out = []
    for fold, spec in FOLDS.items():
        out.append((fold, f"{fold}/mixed3_f0.50", [f"{fold}/single_{f}" for f in spec["seen"]], spec["unseen"]))
    return out


def secondary_specs(corr):
    out = []
    for fold, spec in FOLDS.items():
        m = f"{fold}/mixed3_f0.50"
        singles = [f"{fold}/single_{f}" for f in spec["seen"]]
        unseen_acc = {s: per_item(corr, s, spec["unseen"], sorted(corr[s])).mean() for s in singles}
        best = max(singles, key=lambda s: unseen_acc[s])
        out += [
            (fold, "clean baseline", "unseen", m, ["shared/clean"], spec["unseen"]),
            (fold, "NEFTune", "unseen", m, ["shared/neftune"], spec["unseen"]),
            (fold, "EDA augmentation", "unseen", m, [f"{fold}/augment"], spec["unseen"]),
            (fold, f"best single ({best.split('/single_')[1]})", "unseen", m, [best], spec["unseen"]),
            (fold, "mean single", "seen", m, singles, spec["seen"]),
            (fold, "mean single", "clean", m, singles, ["clean"]),
            (fold, "clean baseline", "clean", m, ["shared/clean"], ["clean"]),
        ]
    return out


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def write(name: str, text: str = None, obj=None):
    if text is not None:
        (OUT / f"{name}.md").write_text(text, encoding="utf-8")
    if obj is not None:
        (OUT / f"{name}.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")
