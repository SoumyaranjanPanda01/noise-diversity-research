"""Step 4: the distractor answer-preference shift (exploratory, not pre-registered).

For every run (54 trained + untuned) on the clean and distractor test sets (same 1,000 questions):
* confusion matrix, predicted-Yes rate, accuracy on gold-Yes / gold-No items
* ROC AUC of p(Yes) (threshold-free discrimination)
* mean log-odds shift  logit p_distractor - logit p_clean  (per item, averaged)
* "label-free re-thresholding": the threshold on the distractor set that reproduces the SAME model's
  predicted-Yes rate on the clean versions of the same questions. It uses no labels, but it does use the
  test inputs, so it is a diagnostic of how much of the drop a single global shift can explain, not a method.
* flips clean -> distractor, split by direction (Yes->No vs No->Yes)
Plus: per-distractor-sentence shifts, a dose-response over the number of distractor training examples,
and the across-run association between the Yes-rate shift and the accuracy drop.

    python statistical_reanalysis/04_distractor_bias.py
"""

import csv
import json

import numpy as np

from common import OUT, ROOT, Resampler, load_jsonl, load_labels, load_probs, write

ORDER = ["untuned", "shared/clean", "shared/neftune",
         "fold_a/single_typo", "fold_a/single_formatting", "fold_a/single_deletion", "fold_a/mixed3_f0.10",
         "fold_a/mixed3_f0.25", "fold_a/mixed3_f0.50", "fold_a/mixed3_f1.00", "fold_a/augment",
         "fold_b/single_shuffle", "fold_b/single_synonym", "fold_b/augment",
         "fold_b/mixed3_f0.10", "fold_b/mixed3_f0.25", "fold_b/mixed3_f0.50", "fold_b/mixed3_f1.00",
         "fold_b/single_distractor"]
N_DISTRACTOR_TRAIN = {"shared/clean": 0, "fold_b/augment": 0, "fold_b/single_shuffle": 0, "fold_b/single_synonym": 0,
                      "fold_b/mixed3_f0.10": 133, "fold_b/mixed3_f0.25": 333, "fold_b/mixed3_f0.50": 667,
                      "fold_b/mixed3_f1.00": 1333, "fold_b/single_distractor": 2000}


def auc(p, y):
    from scipy.stats import rankdata
    r = rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def run_stats(pc, pd, y):
    yc, yd = pc > 0.5, pd > 0.5
    rate_c = yc.mean()
    thr = np.quantile(pd, 1 - rate_c)  # label-free: match the clean predicted-Yes rate
    yr = pd > thr if (pd > thr).mean() > 0 else pd >= thr
    cm = lambda yp: {"TP": int((yp & y).sum()), "FN": int((~yp & y).sum()), "FP": int((yp & ~y).sum()),
                     "TN": int((~yp & ~y).sum())}
    corr_c, corr_d = yc == y, yd == y
    lost = corr_c & ~corr_d
    gained = ~corr_c & corr_d
    return {
        "yes_rate_clean": float(rate_c), "yes_rate_distractor": float(yd.mean()),
        "acc_clean": float(corr_c.mean()), "acc_distractor": float(corr_d.mean()),
        "acc_gold_yes_clean": float(yc[y].mean()), "acc_gold_no_clean": float((~yc)[~y].mean()),
        "acc_gold_yes_distractor": float(yd[y].mean()), "acc_gold_no_distractor": float((~yd)[~y].mean()),
        "auc_clean": auc(pc, y), "auc_distractor": auc(pd, y),
        "mean_logodds_shift": float((logit(pd) - logit(pc)).mean()),
        "acc_distractor_rethresholded": float((yr == y).mean()), "rethreshold": float(thr),
        "flips_lost_yes_to_no": int((lost & yc & ~yd).sum()), "flips_lost_no_to_yes": int((lost & ~yc & yd).sum()),
        "flips_gained": int(gained.sum()),
        "confusion_clean": cm(yc), "confusion_distractor": cm(yd),
    }


def agg(rows, k):
    v = [r[k] for r in rows]
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


def main():
    y = load_labels()
    probs = load_probs()
    flags = list(csv.DictReader(open(OUT / "test_item_flags.csv", encoding="utf-8")))
    rs = Resampler(len(y), np.array([int(r["cluster"]) for r in flags]))

    per_run, res = [], {"gold_yes_rate": float(y.mean()), "conditions": {}}
    for cond in ORDER:
        rows = []
        for s in sorted(probs[cond]):
            st = run_stats(probs[cond][s]["clean"], probs[cond][s]["distractor"], y)
            st.update({"condition": cond, "seed": s})
            rows.append(st)
            per_run.append(st)
        # cluster-bootstrap CI of the seed-averaged Yes-rate shift and accuracy drop
        dyes = np.mean([(probs[cond][s]["distractor"] > 0.5).astype(float) - (probs[cond][s]["clean"] > 0.5)
                        for s in sorted(probs[cond])], axis=0)
        dacc = np.mean([((probs[cond][s]["distractor"] > 0.5) == y).astype(float) - ((probs[cond][s]["clean"] > 0.5) == y)
                        for s in sorted(probs[cond])], axis=0)
        by = rs.boot(dyes)
        ba = rs.boot(dacc)
        res["conditions"][cond] = {
            "n_runs": len(rows),
            **{k: agg(rows, k) for k in rows[0] if isinstance(rows[0][k], float) or isinstance(rows[0][k], int)
               if k not in ("seed",)},
            "yes_rate_shift_pp": [100 * float(dyes.mean()), 100 * float(np.percentile(by, 2.5)), 100 * float(np.percentile(by, 97.5))],
            "acc_change_pp": [100 * float(dacc.mean()), 100 * float(np.percentile(ba, 2.5)), 100 * float(np.percentile(ba, 97.5))],
            "per_seed_yes_rate_shift_pp": [100 * (r["yes_rate_distractor"] - r["yes_rate_clean"]) for r in rows],
            "per_seed_acc_change_pp": [100 * (r["acc_distractor"] - r["acc_clean"]) for r in rows],
        }

    # per distractor sentence (which of the 8 sentences was appended to each test item)
    import sys
    sys.path.insert(0, str(ROOT))
    sentences = [
        "Photosynthesis is the process used by plants to convert light into energy.",
        "Mercury is the smallest planet in our solar system.",
        "Paris hosted the Summer Olympic Games in 2024.",
        "The Pacific Ocean is the largest and deepest ocean basin on Earth.",
        "The printing press was developed by Johannes Gutenberg in the fifteenth century.",
        "Mount Everest is the highest mountain peak above sea level.",
        "Jupiter has the shortest day of all the planets in the solar system.",
        "Honey never completely spoils if kept in an airtight container.",
    ]
    qd = [r["question"] for r in load_jsonl(ROOT / "data/perturbed/test_distractor.jsonl")]
    sid = np.array([next(i for i, s in enumerate(sentences) if q.endswith(s)) for q in qd])
    per_sent = {}
    for cond in ["untuned", "shared/clean", "fold_a/mixed3_f0.50", "fold_b/mixed3_f0.50"]:
        per_sent[cond] = {}
        for i, s in enumerate(sentences):
            m = sid == i
            sh = np.mean([(probs[cond][sd]["distractor"][m] > 0.5).mean() - (probs[cond][sd]["clean"][m] > 0.5).mean()
                          for sd in sorted(probs[cond])])
            ac = np.mean([((probs[cond][sd]["distractor"][m] > 0.5) == y[m]).mean() -
                          ((probs[cond][sd]["clean"][m] > 0.5) == y[m]).mean() for sd in sorted(probs[cond])])
            per_sent[cond][s] = {"n_items": int(m.sum()), "yes_rate_shift_pp": 100 * float(sh), "acc_change_pp": 100 * float(ac)}
    res["per_sentence"] = per_sent

    # Is the loss of discrimination (AUC) explained by sentence-specific shifts? Compare AUC computed
    # within each sentence group (then averaged, weighted by group size) on clean vs distractor inputs,
    # and a label-free re-threshold applied separately within each sentence group.
    within = {}
    for cond in ORDER:
        rows = []
        for sd in sorted(probs[cond]):
            pc_, pd_ = probs[cond][sd]["clean"], probs[cond][sd]["distractor"]
            w_auc_c = w_auc_d = 0.0
            yr = np.zeros(len(y), bool)
            for i in range(len(sentences)):
                m = sid == i
                w_auc_c += m.sum() * auc(pc_[m], y[m])
                w_auc_d += m.sum() * auc(pd_[m], y[m])
                rate = (pc_[m] > 0.5).mean()
                thr = np.quantile(pd_[m], 1 - rate)
                yr[m] = pd_[m] > thr
            rows.append({"auc_within_clean": w_auc_c / len(y), "auc_within_distractor": w_auc_d / len(y),
                         "acc_rethreshold_per_sentence": float((yr == y).mean())})
        within[cond] = {k: agg(rows, k) for k in rows[0]}
    res["within_sentence"] = within

    # dose-response over the number of distractor training examples (fold B family + clean)
    res["dose_response"] = {c: {"n_distractor_train": n, "acc_distractor": res["conditions"][c]["acc_distractor"],
                                "yes_rate_distractor": res["conditions"][c]["yes_rate_distractor"]}
                            for c, n in N_DISTRACTOR_TRAIN.items()}

    # across-run association (trained runs that never saw distractor data)
    unexposed = [r for r in per_run if r["condition"] != "untuned" and N_DISTRACTOR_TRAIN.get(r["condition"], 0) == 0]
    xs = np.array([r["yes_rate_distractor"] - r["yes_rate_clean"] for r in unexposed])
    ys = np.array([r["acc_distractor"] - r["acc_clean"] for r in unexposed])
    from scipy.stats import pearsonr, spearmanr
    res["association_unexposed_runs"] = {"n_runs": len(unexposed), "pearson_r": float(pearsonr(xs, ys)[0]),
                                         "spearman_rho": float(spearmanr(xs, ys)[0])}
    res["per_run"] = per_run
    write("04_distractor_bias", None, res)
    with open(OUT / "04_distractor_per_run.csv", "w", newline="", encoding="utf-8") as f:
        flat = [{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()} for r in per_run]
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)

    pc = lambda c, k, d=1: (f"{100 * res['conditions'][c][k][0]:.{d}f} ± {100 * res['conditions'][c][k][1]:.{d}f}")
    L = ["# 04 Distractor answer-preference shift (exploratory; auto-generated)", "",
         f"Gold Yes rate: {100 * y.mean():.1f}%. Values: mean ± SD over seeds (untuned: 1 run). "
         "CIs: passage-cluster bootstrap over test items, seed-averaged.", "",
         "## Yes rate, accuracy and discrimination, clean vs distractor", "",
         "| condition | Yes% clean | Yes% distractor | Yes-rate shift [CI] | acc clean | acc distractor | acc change [CI] | AUC clean | AUC distractor | acc after label-free re-threshold |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for c in ORDER:
        x = res["conditions"][c]
        L.append(f"| {c} | {pc(c, 'yes_rate_clean')} | {pc(c, 'yes_rate_distractor')} | "
                 f"{x['yes_rate_shift_pp'][0]:+.1f} [{x['yes_rate_shift_pp'][1]:+.1f}, {x['yes_rate_shift_pp'][2]:+.1f}] | "
                 f"{pc(c, 'acc_clean')} | {pc(c, 'acc_distractor')} | "
                 f"{x['acc_change_pp'][0]:+.1f} [{x['acc_change_pp'][1]:+.1f}, {x['acc_change_pp'][2]:+.1f}] | "
                 f"{x['auc_clean'][0]:.3f} ± {x['auc_clean'][1]:.3f} | {x['auc_distractor'][0]:.3f} ± {x['auc_distractor'][1]:.3f} | "
                 f"{pc(c, 'acc_distractor_rethresholded')} |")
    L += ["", "## Per-seed shifts (pp): Yes-rate shift / accuracy change", ""]
    for c in ORDER:
        x = res["conditions"][c]
        L.append(f"* {c}: " + "; ".join(f"{a:+.1f} / {b:+.1f}" for a, b in
                                        zip(x["per_seed_yes_rate_shift_pp"], x["per_seed_acc_change_pp"])))
    L += ["", "## Where the lost items go (mean over seeds)", "",
          "| condition | correct->wrong via Yes->No | correct->wrong via No->Yes | wrong->correct | mean log-odds shift |",
          "|---|---|---|---|---|"]
    for c in ORDER:
        x = res["conditions"][c]
        L.append(f"| {c} | {x['flips_lost_yes_to_no'][0]:.1f} | {x['flips_lost_no_to_yes'][0]:.1f} | "
                 f"{x['flips_gained'][0]:.1f} | {x['mean_logodds_shift'][0]:+.2f} ± {x['mean_logodds_shift'][1]:.2f} |")
    L += ["", "## By appended sentence (Yes-rate shift / accuracy change, pp, mean over seeds)", "",
          "| sentence | n | untuned | clean | A Mixed-3 | B Mixed-3 |", "|---|---|---|---|---|---|"]
    for s in sentences:
        cells = [f"{per_sent[c][s]['yes_rate_shift_pp']:+.1f} / {per_sent[c][s]['acc_change_pp']:+.1f}"
                 for c in ["untuned", "shared/clean", "fold_a/mixed3_f0.50", "fold_b/mixed3_f0.50"]]
        L.append(f"| {s[:48]}... | {per_sent['shared/clean'][s]['n_items']} | " + " | ".join(cells) + " |")
    L += ["", "## Within-sentence discrimination and per-sentence label-free re-thresholding", "",
          "AUC is computed inside each of the 8 sentence groups and averaged (weighted by group size). If within-group AUC",
          "on distractor inputs stays close to clean, the pooled AUC loss comes mostly from sentence-specific shifts.", "",
          "| condition | within AUC clean | within AUC distractor | acc distractor | acc, global re-threshold | acc, per-sentence re-threshold |",
          "|---|---|---|---|---|---|"]
    for c in ORDER:
        w = within[c]
        L.append(f"| {c} | {w['auc_within_clean'][0]:.3f} | {w['auc_within_distractor'][0]:.3f} | {pc(c, 'acc_distractor')} | "
                 f"{pc(c, 'acc_distractor_rethresholded')} | {100 * w['acc_rethreshold_per_sentence'][0]:.1f} ± "
                 f"{100 * w['acc_rethreshold_per_sentence'][1]:.1f} |")
    L += ["", "## Dose-response: distractor training examples vs distractor-test accuracy", "",
          "| condition | distractor train examples | acc distractor | Yes% distractor |", "|---|---|---|---|"]
    for c, d in res["dose_response"].items():
        L.append(f"| {c} | {d['n_distractor_train']} | {100 * d['acc_distractor'][0]:.1f} ± {100 * d['acc_distractor'][1]:.1f} | "
                 f"{100 * d['yes_rate_distractor'][0]:.1f} ± {100 * d['yes_rate_distractor'][1]:.1f} |")
    a = res["association_unexposed_runs"]
    L += ["", f"Across the {a['n_runs']} trained runs with no distractor training data, the per-run Yes-rate shift and the "
          f"per-run accuracy change correlate with Pearson r = {a['pearson_r']:.2f} (Spearman {a['spearman_rho']:.2f}). "
          "This is an association across runs, not a causal estimate."]
    write("04_distractor_bias", "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
