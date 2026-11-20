"""Step 3: primary and secondary contrasts under alternative uncertainty models and subsets.

Settings (all post hoc, none replaces the pre-registered analysis):
  S0 original      : item bootstrap, all 1,000 items, ties -> "No"   (must reproduce scripts/analyze.py)
  S1 cluster       : passage-cluster bootstrap (966 clusters), all items
  S2 disjoint-exact: cluster bootstrap, items whose passage is NOT verbatim in train (864)
  S3 disjoint-near : cluster bootstrap, items with no verbatim/near-duplicate train passage (827)
  S4 ties-half     : cluster bootstrap, exact p=0.5 ties earn 0.5 credit
  S5 ties-yes      : cluster bootstrap, ties -> "Yes"
  S6 p(gold)       : cluster bootstrap, threshold-free credit = probability given to the gold answer

For every contrast we report: effect, 95% percentile CI, per-seed effects, seed SD, a t-interval over
the 3 seeds (training-variance only), and an approximate combined interval
  effect +- 1.96 * sqrt(SE_items^2 + SD_seed^2 / n_seeds)
that adds training-seed variance to test-item variance. For the primary contrast we also run a
two-stage (seed x cluster) bootstrap as a cross-check, leave-one-seed-out, a per-family breakdown,
and Mixed-3 vs each individual single-family model.

    python statistical_reanalysis/03_contrasts_reanalysis.py
"""

import csv

import numpy as np

from common import (ALL_FAMILIES, FOLDS, OUT, Resampler, contrast_diff, correctness, hierarchical_boot,
                    holm, load_labels, load_probs, per_item, primary_specs, secondary_specs, summarize, write)


def load_flags():
    rows = list(csv.DictReader(open(OUT / "test_item_flags.csv", encoding="utf-8")))
    cl = np.array([int(r["cluster"]) for r in rows])
    exact = np.array([r["train_passage_exact"] == "True" for r in rows])
    near = np.array([r["train_passage_near"] == "True" for r in rows])
    return cl, exact, near


def primary_block(corr, rs):
    out, diffs, Ds = {}, [], []
    for fold, a, b, fams in primary_specs():
        d, D, seeds = contrast_diff(corr, a, b, fams)
        out[fold] = summarize(d, D, rs)
        diffs.append(d)
        Ds.append(D)
    out["pooled"] = summarize(np.mean(diffs, axis=0), np.mean(Ds, axis=0), rs)
    return out


def secondary_block(corr, rs):
    out = {}
    for fold, name, tf, a, b, fams in secondary_specs(corr):
        d, D, _ = contrast_diff(corr, a, b, fams)
        out[f"{fold} | {name} | {tf}"] = summarize(d, D, rs)
    return out


def f(c, key="ci95_pp"):
    lo, hi = c[key]
    return f"{c['effect_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def main():
    labels = load_labels()
    probs = load_probs()
    cl, exact, near = load_flags()
    n = len(labels)
    corr = {t: correctness(probs, labels, t) for t in ("no", "half", "yes", "prob")}

    settings = {
        "S0 original (item bootstrap)": (corr["no"], Resampler(n, None)),
        "S1 cluster bootstrap": (corr["no"], Resampler(n, cl)),
        "S2 cluster, exclude verbatim train passages": (corr["no"], Resampler(n, cl, ~exact)),
        "S3 cluster, exclude verbatim+near-duplicate": (corr["no"], Resampler(n, cl, ~near)),
        "S4 cluster, ties = half credit": (corr["half"], Resampler(n, cl)),
        "S5 cluster, ties = Yes": (corr["yes"], Resampler(n, cl)),
        "S6 cluster, p(gold) (threshold-free)": (corr["prob"], Resampler(n, cl)),
    }
    res = {"settings": {}}
    for name, (c, rs) in settings.items():
        res["settings"][name] = {"primary": primary_block(c, rs), "secondary": secondary_block(c, rs)}

    # Holm adjustment of the 14 secondary contrasts (bootstrap p-values, cluster setting S1)
    sec = res["settings"]["S1 cluster bootstrap"]["secondary"]
    keys = list(sec)
    adj = holm([sec[k]["boot_p_two_sided"] for k in keys])
    for k, a in zip(keys, adj):
        sec[k]["holm_p"] = a

    # ---- primary deep-dive on S1
    c, rs = settings["S1 cluster bootstrap"]
    deep = {}
    hb_all = []
    for fold, a, b, fams in primary_specs():
        hb = hierarchical_boot(c, a, b, fams, rs)
        hb_all.append((a, b, fams))
        d, D, seeds = contrast_diff(c, a, b, fams)
        loso = {}
        for drop in seeds:
            keep = [i for i, s in enumerate(seeds) if s != drop]
            loso[f"without seed {drop}"] = 100 * float(D[keep].mean())
        per_fam = {}
        for fam in fams:
            dd, DD, _ = contrast_diff(c, a, b, [fam])
            per_fam[fam] = summarize(dd, DD, rs)
        vs_each = {}
        for s in b:
            dd, DD, _ = contrast_diff(c, a, [s], fams)
            vs_each[s] = summarize(dd, DD, rs)
        per_seed_ci = {}
        for i, s in enumerate(seeds):
            bt = rs.boot(D[i])
            per_seed_ci[s] = [100 * float(D[i].mean()), 100 * float(np.percentile(bt, 2.5)),
                              100 * float(np.percentile(bt, 97.5))]
        deep[fold] = {"hierarchical_ci95_pp": [100 * float(np.percentile(hb, 2.5)), 100 * float(np.percentile(hb, 97.5))],
                      "leave_one_seed_out_pp": loso, "per_family": per_fam, "vs_each_single": vs_each,
                      "per_seed_effect_with_item_ci_pp": per_seed_ci, "_hb": hb}
    # pooled hierarchical: average of the two folds' draws (independent seed draws per condition)
    hbp = (deep["fold_a"]["_hb"] + deep["fold_b"]["_hb"]) / 2
    deep["pooled"] = {"hierarchical_ci95_pp": [100 * float(np.percentile(hbp, 2.5)), 100 * float(np.percentile(hbp, 97.5))]}
    for k in ("fold_a", "fold_b"):
        deep[k].pop("_hb")
    res["primary_deep_dive_S1"] = deep

    # ---- transfer matrix with cluster CIs (single-family model minus clean baseline)
    tm = {}
    for fold, spec in FOLDS.items():
        for fam in spec["seen"]:
            tm[fam] = {}
            for e in ALL_FAMILIES + ["clean"]:
                d, D, _ = contrast_diff(c, f"{fold}/single_{fam}", ["shared/clean"], [e])
                tm[fam][e] = summarize(d, D, rs)
    res["transfer_matrix_S1"] = tm

    # ---- does overlap raise accuracy, and does it differ by condition? (descriptive)
    ov = {}
    for cond in ["shared/clean", "fold_a/mixed3_f0.50", "fold_b/mixed3_f0.50"] + \
            [f"{fo}/single_{fa}" for fo, sp in FOLDS.items() for fa in sp["seen"]]:
        seeds = sorted(c[cond])
        allfam = per_item(c, cond, ALL_FAMILIES, seeds).mean(axis=0)
        clean = per_item(c, cond, ["clean"], seeds).mean(axis=0)
        ov[cond] = {"clean_acc_overlap": 100 * float(clean[exact].mean()),
                    "clean_acc_no_overlap": 100 * float(clean[~exact].mean()),
                    "perturbed_acc_overlap": 100 * float(allfam[exact].mean()),
                    "perturbed_acc_no_overlap": 100 * float(allfam[~exact].mean())}
    res["accuracy_by_overlap"] = ov
    write("03_contrasts_reanalysis", None, res)

    # ---- markdown
    L = ["# 03 Primary and secondary contrasts: post-hoc sensitivity (auto-generated)", "",
         "Effects are Mixed-3 (50%) minus comparator, in percentage points (pp). Brackets: 95% CI.",
         "`item/cluster CI` = test-item uncertainty only (bootstrap). `seed t-CI` = training-seed uncertainty only",
         "(t interval, df = 2). `combined` = effect +- 1.96 sqrt(SE_item^2 + SD_seed^2/3), an approximation.", "",
         "## Primary contrast under every setting", "",
         "| setting | n items | fold A | fold B | pooled |", "|---|---|---|---|---|"]
    for name, blk in res["settings"].items():
        p = blk["primary"]
        L.append(f"| {name} | {p['pooled']['n_items']} | {f(p['fold_a'])} | {f(p['fold_b'])} | {f(p['pooled'])} |")
    L += ["", "## Primary contrast: separating item and seed uncertainty (S1, all items)", "",
          "| fold | effect | cluster CI | per-seed effects | seed SD | seed t-CI | combined CI | seed x cluster bootstrap CI | protocol verdict |",
          "|---|---|---|---|---|---|---|---|---|"]
    p = res["settings"]["S1 cluster bootstrap"]["primary"]
    for k in ("fold_a", "fold_b", "pooled"):
        x = p[k]
        hb = deep[k]["hierarchical_ci95_pp"]
        L.append(f"| {k} | {x['effect_pp']:+.2f} | [{x['ci95_pp'][0]:+.2f}, {x['ci95_pp'][1]:+.2f}] | "
                 + ", ".join(f"{v:+.2f}" for v in x["per_seed_pp"]) +
                 f" | {x['seed_sd_pp']:.2f} | [{x['seed_t_ci95_pp'][0]:+.2f}, {x['seed_t_ci95_pp'][1]:+.2f}] | "
                 f"[{x['combined_ci95_pp'][0]:+.2f}, {x['combined_ci95_pp'][1]:+.2f}] | [{hb[0]:+.2f}, {hb[1]:+.2f}] | "
                 f"{x['protocol_verdict']} |")
    p0 = res["settings"]["S0 original (item bootstrap)"]["primary"]
    L += ["", f"Item-bootstrap SE (S0) vs cluster-bootstrap SE (S1), pooled: "
          f"{p0['pooled']['se_item_pp']:.3f} vs {p['pooled']['se_item_pp']:.3f} pp.", "",
          "## Leave-one-seed-out (S1)", ""]
    for k in ("fold_a", "fold_b"):
        L.append(f"* {k}: " + ", ".join(f"{a}: {v:+.2f}" for a, v in deep[k]["leave_one_seed_out_pp"].items()))
    L += ["", "## Per-seed effect with its own cluster CI (S1)", ""]
    for k in ("fold_a", "fold_b"):
        L.append(f"* {k}: " + "; ".join(f"seed {s}: {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]"
                                         for s, v in deep[k]["per_seed_effect_with_item_ci_pp"].items()))
    L += ["", "## Primary contrast by unseen family (S1)", "",
          "| fold | family | effect | cluster CI | seed SD | combined CI |", "|---|---|---|---|---|---|"]
    for k in ("fold_a", "fold_b"):
        for fam, x in deep[k]["per_family"].items():
            L.append(f"| {k} | {fam} | {x['effect_pp']:+.2f} | [{x['ci95_pp'][0]:+.2f}, {x['ci95_pp'][1]:+.2f}] | "
                     f"{x['seed_sd_pp']:.2f} | [{x['combined_ci95_pp'][0]:+.2f}, {x['combined_ci95_pp'][1]:+.2f}] |")
    L += ["", "## Mixed-3 vs each single-family model, unseen families (S1)", "",
          "| fold | comparator | effect | cluster CI | combined CI |", "|---|---|---|---|---|"]
    for k in ("fold_a", "fold_b"):
        for s, x in deep[k]["vs_each_single"].items():
            L.append(f"| {k} | {s.split('/')[1]} | {x['effect_pp']:+.2f} | [{x['ci95_pp'][0]:+.2f}, {x['ci95_pp'][1]:+.2f}] | "
                     f"[{x['combined_ci95_pp'][0]:+.2f}, {x['combined_ci95_pp'][1]:+.2f}] |")
    L += ["", "## Secondary contrasts (S0 original vs S1 cluster; S1 adds combined CI and Holm-adjusted p)", "",
          "| contrast | S0 effect [item CI] | S1 [cluster CI] | S3 disjoint-near [CI] | combined CI | seed SD | Holm p | protocol verdict (S1) |",
          "|---|---|---|---|---|---|---|---|"]
    s0 = res["settings"]["S0 original (item bootstrap)"]["secondary"]
    s3 = res["settings"]["S3 cluster, exclude verbatim+near-duplicate"]["secondary"]
    for k in keys:
        x = sec[k]
        L.append(f"| {k} | {f(s0[k])} | {f(x)} | {f(s3[k])} | [{x['combined_ci95_pp'][0]:+.2f}, {x['combined_ci95_pp'][1]:+.2f}] | "
                 f"{x['seed_sd_pp']:.2f} | {x['holm_p']:.3f} | {x['protocol_verdict']} |")
    L += ["", "## Transfer matrix with cluster CIs (single-family model minus clean baseline, pp; S1)", "",
          "| trained on | " + " | ".join(ALL_FAMILIES + ["clean"]) + " |", "|---|" + "---|" * 7]
    for fam, row in tm.items():
        L.append(f"| {fam} | " + " | ".join(
            f"{row[e]['effect_pp']:+.1f} [{row[e]['ci95_pp'][0]:+.1f}, {row[e]['ci95_pp'][1]:+.1f}]" for e in ALL_FAMILIES + ["clean"]) + " |")
    L += ["", "## Accuracy on test items with vs without a verbatim train passage (descriptive, %)", "",
          "| condition | clean: overlap | clean: no overlap | 6 perturbed sets: overlap | 6 perturbed: no overlap |",
          "|---|---|---|---|---|"]
    for cond, x in ov.items():
        L.append(f"| {cond} | {x['clean_acc_overlap']:.1f} | {x['clean_acc_no_overlap']:.1f} | "
                 f"{x['perturbed_acc_overlap']:.1f} | {x['perturbed_acc_no_overlap']:.1f} |")
    write("03_contrasts_reanalysis", "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
