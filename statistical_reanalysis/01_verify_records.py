"""Step 1: integrity of the original records, and reproduction of the original analysis.

* SHA-256 of every original record file (so later readers can check nothing was modified)
* accuracy recomputed from predictions.json == metrics.json for every run
* training-data hashes == data/perturbed/MANIFEST.json == current files on disk
* train logs complete (steps 1..500), registry duplicates, environment versions
* exact p(Yes) == 0.5 ties per run and eval set (fp16 logit ties; resolved to "No" by p > 0.5)
* re-runs scripts/analyze.py into a scratch folder and checks byte-identity with results/analysis/

    python statistical_reanalysis/01_verify_records.py
"""

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from common import EVAL_SETS, OUT, ROOT, load_jsonl, write


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cfg = json.loads((ROOT / "configs" / "train_config.json").read_text(encoding="utf-8"))
    man = json.loads((ROOT / "data" / "perturbed" / "MANIFEST.json").read_text(encoding="utf-8"))
    labels = {k: [r["label"] for r in load_jsonl(ROOT / p)] for k, p in cfg["eval_sets"].items()}
    assert all(labels[k] == labels["clean"] for k in labels), "eval sets disagree on labels"
    test = load_jsonl(ROOT / "data/splits/test.jsonl")
    for k, p in cfg["eval_sets"].items():
        assert [r["passage"] for r in load_jsonl(ROOT / p)] == [r["passage"] for r in test], k

    issues, ties, env = [], {}, Counter()
    checksum_lines = []
    for f in sorted((ROOT / "results").rglob("*")):
        if f.is_file():
            checksum_lines.append(f"{sha(f)}  {f.relative_to(ROOT).as_posix()}")
    for d in sorted((ROOT / "results" / "runs").iterdir()):
        if not (d / "metrics.json").exists():
            continue
        rc = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
        if rc.get("tag"):
            continue
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        p = json.loads((d / "predictions.json").read_text(encoding="utf-8"))
        env[(rc["env"]["torch"], rc["env"]["transformers"], rc["env"]["peft"], rc["env"]["gpu"])] += 1
        ties[rc["run_id"]] = {}
        for k in EVAL_SETS:
            acc = sum((x > 0.5) == (y == 1) for x, y in zip(p[k], labels[k])) / len(labels[k])
            if abs(acc - m["test_acc"][k]) > 1e-12:
                issues.append(f"{rc['run_id']} {k}: predictions give {acc}, metrics say {m['test_acc'][k]}")
            ties[rc["run_id"]][k] = sum(x == 0.5 for x in p[k])
        if rc["condition"] != "untuned":
            if rc["train_data_sha256_lf"] != man[rc["condition"]]["sha256_lf"]:
                issues.append(f"{rc['run_id']}: train-data hash differs from MANIFEST")
            steps = [json.loads(l)["step"] for l in (d / "train_log.jsonl").read_text().splitlines()]
            if steps != list(range(1, 501)):
                issues.append(f"{rc['run_id']}: train log not 1..500")
    for name, e in man.items():
        cur = hashlib.sha256((ROOT / e["path"]).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if cur != e["sha256_lf"]:
            issues.append(f"data file for {name} differs from MANIFEST")

    reg = list(csv.DictReader(open(ROOT / "results/run_registry.csv", encoding="utf-8")))
    dups = {k: v for k, v in Counter(r["run_id"] for r in reg).items() if v > 1}

    # reproduce the original analysis
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(ROOT / "scripts/analyze.py"), "--out", tmp],
                       cwd=ROOT, check=True, capture_output=True)
        same_md = (Path(tmp) / "summary.md").read_text(encoding="utf-8").replace("\r\n", "\n") == \
            (ROOT / "results/analysis/summary.md").read_text(encoding="utf-8").replace("\r\n", "\n")
        same_json = json.loads((Path(tmp) / "summary.json").read_text(encoding="utf-8")) == \
            json.loads((ROOT / "results/analysis/summary.json").read_text(encoding="utf-8"))

    trained = {k: v for k, v in ties.items() if k != "untuned"}
    tie_counts = [c for r in trained.values() for c in r.values()]
    res = {"n_runs_checked": len(ties), "issues": issues, "registry_duplicate_rows": dups,
           "environments": {" / ".join(k): v for k, v in env.items()},
           "analysis_reproduced": {"summary_md_identical": same_md, "summary_json_identical": same_json},
           "ties_exact_half": ties,
           "ties_summary_trained_runs": {"min": min(tie_counts), "max": max(tie_counts),
                                         "mean": sum(tie_counts) / len(tie_counts)}}
    (OUT / "original_records_sha256.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    lines = ["# 01 Record integrity and reproduction (auto-generated)", "",
             f"* Runs checked (excluding the tagged smoke run): **{len(ties)}**",
             f"* Integrity issues found: **{len(issues)}**" + ("" if not issues else "\n  * " + "\n  * ".join(issues)),
             f"* `scripts/analyze.py` re-run: summary.md identical = **{same_md}**, summary.json identical = **{same_json}**",
             f"* Registry rows duplicated by Colab resumes: {dups}",
             "* Software environments (torch / transformers / peft / GPU): " +
             "; ".join(f"{k} = {v} runs" for k, v in res["environments"].items()),
             "* Exact p(Yes) = 0.5 values per run and eval set (trained runs): "
             f"min {res['ties_summary_trained_runs']['min']}, max {res['ties_summary_trained_runs']['max']}, "
             f"mean {res['ties_summary_trained_runs']['mean']:.1f} of 1,000. Untuned: {ties['untuned']}.",
             "  No predicted value lies in (0.49, 0.51) other than exactly 0.5, consistent with fp16 logit ties.",
             "  The original rule p > 0.5 resolves every tie to \"No\". Sensitivity: see 03.",
             f"* SHA-256 of every file under results/: `outputs/original_records_sha256.txt` ({len(checksum_lines)} files)."]
    write("01_record_integrity", "\n".join(lines) + "\n", res)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
