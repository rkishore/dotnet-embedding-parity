"""Score token ids from mltokenizers/ against Hugging Face's, by id and by cosine.

For every run (one per globalization mode), configuration and probe: whether the ids
equal Hugging Face's, and the cosine between the two id sequences embedded with the
reference pooling (onnxruntime, masked mean, L2). The cosine puts a divergence on the
same scale as results/summary.md.

Usage (same environment as reference.py):
  reference/score_ids.py --model M --hf results/mltokenizers/accent-ids.json \
      --runs results/mltokenizers/accent-icu.json results/mltokenizers/accent-invariant.json \
      --out results/mltokenizers/accent-scores.json > results/mltokenizers/accent-summary.md
"""
import argparse
import json

import numpy as np
import onnxruntime as ort


def embed_ids(session, ids):
    x = np.array([ids], dtype=np.int64)
    hidden = session.run(None, {"input_ids": x, "attention_mask": np.ones_like(x), "token_type_ids": np.zeros_like(x)})[0]
    v = hidden[0].mean(axis=0).astype(np.float64)
    return v / np.linalg.norm(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    hf = {p["id"]: p["token_ids"] for p in json.load(open(args.hf, encoding="utf-8"))["probes"]}
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    hf_vec = {pid: embed_ids(session, ids) for pid, ids in hf.items()}

    runs = [json.load(open(path, encoding="utf-8")) for path in args.runs]
    configs = list(runs[0]["Ids"])
    report = {"runs": []}
    cells = {}
    for run in runs:
        mode = "invariant" if run["InvariantGlobalization"] else "ICU"
        entry = {"mode": mode, "package_version": run["PackageVersion"], "runtime": run["Runtime"],
                 "normalize_formd_length_of_e9": run["NormalizeFormDLengthOfE9"], "configurations": {}}
        for config, per_probe in run["Ids"].items():
            rows = {}
            for pid, ids in per_probe.items():
                c = float(embed_ids(session, ids) @ hf_vec[pid])
                rows[pid] = {"ids_equal": ids == hf[pid], "cosine": c, "ids": ids}
                cells[(pid, mode, config)] = rows[pid]
            entry["configurations"][config] = rows
        report["runs"].append(entry)
    json.dump(report, open(args.out, "w", encoding="utf-8"), indent=2)

    modes = [r["mode"] for r in report["runs"]]
    print("# Microsoft.ML.Tokenizers BertTokenizer against Hugging Face\n")
    for r in report["runs"]:
        print(f"- {r['mode']}: package {r['package_version'].split('+')[0]}, {r['runtime']}, "
              f"`\"\\u00e9\".Normalize(FormD).Length` = {r['normalize_formd_length_of_e9']}")
    print("\nEach cell: ids equal to Hugging Face's (`=`) or not (`≠`), and the cosine between the")
    print("two id sequences embedded with the reference pooling.\n")
    header = [f"{m}, {c}" for m in modes for c in configs]
    print("| probe | " + " | ".join(header) + " |")
    print("|---|" + "--:|" * len(header))
    for pid in hf:
        row = []
        for m in modes:
            for c in configs:
                cell = cells[(pid, m, c)]
                mark = "=" if cell["ids_equal"] else "≠"
                text = f"{mark} {cell['cosine']:.6f}"
                row.append(text if cell["ids_equal"] else f"**{text}**")
        print(f"| {pid} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
