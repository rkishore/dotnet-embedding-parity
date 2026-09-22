"""Score each library's embeddings against the reference.

For every probe text and library: cosine against the reference truncated at the
model's max_seq_length (256), cosine against the same recipe truncated at 512 (to
attribute long-input divergences to truncation policy), the vector's L2 norm, and
whether the library's multi-text call agrees with its own single-text call.

Usage: python3 compare.py results/reference.json results/<lib>.json [...] > results/summary.md
"""
import json
import math
import sys

THRESHOLD = 0.9999  # "agrees": the bar PREDICTIONS.md set before the run


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def norm(a):
    return math.sqrt(sum(x * x for x in a))


def mark(c):
    return f"**{c:.6f}**" if c < THRESHOLD else f"{c:.6f}"


def main():
    ref = json.load(open(sys.argv[1]))
    r256 = {r["id"]: r for r in ref["references"]["trunc256"]}
    r512 = {r["id"]: r for r in ref["references"]["trunc512"]}

    print("# Oracle check results\n")
    print(f"Reference: {ref['recipe']}; {ref['versions']}; model sha256 `{ref['model_sha256'][:12]}`.")
    print(f"Cosine below {THRESHOLD} is **bold**.\n")

    for path in sys.argv[2:]:
        res = json.load(open(path))
        print(f"## {res['Library']} {res.get('PackageVersion') or ''}\n")
        print(f"- ONNX Runtime: {res.get('OnnxRuntimeVersion')}")
        sha = res.get("ModelSha256")
        same = "identical to reference" if sha == ref["model_sha256"] else "**differs from reference**"
        print(f"- Model file: `{res.get('ModelFile')}`, sha256 `{(sha or 'unknown')[:12]}` ({same if sha else 'not hashed'})")
        print(f"- Configuration: {res.get('Configuration')}\n")
        print("| text | tokens@256 | cos vs ref@256 | cos vs ref@512 | L2 norm | batch vs single |")
        print("|---|--:|--:|--:|--:|--:|")

        batch = res["Batch"]
        batch_vec = {}
        if batch.get("Vectors"):
            batch_vec = dict(zip(batch["Ids"], batch["Vectors"]))

        for s in res["Single"]:
            tid = s["Id"]
            if s.get("Error"):
                print(f"| {tid} | {r256[tid]['token_count']} | error: {s['Error'][:80]} | | | |")
                continue
            v = s["Vector"]
            bvs = f"{cosine(batch_vec[tid], v):.6f}" if tid in batch_vec else "n/a"
            print(f"| {tid} | {r256[tid]['token_count']} | {mark(cosine(v, r256[tid]['vector']))} | "
                  f"{cosine(v, r512[tid]['vector']):.6f} | {norm(v):.4f} | {bvs} |")
        if batch.get("Error"):
            print(f"\nBatch call failed: `{batch['Error']}`")
        print()


if __name__ == "__main__":
    main()
