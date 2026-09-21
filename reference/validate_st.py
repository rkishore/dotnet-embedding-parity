"""Control: validate the numpy reference against sentence-transformers itself.

sentence-transformers runs the PyTorch safetensors weights, not the ONNX export, and
its own tokenizer and pooling modules. Agreement here means the reference is the
model's canonical output and not merely self-consistent. Disagreement means the
reference is wrong and no library result may be read.

Usage:
  uv run --python 3.12 --with sentence-transformers==6.0.1 --with numpy \
      validate_st.py --reference REF --texts T [--revision SHA]
"""
import argparse
import json
import sys

import numpy as np
import sentence_transformers
from sentence_transformers import SentenceTransformer

MIN_COSINE = 0.99999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--revision", default="1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    texts = json.load(open(args.texts, encoding="utf-8"))["texts"]
    ref = {r["id"]: np.array(r["vector"]) for r in json.load(open(args.reference))["references"]["trunc256"]}

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", revision=args.revision, device="cpu")
    vecs = model.encode([t["text"] for t in texts], normalize_embeddings=True, convert_to_numpy=True)

    rows, ok = [], True
    for t, v in zip(texts, vecs):
        r = ref[t["id"]]
        cos = float(np.dot(v, r) / (np.linalg.norm(v) * np.linalg.norm(r)))
        ok &= cos >= MIN_COSINE
        rows.append({"id": t["id"], "cosine": cos, "max_abs_diff": float(np.max(np.abs(v - r)))})
        print(f'{t["id"]:<12} cosine={cos:.8f} max|diff|={rows[-1]["max_abs_diff"]:.2e}')

    json.dump({"sentence_transformers": sentence_transformers.__version__, "revision": args.revision,
               "max_seq_length": model.max_seq_length, "min_cosine": MIN_COSINE, "passed": ok, "rows": rows},
              open(args.out, "w"), indent=2)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
