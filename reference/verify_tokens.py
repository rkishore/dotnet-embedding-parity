"""Confirm a library's token ids explain its vectors, and show where they differ.

Re-embeds each library's own token ids with the reference pooling. If the result
matches the library's vector (cosine >= 0.99999), tokenization alone accounts for
any divergence from the reference, and the token diff below shows the defect.

Usage: verify_tokens.py --model M --vocab V --reference R --results X --tokens T --out O
"""
import argparse
import json

import numpy as np
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser()
    for a in ("--model", "--vocab", "--reference", "--results", "--tokens", "--out"):
        ap.add_argument(a, required=True)
    args = ap.parse_args()

    vocab = [line.rstrip("\n") for line in open(args.vocab, encoding="utf-8")]
    ref = {r["id"]: r for r in json.load(open(args.reference))["references"]["trunc256"]}
    res = json.load(open(args.results))
    lib_vec = {s["Id"]: s["Vector"] for s in res["Single"] if s.get("Vector")}
    lib_ids = json.load(open(args.tokens))
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    out = {}
    print(f"## {res['Library']}")
    for tid, ids in lib_ids.items():
        x = np.array([ids], dtype=np.int64)
        h = session.run(None, {"input_ids": x, "attention_mask": np.ones_like(x), "token_type_ids": np.zeros_like(x)})[0]
        v = h[0].mean(axis=0)
        lv = np.array(lib_vec[tid])
        explained = float(v @ lv / (np.linalg.norm(v) * np.linalg.norm(lv)))
        same = ids == ref[tid]["token_ids"]
        lib_tok = [vocab[i] for i in ids]
        ref_tok = [vocab[i] for i in ref[tid]["token_ids"]]
        out[tid] = {"ids_reproduce_library_vector": explained, "tokens_equal_reference": same,
                    "library_tokens": lib_tok, "reference_tokens": ref_tok}
        if not same:
            print(f"- {tid}: library ids reproduce its vector at {explained:.6f}")
            print(f"    library  ({len(lib_tok)}): {' '.join(lib_tok[:40])}")
            print(f"    reference({len(ref_tok)}): {' '.join(ref_tok[:40])}")
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
