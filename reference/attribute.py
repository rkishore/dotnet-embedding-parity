"""Attribute each library divergence to a specific tokenizer behaviour.

For every text a library disagrees on, re-embed that text with the reference recipe
under tokenizer variants, each of which isolates one hypothesis. If a library's
vector matches a variant to cosine >= 0.99999 while failing the reference, that
variant reproduces the library's behaviour exactly, and the cause is established
by measurement rather than by reading the library's code.

Usage (same environment as reference.py):
  reference/attribute.py --model M --vocab V --texts T --results results/x.json [...]
"""
import argparse
import json

import numpy as np
import onnxruntime as ort
from tokenizers import BertWordPieceTokenizer

MATCH = 0.99999
REFERENCE_OK = 0.9999

VARIANTS = {
    "reference (lowercase, strip accents, split CJK), trunc 256": dict(lowercase=True, max_len=256),
    "trunc 512 instead of 256": dict(lowercase=True, max_len=512),
    "lowercase but do not strip accents": dict(lowercase=True, strip_accents=False, max_len=512),
    "no lowercasing, no accent stripping": dict(lowercase=False, strip_accents=False, max_len=512),
    "no lowercasing, no accent stripping, no CJK splitting":
        dict(lowercase=False, strip_accents=False, handle_chinese_chars=False, max_len=512),
    "no CJK splitting": dict(lowercase=True, handle_chinese_chars=False, max_len=512),
    "no text cleaning (control chars kept)": dict(lowercase=True, clean_text=False, max_len=512),
}


def embed_ids(session, ids):
    ids = np.array([ids], dtype=np.int64)
    mask = np.ones_like(ids)
    hidden = session.run(None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)})[0]
    v = hidden[0].mean(axis=0)
    return v / np.linalg.norm(v)


def cos(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    texts = {t["id"]: t["text"] for t in json.load(open(args.texts, encoding="utf-8"))["texts"]}
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    tokenizers = {}
    for name, cfg in VARIANTS.items():
        cfg = dict(cfg)
        max_len = cfg.pop("max_len")
        tok = BertWordPieceTokenizer(args.vocab, **cfg)
        tok.enable_truncation(max_length=max_len)
        tokenizers[name] = tok

    ref_name = next(iter(VARIANTS))
    report = {}
    for path in args.results:
        res = json.load(open(path))
        lib = res["Library"]
        report[lib] = {}
        print(f"\n## {lib}")
        for s in res["Single"]:
            if not s.get("Vector"):
                continue
            tid, vec = s["Id"], s["Vector"]
            scores, tokens = {}, {}
            for name, tok in tokenizers.items():
                enc = tok.encode(texts[tid])
                scores[name] = cos(vec, embed_ids(session, enc.ids))
                tokens[name] = enc.tokens
            if scores[ref_name] >= REFERENCE_OK:
                continue
            matches = [n for n, c in scores.items() if c >= MATCH]
            report[lib][tid] = {"scores": scores, "matches": matches,
                                "reference_tokens": tokens[ref_name],
                                "matching_tokens": {m: tokens[m] for m in matches}}
            print(f"- {tid}: reference {scores[ref_name]:.6f}; reproduced by: {matches or 'NONE'}")
            if not matches:
                best = max(scores, key=scores.get)
                print(f"    closest: {best} at {scores[best]:.6f}")

    json.dump(report, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
