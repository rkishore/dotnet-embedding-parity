"""Reference embeddings for the competitor oracle check.

The recipe: the model's own WordPiece tokenizer, onnxruntime, attention-masked mean
pooling in numpy, then L2 normalisation. Each text runs alone, unpadded, so padding
cannot enter the reference.

Token ids cannot be injected into the libraries under test, so the reference tokenizes
too, and records its ids so that a divergence can be attributed to tokenization or to
pooling.

Truncation is computed at two lengths. 256 is the model's max_seq_length
(sentence_bert_config.json) and is the reference. 512 is the positional limit that
the libraries default to, kept only to attribute divergences on long input.

Usage:
  uv run --python 3.12 --with numpy==2.2.5 --with onnxruntime==1.22.0 \
      --with tokenizers==0.21.1 reference.py --model M --vocab V --texts T --out O
"""
import argparse
import hashlib
import json

import numpy as np
import onnxruntime as ort
import tokenizers
from tokenizers import BertWordPieceTokenizer


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def embed(session, tokenizer, text):
    enc = tokenizer.encode(text)
    ids = np.array([enc.ids], dtype=np.int64)
    mask = np.array([enc.attention_mask], dtype=np.int64)
    feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
    hidden = session.run(None, feeds)[0]  # [1, seq, 384]
    m = mask[..., None].astype(np.float32)
    pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
    vec = pooled[0] / np.linalg.norm(pooled[0])
    return enc.ids, vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    texts = json.load(open(args.texts, encoding="utf-8"))["texts"]
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    out = {
        "recipe": "tokenizers BertWordPieceTokenizer(lowercase=True) + onnxruntime + masked mean + L2",
        "versions": {"onnxruntime": ort.__version__, "tokenizers": tokenizers.__version__, "numpy": np.__version__},
        "model_sha256": sha256(args.model),
        "vocab_sha256": sha256(args.vocab),
        "references": {},
    }
    for max_len in (256, 512):
        tok = BertWordPieceTokenizer(args.vocab, lowercase=True)
        tok.enable_truncation(max_length=max_len)
        rows = []
        for t in texts:
            ids, vec = embed(session, tok, t["text"])
            rows.append({"id": t["id"], "token_count": len(ids), "token_ids": ids,
                         "vector": [float(x) for x in vec]})
        out["references"][f"trunc{max_len}"] = rows

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    for r256, r512 in zip(out["references"]["trunc256"], out["references"]["trunc512"]):
        print(f'{r256["id"]:<12} tokens@256={r256["token_count"]:<4} tokens@512={r512["token_count"]}')


if __name__ == "__main__":
    main()
