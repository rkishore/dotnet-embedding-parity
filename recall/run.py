"""Recall@10 with correct and with unstripped-accent tokenization, everything else fixed.

Token ids are generated per arm; both arms then run through one onnxruntime session with
one pooling (attention-masked mean, L2) and one 256-token cap, so only tokenization can
move the result. Passages whose ids are the same in both arms are embedded once.

Before anything is scored, the arms are checked against the committed results: the correct
arm must reproduce results/reference.json's token ids, and the unstripped arm must
reproduce ElBruno's `accents` vector (results/elbruno.json) at cosine >= 0.99999.

Usage (see recall/README.md for the environment):
  recall/run.py --corpus {miracl-fr,miracl-es,miracl-de,msmarco}
"""
import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort
from tokenizers import BertWordPieceTokenizer

SEED = 20260921
MAX_LEN = 256
K = 10
BOOTSTRAP = 10_000
ARMS = {"correct": dict(lowercase=True), "unstripped": dict(lowercase=True, strip_accents=False)}


def tokenizer(vocab, **cfg):
    tok = BertWordPieceTokenizer(vocab, **cfg)
    tok.enable_truncation(max_length=MAX_LEN)
    return tok


def encode(tok, texts):
    return [e.ids for e in tok.encode_batch(texts)]


def embed(session, id_lists, batch=128, label=""):
    """Masked mean pooling and L2 over id sequences, batched by length."""
    out = np.zeros((len(id_lists), 384), dtype=np.float32)
    order = np.argsort([len(x) for x in id_lists], kind="stable")
    start = time.time()
    for b in range(0, len(order), batch):
        rows = order[b:b + batch]
        width = max(len(id_lists[i]) for i in rows)
        ids = np.zeros((len(rows), width), dtype=np.int64)
        mask = np.zeros_like(ids)
        for r, i in enumerate(rows):
            ids[r, :len(id_lists[i])] = id_lists[i]
            mask[r, :len(id_lists[i])] = 1
        hidden = session.run(None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)})[0]
        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        out[rows] = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        if label and (b // batch) % 200 == 0:
            done = b + len(rows)
            print(f"  {label}: {done:,}/{len(order):,} ({done / (time.time() - start):,.0f}/s)", flush=True)
    return out


def top_k(queries, passages, k=K, chunk=256):
    hits = np.zeros((len(queries), k), dtype=np.int64)
    for s in range(0, len(queries), chunk):
        # numpy's Accelerate backend on macOS raises spurious floating-point warnings in
        # matmul; the assert is the real check.
        with np.errstate(all="ignore"):
            scores = queries[s:s + chunk] @ passages.T
        assert np.isfinite(scores).all()
        part = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
        hits[s:s + chunk] = np.take_along_axis(part, order, axis=1)
    return hits


def recall_at_k(hits, relevant):
    return np.array([len(set(h.tolist()) & rel) / len(rel) for h, rel in zip(hits, relevant)])


def relative_change(ra, rb, rng):
    """Point estimate and paired-bootstrap 95% interval of mean(rb) / mean(ra) - 1."""
    point = rb.mean() / ra.mean() - 1
    n, stats = len(ra), []
    for _ in range(BOOTSTRAP // 1000):
        idx = rng.integers(0, n, size=(1000, n))
        stats.append(rb[idx].mean(axis=1) / ra[idx].mean(axis=1) - 1)
    lo, hi = np.percentile(np.concatenate(stats), [2.5, 97.5])
    return float(point), [float(lo), float(hi)]


def check_arms(session, toks, root):
    ref = json.load(open(os.path.join(root, "results/reference.json")))
    texts = {t["id"]: t["text"] for t in json.load(open(os.path.join(root, "texts.json"), encoding="utf-8"))["texts"]}
    for r in ref["references"]["trunc256"]:
        assert toks["correct"].encode(texts[r["id"]]).ids == r["token_ids"], f"correct arm differs on {r['id']}"
    elbruno = {s["Id"]: np.array(s["Vector"]) for s in json.load(open(os.path.join(root, "results/elbruno.json")))["Single"]}
    v = embed(session, [toks["unstripped"].encode(texts["accents"]).ids])[0].astype(np.float64)
    c = float(v @ elbruno["accents"] / np.linalg.norm(elbruno["accents"]))
    assert c >= 0.99999, f"unstripped arm reproduces ElBruno's accents vector only at {c:.6f}"
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=["miracl-fr", "miracl-es", "miracl-de", "msmarco"])
    ap.add_argument("--data", default=".cache/recall/data")
    ap.add_argument("--model", default="model/model.onnx")
    ap.add_argument("--vocab", default="model/vocab.txt")
    ap.add_argument("--out", default="recall/results")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    toks = {arm: tokenizer(args.vocab, **cfg) for arm, cfg in ARMS.items()}
    check = check_arms(session, toks, root)
    print(f"arms check: correct arm reproduces reference ids; unstripped arm reproduces ElBruno accents at {check:.6f}")

    folder = os.path.join(args.data, args.corpus)
    passages = [json.loads(line) for line in open(os.path.join(folder, "passages.jsonl"), encoding="utf-8")]
    queries = [json.loads(line) for line in open(os.path.join(folder, "queries.jsonl"), encoding="utf-8")]
    index = {p["id"]: i for i, p in enumerate(passages)}
    relevant = {q["id"]: set() for q in queries}
    for line in open(os.path.join(folder, "qrels.tsv"), encoding="utf-8"):
        q, p, label = line.rstrip("\n").split("\t")
        if int(label) >= 1:
            relevant[q].add(index[p])
    rel = [relevant[q["id"]] for q in queries]

    t0 = time.time()
    p_ids = {arm: encode(tok, [p["text"] for p in passages]) for arm, tok in toks.items()}
    q_ids = {arm: encode(tok, [q["text"] for q in queries]) for arm, tok in toks.items()}
    p_diff = np.array([a != b for a, b in zip(p_ids["correct"], p_ids["unstripped"])])
    q_diff = np.array([a != b for a, b in zip(q_ids["correct"], q_ids["unstripped"])])
    print(f"{args.corpus}: {len(passages):,} passages ({p_diff.mean():.1%} tokenize differently), "
          f"{len(queries):,} queries ({q_diff.mean():.1%} differently)", flush=True)

    p_vec = {"correct": embed(session, p_ids["correct"], label="passages, correct")}
    p_vec["unstripped"] = p_vec["correct"].copy()
    changed = np.flatnonzero(p_diff)
    if len(changed):
        p_vec["unstripped"][changed] = embed(session, [p_ids["unstripped"][i] for i in changed],
                                             label="passages, unstripped (changed only)")
    q_vec = {arm: embed(session, q_ids[arm]) for arm in ARMS}
    embed_seconds = time.time() - t0

    recall = {arm: recall_at_k(top_k(q_vec[arm], p_vec[arm]), rel) for arm in ARMS}
    rng = np.random.default_rng(SEED)
    delta, interval = relative_change(recall["correct"], recall["unstripped"], rng)
    accented = np.array([any(p_diff[i] for i in r) for r in rel])
    subset = None
    if accented.any():
        sd, si = relative_change(recall["correct"][accented], recall["unstripped"][accented], rng)
        subset = {"queries": int(accented.sum()), "recall_correct": float(recall["correct"][accented].mean()),
                  "recall_unstripped": float(recall["unstripped"][accented].mean()),
                  "relative_change": sd, "interval95": si}

    result = {
        "corpus": args.corpus, "passages": len(passages), "queries": len(queries),
        "passages_tokenized_differently": float(p_diff.mean()), "queries_tokenized_differently": float(q_diff.mean()),
        "recall_at_10": {arm: float(r.mean()) for arm, r in recall.items()},
        "relative_change": delta, "interval95": interval,
        "subset_accented_relevant_passage": subset,
        "setup": {"max_len": MAX_LEN, "k": K, "bootstrap": BOOTSTRAP, "seed": SEED,
                  "onnxruntime": ort.__version__, "numpy": np.__version__,
                  "arms_check_unstripped_vs_elbruno_accents": check, "embed_seconds": round(embed_seconds)},
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.corpus}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ("recall_at_10", "relative_change", "interval95",
                                             "subset_accented_relevant_passage")}, indent=2))


if __name__ == "__main__":
    main()
