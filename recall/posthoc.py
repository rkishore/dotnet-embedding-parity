"""Post-hoc analyses, added after the results and not covered by any prediction.

1. MS MARCO, mojibake. The pre-registered subset (queries with a relevant passage that
   tokenizes differently between the arms) turned out to be entirely mojibake: UTF-8
   decoded as Latin-1, so that ’ reads as â + U+0080 + U+0099. This counts how many of those passages
   still tokenize differently once mojibake sequences are removed, which is to say how
   many contain genuine accented characters.
2. Query-side observations for all four corpora, from the text alone: how many queries
   have a word that the unstripped arm turns into [UNK], and how many such tokens.

Usage (same environment as run.py): recall/posthoc.py > recall/results/posthoc.json
"""
import json
import os
import re

import numpy as np

from run import ARMS, encode, tokenizer

UNK = 100
DATA = ".cache/recall/data"
# A UTF-8 multi-byte sequence decoded as Latin-1 or cp1252: a lead byte shown as one of
# Â–ß (two-byte) or à–ï (three-byte), followed by continuation bytes shown as U+0080–U+00BF
# or as the cp1252 characters that occupy 0x80–0x9F.
CONT = "\u0080-¿ŒœŠšŸŽžƒˆ˜–—‘-„†-•…‰‹›€™"
MOJIBAKE = re.compile(f"[Â-ß][{CONT}]|[à-ï][{CONT}]{{2}}")


def load(corpus):
    folder = os.path.join(DATA, corpus)
    passages = [json.loads(line) for line in open(os.path.join(folder, "passages.jsonl"), encoding="utf-8")]
    queries = [json.loads(line) for line in open(os.path.join(folder, "queries.jsonl"), encoding="utf-8")]
    index = {p["id"]: i for i, p in enumerate(passages)}
    relevant = {q["id"]: set() for q in queries}
    for line in open(os.path.join(folder, "qrels.tsv"), encoding="utf-8"):
        q, p, label = line.rstrip("\n").split("\t")
        if int(label) >= 1:
            relevant[q].add(index[p])
    return passages, queries, [relevant[q["id"]] for q in queries]


def msmarco(toks):
    """How much of the pre-registered subset is genuine accents rather than mojibake."""
    passages, _, rel = load("msmarco")
    texts = [p["text"] for p in passages]
    a, b = (encode(toks[arm], texts) for arm in ARMS)
    changed = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    cleaned = [MOJIBAKE.sub(" ", texts[i]) for i in changed]
    ca, cb = (encode(toks[arm], cleaned) for arm in ARMS)
    genuine = {i for i, x, y in zip(changed, ca, cb) if x != y}
    return {
        "passages": len(passages),
        "passages_tokenized_differently": len(changed),
        "of_which_contain_mojibake": sum(1 for i in changed if MOJIBAKE.search(texts[i])),
        "of_which_still_differ_with_mojibake_removed": len(genuine),
        "queries_in_preregistered_subset": sum(1 for r in rel if r & set(changed)),
        "queries_with_a_genuinely_accented_relevant_passage": sum(1 for r in rel if r & genuine),
    }


def query_side(toks):
    out = {}
    for corpus in ("miracl-fr", "miracl-es", "miracl-de", "msmarco"):
        _, queries, _ = load(corpus)
        ids = encode(toks["unstripped"], [q["text"] for q in queries])
        unk = np.array([x.count(UNK) for x in ids])
        length = np.array([len(x) - 2 for x in ids])
        out[corpus] = {"queries": len(queries),
                       "queries_with_an_unk": float((unk > 0).mean()),
                       "mean_unk_tokens_per_query": float(unk.mean()),
                       "unk_share_of_query_tokens": float(unk.sum() / length.sum())}
    return out


def main():
    toks = {arm: tokenizer("model/vocab.txt", **cfg) for arm, cfg in ARMS.items()}
    print(json.dumps({"msmarco_mojibake": msmarco(toks), "query_side": query_side(toks)}, indent=2))


if __name__ == "__main__":
    main()
