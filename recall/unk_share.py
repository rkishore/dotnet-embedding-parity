"""Share of tokens that are [UNK] in each arm, over a fixed sample of each subsample.

Descriptive and post hoc: added after the recall results, to show how much text each arm
loses per corpus. Not part of the pre-registered design.

Usage (same environment as run.py): recall/unk_share.py [--sample 20000]
"""
import argparse
import json
import os
import random

from tokenizers import BertWordPieceTokenizer

from run import ARMS, MAX_LEN, SEED

UNK = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".cache/recall/data")
    ap.add_argument("--vocab", default="model/vocab.txt")
    ap.add_argument("--sample", type=int, default=20_000)
    args = ap.parse_args()
    toks = {}
    for arm, cfg in ARMS.items():
        toks[arm] = BertWordPieceTokenizer(args.vocab, **cfg)
        toks[arm].enable_truncation(max_length=MAX_LEN)
    for corpus in ("miracl-fr", "miracl-es", "miracl-de", "msmarco"):
        path = os.path.join(args.data, corpus, "passages.jsonl")
        texts = [json.loads(line)["text"] for line in open(path, encoding="utf-8")]
        texts = random.Random(SEED).sample(texts, args.sample)
        shares = []
        for arm in ARMS:
            enc = toks[arm].encode_batch(texts)
            shares.append(sum(e.ids.count(UNK) for e in enc) / sum(len(e.ids) - 2 for e in enc))
        print(f"{corpus}: [UNK] share of tokens, " + ", ".join(f"{a} {s:.2%}" for a, s in zip(ARMS, shares)))


if __name__ == "__main__":
    main()
