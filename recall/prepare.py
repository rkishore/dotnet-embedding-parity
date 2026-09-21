"""Download a corpus at a pinned revision and build its fixed retrieval subsample.

Keeps every passage judged for any dev query, relevant or not, and adds distractors drawn
uniformly at random from the rest of the corpus by reservoir sampling in file order, to a
fixed total. The seed and the total are fixed, so the subsample is the same on every
machine, and both arms of run.py read the same one.

Writes <out>/<corpus>/passages.jsonl ({"id", "text"}), queries.jsonl ({"id", "text"},
dev queries with at least one relevant passage) and qrels.tsv (query, passage, label).
A MIRACL passage's text is its title and its text joined by a space.

Usage (see recall/README.md for the environment):
  recall/prepare.py --corpus {miracl-fr,miracl-es,miracl-de,msmarco} [--total 300000]
"""
import argparse
import csv
import gzip
import io
import json
import os
import random

from huggingface_hub import HfApi, hf_hub_download

SEED = 20260921
MIRACL = ("miracl/miracl", "5be20db9509754dadad47689368639fcec739c00")
MIRACL_CORPUS = ("miracl/miracl-corpus", "d921ec7e349ce0d28daf30b2da9da5ee698bef0d")
MSMARCO = ("BeIR/msmarco", "a918e0d11a77ed33f42f29d98340b655593b96ad")
MSMARCO_QRELS = ("BeIR/msmarco-qrels", "253fbf8a3f8d4a0932b63882b5162bedc84779f5")


def fetch(repo, filename, cache):
    return hf_hub_download(repo[0], filename, repo_type="dataset", revision=repo[1], cache_dir=cache)


def miracl(lang, cache):
    base = f"miracl-v1.0-{lang}"
    queries = {}
    with open(fetch(MIRACL, f"{base}/topics/topics.{base}-dev.tsv", cache), encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            queries[row[0]] = row[1]
    qrels = []
    with open(fetch(MIRACL, f"{base}/qrels/qrels.{base}-dev.tsv", cache), encoding="utf-8") as f:
        for line in f:
            qid, _, pid, label = line.split()
            qrels.append((qid, pid, int(label)))

    def passages():
        folder = f"miracl-corpus-v1.0-{lang}"
        files = HfApi().list_repo_files(MIRACL_CORPUS[0], repo_type="dataset", revision=MIRACL_CORPUS[1])
        shards = sorted((p for p in files if p.startswith(folder + "/") and p.endswith(".jsonl.gz")),
                        key=lambda p: int(p.rsplit("-", 1)[1].split(".")[0]))
        for shard in shards:
            with gzip.open(fetch(MIRACL_CORPUS, shard, cache), "rt", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    yield d["docid"], f"{d['title']} {d['text']}"

    return queries, qrels, passages


def msmarco(cache):
    import pyarrow.parquet as pq

    qrels, dev = [], set()
    with open(fetch(MSMARCO_QRELS, "dev.tsv", cache), encoding="utf-8") as f:
        next(f)
        for line in f:
            qid, pid, label = line.split("\t")
            qrels.append((qid, pid, int(label)))
            dev.add(qid)
    queries = {}
    for batch in pq.ParquetFile(fetch(MSMARCO, "queries/queries-00000-of-00001.parquet", cache)).iter_batches(
            columns=["_id", "text"]):
        for qid, text in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            if qid in dev:
                queries[qid] = text

    def passages():
        corpus = pq.ParquetFile(fetch(MSMARCO, "corpus/corpus-00000-of-00001.parquet", cache))
        for batch in corpus.iter_batches(columns=["_id", "text"], batch_size=65536):
            yield from zip(batch.column(0).to_pylist(), batch.column(1).to_pylist())

    return queries, qrels, passages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=["miracl-fr", "miracl-es", "miracl-de", "msmarco"])
    ap.add_argument("--total", type=int, default=300_000)
    ap.add_argument("--cache", default=".cache/recall/hf")
    ap.add_argument("--out", default=".cache/recall/data")
    args = ap.parse_args()

    if args.corpus == "msmarco":
        queries, qrels, passages = msmarco(args.cache)
    else:
        queries, qrels, passages = miracl(args.corpus.split("-")[1], args.cache)

    relevant = {q for q, _, label in qrels if label >= 1 and q in queries}
    queries = {q: t for q, t in queries.items() if q in relevant}
    qrels = [r for r in qrels if r[0] in queries]
    judged_ids = {p for _, p, _ in qrels}
    if len(judged_ids) >= args.total:
        raise SystemExit(f"{len(judged_ids)} judged passages already exceed --total {args.total}")

    rng = random.Random(SEED)
    k = args.total - len(judged_ids)
    judged, reservoir, seen = {}, [], 0
    for pid, text in passages():
        if pid in judged_ids:
            judged[pid] = text
            continue
        seen += 1
        if len(reservoir) < k:
            reservoir.append((pid, text))
        else:
            j = rng.randrange(seen)
            if j < k:
                reservoir[j] = (pid, text)
    missing = judged_ids - judged.keys()
    if missing:
        raise SystemExit(f"{len(missing)} judged passages not found in the corpus, e.g. {sorted(missing)[:3]}")

    out = os.path.join(args.out, args.corpus)
    os.makedirs(out, exist_ok=True)
    with io.open(os.path.join(out, "passages.jsonl"), "w", encoding="utf-8") as f:
        for pid in sorted(judged):
            f.write(json.dumps({"id": pid, "text": judged[pid]}, ensure_ascii=False) + "\n")
        for pid, text in reservoir:
            f.write(json.dumps({"id": pid, "text": text}, ensure_ascii=False) + "\n")
    with io.open(os.path.join(out, "queries.jsonl"), "w", encoding="utf-8") as f:
        for qid in sorted(queries):
            f.write(json.dumps({"id": qid, "text": queries[qid]}, ensure_ascii=False) + "\n")
    with io.open(os.path.join(out, "qrels.tsv"), "w", encoding="utf-8") as f:
        for q, p, label in qrels:
            f.write(f"{q}\t{p}\t{label}\n")
    print(f"{args.corpus}: {len(queries)} dev queries with a relevant passage, {len(judged)} judged passages, "
          f"{len(reservoir)} distractors from {seen:,} unjudged, {len(judged) + len(reservoir):,} total -> {out}")


if __name__ == "__main__":
    main()
