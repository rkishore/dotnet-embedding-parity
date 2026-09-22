# Retrieval recall: correct against unstripped-accent tokenization

The design and the predictions, recorded before this code was written, are in
[`PREDICTIONS.md`](PREDICTIONS.md). In short: MIRACL French, Spanish and German, and MS MARCO
as the English control, each subsampled to 300,000 passages with a fixed seed. Recall@10
with all-MiniLM-L6-v2 is compared between two arms that differ only in whether the
tokenizer strips accents, reported as a relative change.

- `prepare.py` downloads a corpus at a pinned dataset revision and writes the subsample
  to `.cache/recall/data/<corpus>/`.
- `run.py` checks that the two arms are what they claim to be against the committed
  results, tokenizes per arm, embeds through one onnxruntime session, searches exactly,
  and writes `results/<corpus>.json`.

From the repository root, in bash. The MIRACL corpora are about 6 GB to download, MS MARCO
about 1.6 GB; everything lands under `.cache/recall/`, which git ignores.

```sh
bash model/fetch.sh
PREP="uv run --no-project --python 3.12 --with huggingface-hub==1.32.0 --with pyarrow==25.0.1"
RUN="uv run --no-project --python 3.12 --with numpy==2.2.5 --with onnxruntime==1.22.0 --with tokenizers==0.21.1"
for c in miracl-fr miracl-es miracl-de msmarco; do
  $PREP recall/prepare.py --corpus $c
  $RUN recall/run.py --corpus $c
done
$RUN recall/unk_share.py    # the post-hoc [UNK] count in RESULTS.md
```

Results and the scored predictions: [`RESULTS.md`](RESULTS.md).
