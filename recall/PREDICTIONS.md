# Predictions: does unstripped-accent tokenization cost retrieval recall?

_Recorded 2026-09-21 and committed alone, before any code for this experiment was written
and before any corpus was downloaded. This repository's history is the evidence of that
ordering. Scored in `recall/RESULTS.md` after the run, including the ones that are wrong._

## The question

The token-level results in [`../RESULTS.md`](../RESULTS.md) show that ElBruno, and SK
under invariant globalization, do not strip accents, so an accented word becomes `[UNK]`.
A single sentence then scores 0.33 against sentence-transformers. That is a vector-level
number. This experiment asks what it costs a developer who uses such a library for
retrieval, with the default English model, on text in languages that use accents.

## Design

- **Model.** `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a24`,
  `model/model.onnx` (sha256 `6fd5d72f…`), the realistic case of a developer picking the
  default English model. One onnxruntime session, CPU, masked mean pooling, L2
  normalisation, every input truncated at 256 tokens, the model's `max_seq_length`.
- **Arms.** Only tokenization differs. Token ids are generated per arm; every arm then
  runs through the same session and pooling, so truncation policy and pooling cannot
  confound the result.
  - **correct:** `tokenizers.BertWordPieceTokenizer(vocab, lowercase=True)`, the recipe of
    `reference/reference.py`, which matches sentence-transformers.
  - **unstripped:** the same with `strip_accents=False`. This variant reproduces the
    ElBruno and SK-under-invariant vectors for `accents` at cosine ≥ 0.99999
    (`results/attribution.json`, "lowercase but do not strip accents").
  - LMSupply is left out: it is fixed upstream in 0.70.0.
  - Queries and passages are tokenized by the same arm, as they would be in an
    application that uses one library for both.
- **Corpora.**
  - Primary: MIRACL v1.0 French, Spanish and German, dev queries
    (`miracl/miracl` at `5be20db`, `miracl/miracl-corpus` at `d921ec7`).
  - English control: MS MARCO passage ranking, dev queries (the BEIR copy:
    `BeIR/msmarco` at `a918e0d`, `BeIR/msmarco-qrels` at `253fbf8`, `dev.tsv`).
- **Subsample.** For each corpus, every passage judged for any dev query (relevant or
  not), plus distractors drawn uniformly at random from the rest of the corpus, to a total
  of **300,000 passages**. Distractors come from reservoir sampling over the corpus in file
  order with seed **20260921**. Both arms use the same subsample.
- **Retrieval.** Exact inner-product search over the whole subsample, top 10.
- **Metric.** Recall@10 per query: relevant passages in the top 10, divided by the
  query's relevant passages (relevance label ≥ 1), averaged over the dev queries that have
  at least one. Reported as a **relative** change between arms,
  Δ = (R_unstripped − R_correct) / R_correct, because the English model's low baseline on
  these languages compresses absolute differences. Each Δ carries a 95% interval from a
  paired bootstrap over queries (10,000 resamples, seed 20260921).
- **Subset.** For MS MARCO, also the queries with at least one relevant passage whose
  token ids differ between the arms, which is to say a relevant passage containing a
  character that accent stripping changes.

**"Barely moves" means |Δ| < 2%.** A prediction is scored on the point estimate of Δ; the
interval is reported beside it, and a prediction whose interval straddles its threshold is
reported as met or refuted with that caveat stated.

## Predictions

- **R1. MIRACL French: recall@10 drops by at least 10% relative** (Δ ≤ −10%). Medium
  confidence. French marks a large share of common words (`é` alone covers `été`,
  `également`, most past participles), each of those words becomes `[UNK]` in both the
  query and the passage, and the lexical overlap that the English model relies on for a
  foreign language is exactly what is lost.
- **R2. MIRACL Spanish: recall@10 drops by at least 5% relative** (Δ ≤ −5%). Medium–low
  confidence. Accents are frequent (`más`, `también`, every `-ción` word) but a smaller
  share of words than in French.
- **R3. MIRACL German: recall@10 drops by at least 5% relative** (Δ ≤ −5%). Low–medium
  confidence. Umlauts are common (`für`, `über`, `können`); `ß` has no decomposition and
  is tokenized identically by both arms.
- **R4. Ordering: the French drop is the largest of the three.** Low confidence.
- **R5. MS MARCO, all dev queries: barely moves** (|Δ| < 2%). High confidence: English
  passages rarely contain accented characters, so almost every query and passage
  tokenizes identically in both arms.
- **R6. MS MARCO, the accented-relevant-passage subset: recall@10 drops by at least 2%
  relative** (Δ ≤ −2%). Low confidence. The subset is expected to be small, so its
  interval will be wide; if the interval includes zero, the result is reported as
  unresolved whatever the point estimate.

Absolute recall is not predicted.
