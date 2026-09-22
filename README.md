# dotnet-embedding-parity

As part of a learning experiment, three .NET embedding libraries were run against a
reference implementation of `sentence-transformers/all-MiniLM-L6-v2`. The reference is
checked against sentence-transformers itself. This repository holds the probes, the
runners, their results, the predictions written before they ran, and two follow-ups: a
measurement of `Microsoft.ML.Tokenizers` under invariant globalization, and a
retrieval-recall experiment.

All three libraries we measured diverge from sentence-transformers in at least one
configuration, and every divergence is in tokenization.

> The divergence lives before the input tensor. Every parity tool in every ecosystem
> starts at the tensor; .NET embedding libraries go wrong in tokenization, and one class
> of failure depends on the deployment image, not the code.

## What it shows

- **Three external libraries were measured, and all three diverge from sentence-transformers
  in at least one configuration.**
  - **LMSupply.Embedder 0.68.0** skips BERT basic tokenization entirely, and a realistic
    English passage scores 0.42. **Fixed in 0.70.0** after the upstream report.
  - **ElBruno.LocalEmbeddings 1.6.1** turns accented words into `[UNK]` (0.333) and glues
    together words separated only by a newline (0.936).
  - **Semantic Kernel `Connectors.Onnx` 1.80.1-alpha** is exact under ICU, and loses accents
    under invariant globalization.
- **Pooling is correct in all three, and every divergence is tokenization.** For ElBruno and
  LMSupply, re-embedding each library's own token ids reproduces its vectors at 1.000000.
  SK is exact under ICU. Its invariant-mode vectors are reproduced at ≥ 0.99999 by the
  reference tokenizer with accent stripping turned off.
- **Under invariant globalization, `String.Normalize(FormD)` returns its input unchanged,
  by [documented design](https://github.com/dotnet/runtime/blob/6f4751a142ca0e879d60cb4091356bb9d346143e/docs/design/features/globalization-invariant-mode.md#string-normalization).**
  A tokenizer that strips accents through it degrades silently. SK is measured. So is
  `Microsoft.ML.Tokenizers` 2.0.0 with `RemoveNonSpacingMarks = true`: under invariant mode
  the option does nothing on precomposed text, so `Café crème brûlée in São Paulo, naïve
  résumé` scores 0.333027, the same as with the option off. On text that arrives already
  decomposed, it still works. Reported as
  [dotnet/machinelearning#7728](https://github.com/dotnet/machinelearning/issues/7728).

Invariant globalization is common where .NET is deployed:

- **Container images.** Microsoft's Alpine and Ubuntu Chiseled .NET images "do not include
  `icu` or `tzdata`, meaning that these images only work with apps that are configured for
  globalization-invariant mode". Their `extra` variants add ICU
  ([`dotnet-docker` image variants](https://github.com/dotnet/dotnet-docker/blob/7a1cdd5dd426ae782d7304ab8af476855790186a/documentation/image-variants.md)).
- **Native AOT templates.** Native AOT does not require invariant mode, but the .NET 10
  project templates for it turn it on. `dotnet new console --aot`, `dotnet new worker --aot`
  and `dotnet new webapiaot` all set `<InvariantGlobalization>true</InvariantGlobalization>`
  ([console](https://github.com/dotnet/sdk/blob/fd7d9df34dec5bf71ff2ad9869335644a33b9925/template_feed/Microsoft.DotNet.Common.ProjectTemplates.10.0/content/ConsoleApplication-CSharp/Company.ConsoleApplication1.csproj), [worker](https://github.com/dotnet/aspnetcore/blob/0ef4bbfa3291b306a21e5001cb0491277bdd35bc/src/ProjectTemplates/Web.ProjectTemplates/Worker-CSharp.csproj.in),
  [webapiaot](https://github.com/dotnet/aspnetcore/blob/0ef4bbfa3291b306a21e5001cb0491277bdd35bc/src/ProjectTemplates/Web.ProjectTemplates/WebApiAot-CSharp.csproj.in)).

So the same code can be exact on a developer's machine and wrong in one of those images.

## The numbers

Cosine against the reference, truncated at the model's `max_seq_length` of 256. Bold
marks a defect. Full tables are in [`RESULTS.md`](RESULTS.md), and per-text results are in
[`results/summary.md`](results/summary.md) and
[`results/invariant-globalization/summary.md`](results/invariant-globalization/summary.md).

| | short ASCII | case + punctuation | realistic passage | accents | newline / tab | CJK + emoji |
|---|--:|--:|--:|--:|--:|--:|
| SK `Connectors.Onnx` 1.80.1-alpha, ICU | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| SK `Connectors.Onnx` 1.80.1-alpha, invariant | 1.000000 | 1.000000 | 1.000000 | **0.333027** | 1.000000 | 1.000000 |
| LMSupply.Embedder 0.68.0 | 1.000000 | **0.671828** | **0.420103** | **−0.001147** | 1.000000 | **0.517515** |
| LMSupply.Embedder 0.70.0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| ElBruno.LocalEmbeddings 1.6.1 | 1.000000 | 1.000000 | 1.000000 | **0.333027** | **0.936251** | 0.996960 |

ElBruno and LMSupply score the same under both globalization modes. All three libraries
truncate at 512 tokens rather than 256. That is a policy difference, not a defect, and it
is left out of this table.

`Microsoft.ML.Tokenizers` 2.0.0 `BertTokenizer` on its own, compared by token id with
Hugging Face and by cosine after embedding
([`results/mltokenizers/accent-summary.md`](results/mltokenizers/accent-summary.md)):

| probe | ICU, default | ICU, `RemoveNonSpacingMarks` | invariant, default | invariant, `RemoveNonSpacingMarks` |
|---|--:|--:|--:|--:|
| accents, precomposed | **0.333027** | 1.000000 | **0.333027** | **0.333027** |
| accents, decomposed | **0.333027** | 1.000000 | **0.333027** | 1.000000 |
| a French sentence | **0.545866** | 1.000000 | **0.545866** | **0.545866** |

## What it costs retrieval

A cosine of 0.333 on one sentence is a vector-level number. [`recall/`](recall/) asks what
a missing accent-stripping step costs retrieval. It compares two tokenizers that differ
only in whether they strip accents, using the default English model and 300,000-passage
subsamples ([`recall/RESULTS.md`](recall/RESULTS.md)):

| corpus | recall@10, correct | recall@10, unstripped | absolute change | relative change (95% interval) |
|---|--:|--:|--:|--:|
| MIRACL French | 0.409 | 0.264 | −14.5 points | **−35.5%** (−43.0% to −27.5%) |
| MIRACL Spanish | 0.282 | 0.221 | −6.1 points | **−21.6%** (−27.7% to −15.3%) |
| MIRACL German | 0.279 | 0.276 | −0.3 points | −1.0% (−9.4% to +7.8%): inconclusive |
| MS MARCO (English) | 0.913 | 0.909 | −0.4 points | −0.5% (−0.7% to −0.2%) |

- **The baselines are low.** An English model finds only 28–41% of the relevant French,
  Spanish and German passages in its top 10, even with correct tokenization. The design
  fixed the relative change in advance as the headline, because a given absolute loss is a
  larger share of a weak baseline. The absolute numbers are beside it so the relative
  figure can be checked.
- **German is inconclusive, not unaffected.** Its interval holds both a 9% loss and an 8%
  gain.
- **MS MARCO is an encoding control, not an accent control.** Its only non-ASCII text is
  mojibake, and no relevant passage contains a genuine accented character.

The predictions were committed before the code. Four of six were met. German was predicted
to drop by at least 5%, and the result can neither confirm nor rule that out.

## Method

- **Predictions first.** [`PREDICTIONS.md`](PREDICTIONS.md) was written before any probe
  ran, in a private repository. It is reproduced here without an addendum about the
  author's own library, so its date is stated, not demonstrated. The invariant-globalization
  predictions (MT1–MT7) and [`recall/PREDICTIONS.md`](recall/PREDICTIONS.md) were committed
  in this repository before their code existed, so the history shows the order. Every
  prediction is scored, including the wrong ones.
- **A control.** The numpy reference matches sentence-transformers 6.0.1, which runs the
  PyTorch weights and has no ONNX involved, to a minimum cosine of 0.99999996
  ([`results/control-sentence-transformers.json`](results/control-sentence-transformers.json)).
- **Attribution by measurement.** Each divergence is reproduced from token ids
  ([`reference/verify_tokens.py`](reference/verify_tokens.py)) or from a tokenizer variant
  ([`reference/attribute.py`](reference/attribute.py)), not inferred from reading code.
- **A harness error, kept.** The first run was built with invariant globalization by
  habit. SK then scored `accents` at exactly ElBruno's value, and that coincidence is how
  the platform dependence was found. See [`RESULTS.md`](RESULTS.md).

## Reproduce

Requirements: the .NET 10 SDK, [uv](https://docs.astral.sh/uv/), and bash.

```sh
bash model/fetch.sh    # model.onnx at revision 1110a24, sha256-checked against results/reference.json
```

- The reference, the control, the three libraries under both globalization modes, the
  token-level attribution and the `Microsoft.ML.Tokenizers` measurement:
  [`RESULTS.md`, "Reproduce"](RESULTS.md#reproduce).
- The recall experiment: [`recall/README.md`](recall/README.md).
- The invariant-globalization demonstration, about twenty lines:
  `dotnet run --project invariant` refuses to start, and
  `dotnet run --project invariant -p:InvariantGlobalization=false` does not.

Every committed result was produced by this repository's own runners and scripts, on
macOS arm64 with .NET 10.0.12. They reproduce, bit for bit, the vectors of the original run
that `PREDICTIONS.md` was written for.

## Upstream reports

- [dotnet/machinelearning#7724](https://github.com/dotnet/machinelearning/issues/7724):
  `BertTokenizer` merges words separated only by `\n`, `\t` or `\r`.
- [dotnet/machinelearning#7725](https://github.com/dotnet/machinelearning/issues/7725):
  `BertTokenizer` silently drops symbol characters.
- [dotnet/machinelearning#7728](https://github.com/dotnet/machinelearning/issues/7728):
  `RemoveNonSpacingMarks = true` fails to strip accents from precomposed characters under
  invariant globalization.
- [dotnet/machinelearning#7728](https://github.com/dotnet/machinelearning/issues/7728):
  `RemoveNonSpacingMarks = true` fails to strip accents from precomposed characters under
  invariant globalization.
- [elbruno/elbruno.localembeddings#56](https://github.com/elbruno/elbruno.localembeddings/issues/56):
  accented words become `[UNK]` with the default model.
- [iyulab/lm-supply#12](https://github.com/iyulab/lm-supply/issues/12): embeddings diverge
  from sentence-transformers because of WordPiece tokenization. Fixed in 0.70.0.

## Layout

| path | contents |
|---|---|
| `texts.json`, `tokenizer-probes.json` | the probe texts: ten for the cosine comparison, 49 for token ids |
| `reference/` | the Python reference, the sentence-transformers control, and the attribution scripts |
| `sk/`, `elbruno/`, `lmsupply/` | one console runner per library, each pinning its own ONNX Runtime |
| `mltokenizers/` | `Microsoft.ML.Tokenizers` `BertTokenizer` on its own, and the accent probes |
| `shared/` | the result format every runner writes |
| `results/` | every committed result, and `compare.py`'s summaries |
| `invariant/` | the `Normalize(FormD)` demonstration, and a startup guard |
| `recall/` | the retrieval-recall experiment |
| `model/` | the vocabulary, and `fetch.sh` for the model |

Licensed under Apache-2.0.
