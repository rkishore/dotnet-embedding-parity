# Competitor oracle check: results

_Run 2026-09-17 on macOS arm64 (Apple M5 Pro), .NET runtime 10.0.12 (SDK 10.0.401), model
`sentence-transformers/all-MiniLM-L6-v2` at revision `1110a24`. Predictions were recorded
before the run ([`PREDICTIONS.md`](PREDICTIONS.md))._

## Bottom line

**Pooling is correct in all three libraries. Every divergence found is tokenization.**
Re-embedding each library's own token ids reproduces its vectors at cosine 1.000000, so
this is established by measurement, not by reading code.

| | short ASCII | case + punctuation | realistic passage | accents | newline / tab | CJK + emoji | > 256 tokens |
|---|--:|--:|--:|--:|--:|--:|--:|
| **SK Connectors.Onnx** 1.80.1-alpha | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.986514 ¹ |
| **LMSupply.Embedder** 0.68.0 | 1.000000 | **0.671828** | **0.420103** | **−0.001147** | 1.000000 | **0.517515** | **0.396909** |
| **LMSupply.Embedder** 0.70.0, after the report ² | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.986514 ¹ |
| **ElBruno.LocalEmbeddings** 1.6.1 | 1.000000 | 1.000000 | 1.000000 | **0.333027** | **0.936251** | 0.996960 | 0.986514 ¹ |

Cosine against the reference truncated at the model's `max_seq_length` of 256. "Short
ASCII" is the worst of `short`, `medium` and `query`. Bold marks a real defect;
unbolded values below 0.9999 are policy or minor.
¹ Truncates at 512 instead of 256, and agrees with the 512 reference at 1.000000.
² Added 2026-09-21. Reported as iyulab/lm-supply#12 on 2026-09-20; the maintainer confirmed
it and released 0.70.0 the next day, which runs BERT basic tokenization before WordPiece,
configured from the model's `tokenizer.json` or `tokenizer_config.json`. All nine texts up to
256 tokens now score 1.000000 (`results/lmsupply-0.70.0.json`). On token ids against
Hugging Face over `tokenizer-probes.json`, now 49 inputs, it matches 48 at both the 256 and
the 512 cap. The exception is `literal_sep`: a special token typed as text. sentence-
transformers turns `a [SEP] b` into the `[SEP]` token and 0.70.0 keeps it as text, so
end to end it scores 0.535 against sentence-transformers, and `a [CLS] b` scores 0.602.
The 256-cap run shows `MaxSequenceLength = 256` reproduces the reference truncation, at
token level. The token-id check ran from a throwaway program and is not committed.

- **LMSupply 0.68.0 was broken for ordinary English.** Any word with a capital letter or
  attached punctuation was mis-tokenized, and a plain 109-token English passage scored 0.42.
  **Fixed in 0.70.0**, released 2026-09-21, a day after the report: exact on every text up
  to 256 tokens.
- **ElBruno has two defects**, both inherited from `Microsoft.ML.Tokenizers`: accented
  Latin words become `[UNK]`, and words separated only by a newline or tab are glued
  together. Everything else is exact.
- **SK is exact** on every text up to 256 tokens, under .NET's default globalization.

## The control

The numpy reference (`reference/reference.py`) matches sentence-transformers 6.0.1, which
runs PyTorch weights and its own tokenizer with no ONNX involved, on all ten probes: max
absolute difference 3.4e-7, minimum cosine 0.99999996
(`results/control-sentence-transformers.json`). All three libraries loaded a model
file byte-identical to the reference (sha256 `6fd5d72f…`). For LMSupply this was
established from its download cache, which held only `onnx/model.onnx`, because the
probe's `IModelRuntimeInfo` cast returned null.

## Attribution

| defect | libraries | cause | evidence |
|---|---|---|---|
| Accented words → `[UNK]` | ElBruno | `Microsoft.ML.Tokenizers` `BertOptions.RemoveNonSpacingMarks` defaults to `false`. Uncased BERT strips accents when lowercasing, so this is an **option default**. Design review dotnet/machinelearning#7281 suggests the default is intentional. | `mltokenizers/`: `RemoveNonSpacingMarks = true` yields `cafe cr ##eme br ##ule ##e …`, identical to HF |
| Word after a bare `\n` or `\t` glued on as `##…` | ElBruno | **Defect in `Microsoft.ML.Tokenizers` `BertTokenizer`**: `\n` and `\t` are removed rather than treated as whitespace. No option changes it. Present in 2.0.0 and 3.0.0-preview.26457.2. No existing issue found. | `one\nline` → `one ##line`; HF gives `one line` |
| Emoji dropped instead of `[UNK]` | ElBruno | Same package, same versions. Minor (0.99696). | `🚀` → `[CLS] [SEP]`; HF gives `[CLS] [UNK] [SEP]` |
| No lowercasing, no punctuation splitting, no accent stripping, no CJK splitting | LMSupply | **LMSupply 0.68.0 builds a bare `WordPieceTokenizer`** (`LMSupply.Text.Core/TokenizerFactory.cs`, `WordPieceTokenizer.Create(vocabStream)`), which skips BERT's basic tokenization entirely. Fixed in 0.70.0 after iyulab/lm-supply#12. | `The QUICK … didn't … dog?!` → `[UNK] [UNK] … didn ##' ##t … dog ##? ##!` |
| Truncates at 512, not 256 | SK, ElBruno, LMSupply | Library default `MaximumTokens` / `MaxSequenceLength` = 512, against the model's `max_seq_length` of 256. A **policy difference**, not a defect. | `over_256` agrees with the 512 reference at 1.000000 (SK, ElBruno) |
| Accents wrong under `InvariantGlobalization=true` | SK | Under invariant globalization FastBertTokenizer does not strip accents. The mechanism is presumably Unicode normalisation under invariant mode, but that was not verified. **Platform-dependent behaviour**, and invariant mode is common in container images. | `results/invariant-globalization/`; reproduced exactly by "no accent stripping" |

Token-level evidence: `results/token-diff-elbruno.json`, `results/token-diff-lmsupply.json`,
`results/attribution.json`.

## Predictions, scored

| id | prediction | outcome |
|---|---|---|
| C1 | reference ≡ sentence-transformers, ≥ 0.99999 | **met** (min 0.99999996) |
| SK1 | agrees on all texts ≤ 256 tokens, incl. accents and CJK | **met** under default globalization; not anticipated: fails `accents` under invariant globalization |
| SK2 | `over_256` diverges at 256 (0.95–0.99), agrees at 512 | **met** (0.986514 / 1.000000) |
| SK3 | norm ≠ 1 | **met** |
| SK4 | batch ≡ single | **met** |
| SK5 | `empty` agrees | **met** |
| LM1 | loads `onnx/model.onnx` | **met** |
| LM2 | agrees on the six ASCII texts | **refuted**: 4 of 6; `case_punct` 0.671828, `passage` 0.420103 |
| LM3 | `accents` diverges | **met**, for a broader reason than predicted |
| LM4 | `cjk_emoji` may diverge | diverges, 0.517515 |
| LM5 | `over_256` diverges at 256, agrees at 512 | **half refuted**: 0.415340 at 512 |
| LM6 | batch ≡ single | **met** |
| EB1 | loads `onnx/model.onnx` | **met** |
| EB2 | agrees on ASCII texts | **refuted** on `whitespace` (0.936251); met on the rest |
| EB3 | `accents` agrees | **refuted** (0.333027) |
| EB4 | `cjk_emoji` agrees | **refuted** (0.996960) |
| EB5 | `over_256` diverges at 256, agrees at 512 | **met** |
| EB6 | norm ≠ 1 | **met** |
| EB7 | batch ≡ single | **met** |

Of the 19 uninformed predictions, 13 were met, 1 was half refuted, 4 were refuted, and 1
was open (LM4). Every refutation came from tokenization, which reading the libraries'
pooling code could not have caught.

## `Microsoft.ML.Tokenizers` under invariant globalization

_Added 2026-09-21. Predictions MT1–MT7 in [`PREDICTIONS.md`](PREDICTIONS.md) were committed
in this repository before the probe was written. Package 2.0.0, .NET 10.0.12, macOS arm64.
Full tables: [`results/mltokenizers/accent-summary.md`](results/mltokenizers/accent-summary.md)
and, over all 49 inputs of `tokenizer-probes.json`,
[`results/mltokenizers/probes-summary.md`](results/mltokenizers/probes-summary.md)._

`BertTokenizer.Create(vocab, options)` on its own, with no library around it, checked by
token id against Hugging Face and scored by cosine after embedding both id sequences with
the reference pooling. `"\u00e9".Normalize(FormD)` returned 2 code units under ICU and 1
under invariant mode, recorded by the probe process itself.

| probe | ICU, default | ICU, `RemoveNonSpacingMarks` | invariant, default | invariant, `RemoveNonSpacingMarks` |
|---|--:|--:|--:|--:|
| `accents` (precomposed) | **0.333027** | = | **0.333027** | **0.333027** |
| `accents_nfd` (decomposed) | **0.333027** | = | **0.333027** | = |
| `accents_upper` | **0.089375** | = | **0.089375** | **0.089375** |
| `french` | **0.545866** | = | **0.545866** | **0.545866** |
| `spanish` | **0.502484** | = | **0.502484** | **0.502484** |
| `german` | **0.572320** | = | **0.572320** | **0.572320** |
| `hangul` | **0.458831** | **0.458831** | **0.458831** | **0.458831** |
| `ascii` | = | = | = | = |

`=` means token ids identical to Hugging Face's. Under invariant globalization,
`RemoveNonSpacingMarks = true` produces exactly the ids of the default options on every
precomposed probe: the option silently does nothing, and nothing throws. It still works on
text that arrives already decomposed, because the mark test (`CharUnicodeInfo`) does not
depend on ICU; only the decomposition does. The `accents` value, 0.333027, is the value
measured for ElBruno and for SK under invariant mode. For SK this is consistent with the
same mechanism, but SK's tokenizer was not read or instrumented here.

| id | prediction | outcome |
|---|---|---|
| MT1 | ICU + option: every Latin-accent probe matches | **met** |
| MT2 | invariant + option: precomposed probes fail, ids identical to option off, `accents` 0.333027, no exception | **met** |
| MT3 | invariant + option: `accents_nfd` matches | **met** |
| MT4 | `hangul` fails in all four, including ICU + option | **met**: `한국어` → `[UNK]`, where Hugging Face gives conjoining jamo; consistent with the closing `Normalize(FormC)` recomposing them |
| MT5 | option off: no accented probe matches in either mode | **met** |
| MT6 | `ascii` matches in all four | **met** |
| MT7 | over `tokenizer-probes.json`, ICU and invariant differ on `accents` and `hangul` only | **refuted in part**: they differ on `accents` only. `hangul` fails identically in both modes, which MT4 itself implied; the two predictions contradicted each other |

## A harness error, caught and kept

The first run set `InvariantGlobalization=true` in `Directory.Build.props`, a habit rather
than a decision. It made SK fail on `accents` at 0.333027, exactly ElBruno's value. That
coincidence prompted the re-run under .NET's default globalization, where SK is exact.
The invariant-mode results are kept in `results/invariant-globalization/`, because
container images commonly run that way.

## Skipped

- **Overfit** (3,731 downloads), **OnnxTextEmbeddings.NET** (1,154) and **Vectantic** (160):
  outside the top three by NuGet downloads.
  - Overfit loads safetensors, not ONNX, so it would need a different model artifact.
  - OnnxTextEmbeddings.NET's MiniLM support is unverified; its defaults target a 600M model.
  - Vectantic's code shows a likely `bge-small-en-v1.5` pooling defect, read from source and
    not measured, but MiniLM is the only model this harness checks.

## Reproduce

From the repository root, in bash: `$PY` relies on word splitting, which zsh does not do.
`model/fetch.sh` downloads the pinned `model.onnx` and checks its sha256 against
`results/reference.json`. ElBruno and LMSupply download the model themselves into
`.cache/`.

```sh
bash model/fetch.sh
PY="uv run --no-project --python 3.12 --with numpy==2.2.5 --with onnxruntime==1.22.0 --with tokenizers==0.21.1"
$PY reference/reference.py --model model/model.onnx --vocab model/vocab.txt --texts texts.json --out results/reference.json
uv run --no-project --python 3.12 --with sentence-transformers==6.0.1 --with numpy \
  reference/validate_st.py --reference results/reference.json --texts texts.json --out results/control-sentence-transformers.json

# .NET default (ICU) globalization
dotnet run --project sk -c Release -- texts.json model/model.onnx model/vocab.txt results/sk.json
dotnet run --project elbruno -c Release -- texts.json .cache/elbruno results/elbruno.json
dotnet run --project lmsupply -c Release -- texts.json .cache/lmsupply results/lmsupply.json
python3 compare.py results/reference.json results/{sk,elbruno,lmsupply}.json > results/summary.md

# Invariant globalization, the configuration of chiseled, Alpine and Native AOT images
I=results/invariant-globalization
dotnet run --project sk -c Release -p:InvariantGlobalization=true -- texts.json model/model.onnx model/vocab.txt $I/sk.json
dotnet run --project elbruno -c Release -p:InvariantGlobalization=true -- texts.json .cache/elbruno $I/elbruno.json
dotnet run --project lmsupply -c Release -p:InvariantGlobalization=true -- texts.json .cache/lmsupply $I/lmsupply.json
python3 compare.py results/reference.json $I/{sk,elbruno,lmsupply}.json > $I/summary.md

# Token-level attribution: each library's own ids re-embedded, and tokenizer variants
E=.cache/elbruno/sentence-transformers_all-MiniLM-L6-v2
L=.cache/lmsupply/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/main/onnx
dotnet run --project elbruno -c Release -- tokens texts.json $E results/tokens-elbruno.json
dotnet run --project lmsupply -c Release -- tokens texts.json $L results/tokens-lmsupply.json
for lib in elbruno lmsupply; do
  $PY reference/verify_tokens.py --model model/model.onnx --vocab model/vocab.txt --reference results/reference.json \
    --results results/$lib.json --tokens results/tokens-$lib.json --out results/token-diff-$lib.json
done
$PY reference/attribute.py --model model/model.onnx --vocab model/vocab.txt --texts texts.json \
  --results $I/sk.json results/elbruno.json results/lmsupply.json --out results/attribution.json

# LMSupply after the fix; delete lmsupply/bin and lmsupply/obj first
dotnet run --project lmsupply -c Release -p:LmSupplyVersion=0.70.0 -- texts.json .cache/lmsupply results/lmsupply-0.70.0.json

# Microsoft.ML.Tokenizers BertTokenizer on its own
dotnet run --project mltokenizers -c Release -- model/vocab.txt

# ... and by token id against Hugging Face, under both globalization modes
O=results/mltokenizers
$PY reference/tokenizer_ids.py --vocab model/vocab.txt --probes mltokenizers/accent-probes.json --out $O/accent-ids.json
dotnet run --project mltokenizers -c Release -- ids model/vocab.txt mltokenizers/accent-probes.json $O/accent-icu.json
dotnet run --project mltokenizers -c Release -p:InvariantGlobalization=true -- ids model/vocab.txt mltokenizers/accent-probes.json $O/accent-invariant.json
$PY reference/score_ids.py --model model/model.onnx --hf $O/accent-ids.json \
  --runs $O/accent-icu.json $O/accent-invariant.json --out $O/accent-scores.json > $O/accent-summary.md
dotnet run --project mltokenizers -c Release -- ids model/vocab.txt tokenizer-probes.json $O/probes-icu.json
dotnet run --project mltokenizers -c Release -p:InvariantGlobalization=true -- ids model/vocab.txt tokenizer-probes.json $O/probes-invariant.json
$PY reference/score_ids.py --model model/model.onnx --hf results/tokenizer-ids.json \
  --runs $O/probes-icu.json $O/probes-invariant.json --out $O/probes-scores.json > $O/probes-summary.md
```
