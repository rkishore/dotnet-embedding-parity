# Predictions: competitor oracle check

> Recorded 2026-09-17, before any probe ran, in a private repository; reproduced here
> without a later addendum about the author's own library. This repository's history
> does not prove the ordering, so the date is stated rather than demonstrated.

_Recorded 2026-09-17, before any probe or reference was run. The probe projects had been
compiled but not executed. Scored in [`RESULTS.md`](RESULTS.md) after the run, including
the ones that are wrong._

"Agrees" means cosine ≥ 0.9999 against the reference truncated at 256 tokens, the
model's `max_seq_length`. That is the same bar as the benchmark's oracle gate. Each
prediction carries a confidence and the code it rests on (read from source, cited in
[`SOURCES.md`](SOURCES.md)).

## Control

**C1. The numpy reference agrees with sentence-transformers 6.0.1 to cosine ≥ 0.99999 on
all ten texts**, including `empty` and `over_256`. High confidence: same model revision
and same WordPiece vocabulary, and mean pooling plus normalise is what `modules.json`
declares. The expected differences are ONNX-vs-PyTorch float rounding. **If this fails,
no library result is read.**

## Semantic Kernel `Connectors.Onnx` 1.80.1-alpha, default options

Defaults: Mean pooling, `NormalizeEmbeddings = false`, `MaximumTokens = 512`,
FastBertTokenizer, lowercase.

- **SK1.** Agrees on every text ≤ 256 tokens, including `accents` and `cjk_emoji`. High
  confidence for ASCII: its own integration test pins all-MiniLM-L6-v2 against
  sentence-transformers at 8e-7. Medium for accents and CJK, which that test does not
  cover.
- **SK2.** `over_256` does **not** agree at 256, because it truncates at 512. It does
  agree with the 512 reference. High confidence. Guessed magnitude at 256: cosine
  0.95–0.99, low confidence.
- **SK3.** L2 norm ≠ 1, because normalisation is off by default. Cosine is unaffected.
  High confidence.
- **SK4.** Batch equals single to ≥ 0.99999, since the multi-text call loops single runs.
  High confidence.
- **SK5.** `empty` returns a vector that agrees. Low confidence; it may throw.

## LMSupply.Embedder 0.68.0, repo id `sentence-transformers/all-MiniLM-L6-v2`, default options

Defaults: Mean pooling, normalise on, lowercase on, `MaxSequenceLength = 512`, a
`Microsoft.ML.Tokenizers` WordPiece model, file auto-discovery by hardware tier.

- **LM1.** It loads `onnx/model.onnx` (sha256 `6fd5d72f…`). **Low–medium confidence.**
  The repo also contains O1–O4 and quantised variants. If discovery picks a quantised
  file, every text drops to roughly 0.97–0.995, and that is a model-selection finding,
  not a pooling one.
- **LM2.** If LM1 holds, it agrees on the ASCII texts ≤ 256 tokens: `short`, `medium`,
  `query`, `case_punct`, `whitespace`, `passage`. Medium confidence; masked mean pooling
  reads as correct.
- **LM3.** `accents` does **not** agree, because nothing found in its tokenizer path
  strips accents after lowercasing. **Low confidence**; this is the prediction most
  likely to be wrong.
- **LM4.** `cjk_emoji`: no prediction beyond "may diverge"; not enough evidence either way.
- **LM5.** `over_256` does not agree at 256 but does at 512. High confidence.
- **LM6.** Batch equals single to ≥ 0.99999. Medium confidence: the batch path pads to
  the longest input but runs one sequence at a time with its mask.

## ElBruno.LocalEmbeddings 1.6.1, default options

Defaults: downloads `onnx/model.onnx`, `PreferQuantized = false`, every input padded to
a fixed 512, masked mean, `NormalizeEmbeddings = false`, `Microsoft.ML.Tokenizers`
`BertTokenizer` with default options.

- **EB1.** It loads `onnx/model.onnx`, identical to the reference. High confidence.
- **EB2.** Agrees on the ASCII texts ≤ 256 tokens. Medium–high confidence: padding to 512
  is masked in both attention and pooling, so it should cost compute, not correctness.
  Expected ≥ 0.99999.
- **EB3.** `accents` agrees. Low confidence; it depends on `BertTokenizer`'s default
  accent handling, which was not checked.
- **EB4.** `cjk_emoji` agrees. Low confidence, for the same reason.
- **EB5.** `over_256` does not agree at 256 but does at 512. High confidence.
- **EB6.** L2 norm ≠ 1 (normalise off by default). High confidence.
- **EB7.** Batch equals single to ≥ 0.99999. High confidence: one padded `Run` either way.

## What would change the plan

- **Any divergence on a short ASCII text** (`short`, `medium`, `query`) would be a real
  correctness defect. Draft a bug report.
- **Divergence only on `over_256`** is a truncation-policy difference: 512 against the
  model's declared 256. Report it as a documented behaviour difference, not a bug, unless
  the library claims sentence-transformers parity.
- **Divergence only on `accents` or `cjk_emoji`** is a tokenizer defect, and is reportable
  because the model's `tokenizer_config.json` specifies the behaviour.

## `Microsoft.ML.Tokenizers` under invariant globalization

_Recorded 2026-09-21 in this repository, committed before the probe below was written or
run. Not blind: the SK invariant-mode result above, and `mltokenizers/`'s ICU output for
`accents`, were already known. The mechanism is read from `BertNormalizer.cs` at the
commit in [`SOURCES.md`](SOURCES.md): with `RemoveNonSpacingMarks = true` it calls
`Normalize(FormD)` (line 38), drops `NonSpacingMark` characters (line 77), and always
finishes with `Normalize(FormC)` (line 116)._

Package 2.0.0, `BertTokenizer.Create(vocab, options)`, over
[`mltokenizers/accent-probes.json`](mltokenizers/accent-probes.json). "Matches" means
token ids identical to Hugging Face's `BertWordPieceTokenizer(lowercase=True)`. Four
configurations: `RemoveNonSpacingMarks` false or true, under ICU or with
`InvariantGlobalization=true`.

- **MT1.** Under ICU with `RemoveNonSpacingMarks = true`, every Latin-accent probe matches:
  `accents`, `accents_nfd`, `accents_upper`, `french`, `spanish`, `german`. High confidence.
- **MT2.** Under invariant globalization with `RemoveNonSpacingMarks = true`, the
  precomposed probes (`accents`, `accents_upper`, `french`, `spanish`, `german`) do
  **not** match, and their ids are identical to the same mode's ids with the option
  off: the option silently does nothing. `accents` scores 0.333027 end to end, the value
  measured for ElBruno and for SK under invariant mode. Nothing throws. High confidence:
  `Normalize(FormD)` returns its input unchanged in invariant mode, so `é` never becomes
  `e` + U+0301 and there is no mark to drop.
- **MT3.** Under invariant globalization with the option on, `accents_nfd` **does**
  match. The input already carries separate combining marks, and the `NonSpacingMark`
  test uses `CharUnicodeInfo`, which ships its own data and works in invariant mode.
  Medium confidence. If it holds, the failure depends on the input's normalisation form
  as well as on the deployment image.
- **MT4.** `hangul` does not match in any of the four configurations, including ICU with
  the option on: `FormD` splits the syllables into conjoining jamo, which are letters and
  are kept, and the closing `Normalize(FormC)` recomposes them into syllables that are not
  in the vocabulary. Medium–low confidence.
- **MT5.** With the option off, no accented probe matches in either mode. High confidence;
  this is ElBruno's configuration.
- **MT6.** `ascii` matches in all four configurations. High confidence.
- **MT7.** Over the 49 inputs of `tokenizer-probes.json`, ICU and invariant mode produce
  identical ids for every probe except those containing a character with a canonical
  decomposition: `accents` and `hangul`. Medium confidence.
