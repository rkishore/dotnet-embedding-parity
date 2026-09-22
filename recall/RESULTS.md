# Retrieval recall: results

_Run 2026-09-21 on macOS arm64 (Apple M5 Pro), onnxruntime 1.22.0, numpy 2.2.5, tokenizers
0.21.1. The design and the predictions were committed before any code
([`PREDICTIONS.md`](PREDICTIONS.md)). Per-corpus output is in [`results/`](results/)._

## Bottom line

**For a developer using the default English model on French or Spanish text, a tokenizer
that does not strip accents costs about a third, or a fifth, of recall@10.** In German
the loss is too small to measure with 305 queries. In English it is under half a percent.

| corpus | dev queries | passages tokenized differently | recall@10, correct | recall@10, unstripped | relative change (95% interval) |
|---|--:|--:|--:|--:|--:|
| MIRACL French | 343 | 96.7% | 0.4086 | 0.2636 | **−35.5%** (−43.0% to −27.5%) |
| MIRACL Spanish | 648 | 94.7% | 0.2821 | 0.2212 | **−21.6%** (−27.7% to −15.3%) |
| MIRACL German | 305 | 86.9% | 0.2785 | 0.2757 | −1.0% (−9.4% to +7.8%) |
| MS MARCO, all dev | 6,980 | 23.0% | 0.9133 | 0.9091 | −0.5% (−0.7% to −0.2%) |
| MS MARCO, accented-relevant subset | 1,641 | — | 0.8929 | 0.8753 | −2.0% (−3.0% to −0.9%) |

Each corpus is subsampled to 300,000 passages: every judged passage, plus random
distractors with seed 20260921. Before scoring, every run checked that the correct arm
reproduces `results/reference.json`'s token ids and that the unstripped arm reproduces
ElBruno's `accents` vector (cosine 1.000000).

## Predictions, scored

| id | prediction | outcome |
|---|---|---|
| R1 | French: Δ ≤ −10% | **met**: −35.5% (−43.0% to −27.5%) |
| R2 | Spanish: Δ ≤ −5% | **met**: −21.6% (−27.7% to −15.3%) |
| R3 | German: Δ ≤ −5% | **refuted**: −1.0%. The interval, −9.4% to +7.8%, is wide enough to hold both a 5% drop and no change, so this is a failure to show a drop, not evidence of none |
| R4 | French drop is the largest of the three | **met** |
| R5 | MS MARCO, all dev: \|Δ\| < 2% | **met**: −0.5%. The interval excludes zero, so the effect is small but real |
| R6 | MS MARCO subset: Δ ≤ −2%, subset expected small | **refuted, narrowly**: −1.97% against a −2% threshold, and the interval (−3.0% to −0.9%) straddles it. The subset was not small: 1,641 of 6,980 queries, for the reason below |

Of six predictions, four were met and two were refuted.

## What was not expected

**MS MARCO's accented text is mostly mojibake.** 23% of the English passages tokenize
differently between the arms. The most frequent characters responsible are `â`, `Â` and
`Ã`: UTF-8 text decoded as Latin-1, so that `’` appears as `â€™`. The correct arm strips
these to `a` and the unstripped arm turns the word into `[UNK]`. So the English subset
measures mostly how each arm handles encoding damage, not accented English words.

**Every Spanish dev query tokenizes differently.** MIRACL's Spanish queries are questions,
and the interrogatives carry accents (`¿Cómo`, `¿Cuáles`, `qué`). The unstripped arm turns
each of them into `[UNK]`.

**German is the outlier.** 87% of German passages tokenize differently, yet recall barely
moves. A post-hoc count shows how much text each arm loses: the share of tokens that are
`[UNK]` in a 20,000-passage sample of each subsample
([`unk_share.py`](unk_share.py)):

| corpus | correct | unstripped |
|---|--:|--:|
| MIRACL French | 0.01% | 9.41% |
| MIRACL Spanish | 0.01% | 6.29% |
| MIRACL German | 0.01% | 3.66% |
| MS MARCO | 0.00% | 0.68% |

That order is the order of the recall drops. It is consistent with the loss scaling with
the share of words that lose their tokens, but this count was not predicted, and three
corpora do not establish a trend.

## Caveats

- One model, the English all-MiniLM-L6-v2, is the realistic default but a weak one for
  these languages; the baselines are 0.28–0.41. A multilingual model would have different
  baselines, and was not tested.
- Recall is measured on a 300,000-passage subsample, not the full corpora of 10–16 million
  passages. Absolute recall on the full corpus would be lower. The relative change between
  arms is what the design compares, and both arms use the same subsample.
- The unstripped arm reproduces ElBruno's and SK-under-invariant's accent handling, which
  `run.py` checks. It does not reproduce their other defects, such as ElBruno's
  newline gluing, so it isolates the accent effect.

## Reproduce

See [`README.md`](README.md). The runs took 20–85 minutes per corpus on the machine above.
Part of that was time the laptop spent asleep.
