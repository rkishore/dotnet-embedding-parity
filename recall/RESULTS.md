# Retrieval recall: results

_Run 2026-09-21 on macOS arm64 (Apple M5 Pro), onnxruntime 1.22.0, numpy 2.2.5, tokenizers
0.21.1. The design and the predictions were committed before any code
([`PREDICTIONS.md`](PREDICTIONS.md)). Per-corpus output is in [`results/`](results/)._

## Bottom line

**With the default English model, a tokenizer that does not strip accents loses 14.5
points of recall@10 on French, from 0.409 to 0.264, and 6.1 points on Spanish, from 0.282
to 0.221.** Relative to those baselines, that is −35.5% and −21.6%. German is
inconclusive: the interval runs from a 9% loss to an 8% gain. English loses 0.4 points.

**The baselines are low.** An English model retrieves French, Spanish and German poorly
to begin with: the correct arm finds 28–41% of the relevant passages in its top 10. The
change is reported relative to that baseline, as the design fixed in advance, because a
given absolute loss is a larger share of a weak model's performance. The absolute numbers
are printed beside it so the relative figure can be checked against them.

| corpus | dev queries | passages tokenized differently | recall@10, correct | recall@10, unstripped | absolute change | relative change (95% interval) |
|---|--:|--:|--:|--:|--:|--:|
| MIRACL French | 343 | 96.7% | 0.4086 | 0.2636 | −14.50 points | **−35.5%** (−43.0% to −27.5%) |
| MIRACL Spanish | 648 | 94.7% | 0.2821 | 0.2212 | −6.08 points | **−21.6%** (−27.7% to −15.3%) |
| MIRACL German | 305 | 86.9% | 0.2785 | 0.2757 | −0.28 points | −1.0% (−9.4% to +7.8%), inconclusive |
| MS MARCO, all dev | 6,980 | 23.0% | 0.9133 | 0.9091 | −0.43 points | −0.5% (−0.7% to −0.2%) |
| MS MARCO, pre-registered subset ¹ | 1,641 | — | 0.8929 | 0.8753 | −1.76 points | −2.0% (−3.0% to −0.9%) |

A point is one percentage point of recall@10. The intervals come from the paired
bootstrap fixed in the design, and are computed for the relative change only.
¹ This subset does not measure accented text. See below.

Each corpus is subsampled to 300,000 passages: every judged passage, plus random
distractors with seed 20260921. Before scoring, every run checked that the correct arm
reproduces `results/reference.json`'s token ids and that the unstripped arm reproduces
ElBruno's `accents` vector (cosine 1.000000).

## Predictions, scored

| id | prediction | outcome |
|---|---|---|
| R1 | French: Δ ≤ −10% | **met**: −35.5% (−43.0% to −27.5%) |
| R2 | Spanish: Δ ≤ −5% | **met**: −21.6% (−27.7% to −15.3%) |
| R3 | German: Δ ≤ −5% | **refuted**: −1.0% (−0.28 points). The result is **inconclusive**, not "no effect": the interval, −9.4% to +7.8%, holds both a 5% drop and a gain. 305 queries cannot resolve an effect of the size predicted |
| R4 | French drop is the largest of the three | **met** |
| R5 | MS MARCO, all dev: \|Δ\| < 2% | **met**: −0.5%. The interval excludes zero, so the effect is small but real |
| R6 | MS MARCO subset: Δ ≤ −2%, subset expected small | **refuted, narrowly**: −1.97% against a −2% threshold, and the interval (−3.0% to −0.9%) straddles it. **The subset does not measure what R6 was designed to measure.** It was meant to isolate English queries whose relevant passage contains accented words. Every passage in it contains only mojibake, and none contains a genuine accented character (below). R6 is scored as recorded, but it says nothing about accented English text |

Of six predictions, four were met and two were refuted.

## What was not expected

**MS MARCO has no genuinely accented text, only mojibake.** 23% of the English passages
(68,981 of 300,000) tokenize differently between the arms, and every one of them contains
mojibake: UTF-8 decoded as Latin-1. `’` appears as `â` followed by the invisible control
characters U+0080 and U+0099 (48,604 times), `é` as `Ã©` and Greek `α` as `Î±`. The correct arm strips the accented Latin-1 letters in that debris to plain
letters, and the unstripped arm turns the word into `[UNK]`. A post-hoc count removed the
mojibake sequences and re-tokenized. **No passage still tokenized differently, and no dev
query had a relevant passage with a genuine accented character**
([`posthoc.py`](posthoc.py), [`results/posthoc.json`](results/posthoc.json)). So the
English control shows how each arm handles encoding damage. It cannot show what accent
stripping does for English text with real accents, and a genuine-accent subset of MS MARCO
would be empty.

**Every Spanish dev query tokenizes differently.** MIRACL's Spanish queries are questions,
and the interrogatives carry accents (`¿Cómo`, `¿Cuáles`, `qué`). The unstripped arm turns
each of them into `[UNK]`.

**German is inconclusive, and the text suggests why the effect may be smaller.** These
are observations made after the results, not findings. Two counts, both from the text
alone, set German apart from French and Spanish:

| corpus | queries with a word turned into `[UNK]` | `[UNK]` tokens per query | `[UNK]` share of passage tokens | recall@10, correct |
|---|--:|--:|--:|--:|
| MIRACL French | 58.0% | 0.87 | 9.41% | 0.409 |
| MIRACL Spanish | 100.0% | 1.93 | 6.29% | 0.282 |
| MIRACL German | 39.3% | 0.47 | 3.66% | 0.279 |
| MS MARCO | 0.0% | 0.00 | 0.68% | 0.913 |

German loses the fewest query terms and the smallest share of passage text of the three
languages. Its baseline, 0.279, is almost the same as Spanish's, 0.282, so a lower baseline
does not explain the difference from Spanish. Fewer affected terms is consistent with a
smaller effect, but three corpora do not establish that, and the German interval is too
wide to say how large the effect is. The query counts are in
[`results/posthoc.json`](results/posthoc.json). The passage share comes from a
20,000-passage sample of each subsample ([`unk_share.py`](unk_share.py)).

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
