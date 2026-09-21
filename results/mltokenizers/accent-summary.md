# Microsoft.ML.Tokenizers BertTokenizer against Hugging Face

- ICU: package 2.0.0, .NET 10.0.12, `"\u00e9".Normalize(FormD).Length` = 2
- invariant: package 2.0.0, .NET 10.0.12, `"\u00e9".Normalize(FormD).Length` = 1

Each cell: ids equal to Hugging Face's (`=`) or not (`≠`), and the cosine between the
two id sequences embedded with the reference pooling.

| probe | ICU, default BertOptions | ICU, RemoveNonSpacingMarks = true | invariant, default BertOptions | invariant, RemoveNonSpacingMarks = true |
|---|--:|--:|--:|--:|
| accents | **≠ 0.333027** | = 1.000000 | **≠ 0.333027** | **≠ 0.333027** |
| accents_nfd | **≠ 0.333027** | = 1.000000 | **≠ 0.333027** | = 1.000000 |
| accents_upper | **≠ 0.089375** | = 1.000000 | **≠ 0.089375** | **≠ 0.089375** |
| french | **≠ 0.545866** | = 1.000000 | **≠ 0.545866** | **≠ 0.545866** |
| spanish | **≠ 0.502484** | = 1.000000 | **≠ 0.502484** | **≠ 0.502484** |
| german | **≠ 0.572320** | = 1.000000 | **≠ 0.572320** | **≠ 0.572320** |
| hangul | **≠ 0.458831** | **≠ 0.458831** | **≠ 0.458831** | **≠ 0.458831** |
| ascii | = 1.000000 | = 1.000000 | = 1.000000 | = 1.000000 |
