# Oracle check results

Reference: tokenizers BertWordPieceTokenizer(lowercase=True) + onnxruntime + masked mean + L2; {'onnxruntime': '1.22.0', 'tokenizers': '0.21.1', 'numpy': '2.2.5'}; model sha256 `6fd5d72fe458`.
Cosine below 0.9999 is **bold**.

## Semantic Kernel Connectors.Onnx 1.80.1-alpha+d8ec44919265b3641b7898a02038cbfe14590ee5

- ONNX Runtime: 1.23.2+a83fc4d58cb48eb68890dd689f94f28288cf2278
- Model file: `model/model.onnx`, sha256 `6fd5d72fe458` (identical to reference)
- Configuration: {'PoolingMode': 'Mean', 'NormalizeEmbeddings': 'False', 'MaximumTokens': '512', 'CaseSensitive': 'False', 'UnicodeNormalization': 'FormD'}

| text | tokens@256 | cos vs ref@256 | cos vs ref@512 | L2 norm | batch vs single |
|---|--:|--:|--:|--:|--:|
| short | 4 | 1.000000 | 1.000000 | 5.7268 | 1.000000 |
| medium | 11 | 1.000000 | 1.000000 | 6.3745 | 1.000000 |
| query | 10 | 1.000000 | 1.000000 | 6.5268 | 1.000000 |
| case_punct | 17 | 1.000000 | 1.000000 | 4.7432 | 1.000000 |
| accents | 14 | 1.000000 | 1.000000 | 5.2538 | 1.000000 |
| cjk_emoji | 13 | 1.000000 | 1.000000 | 6.1361 | 1.000000 |
| whitespace | 6 | 1.000000 | 1.000000 | 5.9443 | 1.000000 |
| passage | 109 | 1.000000 | 1.000000 | 2.2754 | 1.000000 |
| over_256 | 256 | **0.986514** | 1.000000 | 1.9885 | 1.000000 |
| empty | 2 | 1.000000 | 1.000000 | 6.3657 | n/a |

## ElBruno.LocalEmbeddings 1.6.1+558380032734f9ce5a18be45c1937e11b27fa81a

- ONNX Runtime: 1.24.4+2d924974ef147392ced8409d36bd6d2e7fcc8a74
- Model file: `.cache/elbruno/sentence-transformers_all-MiniLM-L6-v2/model.onnx`, sha256 `6fd5d72fe458` (identical to reference)
- Configuration: {'ModelName': 'sentence-transformers/all-MiniLM-L6-v2', 'MaxSequenceLength': '512', 'NormalizeEmbeddings': 'False', 'PreferQuantized': 'False'}

| text | tokens@256 | cos vs ref@256 | cos vs ref@512 | L2 norm | batch vs single |
|---|--:|--:|--:|--:|--:|
| short | 4 | 1.000000 | 1.000000 | 5.7268 | 1.000000 |
| medium | 11 | 1.000000 | 1.000000 | 6.3745 | 1.000000 |
| query | 10 | 1.000000 | 1.000000 | 6.5268 | 1.000000 |
| case_punct | 17 | 1.000000 | 1.000000 | 4.7432 | 1.000000 |
| accents | 14 | **0.333027** | 0.333027 | 5.8958 | 1.000000 |
| cjk_emoji | 13 | **0.996960** | 0.996960 | 6.3427 | 1.000000 |
| whitespace | 6 | **0.936251** | 0.936251 | 6.4561 | 1.000000 |
| passage | 109 | 1.000000 | 1.000000 | 2.2754 | 1.000000 |
| over_256 | 256 | **0.986514** | 1.000000 | 1.9885 | 1.000000 |
| empty | 2 | 1.000000 | 1.000000 | 6.3657 | n/a |

## LMSupply.Embedder 0.68.0+085832769672793b0b86683e44cf5d0d32f17820

- ONNX Runtime: 1.30.0+f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7
- Model file: `None`, sha256 `unknown` (not hashed)
- Configuration: {'MaxSequenceLength': '512', 'NormalizeEmbeddings': 'True', 'PoolingMode': 'Mean', 'DoLowerCase': 'False', 'Provider': 'Auto', 'ActiveProviders': 'unknown'}

| text | tokens@256 | cos vs ref@256 | cos vs ref@512 | L2 norm | batch vs single |
|---|--:|--:|--:|--:|--:|
| short | 4 | 1.000000 | 1.000000 | 1.0000 | 1.000000 |
| medium | 11 | 1.000000 | 1.000000 | 1.0000 | 1.000000 |
| query | 10 | 1.000000 | 1.000000 | 1.0000 | 1.000000 |
| case_punct | 17 | **0.671828** | 0.671828 | 1.0000 | 1.000000 |
| accents | 14 | **-0.001147** | -0.001147 | 1.0000 | 1.000000 |
| cjk_emoji | 13 | **0.517515** | 0.517515 | 1.0000 | 1.000000 |
| whitespace | 6 | 1.000000 | 1.000000 | 1.0000 | 1.000000 |
| passage | 109 | **0.420103** | 0.420103 | 1.0000 | 1.000000 |
| over_256 | 256 | **0.396909** | 0.415340 | 1.0000 | 1.000000 |
| empty | 2 | 1.000000 | 1.000000 | 1.0000 | n/a |

