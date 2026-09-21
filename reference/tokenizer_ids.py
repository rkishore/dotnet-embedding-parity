"""Generate the Hugging Face token-id fixture the tokenizer tests check against.

Token-id equality is a stricter test than cosine: two different tokenizations can
land close in embedding space, and the pooled vector hides which token moved. It
also needs no model file, so the tests that use this fixture run in a checkout
without model.onnx, where the cosine oracle tests skip.

The probe set is deliberately separate from texts.json. That file is also the
competitor comparison's probe set, so adding rows to it would leave RESULTS.md's
tables for the other libraries missing those rows until every library is re-run.

Recipe: tokenizers.BertWordPieceTokenizer(vocab, lowercase=True), the same
construction reference.py uses, so the ids here and the vectors in reference.json
describe one tokenizer.

Usage: tokenizer_ids.py --vocab V --probes P --out O
"""
import argparse
import json

import tokenizers
from tokenizers import BertWordPieceTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    for a in ("--vocab", "--probes", "--out"):
        ap.add_argument(a, required=True)
    args = ap.parse_args()

    probes = json.load(open(args.probes, encoding="utf-8"))["probes"]
    tok = BertWordPieceTokenizer(args.vocab, lowercase=True)

    out = []
    for probe in probes:
        encoded = tok.encode(probe["text"])
        out.append({
            "id": probe["id"],
            "text": probe["text"],
            "token_ids": encoded.ids,
            "tokens": encoded.tokens,
        })

    json.dump(
        {
            "recipe": "tokenizers BertWordPieceTokenizer(lowercase=True), add_special_tokens default",
            "versions": {"tokenizers": tokenizers.__version__},
            "probes": out,
        },
        open(args.out, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )
    print(f"{len(out)} probes -> {args.out}")


if __name__ == "__main__":
    main()
