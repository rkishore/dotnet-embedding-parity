#!/usr/bin/env bash
# Downloads model.onnx for sentence-transformers/all-MiniLM-L6-v2 at the revision every
# committed result was produced with, and refuses to keep it unless its sha256 matches
# the one recorded in results/reference.json.
set -euo pipefail

REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41
URL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/$REVISION/onnx/model.onnx"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="$here/model.onnx"
expected="$(sed -n 's/.*"model_sha256": *"\([0-9a-f]\{64\}\)".*/\1/p' "$here/../results/reference.json")"
if [ -z "$expected" ]; then
    echo "fetch.sh: no model_sha256 found in results/reference.json" >&2
    exit 1
fi

sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

if [ -f "$out" ] && [ "$(sha256 "$out")" = "$expected" ]; then
    echo "model/model.onnx already present, sha256 $expected"
    exit 0
fi

curl --fail --location --silent --show-error --output "$out.partial" "$URL"
actual="$(sha256 "$out.partial")"
if [ "$actual" != "$expected" ]; then
    rm -f "$out.partial"
    echo "fetch.sh: sha256 MISMATCH for $URL" >&2
    echo "  expected $expected (results/reference.json)" >&2
    echo "  actual   $actual" >&2
    exit 1
fi
mv "$out.partial" "$out"
echo "model/model.onnx at revision ${REVISION:0:7}, sha256 $actual"
