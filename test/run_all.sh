#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dart_hnsw_exe_smoke.XXXXXX")"

cleanup() {
  rm -rf "$OUT_DIR"
}
trap cleanup EXIT

cd "$ROOT_DIR"

NATIVE_TEST_FLAGS=(-std=c++17 -pthread)
if [[ "${DART_HNSW_TSAN:-0}" == "1" ]]; then
  NATIVE_TEST_FLAGS+=(-fsanitize=thread -fno-omit-frame-pointer -g)
fi
"${CXX:-c++}" "${NATIVE_TEST_FLAGS[@]}" \
  -Inative \
  -Ithird_party/hnswlib/hnswlib \
  test/native_concurrency_test.cpp \
  -o "$OUT_DIR/native_concurrency_test"
"$OUT_DIR/native_concurrency_test"

dart analyze
dart test
dart build cli --target bin/basic_example.dart --output "$OUT_DIR"

(
  cd "$OUT_DIR/bundle"
  ./bin/basic_example >/dev/null
)

echo "All checks passed."
