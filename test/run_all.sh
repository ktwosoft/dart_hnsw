#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dart_hnsw_exe_smoke.XXXXXX")"

cleanup() {
  rm -rf "$OUT_DIR"
}
trap cleanup EXIT

case "$(uname -s)" in
  Darwin) SIDE_CAR_NAME="libdart_hnsw.dylib" ;;
  Linux) SIDE_CAR_NAME="libdart_hnsw.so" ;;
  MINGW*|MSYS*|CYGWIN*|Windows_NT) SIDE_CAR_NAME="dart_hnsw.dll" ;;
  *) echo "Unsupported OS for compile smoke test" >&2; exit 1 ;;
esac

cd "$ROOT_DIR"

dart analyze
dart test
dart compile exe example/basic.dart -o "$OUT_DIR/basic_example"
cp ".dart_tool/lib/$SIDE_CAR_NAME" "$OUT_DIR/$SIDE_CAR_NAME"

(
  cd "$OUT_DIR"
  ./basic_example >/dev/null
)

echo "All checks passed."
