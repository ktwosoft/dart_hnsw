#!/usr/bin/env bash
set -euo pipefail
# Activate emsdk 4.0.23 before invoking; no SDK or network download at build time.
cd "$(dirname "$0")/.."
EMXX="${EMXX:-em++}"
if ! "$EMXX" --version | head -n 1 | grep -q '4.0.23'; then
  echo 'Expected Emscripten 4.0.23 (activate that emsdk version).' >&2
  exit 1
fi
OUT="${1:-build/web_assets}"
mkdir -p "$OUT"
"$EMXX" native/dart_hnsw_native.cpp -Ithird_party/hnswlib/hnswlib \
  -std=c++17 -O3 -fexceptions -sDISABLE_EXCEPTION_CATCHING=0 \
  -sMODULARIZE=1 -sEXPORT_NAME=createDartHnsw -sENVIRONMENT=web,worker \
  -sWASM_BIGINT=1 -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=2147483648 \
  -sEXPORTED_RUNTIME_METHODS='["HEAPU8","HEAPF32"]' \
  -sFILESYSTEM=0 -sASSERTIONS=1 \
  -sEXPORTED_FUNCTIONS='["_malloc","_free","_hnsw_abi_version","_hnsw_create","_hnsw_close","_hnsw_context_id","_hnsw_upsert_vector","_hnsw_acquire_exclusive_mutation_access","_hnsw_release_exclusive_mutation_access","_hnsw_registry_size","_hnsw_generation","_hnsw_stats","_hnsw_upsert_batch","_hnsw_search_knn","_hnsw_mark_deleted","_hnsw_resize","_hnsw_set_ef_search","_hnsw_clear","_hnsw_checkpoint","_hnsw_load","_hnsw_wasm_allocated_bytes"]' \
  -o "$OUT/hnsw.js"
