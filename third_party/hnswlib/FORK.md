# dart_hnsw fork of hnswlib

This is a vendored copy of [nmslib/hnswlib](https://github.com/nmslib/hnswlib) with
modifications for the `dart_hnsw` Dart package.

## Changes from upstream

### `hnswalg.h`

`saveIndex(const std::string &location)` and `loadIndex(const std::string &location, ...)`
have been replaced with stream-based overloads that accept `std::ostream` / `std::istream`
instead of file paths. This removes the filesystem dependency from the serialization layer,
enabling pure in-memory checkpoint/load without touching disk.

The binary format is **unchanged** — the same bytes are written and read, just through
`std::stringstream` instead of `std::ofstream`/`std::ifstream`. Index files produced by
upstream hnswlib are fully compatible.

### `bruteforce.h`

Same stream-based replacement for `saveIndex` / `loadIndex`.

## Upstream

- Repository: https://github.com/nmslib/hnswlib
- License: Apache 2.0 (see `LICENSE` in this directory)
