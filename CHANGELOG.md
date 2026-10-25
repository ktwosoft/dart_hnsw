# Changelog

## 0.4.0

- Add a direct wasm32 backend for the existing synchronous `HnswIndex` API,
  with asynchronous web initialization and configurable JS/Wasm asset URLs.
- Preserve native build hooks and isolate sharing; document web sharing and
  checkpoint ABI boundaries, and validate both Dart JS and Dart Wasm in Chrome.

## 0.3.0

- **Breaking change:** renamed `HnswIndex.create()` to `HnswIndex.open()`.
- `open()` is now the only handle-creating operation. `load()` is an instance method that replaces an existing shared context.
- Added `generation` and generation-required `load()`, `checkpoint()`, and `clear()` operations.
- Added `clear()` to reset a shared context without invalidating existing handles.
- Added handle-owned `acquireExclusiveMutationAccess()` / `releaseExclusiveMutationAccess()` controls for stable checkpoint-and-shutdown flows across isolates.
- Added finite-vector validation and safety limits for dimensions, capacity, batch size, search breadth, and checkpoint size.
- Enforced `m >= 2` so every accepted index configuration can be checkpointed and loaded.
- Corrected cosine distances to return `1 - cosine similarity`.
- Added a multi-isolate example where worker isolates open, write, and close a shared index.
- Rejected checkpoints whose serialized graph layout would exceed the native allocation limit during load.
- Locked graph-pointer checks during atomic replacement and clear operations to remove concurrent native pointer races.
- Added fair query/upsert operation groups so searches can run together and hnswlib additions can run together, while the two groups and all administrative calls remain mutually exclusive.
- Hardened the vendored hnswlib random engines for concurrent inserts and updates.
- Removed implicit sidecar loading from the current working directory.
- Added replacement, generation-conflict, clear, and input-validation regression coverage.
- Replaced individual element, deletion, capacity, and EF metadata getters with
  one consistent `HnswStats` snapshot acquired under a single administrative
  lock;

## 0.2.0

- Bumped ABI version to 2. - Breaking Change
- Removed all filesystem dependencies. The index now lives entirely in native memory.
- `checkpoint()` now returns `Uint8List` instead of writing to disk. The caller is responsible for storing the bytes.
- `HnswIndex.open(path: ...)` replaced by `HnswIndex.create(collectionName: ...)` and `HnswIndex.load(collectionName: ..., bytes: ...)`.
- The binary format is unchanged — compatible with hnswlib's native `saveIndex`/`loadIndex` format.
- Removed `path` parameter from all APIs. Collection names replace file paths for shared context identification.
- Removed process safety test and `_process_child.dart` — no longer applicable without file locking.
- Removed `HnswConcurrentProcessException` — no more cross-process file locking.
- Updated all tests to use the new in-memory API.
- Updated README and example to demonstrate the new checkpoint/load workflow.
- Renamed the package example to `example/example.dart` so pub.dev detects it.

## 0.1.2

- Fix README & CHANGELOG version reference to point to latest.

## 0.1.1

- Add dartdoc comments to all public API elements to improve pub.dev documentation score.
- Add `example: example/basic.dart` to pubspec.yaml.
- Update README with native efficiency focus and fix duplicate sections.

## 0.1.0

- Initial release of `dart_hnsw`.
- Added Dart Native bindings for `hnswlib` with `l2`, `innerProduct`, and `cosine` metrics.
- Added explicit `checkpoint()` persistence and shared in-process native index management.
- Added cross-process locking for index ownership.
- Added native-assets support for `dart run` and `dart test`.
- Added compiled executable support via sidecar shared-library fallback for `dart compile exe`.
- Added macOS, Linux, and Windows support declarations.
