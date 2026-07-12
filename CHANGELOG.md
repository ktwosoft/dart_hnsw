# Changelog

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
