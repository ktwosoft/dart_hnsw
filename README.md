# dart_hnsw

Cross-platform Dart Native bindings for [hnswlib](https://github.com/nmslib/hnswlib) with fast approximate nearest neighbor search, explicit persistence, and shared native index management.

---

## Features

- Approximate nearest neighbor search through native `hnswlib`
- Supports `l2`, `innerProduct`, and `cosine` metrics
- Explicit `checkpoint()` persistence
- Shared in-process access to one native index per canonical path
- Cross-process ownership locking
- Works with `dart run`, `dart test`, and `dart compile exe`

---

## Getting Started

### 1. Add dependency

```yaml
dependencies:
  dart_hnsw: ^0.1.0
```

### 2. Import the package

```dart
import 'package:dart_hnsw/dart_hnsw.dart';
```

### 3. Use the index

```dart
final index = HnswIndex.open(
  path: 'example.index',
  dimensions: 3,
  metric: HnswMetric.cosine,
);

index.upsert(1, [1, 0, 0]);
index.upsert(2, [0, 1, 0]);

final results = index.search([1, 0, 0], k: 1);
print(results.first.id);

index.checkpoint();
index.close();
```

---

## How Should I Run It

### `dart run`

Use this for normal development.

```sh
dart run bin/your_app.dart
```

The native library is built and loaded automatically through Dart hooks/code assets. No manual `cmake` step is required.

### `dart compile exe`

Use this when shipping a compiled CLI, but copy the shared library next to the executable.

```sh
dart compile exe bin/your_app.dart -o dist/your_app
```

Sidecar library names:

- macOS: `libdart_hnsw.dylib`
- Linux: `libdart_hnsw.so`
- Windows: `dart_hnsw.dll`

Typical packaging flow on macOS:

```sh
dart compile exe bin/your_app.dart -o dist/your_app
cp .dart_tool/lib/libdart_hnsw.dylib dist/
```

On Linux, copy `libdart_hnsw.so`. On Windows, copy `dart_hnsw.dll`.

At runtime, `dart_hnsw` resolves native code in this order:

1. Dart native-asset mapping, if available
2. `DART_HNSW_LIBRARY_PATH`, if set
3. the executable directory
4. the current working directory

Example override:

```sh
DART_HNSW_LIBRARY_PATH=/opt/dart_hnsw/libdart_hnsw.so ./your_app
```

---

## Persistence

- `close()` only releases the handle. It does not save.
- `checkpoint()` writes the raw `hnswlib` index file.
- Reopening a persisted index is the caller's responsibility: use the same `dimensions`, `metric`, and `allowReplaceDeleted` values that were used to create it.
- The package validates config only when multiple handles in the same process attach to the same active shared context.
- Reopening a file with mismatched config is unsupported and may fail or produce incorrect semantics.

---

## Supported Platforms

Validated platforms:

- macOS
- Linux
- Windows

Web is unsupported. Mobile targets are not declared yet.

---

## API Overview

### `HnswIndex`

| Method / Property | Description |
|-------|-------------|
| `HnswIndex.open(...)` | Open or create an index at a filesystem path |
| `upsert(id, vector)` | Insert or replace one vector |
| `upsertBatch(entries)` | Insert or replace multiple vectors |
| `search(query, {k})` | Approximate nearest neighbor search |
| `markDeleted(id)` | Mark a label deleted |
| `resize(newMaxElements)` | Increase index capacity |
| `setEfSearch(ef)` | Adjust search breadth |
| `checkpoint()` | Persist the current index state |
| `close()` | Release the native handle |
| `elementCount` | Total stored element count |
| `deletedCount` | Deleted entry count |
| `liveCount` | Convenience count of non-deleted entries |
| `maxElements` | Current capacity |
| `efSearch` | Current search breadth |
| `contextId` | Shared native context identifier |

### `HnswMetric`

- `HnswMetric.l2`
- `HnswMetric.innerProduct`
- `HnswMetric.cosine`

---

## Running Tests

```sh
dart test
```

For a full local smoke test, including `dart compile exe` plus sidecar packaging:

```sh
bash test/run_all.sh
```

---

## License

This package is licensed under Apache-2.0. Vendored `hnswlib` sources are included under their Apache-2.0 license in `third_party/hnswlib/LICENSE`.
