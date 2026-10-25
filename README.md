# dart_hnsw

Native and browser Dart bindings for [hnswlib](https://github.com/nmslib/hnswlib) — fast approximate nearest neighbor search through native C++ code with shared in-process indexes, grouped operation locking, and multi-isolate concurrency.

---

## Features

- **Approximate nearest neighbor (ANN)** search through native `hnswlib`
- **Native C++ performance** — search and index operations run in native code via `dart:ffi`, not in the Dart VM. No pure-Dart ANN overhead.
- **Pure in-memory** — no filesystem dependency. The index lives entirely in native memory.
- **One shared native context** per collection name inside a Dart process
- **Grouped operation locking** so searches run together, upserts run together, and administrative calls stay exclusive
- **Symmetric multi-isolate access** without duplicating the same index in each isolate
- **Atomic replacement** — `load()` swaps a complete checkpoint into every live handle
- **Generation guards** — reject stale loads and clears instead of overwriting newer shared state
- **Exclusive mutation access** — one handle can safely checkpoint and shut down while other isolates keep searching
- **Input safety limits** for dimensions, capacity, search breadth, batches, and checkpoint payloads
- **Cross-platform native assets**: macOS, Linux, Windows, iOS, Android
- **WebAssembly backend**: the same C++ hnswlib and synchronous `HnswIndex` API on web

---

## Why This Package

Most local HNSW packages are optimized for single-isolate, pure-Dart usage. `dart_hnsw` exists for the case where multiple isolates in the same Dart process need to work against the same native index efficiently.

Native efficiency. Index construction, vector insertion, and nearest-neighbor search all run in C++ through `dart:ffi` — not in the Dart VM. This means:

- **No serialization overhead** — vectors are passed directly as native float arrays
- **No GC pressure** from large index structures — the entire graph lives off-heap in native memory
- **Shared memory** — one native context per collection name, shared across all isolates in the process
- **Zero-copy reads** — search traverses the native graph directly without marshalling the index into Dart

Core concurrency model:

- opening the same collection name from multiple isolates attaches them to one shared native context
- searches form one concurrent group and upserts form another concurrent group
- query and upsert groups do not overlap; fair phase handoff prevents either group from continuously bypassing the other
- deletes, resize, search-breadth changes, the consolidated `stats` snapshot, `load()`, `clear()`, and `checkpoint()` use the fully exclusive administrative tier
- queued administrative work takes priority when the active query or upsert phase drains
- `acquireExclusiveMutationAccess()` assigns one handle as the only mutation owner while searches continue

This lets you build isolate-heavy CLI or server workloads without copying one large HNSW index into every isolate separately.

### No filesystem dependency

The index lives entirely in native memory. `checkpoint()` serializes the index to a `Uint8List` that you can store anywhere — a file, a database BLOB, Redis, S3, etc. `load()` restores from bytes. The native layer has zero filesystem code, making it truly cross-platform (iOS, Android, macOS, Linux, Windows).

---

## Getting Started

### 1. Add dependency

```yaml
dependencies:
  dart_hnsw: ^0.4.0
```

### 2. Import the package

```dart
import 'package:dart_hnsw/dart_hnsw.dart';
```

### 3. Use the index

```dart
final index = HnswIndex.open(
  collectionName: 'my_collection',
  dimensions: 3,
  metric: HnswMetric.cosine,
);

index.upsert(1, [1, 0, 0]);
index.upsert(2, [0, 1, 0]);

final results = index.search([1, 0, 0], k: 1);
print(results.first.id);

final generation = index.generation;
final bytes = index.checkpoint(expectedGeneration: generation);

index.close();
```

### 4. Restore from bytes

```dart
// `bytes` comes from the application's storage or transport layer.
// dart_hnsw does not read or write files.
final index = HnswIndex.open(
  collectionName: 'my_collection',
  dimensions: 3,
  metric: HnswMetric.cosine,
);
index.load(bytes, expectedGeneration: index.generation);

final results = index.search([1, 0, 0], k: 1);
print(results.first.id);

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

This mode preserves the shared native context behavior for multiple isolates inside the same Dart process.

### Build a distributable CLI

Packages that use Dart build hooks, including `dart_hnsw`, must be compiled
with `dart build cli`. It produces a self-contained application bundle with
the executable in `bin/` and its native libraries in `lib/`.

```sh
dart build cli --target bin/your_app.dart --output dist
```

For example, on macOS the output is:

```text
dist/bundle/bin/your_app
dist/bundle/lib/libdart_hnsw.dylib
```

Ship the entire `dist/bundle/` directory. On Linux the library is named
`libdart_hnsw.so`; on Windows it is `dart_hnsw.dll`.

At runtime, `dart_hnsw` resolves native code in this order:

1. Dart native-asset mapping, if available
2. `DART_HNSW_LIBRARY_PATH`, if set
3. the executable directory

Example override:

```sh
DART_HNSW_LIBRARY_PATH=./lib/libdart_hnsw.so ./your_app
```

Only use an override path controlled by your application deployment.

---

## Persistence

- `close()` only releases the handle. It does not save.
- `checkpoint(expectedGeneration: generation)` serializes the index to a `Uint8List` using the same binary format as hnswlib's native `saveIndex`.
- The caller is responsible for storing the bytes (filesystem, database, blob storage, etc.).
- `open()` is the only operation that creates a handle. `load()` atomically replaces an existing shared context through that handle.
- `load()`, `checkpoint()`, and `clear()` require the current generation, so stale operations fail instead of acting on newer content.
- `acquireExclusiveMutationAccess()` grants one handle exclusive mutation access across every isolate. The owner may mutate; other handles may continue searching but mutation attempts fail until the owner releases access or every handle closes.
- The binary payload does not include all wrapper configuration. Store and restore the same `dimensions`, `metric`, and `allowReplaceDeleted` values yourself; using different values is unsupported.
- Checkpoint bytes use hnswlib's native binary layout, including native `size_t` fields. Restore them on a compatible ABI, rather than treating them as a portable interchange format between 32-bit and 64-bit targets.

### Unordered Isolate Startup

`HnswIndex.open()` attaches to a same-name context or creates an empty one. Calling `index.load()` later replaces that empty context atomically, so isolate startup order does not decide whether checkpoint bytes are used.

Do not write vectors before the intended load completes: that load deliberately replaces the entire shared graph. For stale-write protection, capture the generation before checkpointing and use it when reloading:

```dart
final generation = index.generation;
final bytes = index.checkpoint(expectedGeneration: generation);

index.load(
  bytes,
  expectedGeneration: generation,
);
```

If another mutation, clear, or load happens first, the load throws `HnswGenerationConflictException` instead of overwriting newer content.

### Stable Shutdown Checkpoint

To persist the final state without a worker adding a vector after the
checkpoint, acquire exclusive mutation access before reading its generation.
The owner remains able to mutate and checkpoint. Other handles can keep
searching, but their upserts, batches, deletes, loads, clears, resizes, and
ef-search changes fail.

```dart
index.acquireExclusiveMutationAccess();
try {
  final generation = index.generation;
  final bytes = index.checkpoint(expectedGeneration: generation);
  await saveCheckpoint(bytes); // Application-owned storage.
} catch (_) {
  index.releaseExclusiveMutationAccess(); // Keep workers usable on failure.
  rethrow;
}

// On successful shutdown, do not release before closing.
index.close();
```

See [`example/multi_isolate.dart`](example/multi_isolate.dart) for an application isolate that loads checkpoint bytes, then two worker isolates that open the same collection, write vectors, and close.

### Safety Limits

The package rejects NaN and infinity vector values and enforces limits through `HnswLimits`: 65,536 dimensions, 10,000,000 capacity, `m` from 2 to 10,000, 1,000,000 `ef`, 100,000 search results or batch entries, 16 million batch float values, and 512 MiB checkpoint payloads. Native allocation checks also cap the base index allocation at 1 GiB.

---

## Concurrency Model

- Same process, same collection name: one shared native index context
- Same process, multiple isolates: all isolates attach to that shared context
- Searches: concurrent query group
- Individual and batch upserts: concurrent upsert group
- Queries and upserts: mutually exclusive groups with fair phase handoff
- Deletes, resize, EF changes, the `stats` snapshot, load, clear, and checkpoint: fully exclusive administrative tier
- `acquireExclusiveMutationAccess()` / `releaseExclusiveMutationAccess()`: administrative tier plus shared owner state

This is one of the package's core design goals: isolate-friendly shared native indexing, not just local ANN search.

---

## Supported Platforms

The hook and native source are configured for:

- macOS
- Linux
- Windows

Android and iOS are also declared in `pubspec.yaml` and use the same C++17 hook path. They require validation in the consuming Flutter app for the target ABI/device; this package does not provide a mobile application fixture.

The package hook has been smoke-tested here for Android arm64 (NDK API 21) and iOS arm64 simulator (iOS 13). Actual Flutter app/device runtime validation remains the responsibility of the consuming application.

The native layer has **zero filesystem dependencies** — all index state lives in memory, and checkpoint/load operate on byte buffers.

Web uses the same synchronous `HnswIndex` API after one asynchronous
`HnswIndex.initialize(loaderUrl: ..., wasmUrl: ...)` call. Compile the existing
C++ wrapper with `tool/build_web.sh` and serve its matching `hnsw.js` and
`hnsw.wasm` assets. Dart/Flutter Wasm compilation does not compile this C++ code.

Native isolates retain process-wide graph sharing. Browser handles share within
one Wasm instance; separate workers/tabs have separate memory. Web IDs are
restricted to 0..4,294,967,295, and raw checkpoints require a compatible ABI.
See [web setup, sharing scope and validation](doc/web.md).

---

## API Overview

### `HnswIndex`

| Method / Property | Description |
|-------|-------------|
| `HnswIndex.open(...)` | Attach to a shared index or create an empty one |
| `load(bytes, {expectedGeneration})` | Atomically replace shared content |
| `upsert(id, vector)` | Insert or replace one vector |
| `upsertBatch(entries)` | Insert or replace multiple vectors |
| `search(query, {k})` | Approximate nearest neighbor search |
| `markDeleted(id)` | Mark a label deleted |
| `resize(newMaxElements)` | Increase index capacity |
| `setEfSearch(ef)` | Adjust search breadth |
| `checkpoint(expectedGeneration: generation)` | Serialize the intended shared state to `Uint8List` |
| `clear(expectedGeneration: generation)` | Reset shared content while retaining handles |
| `acquireExclusiveMutationAccess()` | Make this handle the collection's sole mutation owner |
| `releaseExclusiveMutationAccess()` | Release this handle's exclusive mutation ownership |
| `close()` | Release the native handle |
| `stats` | Consistent `HnswStats` snapshot of element, deleted, live, capacity, and EF values under one administrative lock |
| `contextId` | Shared native context identifier for same-process/same-collection handles |
| `generation` | Shared state version for stale-operation protection |

`stats` is intentionally consolidated: reading index metadata takes the fully
exclusive administrative lock, so callers should read it infrequently and
reuse the returned snapshot rather than polling individual values.

### `HnswMetric`

- `HnswMetric.l2`
- `HnswMetric.innerProduct`
- `HnswMetric.cosine`

---

## Running Tests

```sh
dart test
```
