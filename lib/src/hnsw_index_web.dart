import 'dart:convert';
import 'dart:typed_data';
import 'types.dart';
import 'validation.dart';
import 'wasm_bindings.dart';

/// The existing C++ hnswlib compiled to wasm32, with synchronous operations.
/// Await [initialize] once before [open]. Same-name handles share a graph in
/// this JavaScript realm; other browser workers/tabs do not share its memory.
/// All calls execute on the calling thread. Always explicitly [close] handles.
class HnswIndex {
  int? _handle;
  final int _dimensions;
  HnswIndex._(this._handle, this._dimensions);

  /// Loads the application's matching Emscripten JS loader and Wasm binary.
  /// Concurrent/repeated calls with the same URLs reuse initialization. Failures
  /// may be retried. In an application-owned worker, preload the JS loader with
  /// importScripts before calling this method. Native initialization is a no-op.
  static Future<void> initialize({
    required Uri loaderUrl,
    required Uri wasmUrl,
  }) => initializeWasm(loaderUrl, wasmUrl);

  // MARK: Handle lifecycle

  /// Attaches to a named graph or creates it. Configuration and defaults match
  /// the native API. Dimensions, metric and delete-reuse must match live handles.
  static HnswIndex open({
    required String collectionName,
    required int dimensions,
    HnswMetric metric = HnswMetric.cosine,
    int maxElements = 100000,
    int m = 16,
    int efConstruction = 200,
    int efSearch = 64,
    bool allowReplaceDeleted = true,
  }) {
    final module = wasmBindings;
    validateIndexOptions(
      collectionName: collectionName,
      dimensions: dimensions,
      maxElements: maxElements,
      m: m,
      efConstruction: efConstruction,
      efSearch: efSearch,
    );
    if (collectionName.contains('\u0000')) {
      throw ArgumentError('Collection name contains NUL');
    }
    return withWasmMemory((memory) {
      final name = memory.copyBytes([...utf8.encode(collectionName), 0]);
      final error = memory.allocate(4);
      final handle = module.create(
        name,
        dimensions,
        maxElements,
        m,
        efConstruction,
        efSearch,
        metric.index,
        allowReplaceDeleted ? 1 : 0,
        error,
      );
      _check(module.data.getInt32(error, Endian.little), 'open');
      return HnswIndex._(handle, dimensions);
    });
  }

  /// Whether this handle has been released.
  bool get isClosed => _handle == null;
  int get _open => _handle ?? (throw StateError('HnswIndex is closed'));

  /// Releases this handle without saving. Other handles remain valid; closing
  /// the last handle frees the graph. The realm's Wasm runtime remains loaded.
  void close() {
    final handle = _handle;
    if (handle == null) return;
    _check(wasmBindings.close(handle), 'close');
    _handle = null;
  }

  // MARK: Mutation and search

  /// Inserts or updates a vector. Web labels must be in 0..4294967295 (wasm32
  /// hnswlib size_t labels); larger IDs are rejected rather than truncated.
  void upsert(int id, List<double> vector) {
    final handle = _open;
    _label(id);
    _vector(vector);
    withWasmMemory((memory) {
      final pointer = memory.copyFloats(vector);
      _check(wasmBindings.upsert(handle, wasmInt64(id), pointer), 'upsert');
    });
  }

  /// Inserts or updates a bounded batch, with the same partial-failure semantics
  /// as native hnswlib. Vector lengths must match the configured dimensions.
  void upsertBatch(List<HnswVectorEntry> entries) {
    final handle = _open;
    if (entries.isEmpty) return;
    if (entries.length > HnswLimits.maxBatchSize ||
        entries.length > HnswLimits.maxBatchFloatValues ~/ _dimensions) {
      throw ArgumentError('Batch exceeds safety limit');
    }
    for (final entry in entries) {
      _label(entry.id);
      _vector(entry.vector);
    }
    withWasmMemory((memory) {
      final labels = memory.allocate(entries.length * 8);
      final vectors = memory.allocate(entries.length * _dimensions * 4);
      final data = wasmBindings.data, floats = wasmBindings.floats;
      for (var i = 0; i < entries.length; i++) {
        data.setUint32(labels + i * 8, entries[i].id, Endian.little);
        data.setUint32(labels + i * 8 + 4, 0, Endian.little);
        floats.setAll(vectors ~/ 4 + i * _dimensions, entries[i].vector);
      }
      _check(
        wasmBindings.upsertBatch(handle, labels, vectors, entries.length),
        'upsertBatch',
      );
    });
  }

  /// Returns up to k approximate neighbors sorted by increasing distance.
  List<HnswResult> search(List<double> query, {int k = 10}) {
    final handle = _open;
    _range(k, 1, HnswLimits.maxSearchResults, 'k');
    _vector(query);
    return withWasmMemory((memory) {
      final vector = memory.copyFloats(query);
      final labels = memory.allocate(k * 8),
          distances = memory.allocate(k * 4),
          count = memory.allocate(4);
      _check(
        wasmBindings.search(handle, vector, k, labels, distances, count),
        'search',
      );
      final data = wasmBindings.data;
      return [
        for (var i = 0; i < data.getInt32(count, Endian.little); i++)
          HnswResult(
            data.getUint32(labels + i * 8, Endian.little),
            data.getFloat32(distances + i * 4, Endian.little),
          ),
      ];
    });
  }

  /// Marks a live label deleted; missing or already-deleted labels fail.
  void markDeleted(int id) {
    final h = _open;
    _label(id);
    _check(wasmBindings.markDeleted(h, wasmInt64(id)), 'markDeleted');
  }

  /// Changes capacity, subject to the current element count and native limits.
  void resize(int newMaxElements) {
    final h = _open;
    _range(newMaxElements, 1, HnswLimits.maxElements, 'newMaxElements');
    _check(wasmBindings.resize(h, newMaxElements), 'resize');
  }

  /// Changes the search breadth (higher values trade speed for recall).
  void setEfSearch(int ef) {
    final h = _open;
    _range(ef, 1, HnswLimits.maxEf, 'ef');
    _check(wasmBindings.setEfSearch(h, ef), 'setEfSearch');
  }

  // MARK: Shared state and byte serialization

  /// A consistent metadata snapshot. Read infrequently and reuse the result.
  HnswStats get stats {
    final h = _open;
    return withWasmMemory((memory) {
      final p = memory.allocate(32);
      _check(wasmBindings.stats(h, p), 'stats');
      final data = wasmBindings.data;
      return HnswStats(
        elementCount: data.getUint32(p, Endian.little),
        deletedCount: data.getUint32(p + 8, Endian.little),
        maxElements: data.getUint32(p + 16, Endian.little),
        efSearch: data.getInt32(p + 24, Endian.little),
      );
    });
  }

  /// Identifier shared by same-name handles within this Wasm instance.
  int get contextId => wasmInteger(wasmBindings.contextId(_open));

  /// Shared state version used to guard checkpoint, load and clear.
  int get generation => wasmInteger(wasmBindings.generation(_open));

  /// Restricts mutation to this handle; other handles may continue searching.
  void acquireExclusiveMutationAccess() =>
      _check(wasmBindings.acquire(_open), 'acquireExclusiveMutationAccess');

  /// Releases this handle's exclusive mutation ownership. As on native, closing
  /// without release leaves remaining handles mutation-blocked until all close.
  void releaseExclusiveMutationAccess() =>
      _check(wasmBindings.release(_open), 'releaseExclusiveMutationAccess');

  /// Serializes the raw wasm32 hnswlib layout. Bytes are compatible only with
  /// matching builds/ABIs; native 64-bit checkpoints are not portable to wasm32.
  Uint8List checkpoint({required int expectedGeneration}) {
    final h = _open;
    _generation(expectedGeneration);
    return withWasmMemory((memory) {
      final size = memory.allocate(4), expected = wasmInt64(expectedGeneration);
      _check(
        wasmBindings.checkpoint(h, 0, 0, size, expected),
        'checkpoint size',
      );
      final length = wasmBindings.data.getInt32(size, Endian.little);
      if (length <= 0 || length > HnswLimits.maxCheckpointBytes) {
        throw const HnswCheckpointException('Invalid checkpoint size');
      }
      final buffer = memory.allocate(length);
      _check(
        wasmBindings.checkpoint(h, buffer, length, size, expected),
        'checkpoint',
      );
      return Uint8List.fromList(
        wasmBindings.bytes.sublist(buffer, buffer + length),
      );
    });
  }

  /// Atomically replaces the shared graph with compatible raw checkpoint bytes,
  /// rejecting stale generations. A failed load preserves the previous graph.
  void load(Uint8List bytes, {required int expectedGeneration}) {
    final h = _open;
    _generation(expectedGeneration);
    _range(bytes.length, 1, HnswLimits.maxCheckpointBytes, 'bytes.length');
    withWasmMemory((memory) {
      final buffer = memory.copyBytes(bytes);
      _check(
        wasmBindings.load(
          h,
          buffer,
          bytes.length,
          wasmInt64(expectedGeneration),
        ),
        'load',
      );
    });
  }

  /// Clears the graph while retaining handles and configuration.
  void clear({required int expectedGeneration}) {
    final h = _open;
    _generation(expectedGeneration);
    _check(wasmBindings.clear(h, wasmInt64(expectedGeneration)), 'clear');
  }

  void _vector(List<double> values) {
    if (values.length != _dimensions) {
      throw ArgumentError(
        'Expected $_dimensions dimensions, got ${values.length}',
      );
    }
    validateFiniteVector(values, 'vector');
  }
}

void _check(int code, String operation) {
  if (code != 0) throw errorForCode(code, operation);
}

void _range(int value, int min, int max, String name) {
  if (value < min || value > max) {
    throw ArgumentError.value(value, name, 'Expected $min..$max');
  }
}

void _label(int id) => _range(id, 0, 4294967295, 'label');
void _generation(int value) =>
    _range(value, 0, 9007199254740991, 'expectedGeneration');
