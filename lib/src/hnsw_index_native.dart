import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'native_bindings.dart';

import 'types.dart';
import 'validation.dart';
export 'types.dart';

final HnswNativeBindings _nativeBindings = HnswNativeBindings();

const _maxSignedInt64 = 9223372036854775807;
const _maxLabel = _maxSignedInt64;

ffi.Pointer<ffi.Float> _listToFloatPtr(List<double> values) {
  final ptr = calloc<ffi.Float>(values.length);
  for (var i = 0; i < values.length; i++) {
    ptr[i] = values[i];
  }
  return ptr;
}

int _generationValue(int expectedGeneration) {
  if (expectedGeneration < 0 || expectedGeneration > _maxSignedInt64) {
    throw ArgumentError.value(
      expectedGeneration,
      'expectedGeneration',
      'must be a valid generation',
    );
  }
  return expectedGeneration;
}

void _validateLabel(int id, String name) {
  if (id < 0 || id > _maxLabel) {
    throw ArgumentError.value(
      id,
      name,
      'must fit in the supported signed 64-bit label range',
    );
  }
}

int _metricCode(HnswMetric metric) {
  switch (metric) {
    case HnswMetric.l2:
      return 0;
    case HnswMetric.innerProduct:
      return 1;
    case HnswMetric.cosine:
      return 2;
  }
}

/// A shared native HNSW index backed by hnswlib.
///
/// Opening the same [collectionName] from multiple isolates within the same
/// process attaches them to one shared native context. Searches can run
/// concurrently with other searches, and upserts can run concurrently with
/// other upserts. Search and upsert groups do not overlap. Administrative
/// operations such as delete, resize, load, clear, and checkpoint are fully
/// exclusive.
///
/// The index lives entirely in native memory. Use [checkpoint] to serialize
/// the index to a [Uint8List] that you can store anywhere (filesystem,
/// database, blob storage, etc.). Use [load] to restore from bytes.
///
/// {@category Index}
class HnswIndex {
  /// Initializes browser Wasm assets. This is a no-op on native platforms,
  /// where the existing build hook loads the library automatically.
  static Future<void> initialize({
    required Uri loaderUrl,
    required Uri wasmUrl,
  }) async {}

  ffi.Pointer<ffi.Void>? _handle;
  final int _dimensions;

  HnswIndex._(this._handle, this._dimensions);

  /// Opens a shared in-memory HNSW index, creating an empty one if needed.
  ///
  /// If an index with the same [collectionName] is already open in this
  /// process, the existing shared native context is reused. The [dimensions],
  /// [metric], and [allowReplaceDeleted] values must match the existing
  /// context.
  ///
  /// [maxElements] sets the initial capacity. [m] and [efConstruction] control
  /// graph build quality. [efSearch] controls the default search breadth.
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
    validateIndexOptions(
      collectionName: collectionName,
      dimensions: dimensions,
      maxElements: maxElements,
      m: m,
      efConstruction: efConstruction,
      efSearch: efSearch,
    );
    final errorCode = calloc<ffi.Int32>();
    final namePtr = collectionName.toNativeUtf8();

    try {
      final handle = _nativeBindings.hnswCreate(
        namePtr,
        dimensions,
        maxElements,
        m,
        efConstruction,
        efSearch,
        _metricCode(metric),
        allowReplaceDeleted ? 1 : 0,
        errorCode,
      );

      if (errorCode.value != 0) {
        throw errorForCode(
          errorCode.value,
          'HnswIndex.open(\'$collectionName\')',
        );
      }

      return HnswIndex._(handle, dimensions);
    } finally {
      calloc.free(errorCode);
      calloc.free(namePtr);
    }
  }

  /// Replaces this shared index with bytes produced by [checkpoint].
  ///
  /// Existing handles for the same collection remain valid and see either the
  /// old complete graph or the new complete graph. [expectedGeneration] is
  /// required to prevent a stale checkpoint from overwriting newer content.
  void load(Uint8List bytes, {required int expectedGeneration}) {
    final handle = _requireOpenHandle();
    if (bytes.isEmpty || bytes.length > HnswLimits.maxCheckpointBytes) {
      throw ArgumentError.value(
        bytes.length,
        'bytes.length',
        'must be between 1 and ${HnswLimits.maxCheckpointBytes}',
      );
    }
    final expected = _generationValue(expectedGeneration);
    final dataPtr = calloc<ffi.Uint8>(bytes.length);
    for (var i = 0; i < bytes.length; i++) {
      dataPtr[i] = bytes[i];
    }

    try {
      final result = _nativeBindings.hnswLoad(
        handle,
        dataPtr,
        bytes.length,
        expected,
      );

      if (result != 0) throw errorForCode(result, 'HnswIndex.load');
    } finally {
      calloc.free(dataPtr);
    }
  }

  /// Whether the native handle has been released via [close].
  bool get isClosed => _handle == null;

  ffi.Pointer<ffi.Void> _requireOpenHandle() {
    final handle = _handle;
    if (handle == null || handle == ffi.nullptr) {
      throw StateError('HnswIndex is closed');
    }
    return handle;
  }

  /// Inserts or replaces the vector for the given label [id].
  ///
  /// The [vector] length must match the index [dimensions] specified at
  /// open time. If a vector with the same [id] already exists it is replaced.
  /// Upserts from multiple isolates may execute concurrently; searches wait
  /// until the active upsert phase finishes.
  void upsert(int id, List<double> vector) {
    final handle = _requireOpenHandle();
    _validateLabel(id, 'id');
    if (vector.length != _dimensions) {
      throw ArgumentError(
        'Expected $_dimensions dimensions, got ${vector.length}',
      );
    }
    validateFiniteVector(vector, 'vector');

    final vecPtr = _listToFloatPtr(vector);
    try {
      final result = _nativeBindings.hnswUpsertVector(handle, id, vecPtr);
      if (result != 0) {
        throw errorForCode(result, 'upsert(label=$id)');
      }
    } finally {
      calloc.free(vecPtr);
    }
  }

  /// Inserts or replaces multiple vectors in a single native call.
  ///
  /// More efficient than repeated [upsert] calls when adding many vectors.
  /// All entries must match the index [dimensions]. One batch is one member of
  /// the concurrent upsert group and may overlap other upsert calls.
  void upsertBatch(List<HnswVectorEntry> entries) {
    final handle = _requireOpenHandle();
    if (entries.isEmpty) return;
    if (entries.length > HnswLimits.maxBatchSize ||
        entries.length > HnswLimits.maxBatchFloatValues ~/ _dimensions) {
      throw ArgumentError.value(
        entries.length,
        'entries.length',
        'exceeds the batch safety limit',
      );
    }

    for (final entry in entries) {
      _validateLabel(entry.id, 'entry.id');
      if (entry.vector.length != _dimensions) {
        throw ArgumentError(
          'Entry ${entry.id}: expected $_dimensions dimensions, got ${entry.vector.length}',
        );
      }
      validateFiniteVector(entry.vector, 'Entry ${entry.id}.vector');
    }

    final labels = calloc<ffi.Uint64>(entries.length);
    final vectors = calloc<ffi.Float>(entries.length * _dimensions);

    try {
      for (var i = 0; i < entries.length; i++) {
        final entry = entries[i];
        labels[i] = entry.id;
        for (var j = 0; j < _dimensions; j++) {
          vectors[i * _dimensions + j] = entry.vector[j];
        }
      }

      final result = _nativeBindings.hnswUpsertBatch(
        handle,
        labels,
        vectors,
        entries.length,
      );
      if (result != 0) {
        throw errorForCode(result, 'upsertBatch(${entries.length} entries)');
      }
    } finally {
      calloc.free(labels);
      calloc.free(vectors);
    }
  }

  /// Searches for the [k] nearest neighbors of [query].
  ///
  /// Returns up to [k] results sorted by distance (ascending). The distance
  /// interpretation depends on the index [metric].
  List<HnswResult> search(List<double> query, {int k = 10}) {
    final handle = _requireOpenHandle();
    if (k <= 0 || k > HnswLimits.maxSearchResults) {
      throw ArgumentError.value(
        k,
        'k',
        'must be between 1 and ${HnswLimits.maxSearchResults}',
      );
    }
    if (query.length != _dimensions) {
      throw ArgumentError(
        'Expected $_dimensions dimensions, got ${query.length}',
      );
    }
    validateFiniteVector(query, 'query');

    final queryPtr = _listToFloatPtr(query);
    final outLabels = calloc<ffi.Uint64>(k);
    final outDistances = calloc<ffi.Float>(k);
    final outCount = calloc<ffi.Int32>();

    try {
      final result = _nativeBindings.hnswSearchKnn(
        handle,
        queryPtr,
        k,
        outLabels,
        outDistances,
        outCount,
      );
      if (result != 0) {
        throw errorForCode(result, 'search(k=$k)');
      }

      return List.generate(
        outCount.value,
        (index) => HnswResult(outLabels[index], outDistances[index].toDouble()),
      );
    } finally {
      calloc.free(queryPtr);
      calloc.free(outLabels);
      calloc.free(outDistances);
      calloc.free(outCount);
    }
  }

  /// Marks the vector with label [id] as deleted.
  ///
  /// Deleted entries are excluded from search results. Their storage is
  /// reused when [allowReplaceDeleted] was enabled at creation.
  void markDeleted(int id) {
    final handle = _requireOpenHandle();
    _validateLabel(id, 'id');
    final result = _nativeBindings.hnswMarkDeleted(handle, id);
    if (result != 0) {
      throw errorForCode(result, 'markDeleted(label=$id)');
    }
  }

  /// Increases the index capacity to [newMaxElements].
  ///
  /// Cannot shrink the index below the current element count.
  void resize(int newMaxElements) {
    final handle = _requireOpenHandle();
    if (newMaxElements <= 0 || newMaxElements > HnswLimits.maxElements) {
      throw ArgumentError.value(
        newMaxElements,
        'newMaxElements',
        'must be between 1 and ${HnswLimits.maxElements}',
      );
    }
    final result = _nativeBindings.hnswResize(handle, newMaxElements);
    if (result != 0) {
      throw errorForCode(result, 'resize($newMaxElements)');
    }
  }

  /// Sets the search breadth parameter [ef].
  ///
  /// Higher values improve recall at the cost of search speed.
  void setEfSearch(int ef) {
    final handle = _requireOpenHandle();
    if (ef <= 0 || ef > HnswLimits.maxEf) {
      throw ArgumentError.value(
        ef,
        'ef',
        'must be between 1 and ${HnswLimits.maxEf}',
      );
    }
    final result = _nativeBindings.hnswSetEfSearch(handle, ef);
    if (result != 0) {
      throw errorForCode(result, 'setEfSearch($ef)');
    }
  }

  /// Reads all index metadata under one native administrative lock.
  HnswStats get stats {
    final handle = _requireOpenHandle();
    final nativeStats = calloc<NativeHnswStats>();
    try {
      final result = _nativeBindings.hnswStats(handle, nativeStats);
      if (result != 0) {
        throw errorForCode(result, 'stats');
      }
      final value = nativeStats.ref;
      return HnswStats(
        elementCount: value.elementCount,
        deletedCount: value.deletedCount,
        maxElements: value.maxElements,
        efSearch: value.efSearch,
      );
    } finally {
      calloc.free(nativeStats);
    }
  }

  /// Opaque identifier for the shared native context.
  ///
  /// Handles attached to the same collection name share the same context id.
  int get contextId => _nativeBindings.hnswContextId(_requireOpenHandle());

  /// Monotonically increasing shared-index state version.
  ///
  /// Mutations, [load], and [clear] advance this value. Use it with
  /// `expectedGeneration` to prevent stale destructive operations.
  int get generation => _nativeBindings.hnswGeneration(_requireOpenHandle());

  /// Acquires exclusive mutation access for this handle's shared collection.
  ///
  /// This waits for the active native operation phase to drain before assigning
  /// ownership. While held, only this handle may load, clear, upsert, delete,
  /// resize, or change the search breadth. Other handles may continue issuing
  /// searches, subject to the normal query/upsert phase boundary. The owner
  /// must call [releaseExclusiveMutationAccess] if normal mutations should
  /// resume.
  void acquireExclusiveMutationAccess() {
    final result = _nativeBindings.hnswAcquireExclusiveMutationAccess(
      _requireOpenHandle(),
    );
    if (result != 0) {
      throw errorForCode(result, 'acquireExclusiveMutationAccess');
    }
  }

  /// Releases this handle's exclusive mutation access.
  ///
  /// Only the owning handle may release. Closing an owning handle without
  /// release intentionally leaves the remaining shared context mutation-
  /// blocked until every handle has closed.
  void releaseExclusiveMutationAccess() {
    final result = _nativeBindings.hnswReleaseExclusiveMutationAccess(
      _requireOpenHandle(),
    );
    if (result != 0) {
      throw errorForCode(result, 'releaseExclusiveMutationAccess');
    }
  }

  /// Serializes the current index state to a [Uint8List].
  ///
  /// The returned bytes use the same binary format as hnswlib's native
  /// `saveIndex`. The caller is responsible for storing them (filesystem,
  /// database, blob storage, etc.). [close] does **not** save — call
  /// [checkpoint] explicitly before closing to persist changes.
  ///
  /// [expectedGeneration] is required so a checkpoint cannot silently capture
  /// a newer shared state than the caller intended.
  Uint8List checkpoint({required int expectedGeneration}) {
    final handle = _requireOpenHandle();
    final sizeOut = calloc<ffi.Int32>();
    final expected = _generationValue(expectedGeneration);

    try {
      final phase1 = _nativeBindings.hnswCheckpoint(
        handle,
        ffi.Pointer<ffi.Uint8>.fromAddress(0),
        0,
        sizeOut,
        expected,
      );
      if (phase1 != 0) {
        throw errorForCode(phase1, 'checkpoint (size query)');
      }

      final size = sizeOut.value;
      if (size <= 0 || size > HnswLimits.maxCheckpointBytes) {
        throw HnswCheckpointException('checkpoint: invalid size $size');
      }

      final buffer = calloc<ffi.Uint8>(size);
      try {
        final phase2 = _nativeBindings.hnswCheckpoint(
          handle,
          buffer,
          size,
          sizeOut,
          expected,
        );
        if (phase2 != 0) throw errorForCode(phase2, 'checkpoint');

        final written = sizeOut.value;
        if (written <= 0 || written > size) {
          throw HnswCheckpointException(
            'checkpoint: native returned invalid size $written',
          );
        }
        return Uint8List.fromList(buffer.asTypedList(written));
      } finally {
        calloc.free(buffer);
      }
    } finally {
      calloc.free(sizeOut);
    }
  }

  /// Clears all vectors while preserving the shared context configuration.
  ///
  /// Existing handles remain valid and immediately observe the empty index.
  /// Clearing fails if another isolate changed the context first.
  void clear({required int expectedGeneration}) {
    final result = _nativeBindings.hnswClear(
      _requireOpenHandle(),
      _generationValue(expectedGeneration),
    );
    if (result != 0) {
      throw errorForCode(result, 'clear');
    }
  }

  /// Releases the native handle.
  ///
  /// Does **not** save the index — call [checkpoint] first to persist changes.
  /// After calling [close] the instance must not be used further.
  void close() {
    final handle = _handle;
    if (handle == null || handle == ffi.nullptr) {
      return;
    }
    _nativeBindings.hnswClose(handle);
    _handle = null;
  }
}
