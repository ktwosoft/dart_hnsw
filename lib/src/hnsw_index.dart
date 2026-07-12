import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import 'native_bindings.dart';

final HnswNativeBindings _nativeBindings = HnswNativeBindings();

enum HnswErrorCode {
  ok(0),
  nullArg(-1),
  invalidArg(-2),
  configMismatch(-3),
  lockedProcess(-4),
  loadFailed(-5),
  saveFailed(-6),
  renameFailed(-7),
  zeroVector(-8),
  hnswlib(-10);

  final int code;
  const HnswErrorCode(this.code);

  static HnswErrorCode fromCode(int code) => values.firstWhere(
    (value) => value.code == code,
    orElse: () => HnswErrorCode.hnswlib,
  );
}

class HnswException implements Exception {
  final String message;
  const HnswException(this.message);

  @override
  String toString() => 'HnswException: $message';
}

class HnswOpenException extends HnswException {
  const HnswOpenException(super.message);
}

class HnswDimensionMismatchException extends HnswException {
  const HnswDimensionMismatchException(super.message);
}

class HnswCapacityExceededException extends HnswException {
  const HnswCapacityExceededException(super.message);
}

class HnswConcurrentProcessException extends HnswException {
  const HnswConcurrentProcessException(super.message);
}

class HnswLoadException extends HnswException {
  const HnswLoadException(super.message);
}

class HnswCheckpointException extends HnswException {
  const HnswCheckpointException(super.message);
}

class HnswZeroVectorException extends HnswException {
  const HnswZeroVectorException(super.message);
}

class HnswNativeException extends HnswException {
  const HnswNativeException(super.message);
}

enum HnswMetric { l2, innerProduct, cosine }

class HnswResult {
  final int id;
  final double distance;
  const HnswResult(this.id, this.distance);
}

class HnswVectorEntry {
  final int id;
  final List<double> vector;
  const HnswVectorEntry(this.id, this.vector);
}

ffi.Pointer<ffi.Float> _listToFloatPtr(List<double> values) {
  final ptr = calloc<ffi.Float>(values.length);
  for (var i = 0; i < values.length; i++) {
    ptr[i] = values[i];
  }
  return ptr;
}

HnswException _errorForCode(int code, String context) {
  switch (code) {
    case -1:
      return HnswException('$context: null argument');
    case -2:
      return HnswException('$context: invalid argument');
    case -3:
      return HnswDimensionMismatchException(
        '$context: dimensions, metric, or delete-reuse setting mismatch with the active shared index context',
      );
    case -4:
      return HnswConcurrentProcessException(
        '$context: index is locked by another process',
      );
    case -5:
      return HnswLoadException('$context: failed to load index');
    case -6:
      return HnswCheckpointException('$context: failed to save index');
    case -7:
      return HnswCheckpointException(
        '$context: failed to replace checkpoint files',
      );
    case -8:
      return HnswZeroVectorException(
        '$context: zero vector not allowed for cosine metric',
      );
    default:
      return HnswNativeException('$context: native error $code');
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

class HnswIndex {
  ffi.Pointer<ffi.Void>? _handle;
  final int _dimensions;

  HnswIndex._(this._handle, this._dimensions);

  static HnswIndex open({
    required String path,
    required int dimensions,
    HnswMetric metric = HnswMetric.cosine,
    int maxElements = 100000,
    int m = 16,
    int efConstruction = 200,
    int efSearch = 64,
    bool allowReplaceDeleted = true,
  }) {
    final errorCode = calloc<ffi.Int32>();
    final pathPtr = path.toNativeUtf8();

    try {
      final handle = _nativeBindings.hnswOpen(
        pathPtr,
        dimensions,
        maxElements,
        m,
        efConstruction,
        efSearch,
        _metricCode(metric),
        allowReplaceDeleted ? 1 : 0,
        0,
        errorCode,
      );

      if (errorCode.value != 0) {
        throw _errorForCode(errorCode.value, 'HnswIndex.open(\'$path\')');
      }

      return HnswIndex._(handle, dimensions);
    } finally {
      calloc.free(errorCode);
      calloc.free(pathPtr);
    }
  }

  bool get isClosed => _handle == null;

  ffi.Pointer<ffi.Void> _requireOpenHandle() {
    final handle = _handle;
    if (handle == null || handle == ffi.nullptr) {
      throw StateError('HnswIndex is closed');
    }
    return handle;
  }

  void upsert(int id, List<double> vector) {
    if (vector.length != _dimensions) {
      throw ArgumentError(
        'Expected $_dimensions dimensions, got ${vector.length}',
      );
    }

    final vecPtr = _listToFloatPtr(vector);
    try {
      final result = _nativeBindings.hnswUpsertVector(
        _requireOpenHandle(),
        id,
        vecPtr,
      );
      if (result != 0) {
        throw _errorForCode(result, 'upsert(label=$id)');
      }
    } finally {
      calloc.free(vecPtr);
    }
  }

  void upsertBatch(List<HnswVectorEntry> entries) {
    if (entries.isEmpty) return;

    final labels = calloc<ffi.Uint64>(entries.length);
    final vectors = calloc<ffi.Float>(entries.length * _dimensions);

    try {
      for (var i = 0; i < entries.length; i++) {
        final entry = entries[i];
        if (entry.vector.length != _dimensions) {
          throw ArgumentError(
            'Entry ${entry.id}: expected $_dimensions dimensions, got ${entry.vector.length}',
          );
        }

        labels[i] = entry.id;
        for (var j = 0; j < _dimensions; j++) {
          vectors[i * _dimensions + j] = entry.vector[j];
        }
      }

      final result = _nativeBindings.hnswUpsertBatch(
        _requireOpenHandle(),
        labels,
        vectors,
        entries.length,
      );
      if (result != 0) {
        throw _errorForCode(result, 'upsertBatch(${entries.length} entries)');
      }
    } finally {
      calloc.free(labels);
      calloc.free(vectors);
    }
  }

  List<HnswResult> search(List<double> query, {int k = 10}) {
    if (query.length != _dimensions) {
      throw ArgumentError(
        'Expected $_dimensions dimensions, got ${query.length}',
      );
    }

    final queryPtr = _listToFloatPtr(query);
    final outLabels = calloc<ffi.Uint64>(k);
    final outDistances = calloc<ffi.Float>(k);
    final outCount = calloc<ffi.Int32>();

    try {
      final result = _nativeBindings.hnswSearchKnn(
        _requireOpenHandle(),
        queryPtr,
        k,
        outLabels,
        outDistances,
        outCount,
      );
      if (result != 0) {
        throw _errorForCode(result, 'search(k=$k)');
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

  void markDeleted(int id) {
    final result = _nativeBindings.hnswMarkDeleted(_requireOpenHandle(), id);
    if (result != 0) {
      throw _errorForCode(result, 'markDeleted(label=$id)');
    }
  }

  void resize(int newMaxElements) {
    final result = _nativeBindings.hnswResize(
      _requireOpenHandle(),
      newMaxElements,
    );
    if (result != 0) {
      throw _errorForCode(result, 'resize($newMaxElements)');
    }
  }

  void setEfSearch(int ef) {
    final result = _nativeBindings.hnswSetEfSearch(_requireOpenHandle(), ef);
    if (result != 0) {
      throw _errorForCode(result, 'setEfSearch($ef)');
    }
  }

  int get elementCount =>
      _nativeBindings.hnswElementCount(_requireOpenHandle());

  int get deletedCount =>
      _nativeBindings.hnswDeletedCount(_requireOpenHandle());

  int get liveCount => elementCount - deletedCount;

  int get maxElements => _nativeBindings.hnswMaxElements(_requireOpenHandle());

  int get efSearch => _nativeBindings.hnswGetEfSearch(_requireOpenHandle());

  int get contextId => _nativeBindings.hnswContextId(_requireOpenHandle());

  void checkpoint() {
    final result = _nativeBindings.hnswCheckpoint(_requireOpenHandle());
    if (result != 0) {
      throw _errorForCode(result, 'checkpoint');
    }
  }

  void close() {
    final handle = _handle;
    if (handle == null || handle == ffi.nullptr) {
      return;
    }
    _nativeBindings.hnswClose(handle);
    _handle = null;
  }
}
