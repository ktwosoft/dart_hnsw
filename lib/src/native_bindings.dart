@ffi.DefaultAsset('package:dart_hnsw/src/native_bindings.dart')
library;

import 'dart:ffi' as ffi;
import 'dart:io';

import 'package:ffi/ffi.dart';

typedef HnswHandle = ffi.Pointer<ffi.Void>;

const _hnswAbiVersion = 1;
const _libraryPathEnvVar = 'DART_HNSW_LIBRARY_PATH';
const _forceDynamicLibraryEnvVar = 'DART_HNSW_FORCE_DYNAMIC_LIBRARY';

typedef _NativeHnswOpen =
    HnswHandle Function(
      ffi.Pointer<Utf8>,
      ffi.Int32,
      ffi.Int32,
      ffi.Int32,
      ffi.Int32,
      ffi.Int32,
      ffi.Int32,
      ffi.Int32,
      ffi.Int32,
      ffi.Pointer<ffi.Int32>,
    );
typedef _DartHnswOpen =
    HnswHandle Function(
      ffi.Pointer<Utf8>,
      int,
      int,
      int,
      int,
      int,
      int,
      int,
      int,
      ffi.Pointer<ffi.Int32>,
    );

typedef _NativeHnswClose = ffi.Int32 Function(HnswHandle);
typedef _DartHnswClose = int Function(HnswHandle);

typedef _NativeHnswContextId = ffi.Uint64 Function(HnswHandle);
typedef _DartHnswContextId = int Function(HnswHandle);

typedef _NativeHnswRegistrySize = ffi.Uint64 Function();
typedef _DartHnswRegistrySize = int Function();

typedef _NativeHnswUpsertVector =
    ffi.Int32 Function(HnswHandle, ffi.Uint64, ffi.Pointer<ffi.Float>);
typedef _DartHnswUpsertVector =
    int Function(HnswHandle, int, ffi.Pointer<ffi.Float>);

typedef _NativeHnswUpsertBatch =
    ffi.Int32 Function(
      HnswHandle,
      ffi.Pointer<ffi.Uint64>,
      ffi.Pointer<ffi.Float>,
      ffi.Int32,
    );
typedef _DartHnswUpsertBatch =
    int Function(
      HnswHandle,
      ffi.Pointer<ffi.Uint64>,
      ffi.Pointer<ffi.Float>,
      int,
    );

typedef _NativeHnswMarkDeleted = ffi.Int32 Function(HnswHandle, ffi.Uint64);
typedef _DartHnswMarkDeleted = int Function(HnswHandle, int);

typedef _NativeHnswResize = ffi.Int32 Function(HnswHandle, ffi.Int32);
typedef _DartHnswResize = int Function(HnswHandle, int);

typedef _NativeHnswSetEfSearch = ffi.Int32 Function(HnswHandle, ffi.Int32);
typedef _DartHnswSetEfSearch = int Function(HnswHandle, int);

typedef _NativeHnswSearchKnn =
    ffi.Int32 Function(
      HnswHandle,
      ffi.Pointer<ffi.Float>,
      ffi.Int32,
      ffi.Pointer<ffi.Uint64>,
      ffi.Pointer<ffi.Float>,
      ffi.Pointer<ffi.Int32>,
    );
typedef _DartHnswSearchKnn =
    int Function(
      HnswHandle,
      ffi.Pointer<ffi.Float>,
      int,
      ffi.Pointer<ffi.Uint64>,
      ffi.Pointer<ffi.Float>,
      ffi.Pointer<ffi.Int32>,
    );

typedef _NativeHnswElementCount = ffi.Uint64 Function(HnswHandle);
typedef _DartHnswElementCount = int Function(HnswHandle);

typedef _NativeHnswDeletedCount = ffi.Uint64 Function(HnswHandle);
typedef _DartHnswDeletedCount = int Function(HnswHandle);

typedef _NativeHnswMaxElements = ffi.Uint64 Function(HnswHandle);
typedef _DartHnswMaxElements = int Function(HnswHandle);

typedef _NativeHnswGetEfSearch = ffi.Int32 Function(HnswHandle);
typedef _DartHnswGetEfSearch = int Function(HnswHandle);

typedef _NativeHnswCheckpoint = ffi.Int32 Function(HnswHandle);
typedef _DartHnswCheckpoint = int Function(HnswHandle);

typedef _NativeHnswAbiVersion = ffi.Int32 Function();
typedef _DartHnswAbiVersion = int Function();

@ffi.Native<_NativeHnswOpen>(symbol: 'hnsw_open')
external HnswHandle nativeHnswOpen(
  ffi.Pointer<Utf8> path,
  int dimensions,
  int maxElements,
  int m,
  int efConstruction,
  int efSearch,
  int metric,
  int allowReplaceDeleted,
  int reservedOpenFlags,
  ffi.Pointer<ffi.Int32> errorCode,
);

@ffi.Native<_NativeHnswClose>(symbol: 'hnsw_close')
external int nativeHnswClose(HnswHandle handle);

@ffi.Native<_NativeHnswContextId>(symbol: 'hnsw_context_id')
external int nativeHnswContextId(HnswHandle handle);

@ffi.Native<_NativeHnswRegistrySize>(symbol: 'hnsw_registry_size')
external int nativeHnswRegistrySize();

@ffi.Native<_NativeHnswUpsertVector>(symbol: 'hnsw_upsert_vector')
external int nativeHnswUpsertVector(
  HnswHandle handle,
  int label,
  ffi.Pointer<ffi.Float> vector,
);

@ffi.Native<_NativeHnswUpsertBatch>(symbol: 'hnsw_upsert_batch')
external int nativeHnswUpsertBatch(
  HnswHandle handle,
  ffi.Pointer<ffi.Uint64> labels,
  ffi.Pointer<ffi.Float> vectors,
  int count,
);

@ffi.Native<_NativeHnswMarkDeleted>(symbol: 'hnsw_mark_deleted')
external int nativeHnswMarkDeleted(HnswHandle handle, int label);

@ffi.Native<_NativeHnswResize>(symbol: 'hnsw_resize')
external int nativeHnswResize(HnswHandle handle, int newMaxElements);

@ffi.Native<_NativeHnswSetEfSearch>(symbol: 'hnsw_set_ef_search')
external int nativeHnswSetEfSearch(HnswHandle handle, int efSearch);

@ffi.Native<_NativeHnswSearchKnn>(symbol: 'hnsw_search_knn')
external int nativeHnswSearchKnn(
  HnswHandle handle,
  ffi.Pointer<ffi.Float> query,
  int k,
  ffi.Pointer<ffi.Uint64> outLabels,
  ffi.Pointer<ffi.Float> outDistances,
  ffi.Pointer<ffi.Int32> outCount,
);

@ffi.Native<_NativeHnswElementCount>(symbol: 'hnsw_element_count')
external int nativeHnswElementCount(HnswHandle handle);

@ffi.Native<_NativeHnswDeletedCount>(symbol: 'hnsw_deleted_count')
external int nativeHnswDeletedCount(HnswHandle handle);

@ffi.Native<_NativeHnswMaxElements>(symbol: 'hnsw_max_elements')
external int nativeHnswMaxElements(HnswHandle handle);

@ffi.Native<_NativeHnswGetEfSearch>(symbol: 'hnsw_get_ef_search')
external int nativeHnswGetEfSearch(HnswHandle handle);

@ffi.Native<_NativeHnswCheckpoint>(symbol: 'hnsw_checkpoint')
external int nativeHnswCheckpoint(HnswHandle handle);

@ffi.Native<_NativeHnswAbiVersion>(symbol: 'hnsw_abi_version')
external int nativeHnswAbiVersion();

class HnswNativeBindings {
  final String backendDescription;
  final _DartHnswOpen _hnswOpen;
  final _DartHnswClose _hnswClose;
  final _DartHnswContextId _hnswContextId;
  final _DartHnswRegistrySize _hnswRegistrySize;
  final _DartHnswUpsertVector _hnswUpsertVector;
  final _DartHnswUpsertBatch _hnswUpsertBatch;
  final _DartHnswMarkDeleted _hnswMarkDeleted;
  final _DartHnswResize _hnswResize;
  final _DartHnswSetEfSearch _hnswSetEfSearch;
  final _DartHnswSearchKnn _hnswSearchKnn;
  final _DartHnswElementCount _hnswElementCount;
  final _DartHnswDeletedCount _hnswDeletedCount;
  final _DartHnswMaxElements _hnswMaxElements;
  final _DartHnswGetEfSearch _hnswGetEfSearch;
  final _DartHnswCheckpoint _hnswCheckpoint;

  factory HnswNativeBindings([String? libraryPath]) =>
      HnswNativeBindings.withOptions(libraryPath: libraryPath);

  factory HnswNativeBindings.withOptions({
    String? libraryPath,
    bool preferDynamicLibrary = false,
  }) {
    final resolved = _ResolvedBindings.resolve(
      libraryPath: libraryPath,
      preferDynamicLibrary: preferDynamicLibrary,
    );
    return HnswNativeBindings._resolved(resolved);
  }

  HnswNativeBindings._resolved(_ResolvedBindings resolved)
    : backendDescription = resolved.backendDescription,
      _hnswOpen = resolved.hnswOpen,
      _hnswClose = resolved.hnswClose,
      _hnswContextId = resolved.hnswContextId,
      _hnswRegistrySize = resolved.hnswRegistrySize,
      _hnswUpsertVector = resolved.hnswUpsertVector,
      _hnswUpsertBatch = resolved.hnswUpsertBatch,
      _hnswMarkDeleted = resolved.hnswMarkDeleted,
      _hnswResize = resolved.hnswResize,
      _hnswSetEfSearch = resolved.hnswSetEfSearch,
      _hnswSearchKnn = resolved.hnswSearchKnn,
      _hnswElementCount = resolved.hnswElementCount,
      _hnswDeletedCount = resolved.hnswDeletedCount,
      _hnswMaxElements = resolved.hnswMaxElements,
      _hnswGetEfSearch = resolved.hnswGetEfSearch,
      _hnswCheckpoint = resolved.hnswCheckpoint;

  static String libraryFileNameForCurrentPlatform() {
    if (Platform.isMacOS) return 'libdart_hnsw.dylib';
    if (Platform.isLinux) return 'libdart_hnsw.so';
    if (Platform.isWindows) return 'dart_hnsw.dll';
    throw UnsupportedError(
      'dart_hnsw sidecar library is unsupported on ${Platform.operatingSystem}',
    );
  }

  static String? sidecarLibraryPath({String? libraryPath}) {
    if (libraryPath != null) {
      return File(libraryPath).absolute.path;
    }

    final envPath = Platform.environment[_libraryPathEnvVar];
    if (envPath != null && envPath.isNotEmpty) {
      return File(envPath).absolute.path;
    }

    final fileName = libraryFileNameForCurrentPlatform();
    final candidates = <String>{};

    final resolvedExecutable = Platform.resolvedExecutable;
    if (resolvedExecutable.isNotEmpty) {
      candidates.add(File(resolvedExecutable).absolute.parent.path);
    }
    candidates.add(Directory.current.absolute.path);

    for (final directory in candidates) {
      final candidate = File('$directory${Platform.pathSeparator}$fileName');
      if (candidate.existsSync()) {
        return candidate.absolute.path;
      }
    }

    return null;
  }

  static bool get shouldForceDynamicLibrary {
    final value = Platform.environment[_forceDynamicLibraryEnvVar];
    if (value == null) return false;
    switch (value.toLowerCase()) {
      case '1':
      case 'true':
      case 'yes':
      case 'on':
        return true;
      default:
        return false;
    }
  }

  HnswHandle hnswOpen(
    ffi.Pointer<Utf8> path,
    int dimensions,
    int maxElements,
    int m,
    int efConstruction,
    int efSearch,
    int metric,
    int allowReplaceDeleted,
    int reservedOpenFlags,
    ffi.Pointer<ffi.Int32> errorCode,
  ) => _hnswOpen(
    path,
    dimensions,
    maxElements,
    m,
    efConstruction,
    efSearch,
    metric,
    allowReplaceDeleted,
    reservedOpenFlags,
    errorCode,
  );

  int hnswClose(HnswHandle handle) => _hnswClose(handle);

  int hnswContextId(HnswHandle handle) => _hnswContextId(handle);

  int hnswRegistrySize() => _hnswRegistrySize();

  int hnswUpsertVector(
    HnswHandle handle,
    int label,
    ffi.Pointer<ffi.Float> vector,
  ) => _hnswUpsertVector(handle, label, vector);

  int hnswUpsertBatch(
    HnswHandle handle,
    ffi.Pointer<ffi.Uint64> labels,
    ffi.Pointer<ffi.Float> vectors,
    int count,
  ) => _hnswUpsertBatch(handle, labels, vectors, count);

  int hnswMarkDeleted(HnswHandle handle, int label) =>
      _hnswMarkDeleted(handle, label);

  int hnswResize(HnswHandle handle, int newMaxElements) =>
      _hnswResize(handle, newMaxElements);

  int hnswSetEfSearch(HnswHandle handle, int efSearch) =>
      _hnswSetEfSearch(handle, efSearch);

  int hnswSearchKnn(
    HnswHandle handle,
    ffi.Pointer<ffi.Float> query,
    int k,
    ffi.Pointer<ffi.Uint64> outLabels,
    ffi.Pointer<ffi.Float> outDistances,
    ffi.Pointer<ffi.Int32> outCount,
  ) => _hnswSearchKnn(handle, query, k, outLabels, outDistances, outCount);

  int hnswElementCount(HnswHandle handle) => _hnswElementCount(handle);

  int hnswDeletedCount(HnswHandle handle) => _hnswDeletedCount(handle);

  int hnswMaxElements(HnswHandle handle) => _hnswMaxElements(handle);

  int hnswGetEfSearch(HnswHandle handle) => _hnswGetEfSearch(handle);

  int hnswCheckpoint(HnswHandle handle) => _hnswCheckpoint(handle);
}

class _ResolvedBindings {
  final String backendDescription;
  final _DartHnswOpen hnswOpen;
  final _DartHnswClose hnswClose;
  final _DartHnswContextId hnswContextId;
  final _DartHnswRegistrySize hnswRegistrySize;
  final _DartHnswUpsertVector hnswUpsertVector;
  final _DartHnswUpsertBatch hnswUpsertBatch;
  final _DartHnswMarkDeleted hnswMarkDeleted;
  final _DartHnswResize hnswResize;
  final _DartHnswSetEfSearch hnswSetEfSearch;
  final _DartHnswSearchKnn hnswSearchKnn;
  final _DartHnswElementCount hnswElementCount;
  final _DartHnswDeletedCount hnswDeletedCount;
  final _DartHnswMaxElements hnswMaxElements;
  final _DartHnswGetEfSearch hnswGetEfSearch;
  final _DartHnswCheckpoint hnswCheckpoint;

  _ResolvedBindings({
    required this.backendDescription,
    required this.hnswOpen,
    required this.hnswClose,
    required this.hnswContextId,
    required this.hnswRegistrySize,
    required this.hnswUpsertVector,
    required this.hnswUpsertBatch,
    required this.hnswMarkDeleted,
    required this.hnswResize,
    required this.hnswSetEfSearch,
    required this.hnswSearchKnn,
    required this.hnswElementCount,
    required this.hnswDeletedCount,
    required this.hnswMaxElements,
    required this.hnswGetEfSearch,
    required this.hnswCheckpoint,
  });

  factory _ResolvedBindings.resolve({
    String? libraryPath,
    required bool preferDynamicLibrary,
  }) {
    if (libraryPath != null ||
        preferDynamicLibrary ||
        HnswNativeBindings.shouldForceDynamicLibrary) {
      return _ResolvedBindings._fromDynamicLibrary(
        HnswNativeBindings.sidecarLibraryPath(libraryPath: libraryPath),
      );
    }

    try {
      final abiVersion = nativeHnswAbiVersion();
      if (abiVersion != _hnswAbiVersion) {
        throw StateError(
          'dart_hnsw native asset ABI mismatch: expected $_hnswAbiVersion, got $abiVersion',
        );
      }
      return _ResolvedBindings(
        backendDescription: 'native-asset',
        hnswOpen: nativeHnswOpen,
        hnswClose: nativeHnswClose,
        hnswContextId: nativeHnswContextId,
        hnswRegistrySize: nativeHnswRegistrySize,
        hnswUpsertVector: nativeHnswUpsertVector,
        hnswUpsertBatch: nativeHnswUpsertBatch,
        hnswMarkDeleted: nativeHnswMarkDeleted,
        hnswResize: nativeHnswResize,
        hnswSetEfSearch: nativeHnswSetEfSearch,
        hnswSearchKnn: nativeHnswSearchKnn,
        hnswElementCount: nativeHnswElementCount,
        hnswDeletedCount: nativeHnswDeletedCount,
        hnswMaxElements: nativeHnswMaxElements,
        hnswGetEfSearch: nativeHnswGetEfSearch,
        hnswCheckpoint: nativeHnswCheckpoint,
      );
    } catch (_) {
      return _ResolvedBindings._fromDynamicLibrary(
        HnswNativeBindings.sidecarLibraryPath(),
      );
    }
  }

  factory _ResolvedBindings._fromDynamicLibrary(String? libraryPath) {
    if (libraryPath == null) {
      final fileName = HnswNativeBindings.libraryFileNameForCurrentPlatform();
      throw StateError(
        'dart_hnsw could not resolve a native asset or sidecar library. '
        'For compiled executables, place $fileName next to the executable or set $_libraryPathEnvVar.',
      );
    }

    final file = File(libraryPath);
    if (!file.existsSync()) {
      throw StateError(
        'dart_hnsw sidecar library not found at ${file.absolute.path}. '
        'Set $_libraryPathEnvVar to the correct file path.',
      );
    }

    final library = ffi.DynamicLibrary.open(file.absolute.path);
    final abiVersion = library
        .lookupFunction<_NativeHnswAbiVersion, _DartHnswAbiVersion>(
          'hnsw_abi_version',
        )();
    if (abiVersion != _hnswAbiVersion) {
      throw StateError(
        'dart_hnsw sidecar ABI mismatch: expected $_hnswAbiVersion, got $abiVersion.',
      );
    }

    return _ResolvedBindings(
      backendDescription: 'dynamic-library:${file.absolute.path}',
      hnswOpen: library.lookupFunction<_NativeHnswOpen, _DartHnswOpen>(
        'hnsw_open',
      ),
      hnswClose: library.lookupFunction<_NativeHnswClose, _DartHnswClose>(
        'hnsw_close',
      ),
      hnswContextId: library
          .lookupFunction<_NativeHnswContextId, _DartHnswContextId>(
            'hnsw_context_id',
          ),
      hnswRegistrySize: library
          .lookupFunction<_NativeHnswRegistrySize, _DartHnswRegistrySize>(
            'hnsw_registry_size',
          ),
      hnswUpsertVector: library
          .lookupFunction<_NativeHnswUpsertVector, _DartHnswUpsertVector>(
            'hnsw_upsert_vector',
          ),
      hnswUpsertBatch: library
          .lookupFunction<_NativeHnswUpsertBatch, _DartHnswUpsertBatch>(
            'hnsw_upsert_batch',
          ),
      hnswMarkDeleted: library
          .lookupFunction<_NativeHnswMarkDeleted, _DartHnswMarkDeleted>(
            'hnsw_mark_deleted',
          ),
      hnswResize: library.lookupFunction<_NativeHnswResize, _DartHnswResize>(
        'hnsw_resize',
      ),
      hnswSetEfSearch: library
          .lookupFunction<_NativeHnswSetEfSearch, _DartHnswSetEfSearch>(
            'hnsw_set_ef_search',
          ),
      hnswSearchKnn: library
          .lookupFunction<_NativeHnswSearchKnn, _DartHnswSearchKnn>(
            'hnsw_search_knn',
          ),
      hnswElementCount: library
          .lookupFunction<_NativeHnswElementCount, _DartHnswElementCount>(
            'hnsw_element_count',
          ),
      hnswDeletedCount: library
          .lookupFunction<_NativeHnswDeletedCount, _DartHnswDeletedCount>(
            'hnsw_deleted_count',
          ),
      hnswMaxElements: library
          .lookupFunction<_NativeHnswMaxElements, _DartHnswMaxElements>(
            'hnsw_max_elements',
          ),
      hnswGetEfSearch: library
          .lookupFunction<_NativeHnswGetEfSearch, _DartHnswGetEfSearch>(
            'hnsw_get_ef_search',
          ),
      hnswCheckpoint: library
          .lookupFunction<_NativeHnswCheckpoint, _DartHnswCheckpoint>(
            'hnsw_checkpoint',
          ),
    );
  }
}
