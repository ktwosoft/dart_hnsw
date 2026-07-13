@ffi.DefaultAsset('package:dart_hnsw/src/native_bindings.dart')
library;

import 'dart:ffi' as ffi;
import 'dart:io';

import 'package:ffi/ffi.dart';

typedef HnswHandle = ffi.Pointer<ffi.Void>;

final class NativeHnswStats extends ffi.Struct {
  @ffi.Uint64()
  external int elementCount;

  @ffi.Uint64()
  external int deletedCount;

  @ffi.Uint64()
  external int maxElements;

  @ffi.Int32()
  external int efSearch;
}

const _hnswAbiVersion = 3;
const _libraryPathEnvVar = 'DART_HNSW_LIBRARY_PATH';
const _forceDynamicLibraryEnvVar = 'DART_HNSW_FORCE_DYNAMIC_LIBRARY';

// ---------------------------------------------------------------------------
// Native typedefs
// ---------------------------------------------------------------------------

typedef _NativeHnswCreate =
    HnswHandle Function(
      ffi.Pointer<Utf8>,
      ffi.Int32,
      ffi.Int32,
      ffi.Int32,
      ffi.Int32,
      ffi.Int32,
      ffi.Int32,
      ffi.Int32,
      ffi.Pointer<ffi.Int32>,
    );
typedef _DartHnswCreate =
    HnswHandle Function(
      ffi.Pointer<Utf8>,
      int,
      int,
      int,
      int,
      int,
      int,
      int,
      ffi.Pointer<ffi.Int32>,
    );

typedef _NativeHnswLoad =
    ffi.Int32 Function(
      HnswHandle,
      ffi.Pointer<ffi.Uint8>,
      ffi.Int32,
      ffi.Int64,
    );
typedef _DartHnswLoad =
    int Function(HnswHandle, ffi.Pointer<ffi.Uint8>, int, int);

typedef _NativeHnswClose = ffi.Int32 Function(HnswHandle);
typedef _DartHnswClose = int Function(HnswHandle);

typedef _NativeHnswContextId = ffi.Uint64 Function(HnswHandle);
typedef _DartHnswContextId = int Function(HnswHandle);

typedef _NativeHnswGeneration = ffi.Uint64 Function(HnswHandle);
typedef _DartHnswGeneration = int Function(HnswHandle);

typedef _NativeHnswExclusiveMutationAccess = ffi.Int32 Function(HnswHandle);
typedef _DartHnswExclusiveMutationAccess = int Function(HnswHandle);

typedef _NativeHnswRegistrySize = ffi.Uint64 Function();
typedef _DartHnswRegistrySize = int Function();

typedef _NativeHnswClear = ffi.Int32 Function(HnswHandle, ffi.Int64);
typedef _DartHnswClear = int Function(HnswHandle, int);

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

typedef _NativeHnswStats =
    ffi.Int32 Function(HnswHandle, ffi.Pointer<NativeHnswStats>);
typedef _DartHnswStats = int Function(HnswHandle, ffi.Pointer<NativeHnswStats>);

typedef _NativeHnswCheckpoint =
    ffi.Int32 Function(
      HnswHandle,
      ffi.Pointer<ffi.Uint8>,
      ffi.Int32,
      ffi.Pointer<ffi.Int32>,
      ffi.Int64,
    );
typedef _DartHnswCheckpoint =
    int Function(
      HnswHandle,
      ffi.Pointer<ffi.Uint8>,
      int,
      ffi.Pointer<ffi.Int32>,
      int,
    );

typedef _NativeHnswAbiVersion = ffi.Int32 Function();
typedef _DartHnswAbiVersion = int Function();

// ---------------------------------------------------------------------------
// @ffi.Native external declarations (used by native-asset / static linking)
// ---------------------------------------------------------------------------

@ffi.Native<_NativeHnswCreate>(symbol: 'hnsw_create')
external HnswHandle nativeHnswCreate(
  ffi.Pointer<Utf8> collectionName,
  int dimensions,
  int maxElements,
  int m,
  int efConstruction,
  int efSearch,
  int metric,
  int allowReplaceDeleted,
  ffi.Pointer<ffi.Int32> errorCode,
);

@ffi.Native<_NativeHnswLoad>(symbol: 'hnsw_load')
external int nativeHnswLoad(
  HnswHandle handle,
  ffi.Pointer<ffi.Uint8> data,
  int dataSize,
  int expectedGeneration,
);

@ffi.Native<_NativeHnswClose>(symbol: 'hnsw_close')
external int nativeHnswClose(HnswHandle handle);

@ffi.Native<_NativeHnswContextId>(symbol: 'hnsw_context_id')
external int nativeHnswContextId(HnswHandle handle);

@ffi.Native<_NativeHnswGeneration>(symbol: 'hnsw_generation')
external int nativeHnswGeneration(HnswHandle handle);

@ffi.Native<_NativeHnswExclusiveMutationAccess>(
  symbol: 'hnsw_acquire_exclusive_mutation_access',
)
external int nativeHnswAcquireExclusiveMutationAccess(HnswHandle handle);

@ffi.Native<_NativeHnswExclusiveMutationAccess>(
  symbol: 'hnsw_release_exclusive_mutation_access',
)
external int nativeHnswReleaseExclusiveMutationAccess(HnswHandle handle);

@ffi.Native<_NativeHnswRegistrySize>(symbol: 'hnsw_registry_size')
external int nativeHnswRegistrySize();

@ffi.Native<_NativeHnswClear>(symbol: 'hnsw_clear')
external int nativeHnswClear(HnswHandle handle, int expectedGeneration);

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

@ffi.Native<_NativeHnswStats>(symbol: 'hnsw_stats')
external int nativeHnswStats(
  HnswHandle handle,
  ffi.Pointer<NativeHnswStats> outStats,
);

@ffi.Native<_NativeHnswCheckpoint>(symbol: 'hnsw_checkpoint')
external int nativeHnswCheckpoint(
  HnswHandle handle,
  ffi.Pointer<ffi.Uint8> outBuffer,
  int bufferSize,
  ffi.Pointer<ffi.Int32> outWritten,
  int expectedGeneration,
);

@ffi.Native<_NativeHnswAbiVersion>(symbol: 'hnsw_abi_version')
external int nativeHnswAbiVersion();

// ---------------------------------------------------------------------------
// HnswNativeBindings — resolved function pointers
// ---------------------------------------------------------------------------

class HnswNativeBindings {
  final String backendDescription;
  final _DartHnswCreate _hnswCreate;
  final _DartHnswLoad _hnswLoad;
  final _DartHnswClose _hnswClose;
  final _DartHnswContextId _hnswContextId;
  final _DartHnswGeneration _hnswGeneration;
  final _DartHnswExclusiveMutationAccess _hnswAcquireExclusiveMutationAccess;
  final _DartHnswExclusiveMutationAccess _hnswReleaseExclusiveMutationAccess;
  final _DartHnswRegistrySize _hnswRegistrySize;
  final _DartHnswClear _hnswClear;
  final _DartHnswUpsertVector _hnswUpsertVector;
  final _DartHnswUpsertBatch _hnswUpsertBatch;
  final _DartHnswMarkDeleted _hnswMarkDeleted;
  final _DartHnswResize _hnswResize;
  final _DartHnswSetEfSearch _hnswSetEfSearch;
  final _DartHnswSearchKnn _hnswSearchKnn;
  final _DartHnswStats _hnswStats;
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
      _hnswCreate = resolved.hnswCreate,
      _hnswLoad = resolved.hnswLoad,
      _hnswClose = resolved.hnswClose,
      _hnswContextId = resolved.hnswContextId,
      _hnswGeneration = resolved.hnswGeneration,
      _hnswAcquireExclusiveMutationAccess =
          resolved.hnswAcquireExclusiveMutationAccess,
      _hnswReleaseExclusiveMutationAccess =
          resolved.hnswReleaseExclusiveMutationAccess,
      _hnswRegistrySize = resolved.hnswRegistrySize,
      _hnswClear = resolved.hnswClear,
      _hnswUpsertVector = resolved.hnswUpsertVector,
      _hnswUpsertBatch = resolved.hnswUpsertBatch,
      _hnswMarkDeleted = resolved.hnswMarkDeleted,
      _hnswResize = resolved.hnswResize,
      _hnswSetEfSearch = resolved.hnswSetEfSearch,
      _hnswSearchKnn = resolved.hnswSearchKnn,
      _hnswStats = resolved.hnswStats,
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

    final resolvedExecutable = Platform.resolvedExecutable;
    if (resolvedExecutable.isNotEmpty) {
      final candidate = File(
        '${File(resolvedExecutable).absolute.parent.path}'
        '${Platform.pathSeparator}${libraryFileNameForCurrentPlatform()}',
      );
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

  // -- Bindings --

  HnswHandle hnswCreate(
    ffi.Pointer<Utf8> collectionName,
    int dimensions,
    int maxElements,
    int m,
    int efConstruction,
    int efSearch,
    int metric,
    int allowReplaceDeleted,
    ffi.Pointer<ffi.Int32> errorCode,
  ) => _hnswCreate(
    collectionName,
    dimensions,
    maxElements,
    m,
    efConstruction,
    efSearch,
    metric,
    allowReplaceDeleted,
    errorCode,
  );

  int hnswLoad(
    HnswHandle handle,
    ffi.Pointer<ffi.Uint8> data,
    int dataSize,
    int expectedGeneration,
  ) => _hnswLoad(handle, data, dataSize, expectedGeneration);

  int hnswClose(HnswHandle handle) => _hnswClose(handle);

  int hnswContextId(HnswHandle handle) => _hnswContextId(handle);

  int hnswGeneration(HnswHandle handle) => _hnswGeneration(handle);

  int hnswAcquireExclusiveMutationAccess(HnswHandle handle) =>
      _hnswAcquireExclusiveMutationAccess(handle);

  int hnswReleaseExclusiveMutationAccess(HnswHandle handle) =>
      _hnswReleaseExclusiveMutationAccess(handle);

  int hnswRegistrySize() => _hnswRegistrySize();

  int hnswClear(HnswHandle handle, int expectedGeneration) =>
      _hnswClear(handle, expectedGeneration);

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

  int hnswStats(HnswHandle handle, ffi.Pointer<NativeHnswStats> outStats) =>
      _hnswStats(handle, outStats);

  int hnswCheckpoint(
    HnswHandle handle,
    ffi.Pointer<ffi.Uint8> outBuffer,
    int bufferSize,
    ffi.Pointer<ffi.Int32> outWritten,
    int expectedGeneration,
  ) => _hnswCheckpoint(
    handle,
    outBuffer,
    bufferSize,
    outWritten,
    expectedGeneration,
  );
}

// ---------------------------------------------------------------------------
// _ResolvedBindings — resolves to native-asset or dynamic-library
// ---------------------------------------------------------------------------

class _ResolvedBindings {
  final String backendDescription;
  final _DartHnswCreate hnswCreate;
  final _DartHnswLoad hnswLoad;
  final _DartHnswClose hnswClose;
  final _DartHnswContextId hnswContextId;
  final _DartHnswGeneration hnswGeneration;
  final _DartHnswExclusiveMutationAccess hnswAcquireExclusiveMutationAccess;
  final _DartHnswExclusiveMutationAccess hnswReleaseExclusiveMutationAccess;
  final _DartHnswRegistrySize hnswRegistrySize;
  final _DartHnswClear hnswClear;
  final _DartHnswUpsertVector hnswUpsertVector;
  final _DartHnswUpsertBatch hnswUpsertBatch;
  final _DartHnswMarkDeleted hnswMarkDeleted;
  final _DartHnswResize hnswResize;
  final _DartHnswSetEfSearch hnswSetEfSearch;
  final _DartHnswSearchKnn hnswSearchKnn;
  final _DartHnswStats hnswStats;
  final _DartHnswCheckpoint hnswCheckpoint;

  _ResolvedBindings({
    required this.backendDescription,
    required this.hnswCreate,
    required this.hnswLoad,
    required this.hnswClose,
    required this.hnswContextId,
    required this.hnswGeneration,
    required this.hnswAcquireExclusiveMutationAccess,
    required this.hnswReleaseExclusiveMutationAccess,
    required this.hnswRegistrySize,
    required this.hnswClear,
    required this.hnswUpsertVector,
    required this.hnswUpsertBatch,
    required this.hnswMarkDeleted,
    required this.hnswResize,
    required this.hnswSetEfSearch,
    required this.hnswSearchKnn,
    required this.hnswStats,
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
        hnswCreate: nativeHnswCreate,
        hnswLoad: nativeHnswLoad,
        hnswClose: nativeHnswClose,
        hnswContextId: nativeHnswContextId,
        hnswGeneration: nativeHnswGeneration,
        hnswAcquireExclusiveMutationAccess:
            nativeHnswAcquireExclusiveMutationAccess,
        hnswReleaseExclusiveMutationAccess:
            nativeHnswReleaseExclusiveMutationAccess,
        hnswRegistrySize: nativeHnswRegistrySize,
        hnswClear: nativeHnswClear,
        hnswUpsertVector: nativeHnswUpsertVector,
        hnswUpsertBatch: nativeHnswUpsertBatch,
        hnswMarkDeleted: nativeHnswMarkDeleted,
        hnswResize: nativeHnswResize,
        hnswSetEfSearch: nativeHnswSetEfSearch,
        hnswSearchKnn: nativeHnswSearchKnn,
        hnswStats: nativeHnswStats,
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
      hnswCreate: library.lookupFunction<_NativeHnswCreate, _DartHnswCreate>(
        'hnsw_create',
      ),
      hnswLoad: library.lookupFunction<_NativeHnswLoad, _DartHnswLoad>(
        'hnsw_load',
      ),
      hnswClose: library.lookupFunction<_NativeHnswClose, _DartHnswClose>(
        'hnsw_close',
      ),
      hnswContextId: library
          .lookupFunction<_NativeHnswContextId, _DartHnswContextId>(
            'hnsw_context_id',
          ),
      hnswGeneration: library
          .lookupFunction<_NativeHnswGeneration, _DartHnswGeneration>(
            'hnsw_generation',
          ),
      hnswAcquireExclusiveMutationAccess: library
          .lookupFunction<
            _NativeHnswExclusiveMutationAccess,
            _DartHnswExclusiveMutationAccess
          >('hnsw_acquire_exclusive_mutation_access'),
      hnswReleaseExclusiveMutationAccess: library
          .lookupFunction<
            _NativeHnswExclusiveMutationAccess,
            _DartHnswExclusiveMutationAccess
          >('hnsw_release_exclusive_mutation_access'),
      hnswRegistrySize: library
          .lookupFunction<_NativeHnswRegistrySize, _DartHnswRegistrySize>(
            'hnsw_registry_size',
          ),
      hnswClear: library.lookupFunction<_NativeHnswClear, _DartHnswClear>(
        'hnsw_clear',
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
      hnswStats: library.lookupFunction<_NativeHnswStats, _DartHnswStats>(
        'hnsw_stats',
      ),
      hnswCheckpoint: library
          .lookupFunction<_NativeHnswCheckpoint, _DartHnswCheckpoint>(
            'hnsw_checkpoint',
          ),
    );
  }
}
