import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

import 'package:ffi/ffi.dart';
import 'package:test/test.dart';

import 'package:dart_hnsw/src/native_bindings.dart';

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

String get projectRoot {
  final cwd = Directory.current.path;
  if (cwd.endsWith('/test')) {
    return Directory(cwd).parent.path;
  }
  return cwd;
}

final Directory _testRoot = Directory.systemTemp.createTempSync(
  'dart_hnsw_test_',
);

String testIndexPath(String name) => '${_testRoot.path}/$name';

String builtSidecarLibraryPath() {
  String fileName;
  if (Platform.isMacOS) {
    fileName = 'libdart_hnsw.dylib';
  } else if (Platform.isLinux) {
    fileName = 'libdart_hnsw.so';
  } else if (Platform.isWindows) {
    fileName = 'dart_hnsw.dll';
  } else {
    throw UnsupportedError(
      'No sidecar library naming for ${Platform.operatingSystem}',
    );
  }

  return '$projectRoot/.dart_tool/lib/$fileName';
}

// ---------------------------------------------------------------------------
// Pointer helpers
// ---------------------------------------------------------------------------

bool isNullPointer(Pointer pointer) => pointer.address == 0;

// ---------------------------------------------------------------------------
// Worker config / result
// ---------------------------------------------------------------------------

class WorkerConfig {
  final String path;
  final int workerId;
  final int iterations;
  final int sleepMs;
  final SendPort replyPort;

  WorkerConfig({
    required this.path,
    required this.workerId,
    required this.iterations,
    required this.sleepMs,
    required this.replyPort,
  });
}

class WorkerResult {
  final int contextId;
  final int writeCount;
  final String? error;

  WorkerResult({required this.contextId, this.writeCount = 0, this.error});

  WorkerResult.error(this.error) : contextId = 0, writeCount = 0;
}

// ---------------------------------------------------------------------------
// Worker helpers
// ---------------------------------------------------------------------------

void failWorker(SendPort replyPort, String message) {
  Isolate.exit(replyPort, WorkerResult.error(message));
}

Pointer<Void> openOrExit(
  HnswNativeBindings native,
  WorkerConfig config,
  Pointer<Int32> errorCode,
  Pointer<Utf8> pathPtr,
) {
  final handle = native.hnswOpen(
    pathPtr,
    4, // dimensions
    1000, // maxElements
    16, // m
    200, // efConstruction
    64, // efSearch
    0, // metric (l2)
    1, // allowReplaceDeleted
    0, // reservedOpenFlags
    errorCode,
  );

  if (isNullPointer(handle)) {
    failWorker(
      config.replyPort,
      'hnswOpen returned null, errorCode=${errorCode.value}',
    );
  }

  return handle;
}

// ---------------------------------------------------------------------------
// Upsert worker — each worker upserts unique labels
// ---------------------------------------------------------------------------

void upsertWorker(WorkerConfig config) {
  late final HnswNativeBindings native;

  try {
    native = HnswNativeBindings();
  } catch (e) {
    failWorker(config.replyPort, 'HnswNativeBindings failed: $e');
    return;
  }

  final errorCode = calloc<Int32>();
  final pathPtr = config.path.toNativeUtf8();

  try {
    final handle = openOrExit(native, config, errorCode, pathPtr);

    // Each worker gets a unique label range based on its worker id.
    // (SendPort.hashCode is identical across workers that share one
    // ReceivePort, so it cannot be used to derive a unique range.)
    final base = config.workerId * 100000;

    // TODO: dont make them colinear (just on x), doesn't work well with cosine
    for (var i = 0; i < config.iterations; i++) {
      final label = base + i;
      final vector = calloc<Float>(4);
      vector[0] = label.toDouble();
      vector[1] = 0.0;
      vector[2] = 0.0;
      vector[3] = 0.0;

      final result = native.hnswUpsertVector(handle, label, vector);
      calloc.free(vector);

      if (result != 0) {
        failWorker(config.replyPort, 'hnswUpsertVector failed with $result');
        return;
      }
    }

    final id = native.hnswContextId(handle);
    native.hnswClose(handle);

    Isolate.exit(
      config.replyPort,
      WorkerResult(contextId: id, writeCount: config.iterations),
    );
  } catch (e, st) {
    failWorker(config.replyPort, 'upsertWorker exception: $e\n$st');
  } finally {
    calloc.free(errorCode);
    calloc.free(pathPtr);
  }
}

// ---------------------------------------------------------------------------
// Search worker — searches repeatedly and reports nearest label
// ---------------------------------------------------------------------------

void searchWorker(WorkerConfig config) {
  late final HnswNativeBindings native;

  try {
    native = HnswNativeBindings();
  } catch (e) {
    failWorker(config.replyPort, 'HnswNativeBindings failed: $e');
    return;
  }

  final errorCode = calloc<Int32>();
  final pathPtr = config.path.toNativeUtf8();

  try {
    final handle = openOrExit(native, config, errorCode, pathPtr);

    for (var i = 0; i < config.iterations; i++) {
      final query = calloc<Float>(4);
      query[0] = 1.0;
      query[1] = 0.0;
      query[2] = 0.0;
      query[3] = 0.0;

      final outLabels = calloc<Uint64>(1);
      final outDistances = calloc<Float>(1);
      final outCount = calloc<Int32>();

      final result = native.hnswSearchKnn(
        handle,
        query,
        1,
        outLabels,
        outDistances,
        outCount,
      );

      calloc.free(query);
      calloc.free(outLabels);
      calloc.free(outDistances);
      calloc.free(outCount);

      if (result != 0) {
        failWorker(config.replyPort, 'hnswSearchKnn failed with $result');
        return;
      }
    }

    final id = native.hnswContextId(handle);
    native.hnswClose(handle);

    Isolate.exit(config.replyPort, WorkerResult(contextId: id));
  } catch (e, st) {
    failWorker(config.replyPort, 'searchWorker exception: $e\n$st');
  } finally {
    calloc.free(errorCode);
    calloc.free(pathPtr);
  }
}

// ---------------------------------------------------------------------------
// Mixed worker — random upsert/search
// ---------------------------------------------------------------------------

void mixedUpsertSearchWorker(WorkerConfig config) {
  late final HnswNativeBindings native;

  try {
    native = HnswNativeBindings();
  } catch (e) {
    failWorker(config.replyPort, 'HnswNativeBindings failed: $e');
    return;
  }

  final errorCode = calloc<Int32>();
  final pathPtr = config.path.toNativeUtf8();

  try {
    final handle = openOrExit(native, config, errorCode, pathPtr);
    final base = config.workerId * 100000;
    var writeCount = 0;

    for (var i = 0; i < config.iterations; i++) {
      if (i % 3 == 0) {
        // Upsert
        final label = base + i;
        final vector = calloc<Float>(4);
        vector[0] = label.toDouble();
        vector[1] = 0.0;
        vector[2] = 0.0;
        vector[3] = 0.0;

        final result = native.hnswUpsertVector(handle, label, vector);
        calloc.free(vector);

        if (result != 0) {
          failWorker(config.replyPort, 'hnswUpsertVector failed with $result');
          return;
        }
        writeCount++;
      } else {
        // Search
        final query = calloc<Float>(4);
        query[0] = 1.0;
        query[1] = 0.0;
        query[2] = 0.0;
        query[3] = 0.0;

        final outLabels = calloc<Uint64>(1);
        final outDistances = calloc<Float>(1);
        final outCount = calloc<Int32>();

        final result = native.hnswSearchKnn(
          handle,
          query,
          1,
          outLabels,
          outDistances,
          outCount,
        );

        calloc.free(query);
        calloc.free(outLabels);
        calloc.free(outDistances);
        calloc.free(outCount);

        if (result != 0) {
          failWorker(config.replyPort, 'hnswSearchKnn failed with $result');
          return;
        }
      }
    }

    final id = native.hnswContextId(handle);
    native.hnswClose(handle);

    Isolate.exit(
      config.replyPort,
      WorkerResult(contextId: id, writeCount: writeCount),
    );
  } catch (e, st) {
    failWorker(config.replyPort, 'mixedUpsertSearchWorker exception: $e\n$st');
  } finally {
    calloc.free(errorCode);
    calloc.free(pathPtr);
  }
}

// ---------------------------------------------------------------------------
// Spawn helper
// ---------------------------------------------------------------------------

Future<List<WorkerResult>> spawnWorkers({
  required String path,
  required int count,
  required int iterations,
  required int sleepMs,
  required void Function(WorkerConfig) workerFn,
}) async {
  final receivePort = ReceivePort();

  final spawns = <Future<void>>[];

  for (var i = 0; i < count; i++) {
    final config = WorkerConfig(
      path: path,
      workerId: i + 1,
      iterations: iterations,
      sleepMs: sleepMs,
      replyPort: receivePort.sendPort,
    );

    spawns.add(Isolate.spawn(workerFn, config));
  }

  await Future.wait(spawns);

  final results = await receivePort.take(count).cast<WorkerResult>().toList();
  receivePort.close();

  return results;
}

void expectWorkersSucceeded(List<WorkerResult> results) {
  for (final result in results) {
    expect(result.error, isNull, reason: 'Worker error: ${result.error}');
  }
}

// ---------------------------------------------------------------------------
// Cleanup helper — verifies no registry leaks
// ---------------------------------------------------------------------------

void expectCleanRegistry(HnswNativeBindings native) {
  final size = native.hnswRegistrySize();
  if (size != 0) {
    fail('Test leaked $size native registry entries');
  }
}

// ---------------------------------------------------------------------------
// Cleanup helper — removes test index files
// ---------------------------------------------------------------------------

void cleanTestIndex(String path) {
  final file = File(path);
  if (file.existsSync()) file.deleteSync();
  final tmp = File('$path.tmp');
  if (tmp.existsSync()) tmp.deleteSync();
  final lock = File('$path.lock');
  if (lock.existsSync()) lock.deleteSync();
}
