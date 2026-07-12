import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

import 'package:ffi/ffi.dart';
import 'package:test/test.dart';

import 'package:dart_hnsw/src/native_bindings.dart';

// ---------------------------------------------------------------------------
// Paths / names
// ---------------------------------------------------------------------------

String get projectRoot {
  final cwd = Directory.current.path;
  if (cwd.endsWith('/test')) {
    return Directory(cwd).parent.path;
  }
  return cwd;
}

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

/// Unique collection name generator — uses a monotonic counter so tests
/// don't collide on the shared in-process registry.
int _collectionCounter = 0;

String testCollectionName(String label) {
  _collectionCounter++;
  return 'test_${_collectionCounter}_$label';
}

// ---------------------------------------------------------------------------
// Pointer helpers
// ---------------------------------------------------------------------------

bool isNullPointer(Pointer pointer) => pointer.address == 0;

// ---------------------------------------------------------------------------
// Worker config / result
// ---------------------------------------------------------------------------

class WorkerConfig {
  final String collectionName;
  final int workerId;
  final int iterations;
  final int sleepMs;
  final SendPort replyPort;
  final SendPort? cleanupPort;

  WorkerConfig({
    required this.collectionName,
    required this.workerId,
    required this.iterations,
    required this.sleepMs,
    required this.replyPort,
    this.cleanupPort,
  });
}

class WorkerResult {
  final int contextId;
  final int writeCount;
  final String? error;
  final SendPort? releasePort;

  WorkerResult({
    required this.contextId,
    this.writeCount = 0,
    this.error,
    this.releasePort,
  });

  WorkerResult.error(this.error)
    : contextId = 0,
      writeCount = 0,
      releasePort = null;
}

// ---------------------------------------------------------------------------
// Worker helpers
// ---------------------------------------------------------------------------

void failWorker(SendPort replyPort, String message) {
  Isolate.exit(replyPort, WorkerResult.error(message));
}

Pointer<Void> createOrExit(
  HnswNativeBindings native,
  WorkerConfig config,
  Pointer<Int32> errorCode,
  Pointer<Utf8> namePtr,
) {
  final handle = native.hnswCreate(
    namePtr,
    4, // dimensions
    10000, // maxElements
    16, // m
    200, // efConstruction
    64, // efSearch
    0, // metric (l2)
    1, // allowReplaceDeleted
    errorCode,
  );

  if (isNullPointer(handle)) {
    failWorker(
      config.replyPort,
      'hnswCreate returned null, errorCode=${errorCode.value}',
    );
  }

  return handle;
}

void createOnlyWorker(WorkerConfig config) {
  late final HnswNativeBindings native;

  try {
    native = HnswNativeBindings();
  } catch (e) {
    failWorker(config.replyPort, 'HnswNativeBindings failed: $e');
    return;
  }

  final errorCode = calloc<Int32>();
  final namePtr = config.collectionName.toNativeUtf8();
  var handedOff = false;

  try {
    final handle = native.hnswCreate(
      namePtr,
      4,
      100,
      16,
      200,
      64,
      0,
      1,
      errorCode,
    );

    if (isNullPointer(handle)) {
      failWorker(
        config.replyPort,
        'hnswCreate returned null, errorCode=${errorCode.value}',
      );
      return;
    }

    final id = native.hnswContextId(handle);
    final releasePort = ReceivePort();
    releasePort.listen((_) {
      releasePort.close();
      native.hnswClose(handle);
      calloc.free(errorCode);
      calloc.free(namePtr);
      config.cleanupPort?.send(null);
    });
    handedOff = true;
    config.replyPort.send(
      WorkerResult(contextId: id, releasePort: releasePort.sendPort),
    );
  } catch (e, st) {
    failWorker(config.replyPort, 'createOnlyWorker exception: $e\n$st');
  } finally {
    if (!handedOff) {
      calloc.free(errorCode);
      calloc.free(namePtr);
    }
  }
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
  final namePtr = config.collectionName.toNativeUtf8();

  try {
    final handle = createOrExit(native, config, errorCode, namePtr);

    final base = config.workerId * 100000;

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
    calloc.free(namePtr);
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
  final namePtr = config.collectionName.toNativeUtf8();

  try {
    final handle = createOrExit(native, config, errorCode, namePtr);

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
    calloc.free(namePtr);
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
  final namePtr = config.collectionName.toNativeUtf8();

  try {
    final handle = createOrExit(native, config, errorCode, namePtr);
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
    calloc.free(namePtr);
  }
}

// ---------------------------------------------------------------------------
// Spawn helper
// ---------------------------------------------------------------------------

Future<List<WorkerResult>> spawnWorkers({
  required String collectionName,
  required int count,
  required int iterations,
  required int sleepMs,
  required void Function(WorkerConfig) workerFn,
}) async {
  final receivePort = ReceivePort();
  final cleanupReceivePort = ReceivePort();

  final spawns = <Future<void>>[];

  for (var i = 0; i < count; i++) {
    final config = WorkerConfig(
      collectionName: collectionName,
      workerId: i + 1,
      iterations: iterations,
      sleepMs: sleepMs,
      replyPort: receivePort.sendPort,
      cleanupPort: cleanupReceivePort.sendPort,
    );

    spawns.add(Isolate.spawn(workerFn, config));
  }

  await Future.wait(spawns);

  final results = await receivePort.take(count).cast<WorkerResult>().toList();
  receivePort.close();

  final heldResults = results
      .where((result) => result.releasePort != null)
      .toList(growable: false);
  for (final result in heldResults) {
    result.releasePort!.send(null);
  }
  if (heldResults.isNotEmpty) {
    await cleanupReceivePort.take(heldResults.length).toList();
  }
  cleanupReceivePort.close();

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
