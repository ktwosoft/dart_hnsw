import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:test/test.dart';

import 'package:dart_hnsw/src/native_bindings.dart';
import '_fixtures.dart';

// ---------------------------------------------------------------------------
// T13–T16: Persistence tests
// ---------------------------------------------------------------------------

void main() {
  late HnswNativeBindings native;

  setUp(() {
    native = HnswNativeBindings();
  });

  tearDown(() {
    expectCleanRegistry(native);
  });

  // -----------------------------------------------------------------------
  // T13: open(efSearch=64) → upsert → checkpoint → close
  //      → reopen(efSearch=32) → search returns persisted label
  //      → efSearch getter returns 32
  // -----------------------------------------------------------------------
  test('T13: checkpoint and reopen with efSearch reapplied', () {
    final path = testIndexPath('t13_checkpoint_reopen.index');
    cleanTestIndex(path);

    final errorCode = calloc<Int32>();
    final pathPtr = path.toNativeUtf8();

    try {
      // --- First session ---
      final handle1 = native.hnswOpen(
        pathPtr,
        4,
        1000,
        16,
        200,
        64,
        0,
        1,
        0,
        errorCode,
      );
      expect(errorCode.value, equals(0));

      final vector = calloc<Float>(4);
      vector[0] = 1.0;
      vector[1] = 0.0;
      vector[2] = 0.0;
      vector[3] = 0.0;
      expect(native.hnswUpsertVector(handle1, 1, vector), equals(0));
      calloc.free(vector);

      // Checkpoint
      expect(native.hnswCheckpoint(handle1), equals(0));

      expect(native.hnswClose(handle1), equals(0));
      expect(native.hnswRegistrySize(), equals(0));

      // --- Second session with different efSearch ---
      final handle2 = native.hnswOpen(
        pathPtr,
        4,
        1000,
        16,
        200,
        32,
        0,
        1,
        0,
        errorCode,
      );
      expect(errorCode.value, equals(0));

      // efSearch should be reapplied
      expect(native.hnswGetEfSearch(handle2), equals(32));

      // Search persisted vector
      final query = calloc<Float>(4);
      query[0] = 1.0;
      query[1] = 0.0;
      query[2] = 0.0;
      query[3] = 0.0;
      final outLabels = calloc<Uint64>(1);
      final outDistances = calloc<Float>(1);
      final outCount = calloc<Int32>();

      expect(
        native.hnswSearchKnn(
          handle2,
          query,
          1,
          outLabels,
          outDistances,
          outCount,
        ),
        equals(0),
      );
      expect(outCount.value, equals(1));
      expect(outLabels[0], equals(1));

      calloc.free(query);
      calloc.free(outLabels);
      calloc.free(outDistances);
      calloc.free(outCount);

      expect(native.hnswClose(handle2), equals(0));
      expect(native.hnswRegistrySize(), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(pathPtr);
      cleanTestIndex(path);
    }
  });

  // -----------------------------------------------------------------------
  // T14: checkpoint writes via tmp→rename (assert no leftover .tmp)
  // -----------------------------------------------------------------------
  test('T14: checkpoint writes via tmp→rename (assert no leftover .tmp)', () {
    final path = testIndexPath('t14_checkpoint_no_tmp_leftover.index');
    cleanTestIndex(path);

    final errorCode = calloc<Int32>();
    final pathPtr = path.toNativeUtf8();

    try {
      final handle = native.hnswOpen(
        pathPtr,
        4,
        100,
        16,
        200,
        64,
        0,
        1,
        0,
        errorCode,
      );
      expect(errorCode.value, equals(0));

      final vector = calloc<Float>(4);
      vector[0] = 1.0;
      expect(native.hnswUpsertVector(handle, 1, vector), equals(0));
      calloc.free(vector);

      expect(native.hnswCheckpoint(handle), equals(0));

      // Assert no leftover .tmp file
      expect(File('$path.tmp').existsSync(), isFalse);
      expect(File(path).existsSync(), isTrue);

      expect(native.hnswClose(handle), equals(0));
      expect(native.hnswRegistrySize(), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(pathPtr);
      cleanTestIndex(path);
    }
  });

  // -----------------------------------------------------------------------
  // T15: open new path → upsert → close() (no checkpoint) → reopen → elementCount == 0
  // -----------------------------------------------------------------------
  test('T15: close without checkpoint → reopen → elementCount == 0', () {
    final path = testIndexPath('t15_close_no_checkpoint.index');
    cleanTestIndex(path);

    final errorCode = calloc<Int32>();
    final pathPtr = path.toNativeUtf8();

    try {
      // First session
      final handle1 = native.hnswOpen(
        pathPtr,
        4,
        100,
        16,
        200,
        64,
        0,
        1,
        0,
        errorCode,
      );
      expect(errorCode.value, equals(0));

      final vector = calloc<Float>(4);
      vector[0] = 1.0;
      expect(native.hnswUpsertVector(handle1, 1, vector), equals(0));
      calloc.free(vector);

      // Close without checkpoint
      expect(native.hnswClose(handle1), equals(0));
      expect(native.hnswRegistrySize(), equals(0));

      // Reopen — should be a fresh index (no file persisted)
      final handle2 = native.hnswOpen(
        pathPtr,
        4,
        100,
        16,
        200,
        64,
        0,
        1,
        0,
        errorCode,
      );
      expect(errorCode.value, equals(0));

      // If no file existed, it's a new empty index
      expect(native.hnswElementCount(handle2), equals(0));

      expect(native.hnswClose(handle2), equals(0));
      expect(native.hnswRegistrySize(), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(pathPtr);
      cleanTestIndex(path);
    }
  });

  // -----------------------------------------------------------------------
  // T16: upsert → checkpoint → close → reopen → mutation present
  // -----------------------------------------------------------------------
  test('T16: upsert → checkpoint → close → reopen → mutation present', () {
    final path = testIndexPath('t16_checkpoint_persists.index');
    cleanTestIndex(path);

    final errorCode = calloc<Int32>();
    final pathPtr = path.toNativeUtf8();

    try {
      // First session
      final handle1 = native.hnswOpen(
        pathPtr,
        4,
        100,
        16,
        200,
        64,
        0,
        1,
        0,
        errorCode,
      );
      expect(errorCode.value, equals(0));

      final vector = calloc<Float>(4);
      vector[0] = 1.0;
      expect(native.hnswUpsertVector(handle1, 1, vector), equals(0));
      calloc.free(vector);

      expect(native.hnswCheckpoint(handle1), equals(0));
      expect(native.hnswClose(handle1), equals(0));
      expect(native.hnswRegistrySize(), equals(0));

      // Reopen
      final handle2 = native.hnswOpen(
        pathPtr,
        4,
        100,
        16,
        200,
        64,
        0,
        1,
        0,
        errorCode,
      );
      expect(errorCode.value, equals(0));

      expect(native.hnswElementCount(handle2), equals(1));

      final query = calloc<Float>(4);
      query[0] = 1.0;
      final outLabels = calloc<Uint64>(1);
      final outDistances = calloc<Float>(1);
      final outCount = calloc<Int32>();

      expect(
        native.hnswSearchKnn(
          handle2,
          query,
          1,
          outLabels,
          outDistances,
          outCount,
        ),
        equals(0),
      );
      expect(outCount.value, equals(1));
      expect(outLabels[0], equals(1));

      calloc.free(query);
      calloc.free(outLabels);
      calloc.free(outDistances);
      calloc.free(outCount);

      expect(native.hnswClose(handle2), equals(0));
      expect(native.hnswRegistrySize(), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(pathPtr);
      cleanTestIndex(path);
    }
  });
}
