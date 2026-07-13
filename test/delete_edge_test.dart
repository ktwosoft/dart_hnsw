import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:test/test.dart';

import 'package:dart_hnsw/src/native_bindings.dart';
import '_fixtures.dart';

// ---------------------------------------------------------------------------
// T17, T20–T27: In-process config validation, lifecycle edge cases, delete semantics
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
  // T17: same collection opened with wrong dimensions while shared context exists → error (-3)
  // -----------------------------------------------------------------------
  test(
    'T17: same collection opened with wrong dimensions while shared context exists → error (-3)',
    () {
      final name = testCollectionName('t17_config_mismatch');
      final errorCode = calloc<Int32>();
      final namePtr = name.toNativeUtf8();

      try {
        final handle1 = native.hnswCreate(
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
        expect(errorCode.value, equals(0));

        final handle2 = native.hnswCreate(
          namePtr,
          8,
          100,
          16,
          200,
          64,
          0,
          1,
          errorCode,
        );
        expect(errorCode.value, equals(-3));
        expect(isNullPointer(handle2), isTrue);
        expect(native.hnswClose(handle1), equals(0));
        expect(native.hnswRegistrySize(), equals(0));
      } finally {
        calloc.free(errorCode);
        calloc.free(namePtr);
      }
    },
  );

  // -----------------------------------------------------------------------
  // T20: create new index → elementCount == 0
  // -----------------------------------------------------------------------
  test('T20: create new index → elementCount == 0', () {
    final name = testCollectionName('t20_new_index');
    final errorCode = calloc<Int32>();
    final namePtr = name.toNativeUtf8();

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
      expect(errorCode.value, equals(0));
      expect(isNullPointer(handle), isFalse);

      expect(readNativeStats(native, handle).elementCount, equals(0));

      expect(native.hnswClose(handle), equals(0));
      expect(native.hnswRegistrySize(), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(namePtr);
    }
  });

  // -----------------------------------------------------------------------
  // T21: load corrupt bytes → error -5
  // -----------------------------------------------------------------------
  test('T21: load corrupt bytes → error -5', () {
    final name = testCollectionName('t21_corrupt_bytes');
    final errorCode = calloc<Int32>();
    final namePtr = name.toNativeUtf8();

    // Garbage bytes
    final garbage = Uint8List.fromList([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    final dataPtr = calloc<Uint8>(garbage.length);
    for (var i = 0; i < garbage.length; i++) {
      dataPtr[i] = garbage[i];
    }

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
      expect(errorCode.value, equals(0));
      expect(native.hnswLoad(handle, dataPtr, garbage.length, 0), equals(-5));
      expect(native.hnswClose(handle), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(namePtr);
      calloc.free(dataPtr);
    }
  });

  // -----------------------------------------------------------------------
  // T22: cosine metric: upsert zero vector → -8
  // -----------------------------------------------------------------------
  test('T22: cosine metric: upsert zero vector → -8', () {
    final name = testCollectionName('t22_zero_vector');
    final errorCode = calloc<Int32>();
    final namePtr = name.toNativeUtf8();

    try {
      // metric=2 for cosine
      final handle = native.hnswCreate(
        namePtr,
        4,
        100,
        16,
        200,
        64,
        2,
        1,
        errorCode,
      );
      expect(errorCode.value, equals(0));

      final vector = calloc<Float>(4);
      vector[0] = 0.0;
      vector[1] = 0.0;
      vector[2] = 0.0;
      vector[3] = 0.0;

      final result = native.hnswUpsertVector(handle, 1, vector);
      expect(result, equals(-8));

      calloc.free(vector);

      expect(native.hnswClose(handle), equals(0));
      expect(native.hnswRegistrySize(), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(namePtr);
    }
  });

  // -----------------------------------------------------------------------
  // T24: markDeleted(id) → elementCount unchanged
  // -----------------------------------------------------------------------
  test('T24: markDeleted(id) → elementCount unchanged', () {
    final name = testCollectionName('t24_deleted_count');
    final errorCode = calloc<Int32>();
    final namePtr = name.toNativeUtf8();

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
      expect(errorCode.value, equals(0));

      // Insert two vectors
      var vector = calloc<Float>(4);
      vector[0] = 1.0;
      expect(native.hnswUpsertVector(handle, 1, vector), equals(0));
      calloc.free(vector);

      vector = calloc<Float>(4);
      vector[0] = 2.0;
      expect(native.hnswUpsertVector(handle, 2, vector), equals(0));
      calloc.free(vector);

      var stats = readNativeStats(native, handle);
      expect(stats.elementCount, equals(2));
      expect(stats.deletedCount, equals(0));

      // Mark one deleted
      expect(native.hnswMarkDeleted(handle, 1), equals(0));

      // elementCount unchanged (mark-based, not removal)
      stats = readNativeStats(native, handle);
      expect(stats.elementCount, equals(2));
      expect(stats.deletedCount, equals(1));

      expect(native.hnswClose(handle), equals(0));
      expect(native.hnswRegistrySize(), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(namePtr);
    }
  });

  // -----------------------------------------------------------------------
  // T25: same collection opened with different metric while shared context exists fails (-3)
  // -----------------------------------------------------------------------
  test(
    'T25: same collection opened with different metric while shared context exists fails (-3)',
    () {
      final name = testCollectionName('t25_metric_mismatch');
      final errorCode = calloc<Int32>();
      final namePtr = name.toNativeUtf8();

      try {
        final handle1 = native.hnswCreate(
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
        expect(errorCode.value, equals(0));

        final handle2 = native.hnswCreate(
          namePtr,
          4,
          100,
          16,
          200,
          64,
          1,
          1,
          errorCode,
        );
        expect(errorCode.value, equals(-3));
        expect(isNullPointer(handle2), isTrue);
        expect(native.hnswClose(handle1), equals(0));
        expect(native.hnswRegistrySize(), equals(0));
      } finally {
        calloc.free(errorCode);
        calloc.free(namePtr);
      }
    },
  );

  test(
    'T27: same collection opened with different allowReplaceDeleted while shared context exists fails (-3)',
    () {
      final name = testCollectionName('t27_allow_replace_deleted_mismatch');
      final errorCode = calloc<Int32>();
      final namePtr = name.toNativeUtf8();

      try {
        final handle1 = native.hnswCreate(
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
        expect(errorCode.value, equals(0));

        final handle2 = native.hnswCreate(
          namePtr,
          4,
          100,
          16,
          200,
          64,
          0,
          0,
          errorCode,
        );
        expect(errorCode.value, equals(-3));
        expect(isNullPointer(handle2), isTrue);

        expect(native.hnswClose(handle1), equals(0));
        expect(native.hnswRegistrySize(), equals(0));
      } finally {
        calloc.free(errorCode);
        calloc.free(namePtr);
      }
    },
  );
}
