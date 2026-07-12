import 'dart:ffi';
import 'dart:io';

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
  // T17: same path opened with wrong dimensions while shared context exists → error (-3)
  // -----------------------------------------------------------------------
  test(
    'T17: same path opened with wrong dimensions while shared context exists → error (-3)',
    () {
      final path = testIndexPath('t17_config_mismatch.index');
      cleanTestIndex(path);

      final errorCode = calloc<Int32>();
      final pathPtr = path.toNativeUtf8();

      try {
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

        final handle2 = native.hnswOpen(
          pathPtr,
          8,
          100,
          16,
          200,
          64,
          0,
          1,
          0,
          errorCode,
        );
        expect(errorCode.value, equals(-3));
        expect(isNullPointer(handle2), isTrue);
        expect(native.hnswClose(handle1), equals(0));
        expect(native.hnswRegistrySize(), equals(0));
      } finally {
        calloc.free(errorCode);
        calloc.free(pathPtr);
        cleanTestIndex(path);
      }
    },
  );

  // -----------------------------------------------------------------------
  // T20: missing file → open creates a new empty index (elementCount == 0)
  // -----------------------------------------------------------------------
  test('T20: missing file → open creates a new empty index', () {
    final path = testIndexPath('t20_missing_file.index');
    cleanTestIndex(path);

    // Ensure file does not exist
    expect(File(path).existsSync(), isFalse);

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
      expect(isNullPointer(handle), isFalse);

      expect(native.hnswElementCount(handle), equals(0));

      expect(native.hnswClose(handle), equals(0));
      expect(native.hnswRegistrySize(), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(pathPtr);
      cleanTestIndex(path);
    }
  });

  // -----------------------------------------------------------------------
  // T21: corrupt file → open → HnswLoadException (-5)
  // -----------------------------------------------------------------------
  test('T21: corrupt file → open → error -5', () {
    final path = testIndexPath('t21_corrupt_file.index');
    cleanTestIndex(path);

    // Write garbage
    File(path).writeAsBytesSync([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

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
      expect(errorCode.value, equals(-5));
      expect(isNullPointer(handle), isTrue);
    } finally {
      calloc.free(errorCode);
      calloc.free(pathPtr);
      cleanTestIndex(path);
    }
  });

  // -----------------------------------------------------------------------
  // T22: cosine metric: upsert zero vector → -8
  // -----------------------------------------------------------------------
  test('T22: cosine metric: upsert zero vector → -8', () {
    final path = testIndexPath('t22_zero_vector.index');
    cleanTestIndex(path);

    final errorCode = calloc<Int32>();
    final pathPtr = path.toNativeUtf8();

    try {
      // metric=2 for cosine
      final handle = native.hnswOpen(
        pathPtr,
        4,
        100,
        16,
        200,
        64,
        2,
        1,
        0,
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
      calloc.free(pathPtr);
      cleanTestIndex(path);
    }
  });

  // -----------------------------------------------------------------------
  // T24: markDeleted(id) → elementCount unchanged
  // -----------------------------------------------------------------------
  test('T24: markDeleted(id) → elementCount unchanged', () {
    final path = testIndexPath('t24_deleted_count.index');
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

      // Insert two vectors
      var vector = calloc<Float>(4);
      vector[0] = 1.0;
      expect(native.hnswUpsertVector(handle, 1, vector), equals(0));
      calloc.free(vector);

      vector = calloc<Float>(4);
      vector[0] = 2.0;
      expect(native.hnswUpsertVector(handle, 2, vector), equals(0));
      calloc.free(vector);

      expect(native.hnswElementCount(handle), equals(2));
      expect(native.hnswDeletedCount(handle), equals(0));

      // Mark one deleted
      expect(native.hnswMarkDeleted(handle, 1), equals(0));

      // elementCount unchanged (mark-based, not removal)
      expect(native.hnswElementCount(handle), equals(2));
      expect(native.hnswDeletedCount(handle), equals(1));

      expect(native.hnswClose(handle), equals(0));
      expect(native.hnswRegistrySize(), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(pathPtr);
      cleanTestIndex(path);
    }
  });

  // -----------------------------------------------------------------------
  // T25: same path opened with different metric while shared context exists fails (-3)
  // -----------------------------------------------------------------------
  test(
    'T25: same path opened with different metric while shared context exists fails (-3)',
    () {
      final path = testIndexPath('t25_metric_mismatch.index');
      cleanTestIndex(path);

      final errorCode = calloc<Int32>();
      final pathPtr = path.toNativeUtf8();

      try {
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

        final handle2 = native.hnswOpen(
          pathPtr,
          4,
          100,
          16,
          200,
          64,
          1,
          1,
          0,
          errorCode,
        );
        expect(errorCode.value, equals(-3));
        expect(isNullPointer(handle2), isTrue);
        expect(native.hnswClose(handle1), equals(0));
        expect(native.hnswRegistrySize(), equals(0));
      } finally {
        calloc.free(errorCode);
        calloc.free(pathPtr);
        cleanTestIndex(path);
      }
    },
  );

  test(
    'T27: same path opened with different allowReplaceDeleted while shared context exists fails (-3)',
    () {
      final path = testIndexPath('t27_allow_replace_deleted_mismatch.index');
      cleanTestIndex(path);

      final errorCode = calloc<Int32>();
      final pathPtr = path.toNativeUtf8();

      try {
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

        final handle2 = native.hnswOpen(
          pathPtr,
          4,
          100,
          16,
          200,
          64,
          0,
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
        calloc.free(pathPtr);
        cleanTestIndex(path);
      }
    },
  );
}
