import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:test/test.dart';

import 'package:dart_hnsw/src/native_bindings.dart';
import '_fixtures.dart';

// ---------------------------------------------------------------------------
// T18: Process safety test
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
  // T18: process A opens index → process B open same path → -4
  // -----------------------------------------------------------------------
  test('T18: process A opens index → process B open same path → -4', () {
    final path = testIndexPath('t18_process_lock.index');
    cleanTestIndex(path);

    final errorCode = calloc<Int32>();
    final pathPtr = path.toNativeUtf8();

    try {
      // Process A (this process) opens the index
      final handleA = native.hnswOpen(
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
      expect(isNullPointer(handleA), isFalse);

      // Spawn a child Dart process that tries to open the same path
      // The child process should get -4 (locked by another process)
      final result = Process.runSync(
        'dart',
        ['run', '--enable-asserts', 'test/_process_child.dart', path],
        runInShell: true,
        workingDirectory: projectRoot,
      );

      // Parse child output
      final stdout = result.stdout as String;
      final stderr = result.stderr as String;

      // Extract error code from child output
      final errorMatch = RegExp(r"ERROR_CODE='(-?\d+)'").firstMatch(stdout);
      final handleNullMatch = RegExp(
        r"HANDLE_NULL='(true|false)'",
      ).firstMatch(stdout);

      expect(
        errorMatch,
        isNotNull,
        reason: 'Child process output: $stdout\n$stderr',
      );
      expect(
        handleNullMatch,
        isNotNull,
        reason: 'Child process output: $stdout\n$stderr',
      );

      final childErrorCode = int.parse(errorMatch!.group(1)!);
      final childHandleNull = handleNullMatch!.group(1) == 'true';

      expect(
        childErrorCode,
        equals(-4),
        reason: 'Child process should get -4 (locked)',
      );
      expect(
        childHandleNull,
        isTrue,
        reason: 'Child process handle should be null',
      );

      expect(native.hnswClose(handleA), equals(0));
      expect(native.hnswRegistrySize(), equals(0));
    } finally {
      calloc.free(errorCode);
      calloc.free(pathPtr);
      cleanTestIndex(path);
    }
  });
}
