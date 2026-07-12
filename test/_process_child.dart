import 'dart:ffi';
import 'dart:io';

import 'package:dart_hnsw/src/native_bindings.dart';
import 'package:ffi/ffi.dart';

/// Child process for T18: tries to open an index path that another process
/// has already locked. Expects error code -4 (HNSW_ERR_LOCKED_PROCESS).
void main(List<String> args) {
  if (args.isEmpty) {
    print('Usage: dart run test/_process_child.dart <path>');
    exitCode = 1;
    return;
  }

  final path = args[0];
  final native = HnswNativeBindings();

  final errorCode = calloc<Int32>();
  final pathPtr = path.toNativeUtf8();
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
  print("ERROR_CODE='${errorCode.value}'");
  print("HANDLE_NULL='${handle == nullptr}'");
  calloc.free(errorCode);
  calloc.free(pathPtr);
}
