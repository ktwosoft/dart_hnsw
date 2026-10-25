import 'dart:async';
import 'dart:js_interop';
import 'dart:typed_data';
import 'package:web/web.dart' as web;

// MARK: One C++ module per JavaScript realm; no worker/RPC ownership.
WasmBindings? _bindings;
Future<void>? _initializing;
(Uri, Uri)? _urls;
WasmBindings get wasmBindings =>
    _bindings ??
    (throw StateError('Await HnswIndex.initialize before opening an index'));

Future<void> initializeWasm(Uri loaderUrl, Uri wasmUrl) {
  final urls = (Uri.base.resolveUri(loaderUrl), Uri.base.resolveUri(wasmUrl));
  if (_initializing != null) {
    if (_urls != urls) {
      throw StateError('HNSW is already initialized with different assets');
    }
    return _initializing!;
  }
  _urls = urls;
  return _initializing = _initialize(urls.$1, urls.$2).catchError((
    Object error,
    StackTrace stack,
  ) {
    _initializing = null;
    _urls = null;
    Error.throwWithStackTrace(error, stack);
  });
}

@JS('globalThis.document')
external JSObject? get _document;
@JS('globalThis.createDartHnsw')
external JSPromise<WasmBindings> _createModule(_ModuleOptions options);

Future<void> _initialize(Uri loaderUrl, Uri wasmUrl) async {
  // Applications running Dart in their own worker preload the classic loader
  // with importScripts. In a document, load the same script here.
  if (_document != null) {
    final script = web.HTMLScriptElement()..src = loaderUrl.toString();
    final loaded = Completer<void>();
    script.onload = ((web.Event _) {
      loaded.complete();
    }).toJS;
    script.onerror = ((web.Event _) {
      loaded.completeError(StateError('Cannot load $loaderUrl'));
    }).toJS;
    web.document.head!.appendChild(script);
    try {
      await loaded.future;
    } finally {
      script.onload = null;
      script.onerror = null;
      script.remove();
    }
  }
  final module = await _createModule(
    _ModuleOptions(
      locateFile: ((String path, String prefix) => wasmUrl.toString()).toJS,
    ),
  ).toDart;
  if (module.abiVersion() != 3) throw StateError('HNSW C++ ABI mismatch');
  _bindings = module;
}

extension type _ModuleOptions._(JSObject _) implements JSObject {
  external factory _ModuleOptions({JSFunction locateFile});
}

// MARK: Direct C ABI declarations (Emscripten uses JS BigInt for int64).
@JS('BigInt')
external JSBigInt _bigInt(String value);
@JS('String')
external String _string(JSAny value);
JSBigInt wasmInt64(int value) => _bigInt(value.toString());
int wasmInteger(JSBigInt value) {
  final result = int.parse(_string(value));
  if (result < 0 || result > 9007199254740991) {
    throw StateError('Integer exceeds JS safe range');
  }
  return result;
}

extension type WasmBindings._(JSObject _) implements JSObject {
  @JS('_malloc')
  external int malloc(int bytes);
  @JS('_free')
  external void free(int pointer);
  @JS('HEAPU8')
  external JSUint8Array get heapBytes;
  @JS('HEAPF32')
  external JSFloat32Array get heapFloats;
  @JS('_hnsw_abi_version')
  external int abiVersion();
  @JS('_hnsw_create')
  external int create(
    int name,
    int dimensions,
    int capacity,
    int m,
    int efConstruction,
    int efSearch,
    int metric,
    int reuse,
    int error,
  );
  @JS('_hnsw_close')
  external int close(int handle);
  @JS('_hnsw_context_id')
  external JSBigInt contextId(int handle);
  @JS('_hnsw_generation')
  external JSBigInt generation(int handle);
  @JS('_hnsw_upsert_vector')
  external int upsert(int handle, JSBigInt label, int vector);
  @JS('_hnsw_upsert_batch')
  external int upsertBatch(int handle, int labels, int vectors, int count);
  @JS('_hnsw_search_knn')
  external int search(
    int handle,
    int query,
    int k,
    int labels,
    int distances,
    int count,
  );
  @JS('_hnsw_mark_deleted')
  external int markDeleted(int handle, JSBigInt label);
  @JS('_hnsw_resize')
  external int resize(int handle, int capacity);
  @JS('_hnsw_set_ef_search')
  external int setEfSearch(int handle, int ef);
  @JS('_hnsw_stats')
  external int stats(int handle, int output);
  @JS('_hnsw_checkpoint')
  external int checkpoint(
    int handle,
    int output,
    int size,
    int written,
    JSBigInt generation,
  );
  @JS('_hnsw_load')
  external int load(int handle, int bytes, int length, JSBigInt generation);
  @JS('_hnsw_clear')
  external int clear(int handle, JSBigInt generation);
  @JS('_hnsw_acquire_exclusive_mutation_access')
  external int acquire(int handle);
  @JS('_hnsw_release_exclusive_mutation_access')
  external int release(int handle);
  // Internal test diagnostics, not part of the public API.
  @JS('_hnsw_registry_size')
  external JSBigInt registrySize();
  @JS('_hnsw_wasm_allocated_bytes')
  external int allocatedBytes();

  // Re-read heap views after allocations: Wasm memory growth replaces them.
  Uint8List get bytes => heapBytes.toDart;
  Float32List get floats => heapFloats.toDart;
  ByteData get data => ByteData.sublistView(bytes);
}

// MARK: Scoped allocation. Graph handles have explicit close; buffers never escape.
T withWasmMemory<T>(T Function(WasmMemory memory) action) {
  final memory = WasmMemory(wasmBindings);
  try {
    return action(memory);
  } finally {
    memory.dispose();
  }
}

class WasmMemory {
  final WasmBindings module;
  final _allocations = <int>[];
  WasmMemory(this.module);
  int allocate(int bytes) {
    final pointer = module.malloc(bytes == 0 ? 1 : bytes);
    if (pointer == 0) throw StateError('Wasm allocation failed');
    _allocations.add(pointer);
    return pointer;
  }

  int copyBytes(List<int> values) {
    final pointer = allocate(values.length);
    module.bytes.setAll(pointer, values);
    return pointer;
  }

  int copyFloats(List<double> values) {
    final pointer = allocate(values.length * 4);
    module.floats.setAll(pointer ~/ 4, values);
    return pointer;
  }

  void dispose() {
    for (final pointer in _allocations) {
      module.free(pointer);
    }
  }
}
