# Web backend

The existing C++ wrapper and vendored hnswlib compile to wasm32. The public API
is still `HnswIndex`: initialize once asynchronously on web, then use the same
synchronous methods as native. There is no worker owner, RPC protocol, mutation
queue, rebuild service, or new checkpoint format.

## Build and initialize

Run these commands from the package checkout. Keep the pinned Emscripten SDK
in a sibling directory, outside the package:

```sh
git clone --depth 1 --branch 4.0.23 https://github.com/emscripten-core/emsdk.git ../emsdk-hnsw
../emsdk-hnsw/emsdk install 4.0.23
../emsdk-hnsw/emsdk activate 4.0.23
source ../emsdk-hnsw/emsdk_env.sh
tool/build_web.sh # writes build/web_assets/hnsw.js and hnsw.wasm
```

Serve **both matching files** in the application's static assets. For Flutter,
copy them into `web/hnsw/` before building the app. Serve JS as JavaScript and
Wasm as `application/wasm`; avoid SPA fallback HTML for missing assets. Configure
CSP/CORS for the selected script and Wasm URLs. No SharedArrayBuffer or COOP/COEP
is needed. Compiling Dart/Flutter to Wasm does not compile the C++ backend.

```dart
import 'package:dart_hnsw/dart_hnsw.dart';

// Use your actual deployment base, including any subpath.
await HnswIndex.initialize(
  loaderUrl: deploymentBase.resolve('hnsw/hnsw.js'),
  wasmUrl: deploymentBase.resolve('hnsw/hnsw.wasm'),
);
final index = HnswIndex.open(collectionName: 'vectors', dimensions: 3);
try {
  index.upsert(1, [1, 0, 0]);
  final hits = index.search([1, 0, 0], k: 10);
  final bytes = index.checkpoint(expectedGeneration: index.generation);
  index.load(bytes, expectedGeneration: index.generation);
} finally {
  index.close();
}
```

`initialize` is a no-op on native, where the existing build hook still loads the
library automatically. Repeated/concurrent browser initialization with the same
URLs is idempotent; different URLs are rejected after initialization. A failed
initialization may be retried. All other operations are synchronous and execute
on the calling thread. Scheduling, workers and background rebuilding belong to
the application. If an application runs this code in its own browser worker,
it must preload `hnsw.js` with `importScripts` before initialization and supply
its own communication/bootstrap code.

## Sharing, labels and checkpoint boundaries

Native behavior is unchanged: same-name handles across isolates in one process
share the C++ graph and existing locks. On web, same-name handles share the C++
graph **within the same Wasm instance/JavaScript realm**. Separate browser tabs
and workers do not share that memory. This backend does not implement shared
Wasm memory or cross-worker graph access.

Close every handle explicitly. Closing one leaves other handles usable; closing
the last frees the graph. The loaded Wasm runtime and its memory capacity remain
for the lifetime of the realm, like a loaded native library. Temporary buffers
are freed even when C++ operations fail. Exclusive mutation access, generation
guards and atomic `load` retain the native semantics.

Web labels must be integers in **0..4,294,967,295**, matching hnswlib's wasm32
`size_t` labels. Larger values are rejected, never rounded or truncated. Native
label limits remain unchanged. Web generation values must fit JS's safe integer
range (0..9,007,199,254,740,991).

`checkpoint` returns the original raw hnswlib bytes, without an application
storage layer or envelope. Restore only with matching builds, configuration and
ABI. In particular, native 64-bit checkpoints are **not** portable to wasm32.
The existing C++ layout validation rejects the tested native64 fixture. Source
vectors are the portable recovery input; rebuild them on the target ABI.

## Validation

```sh
bash test/run_all.sh # native concurrency, analyzer, native tests, CLI hook
source ../emsdk-hnsw/emsdk_env.sh
tool/test_web.sh     # real headless Chrome: Dart JS and Dart Wasm targets
```

Set `CHROME_EXECUTABLE` if Chrome is not found. The browser suite covers sharing,
empty/deleted graphs, brute-force ranking, updates, deleted-slot reuse, resizing,
invalid inputs, raw checkpoint round-trip, native64 rejection, stale generations,
atomic replacement, exclusive mutation ownership, initialization retry and memory
cleanup. The 256D smoke test reports direct-call timings and Wasm allocation
observations. These calls intentionally occupy the caller; background scheduling
is not tested or provided by the library.

A full Flutter application, other browser engines and mobile/device runtime
integration have not been exercised here. See [test results](validation.md).
