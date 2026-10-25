# Direct-backend validation

Recorded 2026-10-25 on macOS 26.5.1 arm64, Dart 3.12.0, Chrome 155.0.8059.85,
Emscripten 4.0.23.

- All 52 original native tests pass.
- C++ concurrency harness, analyzer and native build-hook CLI smoke pass.
- Real headless Chrome tests pass with both Dart JS and Dart Wasm targets.
  They exercise the synchronous API directly, including initialization failure
  and retry, same-name graph sharing, ranking for every metric, invalid inputs,
  deletes/reuse, resizing, raw checkpoint/load, stale generations, native64 ABI
  rejection, exclusive mutation access, close/reopen and allocation cleanup.
- Wasm heap-growth testing exercises graph creation/resizing beyond the initial
  linear memory capacity, followed by search and checkpoint/load.
- Two builds into different output directories produced identical JS and Wasm
  files with the pinned SDK.

The focused smoke uses 1,000 seeded 256D cosine vectors (M=16,
efConstruction=200, efSearch=64), 100 updates, 20 deleted-slot replacements,
a full temporary graph/checkpoint/load replacement, and 200 serial k=10 searches.
All work runs synchronously on the calling thread; no background scheduling is
included. One observed run (milliseconds, including Dart/native interop):

| Target | Initial build | Incremental work | Replacement | Query p50 | Query p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dart JS | 219.900 | 1081.301 | 220.399 | 0.200 | 0.300 |
| Dart Wasm | 230.200 | 1070.100 | 289.400 | 0.200 | 0.300 |

Before the heap-growth test, both reported 16,777,216 bytes of Wasm capacity,
2,926,304 live allocator bytes after replacement and 632 after closing graphs.
The runtime retains its memory capacity; graph allocations are freed. These
figures exclude Dart/JS heaps and peak replacement memory. Browser timing
resolution and host load matter; this is a smoke case, not a universal benchmark.

Not validated here: full Flutter applications, other browser engines,
application-owned browser worker bootstraps, native Linux/Windows/mobile devices,
or native 32-bit runtime. Native isolate sharing is preserved and tested;
cross-worker/cross-tab Wasm sharing is not implemented.
