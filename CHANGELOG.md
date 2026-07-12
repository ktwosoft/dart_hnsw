# Changelog

## 0.1.2

- Fix README & CHANGELOG version reference to point to latest.

## 0.1.1

- Add dartdoc comments to all public API elements to improve pub.dev documentation score.
- Add `example: example/basic.dart` to pubspec.yaml.
- Update README with native efficiency focus and fix duplicate sections.

## 0.1.0

- Initial release of `dart_hnsw`.
- Added Dart Native bindings for `hnswlib` with `l2`, `innerProduct`, and `cosine` metrics.
- Added explicit `checkpoint()` persistence and shared in-process native index management.
- Added cross-process locking for index ownership.
- Added native-assets support for `dart run` and `dart test`.
- Added compiled executable support via sidecar shared-library fallback for `dart compile exe`.
- Added macOS, Linux, and Windows support declarations.
