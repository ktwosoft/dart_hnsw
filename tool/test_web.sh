#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tool/build_web.sh
# Set CHROME_EXECUTABLE if Chrome is not on the default discovery path.
dart test -p chrome -c dart2js -c dart2wasm test/web_test.dart test/web_initialization_test.dart
