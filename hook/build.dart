import 'package:hooks/hooks.dart';
import 'package:code_assets/code_assets.dart';
import 'package:native_toolchain_c/native_toolchain_c.dart';

void main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) {
      return;
    }

    final builder = CBuilder.library(
      name: 'dart_hnsw',
      assetName: 'src/native_bindings.dart',
      language: Language.cpp,
      std: 'c++17',
      sources: ['native/dart_hnsw_native.cpp'],
      includes: ['third_party/hnswlib/hnswlib'],
      libraries: [if (input.config.code.targetOS == OS.linux) 'pthread'],
    );

    await builder.run(input: input, output: output);
  });
}
