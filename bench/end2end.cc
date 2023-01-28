// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>
#include <iostream>
#include <chrono>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <xnnpack.h>
#include "operator.h"

#include <benchmark/benchmark.h>

#include "bench/utils.h"
#include "models/models.h"
#include "bench/end2end_utils.hpp"

#if XNN_PLATFORM_JIT && XNN_ENABLE_JIT
static void FP32MobileNetV3SmallFused(benchmark::State& state) {
  End2EndBenchmark(state, models::FP32MobileNetV3SmallFused);
}
#endif  // XNN_PLATFORM_JIT && XNN_ENABLE_JIT

#if XNN_PLATFORM_JIT && XNN_ENABLE_JIT
BENCHMARK(FP32MobileNetV3SmallFused)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
#endif  // XNN_PLATFORM_JIT && XNN_ENABLE_JIT

namespace V1 {

#define BENCHMARK_NAME(name) "FP32MobileNetV1_"#name
auto& DenseModel = models::FP32MobileNetV1;
auto& SparseModel = models::FP32SparseMobileNetV1;
auto& SparseModelNano = models::FP32SparseMobileNetV1Nano;

BENCHMARK(ModelBench<DenseModel>)->Name(BENCHMARK_NAME(Dense))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(SparseModelBench<SparseModel, 70>)->Name(BENCHMARK_NAME(Sparse70))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(SparseModelBenchVerified<SparseModelNano, SparseModel, 70>)->Name(BENCHMARK_NAME(Nano70))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(SparseModelBench<SparseModel, 80>)->Name(BENCHMARK_NAME(Sparse80))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(SparseModelBenchVerified<SparseModelNano, SparseModel, 80>)->Name(BENCHMARK_NAME(Nano80))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(SparseModelBench<SparseModel, 90>)->Name(BENCHMARK_NAME(Sparse90))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(SparseModelBenchVerified<SparseModelNano, SparseModel, 90>)->Name(BENCHMARK_NAME(Nano90))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();

#undef BENCHMARK_NAME
}


namespace V2 {

#define BENCHMARK_NAME(name) "FP32MobileNetV2_"#name
auto& DenseModel = models::FP32MobileNetV2;
auto& SparseModel = models::FP32SparseMobileNetV2;
auto& SparseModelNano = models::FP32SparseMobileNetV2Nano;

BENCHMARK(ModelBench<DenseModel>)->Name(BENCHMARK_NAME(Dense))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(SparseModelBench<SparseModel, 70>)->Name(BENCHMARK_NAME(Sparse70))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(SparseModelBenchVerified<SparseModelNano, SparseModel, 70>)->Name(BENCHMARK_NAME(Nano70))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(SparseModelBench<SparseModel, 80>)->Name(BENCHMARK_NAME(Sparse80))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(SparseModelBenchVerified<SparseModelNano, SparseModel, 80>)->Name(BENCHMARK_NAME(Nano80))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(SparseModelBench<SparseModel, 90>)->Name(BENCHMARK_NAME(Sparse90))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(SparseModelBenchVerified<SparseModelNano, SparseModel, 90>)->Name(BENCHMARK_NAME(Nano90))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();

#undef BENCHMARK_NAME
}

namespace V3Small {

#define BENCHMARK_NAME(name) "FP32MobileNetV3Small_"#name
auto& DenseModel = models::FP32MobileNetV3Small;
auto& SparseModel = models::FP32SparseMobileNetV3Small;
auto& SparseModelNano = models::FP32SparseMobileNetV3SmallNano;

BENCHMARK(ModelBench<DenseModel>)->Name(BENCHMARK_NAME(Dense))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(SparseModelBenchVerified<SparseModel, SparseModel, 70>)->Name(BENCHMARK_NAME(Sparse70))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(SparseModelBenchVerified<SparseModelNano, SparseModel, 70>)->Name(BENCHMARK_NAME(Nano70))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(SparseModelBench<SparseModel, 80>)->Name(BENCHMARK_NAME(Sparse80))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(SparseModelBenchVerified<SparseModelNano, SparseModel, 80>)->Name(BENCHMARK_NAME(Nano80))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(SparseModelBench<SparseModel, 90>)->Name(BENCHMARK_NAME(Sparse90))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(SparseModelBenchVerified<SparseModelNano, SparseModel, 90>)->Name(BENCHMARK_NAME(Nano90))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();

#undef BENCHMARK_NAME
}

namespace V3Large {

#define BENCHMARK_NAME(name) "FP32MobileNetV3Large_"#name
auto& DenseModel = models::FP32MobileNetV3Large;
auto& SparseModel = models::FP32SparseMobileNetV3Large;
auto& SparseModelNano = models::FP32SparseMobileNetV3LargeNano;

BENCHMARK(ModelBench<DenseModel>)->Name(BENCHMARK_NAME(Dense))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(SparseModelBench<SparseModel, 70>)->Name(BENCHMARK_NAME(Sparse70))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(SparseModelBenchVerified<SparseModelNano, SparseModel, 70>)->Name(BENCHMARK_NAME(Nano70))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(SparseModelBench<SparseModel, 80>)->Name(BENCHMARK_NAME(Sparse80))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(SparseModelBenchVerified<SparseModelNano, SparseModel, 80>)->Name(BENCHMARK_NAME(Nano80))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(SparseModelBench<SparseModel, 90>)->Name(BENCHMARK_NAME(Sparse90))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(SparseModelBenchVerified<SparseModelNano, SparseModel, 90>)->Name(BENCHMARK_NAME(Nano90))\
    ->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();

#undef BENCHMARK_NAME
}

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
