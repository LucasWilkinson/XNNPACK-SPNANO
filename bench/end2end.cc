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

#include <xnnpack.h>
#include "operator.h"

#include <benchmark/benchmark.h>

#include "bench/utils.h"
#include "models/models.h"


static void End2EndBenchmark(
  benchmark::State& state,
  models::ExecutionPlanFactory model_factory)
{
  if (xnn_initialize(nullptr /* allocator */) != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  const size_t num_threads = state.range(0);
  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> threadpool(
    pthreadpool_create(num_threads), pthreadpool_destroy);

  auto execution_plan = model_factory(threadpool.get());
  if (execution_plan.empty()) {
    state.SkipWithError("failed to create a model");
    return;
  }

  void* final_output = nullptr;
  int final_output_size = 0;
  for (const std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>& op : execution_plan) {
    xnn_status status = xnn_run_operator(op.get(), threadpool.get());
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run a model");
      return;
    }
    final_output = op.get()->output;
    final_output_size = op.get()->output_height * op.get()->output_width * op.get()->output_pixel_stride;
  }

//  float* f32_final_output = (float*) final_output;
//  for (int i = 0; i < final_output_size; i++) {
//    std::cout << f32_final_output[i] << " ";
//  }
//  std::cout << std::endl;


  for (auto _ : state) {
    for (const std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>& op : execution_plan) {
      xnn_status status = xnn_run_operator(op.get(), threadpool.get());
      if (status != xnn_status_success) {
        state.SkipWithError("failed to run a model");
        return;
      }
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

template<typename T>
bool is_within_tol(const T &x, const T &y, const T eps=1e-6) {
  //http://realtimecollisiondetection.net/blog/?p=89
  auto relTol = eps * std::max((T) 1, std::max(std::abs(x), std::abs(y)));

  return std::abs(x - y) < relTol;
}


int conv_output_size(int input_size, int kernel_size, int stride, int pad) {
  return (input_size + 2 * pad - kernel_size) / stride + 1;
}

float *get_op_output(xnn_operator_t op) {
  switch (op->type) {
    case xnn_operator_type_convolution_nchw_f32:
    case xnn_operator_type_convolution_nhwc_f32:
      return (float*) op->output;
    case xnn_operator_type_global_average_pooling_ncw_f32:
      return (float*)  op->context.global_average_pooling_ncw.output;
    default: {
      std::cout << "Unknown layer for output selection please add: " << op->type << std::endl;
      break;
    }
  }
  return nullptr;
}


float *get_op_input(xnn_operator_t op)
{
  switch (op->type) {
    case xnn_operator_type_convolution_nchw_f32:
    case xnn_operator_type_convolution_nhwc_f32:
      return (float*) op->input;
    case xnn_operator_type_global_average_pooling_ncw_f32:
      return (float*)  op->context.global_average_pooling_ncw.input;
    default: {
      std::cout << "Unknown layer for output selection please add: " << op->type << std::endl;
      break;
    }
  }
  return nullptr;
}


static void Verify(int num_threads, models::ExecutionPlanFactory model_factory, models::ExecutionPlanFactory ref_model_factory) {
  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> threadpool(
    pthreadpool_create(num_threads), pthreadpool_destroy);

  // Both plans should be initialized with the same seed
  auto execution_plan = model_factory(threadpool.get());
  auto ref_execution_plan = ref_model_factory(threadpool.get());

  struct LayerVerification {
    float* input;
    float* output;

    int input_size;
    int output_size;

    enum xnn_operator_type type;
  };

  std::vector<LayerVerification> ref_layers;
  int op_idx = 0;

  // Tailored for mobilenetv1 for now
  for (const std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>& op : ref_execution_plan) {
    xnn_status status = xnn_run_operator(op.get(), threadpool.get());

    xnn_operator_t op_ptr = op.get();

    int input_buffer_size = 0;
    int output_buffer_size = 0;

    switch (op_ptr->type) {
      case xnn_operator_type_convolution_nchw_f32:
      case xnn_operator_type_convolution_nhwc_f32: {
        int output_height =
          conv_output_size(op_ptr->input_height, op_ptr->kernel_height, op_ptr->stride_height, op_ptr->padding_top);
        int output_width =
          conv_output_size(op_ptr->input_width, op_ptr->kernel_width, op_ptr->stride_width, op_ptr->padding_left);

        input_buffer_size = op_ptr->input_height * op_ptr->input_width * op_ptr->input_pixel_stride;
        output_buffer_size = output_height * output_width * op_ptr->output_pixel_stride;

        ref_layers.push_back({
          .input = get_op_input(op_ptr),
          .output = get_op_output(op_ptr),
          .input_size = input_buffer_size,
          .output_size = output_buffer_size,
          .type = op_ptr->type});
        break;
      }
      case xnn_operator_type_global_average_pooling_ncw_f32: {
        // NOTE: the input_height and input_width are not set for global average pooling
        //  this will just be 0 I guess and we won't verify it, todo: fix
        input_buffer_size = op_ptr->input_height * op_ptr->input_width * op_ptr->channels;
        output_buffer_size = op_ptr->channels;

        ref_layers.push_back({
          .input = get_op_input(op_ptr),
          .output = get_op_output(op_ptr),
          .input_size = input_buffer_size,
          .output_size = output_buffer_size,
          .type = op_ptr->type});
        break;
      }
      default: {
        std::cout << "Unsupported operator layer type for verification: " << op_ptr->type << std::endl;
        break;
      }
    }

    op_idx++;
  }

  op_idx = 0;
  for (const std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>& op : execution_plan) {
    xnn_status status = xnn_run_operator(op.get(), threadpool.get());

    LayerVerification& ref_layer = ref_layers[op_idx];
    xnn_operator_t op_ptr = op.get();

    if (op_ptr->type != ref_layer.type) {
      std::cout << "Operator type mismatch between reference, layer: " << op_idx << std::endl;
      return;
    }

    float* input = get_op_input(op_ptr);
    float* output = get_op_output(op_ptr);

    float error_tols = op_idx >= 16 ? op_idx >= 22 ? 1e-1 : 1e-1 : 1e-1;

    bool correct = true;
    int print_count = 0;
    for (int i = 0; i < ref_layer.input_size; i++) {
      if (!is_within_tol<float>(ref_layer.input[i], input[i], error_tols) && print_count < 10) {
        std::cout << "(" << i << "): " << ref_layer.input[i] << " " << input[i] << " ";
        correct = false;
        print_count++;
      }
    }

    if (correct) {
      //std::cout << "Layer " << op_idx << " input is correct" << std::endl;
    }
    else {
      std::cout << std::endl << "Layer " << op_idx << " input is incorrect" << std::endl;
    }

    correct = true;
    print_count = 0;
    for (int i = 0; i < ref_layer.output_size; i++) {
      if (!is_within_tol<float>(ref_layer.output[i], output[i], error_tols) && print_count < 10) {
        std::cout << "(" << i << "): " << ref_layer.output[i] << " " << output[i] << " ";
        correct = false;
        print_count++;
      }
    }

    if (correct) {
      //std::cout << "Layer " << op_idx << " output is correct" << std::endl;
    } else {
      std::cout << std::endl << "Layer " << op_idx << " output is incorrect"
                << " type: " << op_ptr->type
                << " output size: " << ref_layer.output_size
                << " input stride: " << op_ptr->input_pixel_stride
                << " output stride: " << op_ptr->output_pixel_stride
                << " spatial stride: " << op_ptr->input_height * op_ptr->input_width
                << " flags: " << op_ptr->flags << std::endl;
    }

    op_idx++;
  }
}

static void End2EndBenchmark(
  benchmark::State& state,
  models::ExecutionPlanFactory model_factory,
  models::ExecutionPlanFactory ref_model_factory)
{
  if (xnn_initialize(nullptr /* allocator */) != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  const size_t num_threads = state.range(0);

  if (true) {
      Verify(num_threads, model_factory, ref_model_factory);
  }

  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> threadpool(
    pthreadpool_create(num_threads), pthreadpool_destroy);

  auto execution_plan = model_factory(threadpool.get());
  if (execution_plan.empty()) {
    state.SkipWithError("failed to create a model");
    return;
  }

  for (auto _ : state) {
    for (const std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>& op : execution_plan) {
      xnn_status status = xnn_run_operator(op.get(), threadpool.get());
      if (status != xnn_status_success) {
        state.SkipWithError("failed to run a model");
        return;
      }
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

static void FP32MobileNetV1(benchmark::State& state) {
  End2EndBenchmark(state, models::FP32MobileNetV1);
}

static void FP32MobileNetV2(benchmark::State& state) {
  End2EndBenchmark(state, models::FP32MobileNetV2);
}

static void FP32MobileNetV3Large(benchmark::State& state) {
  End2EndBenchmark(state, models::FP32MobileNetV3Large);
}

static void FP32MobileNetV3Small(benchmark::State& state) {
  End2EndBenchmark(state, models::FP32MobileNetV3Small);
}

#if XNN_PLATFORM_JIT && XNN_ENABLE_JIT
static void FP32MobileNetV3SmallFused(benchmark::State& state) {
  End2EndBenchmark(state, models::FP32MobileNetV3SmallFused);
}
#endif  // XNN_PLATFORM_JIT && XNN_ENABLE_JIT

static void FP32Sparse90MobileNetV1(benchmark::State& state) {
  End2EndBenchmark(state, [](pthreadpool_t threadpool) {
    return models::FP32SparseMobileNetV1(0.9f, threadpool);
  });
}

static void FP32Sparse90MobileNetV1Nano(benchmark::State& state) {
  End2EndBenchmark(state, [](pthreadpool_t threadpool) {
    return models::FP32SparseMobileNetV1Nano(0.9f, threadpool);
  });
}

static void FP32Sparse80MobileNetV1(benchmark::State& state) {
  End2EndBenchmark(state, [](pthreadpool_t threadpool) {
    return models::FP32SparseMobileNetV1(0.8f, threadpool);
  });
}

static void FP32Sparse80MobileNetV1Nano(benchmark::State& state) {
  End2EndBenchmark(state, [](pthreadpool_t threadpool) {
    return models::FP32SparseMobileNetV1Nano(0.8f, threadpool);
  }, [](pthreadpool_t threadpool) {
    return models::FP32SparseMobileNetV1(0.8f, threadpool);
  });
}

static void FP32Sparse70MobileNetV1(benchmark::State& state) {
  End2EndBenchmark(state, [](pthreadpool_t threadpool) {
    return models::FP32SparseMobileNetV1(0.7f, threadpool);
  });
}

static void FP32Sparse70MobileNetV1Nano(benchmark::State& state) {
  End2EndBenchmark(state, [](pthreadpool_t threadpool) {
      return models::FP32SparseMobileNetV1Nano(0.7f, threadpool);
    }, [](pthreadpool_t threadpool) {
      return models::FP32SparseMobileNetV1(0.7f, threadpool);
    });
}


static void FP32Sparse80MobileNetV2(benchmark::State& state) {
  End2EndBenchmark(state, [](pthreadpool_t threadpool) {
    return models::FP32SparseMobileNetV2(0.8f, threadpool);
  });
}

static void FP32Sparse80MobileNetV3Large(benchmark::State& state) {
  End2EndBenchmark(state, [](pthreadpool_t threadpool) {
    return models::FP32SparseMobileNetV3Large(0.8f, threadpool);
  });
}

static void FP32Sparse80MobileNetV3Small(benchmark::State& state) {
  End2EndBenchmark(state, [](pthreadpool_t threadpool) {
    return models::FP32SparseMobileNetV3Small(0.8f, threadpool);
  });
}

static void FP16MobileNetV1(benchmark::State& state) {
  End2EndBenchmark(state, models::FP16MobileNetV1);
}

static void FP16MobileNetV2(benchmark::State& state) {
  End2EndBenchmark(state, models::FP16MobileNetV2);
}

static void FP16MobileNetV3Large(benchmark::State& state) {
  End2EndBenchmark(state, models::FP16MobileNetV3Large);
}

static void FP16MobileNetV3Small(benchmark::State& state) {
  End2EndBenchmark(state, models::FP16MobileNetV3Small);
}

static void QC8MobileNetV1(benchmark::State& state) {
  End2EndBenchmark(state, models::QC8MobileNetV1);
}

static void QC8MobileNetV2(benchmark::State& state) {
  End2EndBenchmark(state, models::QC8MobileNetV2);
}

static void QS8MobileNetV1(benchmark::State& state) {
  End2EndBenchmark(state, models::QS8MobileNetV1);
}

static void QS8MobileNetV2(benchmark::State& state) {
  End2EndBenchmark(state, models::QS8MobileNetV2);
}

static void QU8MobileNetV1(benchmark::State& state) {
  End2EndBenchmark(state, models::QU8MobileNetV1);
}

static void QU8MobileNetV2(benchmark::State& state) {
  End2EndBenchmark(state, models::QU8MobileNetV2);
}

//BENCHMARK(FP32MobileNetV1)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
//BENCHMARK(FP32MobileNetV2)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
//BENCHMARK(FP32MobileNetV3Large)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
//BENCHMARK(FP32MobileNetV3Small)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
#if XNN_PLATFORM_JIT && XNN_ENABLE_JIT
BENCHMARK(FP32MobileNetV3SmallFused)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
#endif  // XNN_PLATFORM_JIT && XNN_ENABLE_JIT

//BENCHMARK(FP32MobileNetV1)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
//BENCHMARK(FP32MobileNetV1)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK(FP32MobileNetV1)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP32Sparse70MobileNetV1)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP32Sparse70MobileNetV1Nano)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP32Sparse80MobileNetV1)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP32Sparse80MobileNetV1Nano)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();

//BENCHMARK(FP32Sparse70MobileNetV1)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
//BENCHMARK(FP32Sparse70MobileNetV1Nano)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
//BENCHMARK(FP32Sparse80MobileNetV1)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
//BENCHMARK(FP32Sparse80MobileNetV1Nano)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
//BENCHMARK(FP32Sparse90MobileNetV1)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
//BENCHMARK(FP32Sparse90MobileNetV1Nano)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();

//BENCHMARK(FP32Sparse80MobileNetV2)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
//BENCHMARK(FP32Sparse80MobileNetV3Large)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
//BENCHMARK(FP32Sparse80MobileNetV3Small)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();

//BENCHMARK(FP16MobileNetV1)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
//BENCHMARK(FP16MobileNetV2)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
//BENCHMARK(FP16MobileNetV3Large)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
//BENCHMARK(FP16MobileNetV3Small)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
//
//BENCHMARK(QC8MobileNetV1)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
//BENCHMARK(QC8MobileNetV2)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
//
//BENCHMARK(QS8MobileNetV1)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
//BENCHMARK(QS8MobileNetV2)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
//
//BENCHMARK(QU8MobileNetV1)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
//BENCHMARK(QU8MobileNetV2)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
