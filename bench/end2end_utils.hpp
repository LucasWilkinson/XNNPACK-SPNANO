//
// Created by Lucas Wilkinson on 1/21/23.
//

#ifndef XNNPACK_BENCH_END2END_UTILS_HPP_
#define XNNPACK_BENCH_END2END_UTILS_HPP_

template<typename Scalar>
double median(std::vector<Scalar> &v) {
    size_t n = v.size() / 2;
    nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}


inline void End2EndBenchmark(
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

    std::vector<std::vector<double>> operator_times;
    operator_times.resize(execution_plan.size());
    for (auto& operator_time : operator_times) operator_time.reserve(200);

    for (auto _ : state) {
        int layer_idx = 0;

        for (const std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>& op : execution_plan) {
            auto start = std::chrono::high_resolution_clock::now();
            xnn_status status = xnn_run_operator(op.get(), threadpool.get());
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::micro> duration = end - start;
            operator_times[layer_idx++].push_back(duration.count());

            if (status != xnn_status_success) {
                state.SkipWithError("failed to run a model");
                return;
            }
        }
    }

    for (int layer_idx = 0; layer_idx < operator_times.size(); layer_idx++) {
        std::ostringstream str; str << std::setw(2) << std::setfill('0') << layer_idx;
        state.counters["layer_" + str.str()] = median(operator_times[layer_idx]);
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


inline int conv_output_size(int input_size, int kernel_size, int stride, int pad) {
    return (input_size + 2 * pad - kernel_size) / stride + 1;
}

inline float *get_op_output(xnn_operator_t op) {
    switch (op->type) {
        case xnn_operator_type_convolution_nchw_f32:
        case xnn_operator_type_convolution_nhwc_f32:
            return (float*) op->output;
        case xnn_operator_type_global_average_pooling_ncw_f32:
            return (float*)  op->context.global_average_pooling_ncw.output;
        default: {
            // std::cout << "Unknown layer for output selection please add: " << op->type << std::endl;
            break;
        }
    }
    return nullptr;
}


inline float *get_op_input(xnn_operator_t op)
{
    switch (op->type) {
        case xnn_operator_type_convolution_nchw_f32:
        case xnn_operator_type_convolution_nhwc_f32:
            return (float*) op->input;
        case xnn_operator_type_global_average_pooling_ncw_f32:
            return (float*)  op->context.global_average_pooling_ncw.input;
        default: {
            // std::cout << "Unknown layer for output selection please add: " << op->type << std::endl;
            break;
        }
    }
    return nullptr;
}


inline void Verify(int num_threads, models::ExecutionPlanFactory model_factory, models::ExecutionPlanFactory ref_model_factory) {
    std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> threadpool(
      pthreadpool_create(num_threads), pthreadpool_destroy);

    // Both plans should be initialized with the same seed
    auto execution_plan = model_factory(threadpool.get());
    auto ref_execution_plan = ref_model_factory(threadpool.get());

    struct LayerVerification {
      float* input = nullptr;
      float* output = nullptr;

      int input_size = 0;
      int output_size = 0;

      enum xnn_operator_type type;
      bool verify = false;
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
            case xnn_operator_type_convolution_nchw_f32: {
                auto input_hw = op_ptr->input_height * op_ptr->input_width;
                auto input_channels = op_ptr->group_input_channels;

                auto output_hw = op_ptr->input_height * op_ptr->input_width;
                auto output_channels = op_ptr->group_output_channels;

                auto kernel_h = op_ptr->kernel_height;
                auto kernel_w = op_ptr->kernel_width;

                bool is_1x1 = (kernel_h == 1 && kernel_w == 1);

                if (is_1x1) {
                    //std::cout << "Layer " << op_idx << ": (" << input_channels << "x" << input_hw << ") -> (" << output_channels << "x" << output_hw << ")" << std::endl;
                }

                // fall-through
            }
            case xnn_operator_type_convolution_nhwc_f32: {
                int output_height =
                  conv_output_size(op_ptr->input_height, op_ptr->kernel_height, op_ptr->stride_height, op_ptr->padding_top);
                int output_width =
                  conv_output_size(op_ptr->input_width, op_ptr->kernel_width, op_ptr->stride_width, op_ptr->padding_left);

                input_buffer_size = op_ptr->input_height * op_ptr->input_width * op_ptr->input_pixel_stride;
                output_buffer_size = output_height * output_width * op_ptr->output_pixel_stride;

                ref_layers.push_back({.input = get_op_input(op_ptr),
                                      .output = get_op_output(op_ptr),
                                      .input_size = input_buffer_size,
                                      .output_size = output_buffer_size,
                                      .type = op_ptr->type,
                                      .verify = true});
                break;
            }
            case xnn_operator_type_global_average_pooling_ncw_f32: {
                // NOTE: the input_height and input_width are not set for global average pooling
                //  this will just be 0 I guess and we won't verify it, todo: fix
                input_buffer_size = op_ptr->input_height * op_ptr->input_width * op_ptr->channels;
                output_buffer_size = op_ptr->channels;

                ref_layers.push_back({.input = get_op_input(op_ptr),
                                      .output = get_op_output(op_ptr),
                                      .input_size = input_buffer_size,
                                      .output_size = output_buffer_size,
                                      .type = op_ptr->type,
                                      .verify = true});
                break;
            }
            default: {
                //std::cout << "Unsupported operator layer type for verification: " << op_ptr->type << std::endl;
                ref_layers.push_back({.input = nullptr,
                                      .output = nullptr,
                                      .input_size = 0,
                                      .output_size = 0,
                                      .type = op_ptr->type,
                                      .verify = false});
                break;
            }
        }

        op_idx++;
    }

    op_idx = 0;
    for (const std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>& op : execution_plan) {
        LayerVerification& ref_layer = ref_layers[op_idx];
        xnn_operator_t op_ptr = op.get();

        float error_tols = 0.5;
        bool correct = true;
        int print_count = 0;

        const float* input = get_op_input(op_ptr);
        const float* output = get_op_output(op_ptr);

        //ref_layer.verify = false;
        if (ref_layer.verify) {
            if (op_ptr->type != ref_layer.type) {
                std::cout << op_ptr->type << " != " << ref_layer.type << std::endl;
                std::cout << "Operator type mismatch between reference, layer: " << op_idx << std::endl;
                return;
            }

            correct = true;
            print_count = 0;
            for (int i = 0; i < ref_layer.input_size; i++) {
                if (!is_within_tol<float>(ref_layer.input[i], input[i], error_tols) && print_count < 10) {
                    std::cout << "(" << i << "): " << ref_layer.input[i] << " " << input[i] << " ";
                    correct = false;
                    print_count++;
                }
            }

            if (correct) {
                //std::cout << "Layer " << op_idx << " input pre-op is correct, ref_ptr: " << ref_layer.input << " prt " << input << std::endl;
            }
            else {
                std::cout << std::endl << "Layer " << op_idx << " input (pre run) is incorrect, ref_ptr: " << ref_layer.input << " prt " << input
                          << " type: " << op_ptr->type
                          << " output size: " << ref_layer.output_size
                          << " input stride: " << op_ptr->input_pixel_stride
                          << " output stride: " << op_ptr->output_pixel_stride
                          << " spatial stride: " << op_ptr->input_height * op_ptr->input_width
                          << " flags: " << op_ptr->flags << std::endl;
            }
        }

        xnn_status status = xnn_run_operator(op.get(), threadpool.get());

        if (ref_layer.verify) {
            if (op_ptr->type != ref_layer.type) {
                std::cout << op_ptr->type << " != " << ref_layer.type << std::endl;
                std::cout << "Operator type mismatch between reference, layer: " << op_idx << std::endl;
                return;
            }

            correct = true;
            print_count = 0;
            for (int i = 0; i < ref_layer.input_size; i++) {
                if (!is_within_tol<float>(ref_layer.input[i], input[i], error_tols) && print_count < 10) {
                    std::cout << "(" << i << "): " << ref_layer.input[i] << " " << input[i] << " ";
                    correct = false;
                    print_count++;
                }
            }

            if (correct) {
                //std::cout << "Layer " << op_idx << " input is correct, ref_ptr: " << ref_layer.input << " prt " << input << std::endl;
            }
            else {
                std::cout << std::endl << "Layer " << op_idx << " input is incorrect, ref_ptr: " << ref_layer.input << " prt " << input
                          << " type: " << op_ptr->type
                          << " output size: " << ref_layer.output_size
                          << " input stride: " << op_ptr->input_pixel_stride
                          << " output stride: " << op_ptr->output_pixel_stride
                          << " spatial stride: " << op_ptr->input_height * op_ptr->input_width
                          << " flags: " << op_ptr->flags << std::endl;
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
                //std::cout << "Layer " << op_idx << " output is correct, ref_ptr: " << ref_layer.output << " prt " << output << std::endl;
            } else {
                std::cout << std::endl << "Layer " << op_idx << " output is incorrect, ref_ptr: " << ref_layer.output << " prt " << output
                          << " type: " << op_ptr->type
                          << " output size: " << ref_layer.output_size
                          << " input stride: " << op_ptr->input_pixel_stride
                          << " output stride: " << op_ptr->output_pixel_stride
                          << " spatial stride: " << op_ptr->input_height * op_ptr->input_width
                          << " flags: " << op_ptr->flags << std::endl;
            }
        }

        op_idx++;
    }
}

inline void End2EndBenchmark(
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


    enum layer_type: int {
      spnano = 0, spmm, hwc2chw, dwconv, conv2d, add, mul, unknown
    };

    std::vector<std::vector<double>> operator_times;
    std::vector<std::pair<int, int>> kernel_shape;
    std::vector<std::tuple<int, int, int>> mul_shape;
    std::vector<layer_type> layer_type_str;

    kernel_shape.resize(execution_plan.size());
    layer_type_str.resize(execution_plan.size());
    operator_times.resize(execution_plan.size());
    mul_shape.resize(execution_plan.size());

    for (auto& operator_time : operator_times) operator_time.reserve(200);

    for (auto _ : state) {
        int layer_idx = 0;

        for (const std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>& op : execution_plan) {
            auto start = std::chrono::high_resolution_clock::now();
            xnn_status status = xnn_run_operator(op.get(), threadpool.get());
            auto end = std::chrono::high_resolution_clock::now();

            kernel_shape[layer_idx] = {op->kernel_height, op->kernel_width};
            mul_shape[layer_idx] = {-1, -1, -1};

            if (op->type == xnn_operator_type_convolution_nhwc_f32 || op->type == xnn_operator_type_convolution_nchw_f32) {
                if (op->ukernel.type == xnn_ukernel_type_spmm_nano) {
                    layer_type_str[layer_idx] = spnano;
                    mul_shape[layer_idx] = { op->output_pixel_stride, op->input_pixel_stride, op->input_height * op->input_width};
                } else if (op->ukernel.type == xnn_ukernel_type_spmm) {
                    layer_type_str[layer_idx] = spmm;
                    mul_shape[layer_idx] = { op->output_pixel_stride, op->input_pixel_stride, op->input_height * op->input_width};
                } else if (op->ukernel.type == xnn_ukernel_type_conv2d_hwc2chw) {
                    layer_type_str[layer_idx] = hwc2chw;
                } else if (op->ukernel.type == xnn_ukernel_type_dwconv) {
                    layer_type_str[layer_idx] = dwconv;
                } else if (op->ukernel.type == xnn_ukernel_type_subconv2d) {
                    layer_type_str[layer_idx] = conv2d;
                }
            } else if (op->type == xnn_operator_type_add_nd_f32) {
                layer_type_str[layer_idx] = add;
            } else if (op->type == xnn_operator_type_multiply_nd_f32) {
                layer_type_str[layer_idx] = mul;
            } else {
                layer_type_str[layer_idx] = unknown;
            }

            std::chrono::duration<double, std::micro> duration = end - start;
            operator_times[layer_idx++].push_back(duration.count());

            if (status != xnn_status_success) {
                state.SkipWithError("failed to run a model");
                return;
            }
        }
    }

    for (int layer_idx = 0; layer_idx < operator_times.size(); layer_idx++) {
        std::ostringstream layer_idx_str; layer_idx_str << std::setw(2) << std::setfill('0') << layer_idx;

        state.counters["layer_" + layer_idx_str.str()] = median(operator_times[layer_idx]);
        state.counters["layer_enum_" + layer_idx_str.str()] = static_cast<int>(layer_type_str[layer_idx]);
        state.counters["layer_kh_" + layer_idx_str.str()] = kernel_shape[layer_idx].first;
        state.counters["layer_kw_" + layer_idx_str.str()] = kernel_shape[layer_idx].second;
        state.counters["layer_dM_" + layer_idx_str.str()] = std::get<0>(mul_shape[layer_idx]);
        state.counters["layer_dK_" + layer_idx_str.str()] = std::get<1>(mul_shape[layer_idx]);
        state.counters["layer_dN_" + layer_idx_str.str()] = std::get<2>(mul_shape[layer_idx]);
    }

    const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
    if (cpu_frequency != 0) {
        state.counters["cpufreq"] = cpu_frequency;
    }
}

template<models::ExecutionPlan Model(pthreadpool_t)>
void ModelBench(benchmark::State& state) {
    End2EndBenchmark(state, [](pthreadpool_t threadpool) {
      return Model(threadpool);
    });
}

template<models::ExecutionPlan Model(float , pthreadpool_t), int sparsity>
void SparseModelBench(benchmark::State& state) {
    End2EndBenchmark(state, [](pthreadpool_t threadpool) {
      return Model(sparsity / 100.f, threadpool);
    });
}

template<models::ExecutionPlan Model(float , pthreadpool_t), models::ExecutionPlan VerificationModel(float , pthreadpool_t), int sparsity>
void SparseModelBenchVerified(benchmark::State& state) {
    End2EndBenchmark(state, [](pthreadpool_t threadpool) {
      return Model(sparsity / 100.f, threadpool);
    }, [](pthreadpool_t threadpool) {
      return VerificationModel(sparsity / 100.f, threadpool);
    });
}

#endif //XNNPACK_BENCH_END2END_UTILS_HPP_
