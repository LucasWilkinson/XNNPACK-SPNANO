// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <random>
#include <vector>

#include <fp16.h>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>

#include "KernelDesc.h"
#include "MatMulSpecialized.h"

static inline bool is_fp16_zero(uint16_t x) {
  const uint16_t two_x = x + x;
  return two_x == 0;
}

template<typename KernelDesc>
class SpMMNanoMicrokernelTester {
 public:
  inline SpMMNanoMicrokernelTester& executor_id(std::string executor_id) {
    this->executor_id_ = executor_id;
    return *this;
  }

  inline SpMMNanoMicrokernelTester& mapping_id(std::string mapping_id) {
    this->mapping_id_ = mapping_id;
    return *this;
  }

  inline SpMMNanoMicrokernelTester& mr(size_t mr) {
    this->mr_ = mr;
    return *this;
  }

  inline size_t mr() const {
    return this->mr_;
  }

  inline SpMMNanoMicrokernelTester& nr(size_t nr) {
    this->nr_ = nr;
    return *this;
  }

  inline size_t nr() const {
    return this->nr_;
  }

  inline SpMMNanoMicrokernelTester& m(size_t m) {
    this->m_ = m;
    return *this;
  }

  inline size_t m() const {
    return this->m_;
  }

  inline SpMMNanoMicrokernelTester& n(size_t n) {
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline SpMMNanoMicrokernelTester& k(size_t k) {
    this->k_ = k;
    return *this;
  }

  inline size_t k() const {
    return this->k_;
  }

  inline SpMMNanoMicrokernelTester& output_stride(size_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  inline size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return m();
    } else {
      assert(this->output_stride_ >= m());
      return this->output_stride_;
    }
  }

  inline SpMMNanoMicrokernelTester& sparsity(float sparsity) {
    this->sparsity_ = sparsity;
    return *this;
  }

  inline float sparsity() const {
    return this->sparsity_;
  }

  inline SpMMNanoMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline SpMMNanoMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline SpMMNanoMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(float min, float max) const {
    ASSERT_GE(m(), 1);
    ASSERT_GE(n(), 1);
    ASSERT_GE(k(), 1);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;
    std::uniform_real_distribution<float> pdist;

    std::vector<float, AlignedAllocator<float, 64>> input(k() * m());
    // Think of b as (n/nr + n % nr) x k, expansion happens later.
    const size_t ncols = n() / nr() + n() % nr();
    std::vector<float> b(ncols * k());
    std::vector<float> bias(n());
    // Number of non-zero weights per N (output channel).
    std::vector<uint32_t> nmap(n());
    // Mapping from index of non-zero weight to increment of K (input channel) following this index.
    std::vector<int32_t> dmap(n() * k());
    std::vector<float> w(n() * k() + n());
    std::vector<float> output((n() - 1) * output_stride() + m());
    std::vector<float> output_ref(n() * m());

    auto coo = new COO<float>(n(), k());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), nanf(""));
      std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      std::fill(nmap.begin(), nmap.end(), 0);
      std::fill(dmap.begin(), dmap.end(), 0);
      std::fill(w.begin(), w.end(), 0.0f);

      for (float& b_value : b) {
        if (pdist(rng) <= sparsity()) {
          b_value = 0.0f;
        }
      }

      uint32_t nnz = 0;
      uint32_t wcnt = 0;
      size_t last_kk = 0;
      bool first_nzz = true;
      size_t first_kk = 0;
      for (size_t nn = 0; nn < n() / nr(); nn++) {
        for (size_t i = 0; i < nr(); ++i)
          w[wcnt++] = bias[nr() * nn + i];
        for (size_t kk = 0; kk < k(); kk++) {
          if (b[nn * k() + kk] != 0.0f) {
            // Every non-zero actually corresponds to nr adjacent non-zeros.
            for (size_t i = 0; i < nr(); ++i)
              w[wcnt++] = b[nn * k() + kk] + static_cast<float>(i);
            // Skip the very first non-zero weight as we record only the difference.
            if (first_nzz) {
              first_kk = kk;
            } else {
              const int32_t increment = int32_t(kk - last_kk) * int32_t(m() * sizeof(float));
              dmap[nnz++] = increment;
            }
            last_kk = kk;
            first_nzz = false;
            nmap[nn] += 1;

            ASSERT_TRUE(nr() == 1); // For nano we expect unstructured weights.
            // Construct COO matrix for the SPNANO kernel.
            coo->append_nnz({(int) nn, (int) kk, b[nn * k() + kk]});
          }
        }
      }

      // now we've constructed the matrix for the blocked part and switch to the
      // leftovers, which we do as nr=1 always.
      for (size_t nn = n() / nr(); nn < ncols; nn++) {
        w[wcnt++] = bias[(n() / nr()) * nr() + (nn - n() / nr())];
        for (size_t kk = 0; kk < k(); kk++) {
          if (b[nn * k() + kk] != 0.0f) {
            // Every non-zero actually corresponds to nr adjacent non-zeros.
            w[wcnt++] = b[nn * k() + kk];
            // Skip the very first non-zero weight as we record only the difference.
            if (first_nzz) {
              first_kk = kk;
            } else {
              const int32_t increment = int32_t(kk - last_kk) * int32_t(m() * sizeof(float));
              dmap[nnz++] = increment;
            }
            last_kk = kk;
            first_nzz = false;
            nmap[nn] += 1;
          }
        }
      }
      // In the end, we must return input pointer to the initial value.
      const int64_t increment = int32_t(first_kk - last_kk) * int32_t(m() * sizeof(float));
      dmap[nnz++] = increment;

      // Generate expanded b which will be used in reference calculation.
      // Everywhere there is input non-zero in the original we copy it and add an
      // adjacent non-zero with incremented weight value.
      std::vector<float> b_full(n() * k());
      if (nr() == 1) {
         b_full = b;
      }
      else {
        for (size_t nn = 0; nn < n() / nr(); nn++) {
          for (size_t kk = 0; kk < k(); kk++) {
            if (b[nn * k() + kk] != 0.0f) {
              for (size_t i = 0; i < nr(); ++i)
                b_full[nr() * nn * k() + i * k() + kk] = b[nn * k() + kk] + static_cast<float>(i);
            }
          }
        }
        for (size_t nn = n() / nr(); nn < ncols; nn++) {
          for (size_t kk = 0; kk < k(); kk++) {
            if (b[nn * k() + kk] != 0.0f) {
              b_full[nr() * (n() / nr()) * k() + (nn - n() / nr()) * k() + kk] = b[nn * k() + kk];
            }
          }
        }
      }

      for (size_t oc = 0; oc < n(); oc++) {
        for (size_t pxb = 0; pxb < m(); pxb++) {
          output_ref[oc * m() + pxb] = bias[oc];
          for (size_t ic = 0; ic < k(); ic++) {
            output_ref[oc * m() + pxb] += input[ic * m() + pxb] * b_full[oc * k() + ic];
          }
        }
      }

      // Micro-kernel can access one element beyond w and dmap for software pipelining.
      w.resize(wcnt + 1);
      dmap.resize(nnz + 1);

      // Apply minmax to reference
      for (float& output_value : output_ref) {
        output_value = std::max(min, std::min(max, output_value));
      }

      sop::TileConfig tile_config;

      tile_config.M_c = 8;
      tile_config.N_c = 8;
      tile_config.K_c = 32;
      tile_config.tiling_strategy = sop::MANUAL_TILING;

      sop::MatMulSpecialized<KernelDesc> matmul(
        coo, m(), tile_config, 1, executor_id_, mapping_id_);

      matmul.allocate_executor(m());
      matmul(output.data(), input.data(), bias.data(), sop::MINMAX, min, max);

      // Validate micro-kernel outputs.
      for (size_t i = 0; i < m(); i++) {
        for (size_t j = 0; j < n(); j++) {
          ASSERT_NEAR(
              output[j * output_stride() + i],
              output_ref[j * m() + i],
              std::abs(output_ref[j * m() + i]) * 1.0e-6f)
            << "at M index " << i
            << ", N index " << j << " / " << n() << " (tile " << nr() << ")"
            << ", K = " << k()
            << " | " << "executor_id = " << executor_id_
            << ", mapping_id = " << mapping_id_;
        }
      }
    }
  }

  void Test(uint16_t min, uint16_t max) const {
    ASSERT_GE(m(), 1);
    ASSERT_GE(n(), 1);
    ASSERT_GE(k(), 1);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;
    std::uniform_real_distribution<float> pdist;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> input(k() * m());
    // Think of b as (n/nr + n % nr) x k, expansion happens later.
    const size_t ncols = n() / nr() + n() % nr();
    std::vector<uint16_t> b(ncols * k());
    std::vector<uint16_t> bias(n());
    // Number of non-zero weights per N (output channel).
    std::vector<uint32_t> nmap(n());
    // Mapping from index of non-zero weight to increment of K (input channel) following this index.
    std::vector<int32_t> dmap(n() * k());
    std::vector<uint16_t> w(n() * k() + n());
    std::vector<uint16_t> output((n() - 1) * output_stride() + m());
    std::vector<float> output_ref(n() * m());

    auto coo = new COO<uint16_t>(n(), k());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::generate(b.begin(), b.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::generate(bias.begin(), bias.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(output.begin(), output.end(), 0xC000);
      std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      std::fill(nmap.begin(), nmap.end(), 0);
      std::fill(dmap.begin(), dmap.end(), 0);
      std::fill(w.begin(), w.end(), 0);

      for (uint16_t& b_value : b) {
        if (pdist(rng) <= sparsity()) {
          b_value = 0;
        }
      }

      uint32_t nnz = 0;
      uint32_t wcnt = 0;
      size_t last_kk = 0;
      bool first_nzz = true;
      size_t first_kk = 0;
      for (size_t nn = 0; nn < n() / nr(); nn++) {
        for (size_t i = 0; i < nr(); ++i)
          w[wcnt++] = bias[nr() * nn + i];
        for (size_t kk = 0; kk < k(); kk++) {
          if (!is_fp16_zero(b[nn * k() + kk])) {
            // Every non-zero actually corresponds to nr adjacent non-zeros.
            for (size_t i = 0; i < nr(); ++i)
              w[wcnt++] = fp16_ieee_from_fp32_value(fp16_ieee_to_fp32_value(b[nn * k() + kk]) + static_cast<float>(i));
            // Skip the very first non-zero weight as we record only the difference.
            if (first_nzz) {
              first_kk = kk;
            } else {
              const int32_t increment = int32_t(kk - last_kk) * int32_t(m() * sizeof(uint16_t));
              dmap[nnz++] = increment;
            }
            last_kk = kk;
            first_nzz = false;
            nmap[nn] += 1;

            ASSERT_TRUE(nr() == 1); // For nano we expect unstructured weights.
            // Construct COO matrix for the SPNANO kernel.
            coo->append_nnz({(int) nn, (int) kk, b[nn * k() + kk]});
          }
        }
      }

      // now we've constructed the matrix for the blocked part and switch to the
      // leftovers, which we do as nr=1 always.
      for (size_t nn = n() / nr(); nn < ncols; nn++) {
        w[wcnt++] = bias[(n() / nr()) * nr() + (nn - n() / nr())];
        for (size_t kk = 0; kk < k(); kk++) {
          if (!is_fp16_zero(b[nn * k() + kk])) {
            // Every non-zero actually corresponds to nr adjacent non-zeros.
            w[wcnt++] = b[nn * k() + kk];
            // Skip the very first non-zero weight as we record only the difference.
            if (first_nzz) {
              first_kk = kk;
            } else {
              const int32_t increment = int32_t(kk - last_kk) * int32_t(m() * sizeof(uint16_t));
              dmap[nnz++] = increment;
            }
            last_kk = kk;
            first_nzz = false;
            nmap[nn] += 1;
          }
        }
      }
      // In the end, we must return input pointer to the initial value.
      const int64_t increment = int32_t(first_kk - last_kk) * int32_t(m() * sizeof(uint16_t));
      dmap[nnz++] = increment;

      // Construct MatMul
      //sop::MatMul matmul;


      // Generate expanded b which will be used in reference calculation.
      // Everywhere there is input non-zero in the original we copy it and add an
      // adjacent non-zero with incremented weight value.
      std::vector<uint16_t> b_full(n() * k());
      if (nr() == 1) {
         b_full = b;
      }
      else {
        for (size_t nn = 0; nn < n() / nr(); nn++) {
          for (size_t kk = 0; kk < k(); kk++) {
            if (b[nn * k() + kk] != 0.0f) {
              for (size_t i = 0; i < nr(); ++i)
                b_full[nr() * nn * k() + i * k() + kk] = fp16_ieee_from_fp32_value(
                  fp16_ieee_to_fp32_value(b[nn * k() + kk]) + static_cast<float>(i));
            }
          }
        }
        for (size_t nn = n() / nr(); nn < ncols; nn++) {
          for (size_t kk = 0; kk < k(); kk++) {
            if (b[nn * k() + kk] != 0.0f) {
              b_full[nr() * (n() / nr()) * k() + (nn - n() / nr()) * k() + kk] = b[nn * k() + kk];
            }
          }
        }
      }

      for (size_t oc = 0; oc < n(); oc++) {
        for (size_t pxb = 0; pxb < m(); pxb++) {
          output_ref[oc * m() + pxb] = fp16_ieee_to_fp32_value(bias[oc]);
          for (size_t ic = 0; ic < k(); ic++) {
            output_ref[oc * m() + pxb] += fp16_ieee_to_fp32_value(input[ic * m() + pxb]) * fp16_ieee_to_fp32_value(b_full[oc * k() + ic]);
          }
        }
      }

      // Micro-kernel can access one element beyond w and dmap for software pipelining.
      w.resize(wcnt + 1);
      dmap.resize(nnz + 1);

      // Apply minmax to reference
//      for (float& output_value : output_ref) {
//        output_value = std::max(min, std::min(max, output_value));
//      }


      // Prepare parameters.
//      xnn_f16_minmax_params params;
//      init_params(&params,
//        fp16_ieee_from_fp32_value(output_min), fp16_ieee_from_fp32_value(output_max));
//
//      spmm(m() * sizeof(uint16_t), n(),
//        input.data() + first_kk * m(),
//        w.data(), dmap.data(), nmap.data(),
//        output.data(), output_stride() * sizeof(uint16_t),
//        &params);

      ASSERT_TRUE(false);

      // Validate micro-kernel outputs.
      for (size_t i = 0; i < m(); i++) {
        for (size_t j = 0; j < n(); j++) {
          ASSERT_NEAR(
            output[j * output_stride() + i],
            output_ref[j * m() + i],
            std::abs(output_ref[j * m() + i]) * 1.0e-5f)
            << "at M index " << i
            << ", N index " << j << " / " << n() << " (tile " << nr() << ")"
            << ", K = " << k()
            << " | " << "executor_id = " << executor_id_
            << ", mapping_id = " << mapping_id_;
        }
      }
    }
  }

 private:
  std::string executor_id_;
  std::string mapping_id_;

  size_t mr_{1};
  size_t nr_{1};
  size_t m_{1};
  size_t n_{1};
  size_t k_{1};
  size_t output_stride_{0};
  float sparsity_{0.5f};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{1};
};
