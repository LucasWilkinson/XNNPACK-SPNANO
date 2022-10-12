// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: none
//   Generator: tools/generate-spmm-nano-neon-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/spmm.h>
#include "spmm-nano-microkernel-tester.h"


#if XNN_ARCH_ARM64
  TEST(NANO_61fee_c22a5_NEON_128_4x2_KD_PIFloatSplitM_float_relu6_80pct, m49) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_PIFloatSplitM>()
          .mr(1)
          .nr(1)
          .m(49)
          .n(n)
          .k(k)
          .mapping_id("61fee")
          .executor_id("c22a5_NEON_128_4x2")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(NANO_61fee_c22a5_NEON_128_4x2_KD_PIFloatSplitM_float_relu6_80pct, m64) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_PIFloatSplitM>()
          .mr(1)
          .nr(1)
          .m(64)
          .n(n)
          .k(k)
          .mapping_id("61fee")
          .executor_id("c22a5_NEON_128_4x2")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(NANO_61fee_c22a5_NEON_128_4x2_KD_PIFloatSplitM_float_relu6_80pct, m196) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_PIFloatSplitM>()
          .mr(1)
          .nr(1)
          .m(196)
          .n(n)
          .k(k)
          .mapping_id("61fee")
          .executor_id("c22a5_NEON_128_4x2")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(NANO_61fee_c22a5_NEON_128_4x2_KD_PIFloatSplitM_float_relu6_80pct, m784) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_PIFloatSplitM>()
          .mr(1)
          .nr(1)
          .m(784)
          .n(n)
          .k(k)
          .mapping_id("61fee")
          .executor_id("c22a5_NEON_128_4x2")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(NANO_61fee_c22a5_NEON_128_4x2_KD_PIFloatSplitM_float_relu6_80pct, m3136) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_PIFloatSplitM>()
          .mr(1)
          .nr(1)
          .m(3136)
          .n(n)
          .k(k)
          .mapping_id("61fee")
          .executor_id("c22a5_NEON_128_4x2")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_ARM64
