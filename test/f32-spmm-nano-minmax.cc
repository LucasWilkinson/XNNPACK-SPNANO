// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-spmm-nano-minmax.yaml
//   Generator: tools/generate-spmm-nano-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/spmm.h>
#include "spmm-nano-microkernel-tester.h"


#if XNN_ARCH_X86_64
  TEST(NANO_61fee_c22a5_AVX512_512_4x6_float_relu6_80pct, k49) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t m = 64; m <= 1024; m <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatLoadBalancedNKM>()
          .mr(1)
          .nr(1)
          .m(m)
          .n(n)
          .k(49)
          .mapping_id("61fee")
          .executor_id("c22a5_AVX512_512_4x6")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_61fee_c22a5_AVX512_512_4x4_float_relu6_80pct, k49) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t m = 64; m <= 1024; m <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatLoadBalancedNKM>()
          .mr(1)
          .nr(1)
          .m(m)
          .n(n)
          .k(49)
          .mapping_id("61fee")
          .executor_id("c22a5_AVX512_512_4x4")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_400fa_77f9d_AVX512_512_8x3_float_relu6_80pct, k49) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t m = 64; m <= 1024; m <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatLoadBalancedNKM>()
          .mr(1)
          .nr(1)
          .m(m)
          .n(n)
          .k(49)
          .mapping_id("400fa")
          .executor_id("77f9d_AVX512_512_8x3")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_400fa_77f9d_AVX512_512_8x2_float_relu6_80pct, k49) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t m = 64; m <= 1024; m <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatLoadBalancedNKM>()
          .mr(1)
          .nr(1)
          .m(m)
          .n(n)
          .k(49)
          .mapping_id("400fa")
          .executor_id("77f9d_AVX512_512_8x2")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_61fee_c22a5_AVX512_512_4x6_float_relu6_80pct, k64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t m = 64; m <= 1024; m <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatLoadBalancedNKM>()
          .mr(1)
          .nr(1)
          .m(m)
          .n(n)
          .k(64)
          .mapping_id("61fee")
          .executor_id("c22a5_AVX512_512_4x6")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_61fee_c22a5_AVX512_512_4x4_float_relu6_80pct, k64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t m = 64; m <= 1024; m <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatLoadBalancedNKM>()
          .mr(1)
          .nr(1)
          .m(m)
          .n(n)
          .k(64)
          .mapping_id("61fee")
          .executor_id("c22a5_AVX512_512_4x4")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_400fa_77f9d_AVX512_512_8x3_float_relu6_80pct, k64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t m = 64; m <= 1024; m <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatLoadBalancedNKM>()
          .mr(1)
          .nr(1)
          .m(m)
          .n(n)
          .k(64)
          .mapping_id("400fa")
          .executor_id("77f9d_AVX512_512_8x3")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_400fa_77f9d_AVX512_512_8x2_float_relu6_80pct, k64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t m = 64; m <= 1024; m <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatLoadBalancedNKM>()
          .mr(1)
          .nr(1)
          .m(m)
          .n(n)
          .k(64)
          .mapping_id("400fa")
          .executor_id("77f9d_AVX512_512_8x2")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_61fee_c22a5_AVX512_512_4x6_float_relu6_80pct, k196) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t m = 64; m <= 1024; m <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatLoadBalancedNKM>()
          .mr(1)
          .nr(1)
          .m(m)
          .n(n)
          .k(196)
          .mapping_id("61fee")
          .executor_id("c22a5_AVX512_512_4x6")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_61fee_c22a5_AVX512_512_4x4_float_relu6_80pct, k196) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t m = 64; m <= 1024; m <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatLoadBalancedNKM>()
          .mr(1)
          .nr(1)
          .m(m)
          .n(n)
          .k(196)
          .mapping_id("61fee")
          .executor_id("c22a5_AVX512_512_4x4")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_400fa_77f9d_AVX512_512_8x3_float_relu6_80pct, k196) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t m = 64; m <= 1024; m <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatLoadBalancedNKM>()
          .mr(1)
          .nr(1)
          .m(m)
          .n(n)
          .k(196)
          .mapping_id("400fa")
          .executor_id("77f9d_AVX512_512_8x3")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_400fa_77f9d_AVX512_512_8x2_float_relu6_80pct, k196) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t m = 64; m <= 1024; m <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatLoadBalancedNKM>()
          .mr(1)
          .nr(1)
          .m(m)
          .n(n)
          .k(196)
          .mapping_id("400fa")
          .executor_id("77f9d_AVX512_512_8x2")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_61fee_c22a5_AVX512_512_4x6_float_relu6_80pct, k784) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t m = 64; m <= 1024; m <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatLoadBalancedNKM>()
          .mr(1)
          .nr(1)
          .m(m)
          .n(n)
          .k(784)
          .mapping_id("61fee")
          .executor_id("c22a5_AVX512_512_4x6")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_61fee_c22a5_AVX512_512_4x4_float_relu6_80pct, k784) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t m = 64; m <= 1024; m <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatLoadBalancedNKM>()
          .mr(1)
          .nr(1)
          .m(m)
          .n(n)
          .k(784)
          .mapping_id("61fee")
          .executor_id("c22a5_AVX512_512_4x4")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_400fa_77f9d_AVX512_512_8x3_float_relu6_80pct, k784) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t m = 64; m <= 1024; m <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatLoadBalancedNKM>()
          .mr(1)
          .nr(1)
          .m(m)
          .n(n)
          .k(784)
          .mapping_id("400fa")
          .executor_id("77f9d_AVX512_512_8x3")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_400fa_77f9d_AVX512_512_8x2_float_relu6_80pct, k784) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t m = 64; m <= 1024; m <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatLoadBalancedNKM>()
          .mr(1)
          .nr(1)
          .m(m)
          .n(n)
          .k(784)
          .mapping_id("400fa")
          .executor_id("77f9d_AVX512_512_8x2")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64
