// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: none
//   Generator: tools/generate-spmm-nano-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/spmm.h>
#include "spmm-nano-microkernel-tester.h"


#if XNN_ARCH_X86_64
  TEST(NANO_da01e_64487_AVX512_512_4x6_KD_IntelFloatKNM_float_relu6_80pct, m49) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatKNM>()
          .mr(1)
          .nr(1)
          .m(49)
          .n(n)
          .k(k)
          .mapping_id("da01e")
          .executor_id("64487_AVX512_512_4x6")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_da01e_64487_AVX512_512_4x6_KD_IntelFloatNKM_float_relu6_80pct, m49) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatNKM>()
          .mr(1)
          .nr(1)
          .m(49)
          .n(n)
          .k(k)
          .mapping_id("da01e")
          .executor_id("64487_AVX512_512_4x6")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_61fee_c22a5_AVX512_512_4x6_KD_IntelFloatKNM_float_relu6_80pct, m49) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatKNM>()
          .mr(1)
          .nr(1)
          .m(49)
          .n(n)
          .k(k)
          .mapping_id("61fee")
          .executor_id("c22a5_AVX512_512_4x6")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_61fee_c22a5_AVX512_512_4x6_KD_IntelFloatNKM_float_relu6_80pct, m49) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatNKM>()
          .mr(1)
          .nr(1)
          .m(49)
          .n(n)
          .k(k)
          .mapping_id("61fee")
          .executor_id("c22a5_AVX512_512_4x6")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_400fa_77f9d_AVX512_512_8x3_KD_IntelFloatKNM_float_relu6_80pct, m49) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatKNM>()
          .mr(1)
          .nr(1)
          .m(49)
          .n(n)
          .k(k)
          .mapping_id("400fa")
          .executor_id("77f9d_AVX512_512_8x3")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_400fa_77f9d_AVX512_512_8x3_KD_IntelFloatNKM_float_relu6_80pct, m49) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatNKM>()
          .mr(1)
          .nr(1)
          .m(49)
          .n(n)
          .k(k)
          .mapping_id("400fa")
          .executor_id("77f9d_AVX512_512_8x3")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_da01e_64487_AVX512_512_4x6_KD_IntelFloatKNM_float_relu6_80pct, m64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatKNM>()
          .mr(1)
          .nr(1)
          .m(64)
          .n(n)
          .k(k)
          .mapping_id("da01e")
          .executor_id("64487_AVX512_512_4x6")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_da01e_64487_AVX512_512_4x6_KD_IntelFloatNKM_float_relu6_80pct, m64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatNKM>()
          .mr(1)
          .nr(1)
          .m(64)
          .n(n)
          .k(k)
          .mapping_id("da01e")
          .executor_id("64487_AVX512_512_4x6")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_61fee_c22a5_AVX512_512_4x6_KD_IntelFloatKNM_float_relu6_80pct, m64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatKNM>()
          .mr(1)
          .nr(1)
          .m(64)
          .n(n)
          .k(k)
          .mapping_id("61fee")
          .executor_id("c22a5_AVX512_512_4x6")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_61fee_c22a5_AVX512_512_4x6_KD_IntelFloatNKM_float_relu6_80pct, m64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatNKM>()
          .mr(1)
          .nr(1)
          .m(64)
          .n(n)
          .k(k)
          .mapping_id("61fee")
          .executor_id("c22a5_AVX512_512_4x6")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_400fa_77f9d_AVX512_512_8x3_KD_IntelFloatKNM_float_relu6_80pct, m64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatKNM>()
          .mr(1)
          .nr(1)
          .m(64)
          .n(n)
          .k(k)
          .mapping_id("400fa")
          .executor_id("77f9d_AVX512_512_8x3")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_400fa_77f9d_AVX512_512_8x3_KD_IntelFloatNKM_float_relu6_80pct, m64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatNKM>()
          .mr(1)
          .nr(1)
          .m(64)
          .n(n)
          .k(k)
          .mapping_id("400fa")
          .executor_id("77f9d_AVX512_512_8x3")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_da01e_64487_AVX512_512_4x6_KD_IntelFloatKNM_float_relu6_80pct, m196) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatKNM>()
          .mr(1)
          .nr(1)
          .m(196)
          .n(n)
          .k(k)
          .mapping_id("da01e")
          .executor_id("64487_AVX512_512_4x6")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_da01e_64487_AVX512_512_4x6_KD_IntelFloatNKM_float_relu6_80pct, m196) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatNKM>()
          .mr(1)
          .nr(1)
          .m(196)
          .n(n)
          .k(k)
          .mapping_id("da01e")
          .executor_id("64487_AVX512_512_4x6")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_61fee_c22a5_AVX512_512_4x6_KD_IntelFloatKNM_float_relu6_80pct, m196) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatKNM>()
          .mr(1)
          .nr(1)
          .m(196)
          .n(n)
          .k(k)
          .mapping_id("61fee")
          .executor_id("c22a5_AVX512_512_4x6")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_61fee_c22a5_AVX512_512_4x6_KD_IntelFloatNKM_float_relu6_80pct, m196) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatNKM>()
          .mr(1)
          .nr(1)
          .m(196)
          .n(n)
          .k(k)
          .mapping_id("61fee")
          .executor_id("c22a5_AVX512_512_4x6")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_400fa_77f9d_AVX512_512_8x3_KD_IntelFloatKNM_float_relu6_80pct, m196) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatKNM>()
          .mr(1)
          .nr(1)
          .m(196)
          .n(n)
          .k(k)
          .mapping_id("400fa")
          .executor_id("77f9d_AVX512_512_8x3")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_400fa_77f9d_AVX512_512_8x3_KD_IntelFloatNKM_float_relu6_80pct, m196) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatNKM>()
          .mr(1)
          .nr(1)
          .m(196)
          .n(n)
          .k(k)
          .mapping_id("400fa")
          .executor_id("77f9d_AVX512_512_8x3")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_da01e_64487_AVX512_512_4x6_KD_IntelFloatKNM_float_relu6_80pct, m784) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatKNM>()
          .mr(1)
          .nr(1)
          .m(784)
          .n(n)
          .k(k)
          .mapping_id("da01e")
          .executor_id("64487_AVX512_512_4x6")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_da01e_64487_AVX512_512_4x6_KD_IntelFloatNKM_float_relu6_80pct, m784) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatNKM>()
          .mr(1)
          .nr(1)
          .m(784)
          .n(n)
          .k(k)
          .mapping_id("da01e")
          .executor_id("64487_AVX512_512_4x6")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_61fee_c22a5_AVX512_512_4x6_KD_IntelFloatKNM_float_relu6_80pct, m784) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatKNM>()
          .mr(1)
          .nr(1)
          .m(784)
          .n(n)
          .k(k)
          .mapping_id("61fee")
          .executor_id("c22a5_AVX512_512_4x6")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_61fee_c22a5_AVX512_512_4x6_KD_IntelFloatNKM_float_relu6_80pct, m784) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatNKM>()
          .mr(1)
          .nr(1)
          .m(784)
          .n(n)
          .k(k)
          .mapping_id("61fee")
          .executor_id("c22a5_AVX512_512_4x6")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_400fa_77f9d_AVX512_512_8x3_KD_IntelFloatKNM_float_relu6_80pct, m784) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatKNM>()
          .mr(1)
          .nr(1)
          .m(784)
          .n(n)
          .k(k)
          .mapping_id("400fa")
          .executor_id("77f9d_AVX512_512_8x3")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_400fa_77f9d_AVX512_512_8x3_KD_IntelFloatNKM_float_relu6_80pct, m784) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatNKM>()
          .mr(1)
          .nr(1)
          .m(784)
          .n(n)
          .k(k)
          .mapping_id("400fa")
          .executor_id("77f9d_AVX512_512_8x3")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_da01e_64487_AVX512_512_4x6_KD_IntelFloatKNM_float_relu6_80pct, m3136) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatKNM>()
          .mr(1)
          .nr(1)
          .m(3136)
          .n(n)
          .k(k)
          .mapping_id("da01e")
          .executor_id("64487_AVX512_512_4x6")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_da01e_64487_AVX512_512_4x6_KD_IntelFloatNKM_float_relu6_80pct, m3136) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatNKM>()
          .mr(1)
          .nr(1)
          .m(3136)
          .n(n)
          .k(k)
          .mapping_id("da01e")
          .executor_id("64487_AVX512_512_4x6")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_61fee_c22a5_AVX512_512_4x6_KD_IntelFloatKNM_float_relu6_80pct, m3136) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatKNM>()
          .mr(1)
          .nr(1)
          .m(3136)
          .n(n)
          .k(k)
          .mapping_id("61fee")
          .executor_id("c22a5_AVX512_512_4x6")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_61fee_c22a5_AVX512_512_4x6_KD_IntelFloatNKM_float_relu6_80pct, m3136) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatNKM>()
          .mr(1)
          .nr(1)
          .m(3136)
          .n(n)
          .k(k)
          .mapping_id("61fee")
          .executor_id("c22a5_AVX512_512_4x6")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_400fa_77f9d_AVX512_512_8x3_KD_IntelFloatKNM_float_relu6_80pct, m3136) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatKNM>()
          .mr(1)
          .nr(1)
          .m(3136)
          .n(n)
          .k(k)
          .mapping_id("400fa")
          .executor_id("77f9d_AVX512_512_8x3")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64


#if XNN_ARCH_X86_64
  TEST(NANO_400fa_77f9d_AVX512_512_8x3_KD_IntelFloatNKM_float_relu6_80pct, m3136) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 64; k <= 1024; k <<= 1) {
      for (uint32_t n = 64; n <= 1024; n <<= 1) {
        SpMMNanoMicrokernelTester<sop::KD_IntelFloatNKM>()
          .mr(1)
          .nr(1)
          .m(3136)
          .n(n)
          .k(k)
          .mapping_id("400fa")
          .executor_id("77f9d_AVX512_512_8x3")
          .sparsity(0.8f)
          .Test(0.0f, 6.0f);
      }
    }
  }
#endif  // XNN_ARCH_X86_64
