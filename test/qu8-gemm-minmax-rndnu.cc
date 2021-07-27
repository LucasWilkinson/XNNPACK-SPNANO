// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qu8-gemm-minmax-rndnu.yaml
//   Generator: tools/generate-gemm-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/ppmm.h>
#include "gemm-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(83)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(8)
          .k(k)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(8)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(8)
          .k(k)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, no_a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, no_b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X8__NEON_MLAL_LANE, no_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 8; n++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(83)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 8; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, no_a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, no_b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X8__NEON_MLAL_LANE, no_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      for (uint32_t n = 1; n <= 16; n++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(83)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(16)
          .k(k)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(16)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(16)
          .k(k)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 1; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, no_a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, no_b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_1X16__NEON_MLAL_LANE, no_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      for (uint32_t n = 1; n <= 16; n++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(83)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(16)
          .k(k)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(16)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(16)
          .k(k)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t m = 1; m <= 4; m++) {
        for (uint32_t n = 1; n <= 16; n++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, no_a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, no_b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, no_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qu8_conv_minmax_rndnu_neon_params, xnn_init_qu8_requantization_rndnu_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64