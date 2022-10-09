#!/usr/bin/env python
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import bisect
import codecs
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from primes import next_prime
import xngen
import xnncommon


parser = argparse.ArgumentParser(description='XNNPACK generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Spec (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  common_name, target_name = name.split("__", 1)
  common_parts = common_name.split("_")
  param_spec = common_parts[-1]
  mr, nr = map(int, param_spec.split("x"))
  arch, isa = xnncommon.parse_target_name(target_name)
  return mr, nr, arch, isa


TEST_TEMPLATE = """\
TEST(${TEST_NAME}_float_relu6_80pct, k${KBLOCK}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t m = 64; m <= 1024; m <<= 1) {
    for (uint32_t n = 64; n <= 1024; n <<= 1) {
      SpMMNanoMicrokernelTester<sop::${KERNEL_DESC}>()
        .mr(${MR})
        .nr(${NR})
        .m(m)
        .n(n)
        .k(${KBLOCK})
        .mapping_id("${MAPPING_ID}")
        .executor_id("${EXECUTOR_ID}")
        .sparsity(0.8f)
        .Test(0.0f, 6.0f);
    }
  }
}
"""


def generate_test_cases(mr, nr, k_block, isa, mapping_id, executor_id):
  """Generates all tests cases for a GEMM micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    init_fn: C name of the function to initialize microkernel parameters.
    mr: MR parameter of the GEMM micro-kernel.
    nr: NR parameter of the GEMM micro-kernel.
    k_block: Number of K values processed per one iteration of the main loop of
             the micro-kernel.
    is_pipelined: Indicates if the micro-kernel is implemented with software
                  pipelining. Additional test cases are generated for software
                  pipelined micro-kernels to separately test prologue + epiloque
                  of the pipelined loop and iteration of the pipelined loop.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  # _, test_name = ukernel.split("_", 1)
  # _, datatype, ukernel_type, _ = ukernel.split("_", 3)
  # test_args = [ukernel]
  return xngen.preprocess(TEST_TEMPLATE, {
      "TEST_NAME": "NANO_" + mapping_id + "_" + executor_id,
      "TEST_ARGS": "",
      "UKERNEL_TYPE": "",
      "DATATYPE": "float",
      "MR": mr,
      "NR": nr,
      "KBLOCK": k_block,
      "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
      "KERNEL_DESC": "KD_IntelFloatLoadBalancedNKM",
      "MAPPING_ID": mapping_id,
      "EXECUTOR_ID": executor_id,
      "next_prime": next_prime,
    })


def main(args):
  options = parser.parse_args(args)

  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    # spec_yaml = yaml.safe_load(spec_file)
    # if not isinstance(spec_yaml, list):
    #   raise ValueError("expected a list of micro-kernels in the spec")

    tests = """\
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/spmm.h>
#include "spmm-nano-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    executor_mapping_pairs_to_test = [
        ("61fee", "c22a5_AVX512_512_4x6"),
        ("61fee", "c22a5_AVX512_512_4x4"),
        ("400fa", "77f9d_AVX512_512_8x3"),
        ("400fa", "77f9d_AVX512_512_8x2")
    ]


    for k_block in [49, 64, 196, 784]:
        for mapping_id, executor_id in executor_mapping_pairs_to_test:
            # specification can override architecture
            isa = "avx512f"
            arch = ["x86-64"]

            test_case = generate_test_cases(1, 1, k_block, isa, mapping_id, executor_id)
            tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    txt_changed = True
    if os.path.exists(options.output):
      with codecs.open(options.output, "r", encoding="utf-8") as output_file:
        txt_changed = output_file.read() != tests

    if txt_changed:
      with codecs.open(options.output, "w", encoding="utf-8") as output_file:
        output_file.write(tests)


if __name__ == "__main__":
  main(sys.argv[1:])
