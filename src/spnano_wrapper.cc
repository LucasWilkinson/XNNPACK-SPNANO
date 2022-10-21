#include "xnnpack.h"
#include <xnnpack/log.h>
#include <xnnpack/operator.h>
#include <xnnpack/params.h>
#include <cpuinfo.h>

#include "spnano_wrapper.h"

#include <iostream>

#include "COO.h"
#include "KernelDesc.h"
#include "MatMulSpecialized.h"


#include "mapping_io.h"
#include "utils/misc.h"

#include "MicroKernelBase.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>


#include "utils/misc.h"

#include <iostream>
#include <filesystem>


extern "C" spnano_coo_t spnano_allocate_coo_matrix_f32(int rows, int cols) {
  return new COO<float>(rows, cols);
}

extern "C" void spnano_coo_matrix_add_nnz_f32(spnano_coo_t m, int i, int j, float val) {
  COO<float>* coo = reinterpret_cast<COO<float>*>(m);
  coo->append_nnz({i, j, val});
}


static sop::MatMul<float>* create_matmul_avx(COO<float>* coo, int num_threads, int b_cols) {
  std::string mapping_id = "";
  std::string executor_id = "";

  sop::TileConfig tile_config;
  tile_config.tiling_strategy = sop::CAKE_TILING_WITH_TLB_COMPENSATION;

  sop::MatMul<float>* matmul = nullptr;
  bool packed = false;

  if (cpuinfo_has_x86_avx512vl()) {
    packed = (b_cols % 16) != 0 && num_threads == 1;  // packing is less efficient in the parallel
    if (num_threads == 1) {
//      if (b_cols >= 1024) { // TODO: Test orig mapping with new schedule
//        mapping_id = "da01e";
//        executor_id = "64487_AVX512_512_4x6";
//        tile_config.runtimeSchedule = sop::nmMN;
//      }
//      else
      if (b_cols >= 512) {
        mapping_id = "61fee";
        executor_id = "c22a5_AVX512_512_4x6";
        tile_config.runtimeSchedule = sop::nmN;
      }
      else if (b_cols >= 128) {
        mapping_id = "61fee";
        executor_id = "c22a5_AVX512_512_4x6";
        tile_config.runtimeSchedule = sop::nmN;
      }
      else {
        mapping_id = "400fa";
        executor_id = "77f9d_AVX512_512_8x3";
        tile_config.runtimeSchedule = sop::nmKNM;
      }
    }
    else {
      //      if (b_cols >= 1024) { // TODO: Test orig mapping with new schedule
      //        mapping_id = "da01e";
      //        executor_id = "64487_AVX512_512_4x6";
      //        tile_config.runtimeSchedule = sop::nmMN;
      //      }
      //      else
      if (b_cols >= 512) {
        mapping_id = "61fee";
        executor_id = "c22a5_AVX512_512_4x6";
        tile_config.runtimeSchedule = sop::nmN;
      }
      else if (b_cols >= 128) {
        mapping_id = "400fa";
        executor_id = "77f9d_AVX512_512_8x3";
        tile_config.runtimeSchedule = sop::nmKNM;
      }
      else {
        mapping_id = "400fa";
        executor_id = "77f9d_AVX512_512_8x3";
        tile_config.runtimeSchedule = sop::nmKNM;
      }
    }
  } else if (cpuinfo_has_x86_avx2()) {
    if (num_threads >= 1) {
      if (b_cols >= 1024) {
        mapping_id = "da01e";
        executor_id = "64487_AVX2_256_4x3";
        tile_config.runtimeSchedule = sop::nmMN;
      }
      else if (b_cols >= 512) {
        mapping_id = "61fee";
        executor_id = "c22a5_AVX2_256_4x3";
        tile_config.runtimeSchedule = sop::nmNKM;
      }
      else {
        mapping_id = "400fa";
        executor_id = "77f9d_AVX2_256_8x1";
        tile_config.runtimeSchedule = sop::nmNKM;
      }
    }
    else {
      std::cout << "Not yet supported" << std::endl;
      exit(-1);
    }
  }

#define CREATE_MATMUL(_packed, _load_balance_cond, KernelDesc)                          \
    if (packed == _packed && _load_balance_cond) {                                      \
      ERROR_AND_EXIT_IF(matmul, "Already found matching kernel desc");                  \
      matmul = new sop::MatMulSpecialized<KernelDesc>(                                  \
        coo, b_cols, tile_config, num_threads, executor_id, mapping_id                  \
      );                                                                                \
    }

  CREATE_MATMUL(false, (num_threads == 1), sop::KD_IntelFloat);
  CREATE_MATMUL(false, (num_threads >  1), sop::KD_IntelFloatLoadBalanced);
  CREATE_MATMUL(true,  (num_threads == 1), sop::KD_IntelFloatPacked);
  CREATE_MATMUL(true,  (num_threads >  1), sop::KD_IntelFloatLoadBalancedPacked);

  return matmul;
}

static sop::MatMul<float>* create_matmul_neon(COO<float>* coo, int num_threads, int b_cols) {
  std::string mapping_id = "";
  std::string executor_id = "";

  sop::TileConfig tile_config;
  sop::MatMul<float>* matmul = nullptr;

  // NOTE: Executor mapping pairs
  //   "61fee" -> "c22a5", Mr = 4, identity
  //   "da01e" -> "c22a5", Mr = 4
  //   "400fa" -> "77f9d", Mr = 8

  int N_c = 0, K_c = 0, M_c = 0;
  if (num_threads == 1) {
    if (b_cols >= 1024) {
      mapping_id = "61fee";
      executor_id = "c22a5_NEON_128_4x4";
      tile_config.runtimeSchedule = sop::nmKN;
      N_c = 64;
      M_c = coo->rows();
      K_c = 64;
    }
    else if (b_cols >= 512) {
      mapping_id = "61fee";
      executor_id = "c22a5_NEON_128_4x4";
      tile_config.runtimeSchedule = sop::nmKN;
      N_c = 16;
      M_c = coo->rows();
      K_c = 128;
    }
    else if (b_cols >= 128) {
      mapping_id = "61fee";
      executor_id = "c22a5_NEON_128_4x6";
      tile_config.runtimeSchedule = sop::nmKM;
      N_c = coo->cols();
      M_c = 64;
      K_c = 128;
    }
    else {
      mapping_id = "61fee";
      executor_id = "c22a5_NEON_128_4x6";
      tile_config.runtimeSchedule = sop::nmKM;
      N_c = coo->cols();
      M_c = 64;
      K_c = 128;
    }
  } else {
    if (b_cols >= 1024) {
      mapping_id = "61fee";
      executor_id = "c22a5_NEON_128_4x4";
      tile_config.runtimeSchedule = sop::nmKN;
      N_c = 64;
      M_c = coo->rows();
      K_c = 64;
    }
    else if (b_cols >= 512) {
      mapping_id = "61fee";
      executor_id = "c22a5_NEON_128_4x6";
      tile_config.runtimeSchedule = sop::nmKN;
      N_c = 24;
      M_c = coo->rows();
      K_c = 128;
    }
    else if (b_cols >= 128) {
      mapping_id = "61fee";
      executor_id = "c22a5_NEON_128_4x6";
      tile_config.runtimeSchedule = sop::nmKM;
      N_c = coo->cols();
      M_c = 64;
      K_c = 32;
    }
    else {
      mapping_id = "61fee";
      executor_id = "c22a5_NEON_128_4x6";
      tile_config.runtimeSchedule = sop::nmKM;
      N_c = coo->cols();
      M_c = 64;
      K_c = 64;
    }
  }

  tile_config.M_c = M_c;
  tile_config.N_c = N_c;
  tile_config.K_c = K_c;
  tile_config.tiling_strategy = sop::MANUAL_TILING;

  matmul = new sop::MatMulSpecialized<sop::KD_PIFloat>(
    coo, b_cols, tile_config, num_threads, executor_id, mapping_id
  );

  return matmul;
}


extern "C" spnano_matmul_t spnano_allocate_matmul_f32(spnano_coo_t m, int num_threads, int b_cols)
{
  sop::MatMul<float>* matmul = nullptr;
  COO<float>* coo = reinterpret_cast<COO<float>*>(m);

#if defined(__x86_64__)
  if (cpuinfo_has_x86_avx2() || cpuinfo_has_x86_avx512f()) {
    matmul = create_matmul_avx(coo, num_threads, b_cols);
#endif
#ifdef _M_ARM64
  } else if (cpuinfo_has_arm_neon_v8()) {
    matmul = create_matmul_neon(coo, num_threads, b_cols);
#endif
  } else {
    std::cout << "Architecture not yet supported" << std::endl;
    exit(-1);
  }

  matmul->allocate_executor(b_cols);
  return matmul;
}

extern "C" void spnano_delete_matmul_f32(spnano_matmul_t m) {
  sop::MatMul<float>* matmul = reinterpret_cast<sop::MatMul<float>*>(m);
  delete matmul;
}

extern "C" spnano_executor_t spnano_get_executor_f32(spnano_matmul_t m) {
  auto matmul = reinterpret_cast<sop::MatMul<float>*>(m);
  return (void*) matmul->get_executor();
}

extern "C" int spnano_get_num_parallel_tiles_f32(spnano_executor_t e) {
  auto executor = reinterpret_cast<sop::Executor<float>*>(e);
  int par_tiles = executor->num_parallel_tile();
  return par_tiles;
}

extern "C" void spnano_run_f32(spnano_executor_t e, float* output, const float* input, const float* bias, float min, float max) {
  auto executor = reinterpret_cast<sop::Executor<float>*>(e);
  (*executor)(output, input, bias, sop::MINMAX, min, max);
}

extern "C" void spnano_begin_threaded_f32(spnano_executor_t e, float* output, const float* input, const float* bias, float min, float max) {
  auto executor = reinterpret_cast<sop::Executor<float>*>(e);
  executor->begin_threaded(output, input, bias, sop::MINMAX, min, max);
}

extern "C" void spnano_run_thread_f32(spnano_executor_t e, int p, int tid) {
  auto executor = reinterpret_cast<sop::Executor<float>*>(e);
  executor->execute_thread(p, tid);
}

extern "C" enum xnn_status xnn_delete_spnano_operator_f32(xnn_operator_t op)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    //xnn_log_error("failed to delete operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (op == NULL) {
    return xnn_status_invalid_parameter;
  }

  spnano_delete_matmul_f32(op->context.spmm_nano.matmul);

  return xnn_delete_operator(op);
}
