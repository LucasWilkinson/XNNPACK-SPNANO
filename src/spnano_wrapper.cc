#include "xnnpack.h"
#include <xnnpack/log.h>
#include <xnnpack/operator.h>
#include <xnnpack/params.h>

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


extern "C" spnano_matmul_t spnano_allocate_matmul_f32(spnano_coo_t m, int num_threads, int b_cols) {
  std::string mapping_id = "";
  std::string executor_id = "";
  std::string schedule = "KNM";
  bool packed = false;

#ifdef __AVX512VL__
  packed = (b_cols % 16) != 0 && num_threads == 1; // Parallel packing not yet supported
#endif

  //b_cols = b_cols & ~(16-1);

  COO<float>* coo = reinterpret_cast<COO<float>*>(m);

  //std::cout << "BCols " << b_cols << " packed " << packed << std::endl;


  sop::TileConfig tile_config;
  sop::MatMul<float>* matmul = nullptr;

#ifdef __AVX512VL__

  if (num_threads >= 1) {
    if (b_cols >= 1024) {
      mapping_id = "da01e";
      executor_id = "64487_AVX512_512_4x6";
      schedule = "KNM";
    } else if (b_cols >= 512) {
      mapping_id = "61fee";
      executor_id = "c22a5_AVX512_512_4x6";
      schedule = "NKM";
    } else {
      mapping_id = "400fa";
      executor_id = "77f9d_AVX512_512_8x3";
      schedule = "KNM";
    }
  } else {
    std::cout << "Not yet supported" << std::endl;
    exit(-1);
  }

#elif __AVX2__
    if (num_threads >= 1) {
      if (b_cols >= 1024) {
        mapping_id = "da01e";
        executor_id = "64487_AVX2_256_4x3";
        schedule = "KNM";
      }
      else if (b_cols >= 512) {
        mapping_id = "61fee";
        executor_id = "c22a5_AVX2_256_4x3";
        schedule = "NKM";
      }
      else {
        mapping_id = "400fa";
        executor_id = "77f9d_AVX2_256_8x1";
        schedule = "KNM";
      }
    }
    else {
      std::cout << "Not yet supported" << std::endl;
      exit(-1);
    }
#endif

#if (defined(__AVX512F__) || defined(__AVX2__))

#define CREATE_MATMUL(_packed, _load_balance_cond, _sechdule, KernelDesc)    \
    if (packed == _packed && _load_balance_cond && schedule == _sechdule) {  \
      ERROR_AND_EXIT_IF(matmul, "Already found matching kernel desc");       \
      matmul = new sop::MatMulSpecialized<KernelDesc>(                       \
         coo, b_cols, tile_config, num_threads, executor_id, mapping_id      \
      ); \
    }

  CREATE_MATMUL(false, (num_threads == 1), "KNM", sop::KD_IntelFloatKNM);
  CREATE_MATMUL(false, (num_threads == 1), "NKM", sop::KD_IntelFloatNKM);
  CREATE_MATMUL(false, (num_threads >  1), "KNM", sop::KD_IntelFloatLoadBalancedKNM);
  CREATE_MATMUL(false, (num_threads >  1), "NKM", sop::KD_IntelFloatLoadBalancedNKM);
  CREATE_MATMUL(true,  (num_threads == 1), "KNM", sop::KD_IntelFloatPackedKNM);
  // NOTE: NKM schedule or load balancing not supported for packed yet so use unpacked versions
  CREATE_MATMUL(true,  (num_threads == 1), "NKM", sop::KD_IntelFloatNKM);
  CREATE_MATMUL(true,  (num_threads >  1), "KNM", sop::KD_IntelFloatLoadBalancedKNM);
  CREATE_MATMUL(true,  (num_threads >  1), "NKM", sop::KD_IntelFloatLoadBalancedNKM);

#elif defined(__ARM_NEON)
  if (num_threads == 1) {
      int N_c = 12;
      if (b_cols >= 1024) {
        N_c = 48;
      }
      else if (b_cols >= 512) {
        N_c = 12;
      }
      else if (b_cols >= 128) {
        N_c = 96;
      }
      else {
        N_c = 48;
      }

      tile_config.M_c = coo->rows();
      tile_config.N_c = N_c;
      tile_config.K_c = (b_cols >= 1024) ? 64 : 128;
      tile_config.tiling_strategy = sop::MANUAL_TILING;

      mapping_id = "61fee";
      executor_id = "c22a5_NEON_128_4x2";
      schedule = "N";
  } else {
      int N_c = 12;
      if (b_cols >= 1024) {
        N_c = 48;
      } else if (b_cols >= 512) {
        N_c = 12;
      } else if (b_cols >= 128) {
        N_c = 96;
      } else {
        N_c = 48;
      }

      tile_config.M_c = coo->rows();
      tile_config.N_c = N_c;
      tile_config.K_c = (b_cols >= 1024) ? 64 : 128;
      tile_config.tiling_strategy = sop::MANUAL_TILING;

      mapping_id = "61fee";
      executor_id = "c22a5_NEON_128_4x2";
      schedule = "M";
  }

  if (schedule == "M") {
    matmul = new sop::MatMulSpecialized<sop::KD_PIFloatSplitM>(
      coo, b_cols, tile_config, num_threads, executor_id, mapping_id
    );
  } else {
    matmul = new sop::MatMulSpecialized<sop::KD_PIFloatSplitM>(
      coo, b_cols, tile_config, num_threads, executor_id, mapping_id
    );
  }
#endif

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
