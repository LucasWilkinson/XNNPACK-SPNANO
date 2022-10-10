#include "spnano_wrapper.h"

#include <iostream>

#include "COO.h"
#include "KernelDesc.h"
#include "MatMulSpecialized.h"

extern "C" spnano_coo_t allocate_coo_matrix_f32(int m, int n) {
  return new COO<float>(m, n);
}

extern "C" void deallocate_coo_matrix_f32(spnano_coo_t m) {
  //delete reinterpret_cast<COO<float>*>(m);
}

extern "C" void coo_matrix_add_nnz_f32(spnano_coo_t m, int i, int j, float val) {
  COO<float>* coo = reinterpret_cast<COO<float>*>(m);
  coo->append_nnz({i, j, val});
}

extern "C" spnano_matmul_t allocate_matmul_f32(spnano_coo_t m, int num_threads, int b_cols) {
  std::string mapping_id = "";
  std::string executor_id = "";
  std::string schedule = "KNM";
  bool packed = false;

#ifdef ENABLE_AVX512
  //packed = (b_cols % 16) != 0;
#endif

  COO<float>* coo = reinterpret_cast<COO<float>*>(m);

  //std::cout << "BCols " << b_cols << " packed " << packed << std::endl;

  if (num_threads >= 1) {
#ifdef __AVX512VL__
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
#else
#endif
  } else {
    std::cout << "Not yet supported" << std::endl;
    exit(-1);
  }

  sop::TileConfig tile_config;
  sop::MatMul<float>* matmul;

  if (packed) {
    if (num_threads == 1) {
      if (schedule == "KNM") {
        matmul = new sop::MatMulSpecialized<sop::KD_IntelFloatKNM>(
          coo, b_cols, tile_config, num_threads, executor_id, mapping_id
        );
      }
      else if (schedule == "NKM") {
        matmul = new sop::MatMulSpecialized<sop::KD_IntelFloatNKM>(
          coo, b_cols, tile_config, num_threads, executor_id, mapping_id
        );
      }
    } else {
      if (schedule == "KNM") {
        matmul = new sop::MatMulSpecialized<sop::KD_IntelFloatLoadBalancedKNM>(
          coo, b_cols, tile_config, num_threads, executor_id, mapping_id
        );
      }
      else if (schedule == "NKM") {
        matmul = new sop::MatMulSpecialized<sop::KD_IntelFloatLoadBalancedNKM>(
          coo, b_cols, tile_config, num_threads, executor_id, mapping_id
        );
      }
    }
  } else {
    if (num_threads == 1) {
      if (schedule == "KNM") {
        matmul = new sop::MatMulSpecialized<sop::KD_IntelFloatPackedKNM>(
          coo, b_cols, tile_config, num_threads, executor_id, mapping_id
        );
      }
      else if (schedule == "NKM") {
        matmul = new sop::MatMulSpecialized<sop::KD_IntelFloatNKM>(
          coo, b_cols, tile_config, num_threads, executor_id, mapping_id
        );
      }
    } else {
      if (schedule == "KNM") {
        matmul = new sop::MatMulSpecialized<sop::KD_IntelFloatLoadBalancedKNM>(
          coo, b_cols, tile_config, num_threads, executor_id, mapping_id
        );
      }
      else if (schedule == "NKM") {
         matmul = new sop::MatMulSpecialized<sop::KD_IntelFloatLoadBalancedNKM>(
          coo, b_cols, tile_config, num_threads, executor_id, mapping_id
        );
      }
    }
  }

  matmul->allocate_executor(b_cols);
  return matmul;
}

extern "C" spnano_executor_t get_executor_f32(spnano_matmul_t m) {
  auto matmul = reinterpret_cast<sop::MatMul<float>*>(m);
  return (void*) matmul->get_executor();
}

extern "C" int get_num_parallel_tiles_f32(spnano_executor_t e) {
  auto executor = reinterpret_cast<sop::Executor<float>*>(e);
  return executor->num_parallel_tile();
}

extern "C" void begin_threaded_f32(spnano_executor_t e, float* output, const float* input, const float* bias, float min, float max) {
  auto executor = reinterpret_cast<sop::Executor<float>*>(e);
  executor->begin_threaded(output, input, bias, sop::MINMAX, min, max);
}

extern "C" void spnano_run_thread(spnano_executor_t e, int p, int tid) {
  auto executor = reinterpret_cast<sop::Executor<float>*>(e);
  executor->execute_thread(p, tid);
}