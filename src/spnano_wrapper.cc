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

#if defined(RASPBERRY_PI) && RASPBERRY_PI
std::string resolve_path(std::string file, const std::vector<std::string>& search_dirs) {
  // Filepath construction priority
  //  1. if path is an absolute path use as is
  //  2. search of non-empty search paths
  std::string fullpath = file;
  std::vector<std::string> tested_paths;
  bool file_found = false;

  // Assume leading / means absolute path
  if (file[0] == '/') {
    file_found = std::filesystem::exists(fullpath);
  } else {
    for (const auto& search_dir : search_dirs) {
      if (search_dir.empty()) continue;
      fullpath = search_dir + "/" + file;
      tested_paths.push_back(fullpath);
      if (std::filesystem::exists(fullpath)) {
        file_found = true;
        break;
      }
    }
  }

  if (!file_found) {
    std::cerr << "Failed to resolve location of file " << file << " tested:" << std::endl;
    for (const auto& path : tested_paths)
      std::cerr << "  " << path << std::endl;
    exit(-1);
  }

  // Clean path string
  return std::filesystem::path(fullpath).lexically_normal();
}


namespace sop {
static std::string replace(
  std::string str,
  const std::string& from,
  const std::string& to
) {
  size_t start_pos = 0;
  while((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
  }
  return str;
}

static std::vector<std::string> split(std::string strToSplit, char delimeter)
{
  std::stringstream ss(strToSplit);
  std::string item;
  std::vector<std::string> splittedStrings;
  while (std::getline(ss, item, delimeter))
  {
    splittedStrings.push_back(item);
  }
  return splittedStrings;
}

std::shared_ptr<NanoKernelMapping>
read_pattern_mapping(const std::string& id, const std::vector<std::string>& search_dirs)
{
  std::string filepath = resolve_path("mapping_" + id + ".txt", search_dirs);

  std::ifstream file(filepath);
  std::string line;

  if (!file.is_open()) {
    std::cout << "Failed to open " << filepath << std::endl;
    exit(-1);
  }

  std::getline(file, line);
  int M_r = std::stoi(line);

  auto pattern_mapping_ptr = std::make_shared<NanoKernelMapping>(1 << M_r);
  auto& pattern_mapping = *pattern_mapping_ptr;

  pattern_mapping[0].push_back(0);

  while (std::getline(file, line)) {
    // Cleanup
    line = replace(line, "[", "");
    line = replace(line, "]", "");

    auto line_split = split(line, ':');
    int pattern = std::stoi(line_split[0]);
    auto nano_kernel_strings = split(line_split[1], ',');

    // replace all 'x' to 'y'
    for (const auto& nano_kernel_string : nano_kernel_strings) {
      pattern_mapping[pattern].push_back(std::stoi(nano_kernel_string));
    }
  }

  file.close();
  return pattern_mapping_ptr;
}
}

#endif

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
  sop::MatMul<float>* matmul;

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

  if (!packed) {
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
#elif defined(__ARM_NEON)
  if (num_threads >= 1) {
//      int N_c = 12;
//      if (b_cols >= 1024) {
//        N_c = 48;
//      } else if (b_cols >= 512) {
//        N_c = 12;
//      } else if (b_cols >= 128) {
//        N_c = 96;
//      } else {
//        N_c = 48;
//      }
//
//      tile_config.M_c = coo->rows();
//      tile_config.N_c = N_c;
//      tile_config.K_c = (b_cols >= 1024) ? 64 : 128;
//      tile_config.tiling_strategy = sop::MANUAL_TILING;

      tile_config.M_c = 8;
      tile_config.N_c = 8;
      tile_config.K_c = 32;
      tile_config.tiling_strategy = sop::MANUAL_TILING;

      mapping_id = "61fee";
      executor_id = "c22a5_NEON_128_4x2";
      schedule = "M";
  } else {
    std::cout << "Not yet supported" << std::endl;
    exit(-1);
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
