#include "reference_windows.hpp"
#include "tensor_index.hpp"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void print_help() {
  std::cout
      << "Usage:\n"
      << "  candidate_tool --help\n"
      << "  candidate_tool inspect-reference --ref PATH --window N --stride N\n"
      << "  candidate_tool tensor-build --ref PATH --window N --stride N"
         " --dimension N --seed N --hnsw-m N --hnsw-ef-construction N"
         " --hnsw-ef-search N --out-dir PATH [--exact-vectors 0|1]\n"
      << "  candidate_tool tensor-query --index-dir PATH --query DNA"
         " [--top-k N]\n";
}

uint32_t parse_uint32(const std::string& value, const std::string& flag) {
  std::size_t parsed = 0;
  unsigned long long number = 0;
  try {
    number = std::stoull(value, &parsed);
  } catch (const std::exception&) {
    throw std::invalid_argument("invalid value for " + flag + ": " + value);
  }
  if (parsed != value.size() || number > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("invalid value for " + flag + ": " + value);
  }
  return static_cast<uint32_t>(number);
}

bool parse_bool_flag(const std::string& value, const std::string& flag) {
  if (value == "1" || value == "true") {
    return true;
  }
  if (value == "0" || value == "false") {
    return false;
  }
  throw std::invalid_argument("invalid value for " + flag + ": " + value);
}

std::vector<int> encode_query(std::string_view sequence) {
  std::vector<int> encoded;
  encoded.reserve(sequence.size());
  for (char base : sequence) {
    switch (base) {
      case 'A':
      case 'a':
        encoded.push_back(0);
        break;
      case 'C':
      case 'c':
        encoded.push_back(1);
        break;
      case 'G':
      case 'g':
        encoded.push_back(2);
        break;
      case 'T':
      case 't':
        encoded.push_back(3);
        break;
      default:
        throw std::invalid_argument("query contains non-ACGT base");
    }
  }
  return encoded;
}

int inspect_reference(int argc, char** argv) {
  std::string reference_path;
  uint32_t window_length = 0;
  uint32_t stride = 0;
  bool saw_window = false;
  bool saw_stride = false;

  for (int index = 2; index < argc; index += 2) {
    const std::string flag = argv[index];
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for flag: " + flag);
    }
    const std::string value = argv[index + 1];
    if (flag == "--ref") {
      if (!reference_path.empty()) {
        throw std::invalid_argument("duplicate flag: --ref");
      }
      reference_path = value;
    } else if (flag == "--window") {
      if (saw_window) {
        throw std::invalid_argument("duplicate flag: --window");
      }
      window_length = parse_uint32(value, flag);
      saw_window = true;
    } else if (flag == "--stride") {
      if (saw_stride) {
        throw std::invalid_argument("duplicate flag: --stride");
      }
      stride = parse_uint32(value, flag);
      saw_stride = true;
    } else {
      throw std::invalid_argument("unknown flag: " + flag);
    }
  }

  if (reference_path.empty()) {
    throw std::invalid_argument("missing required flag: --ref");
  }
  if (!saw_window) {
    throw std::invalid_argument("missing required flag: --window");
  }
  if (!saw_stride) {
    throw std::invalid_argument("missing required flag: --stride");
  }

  const ReferenceWindows reference = ReferenceWindows::from_fasta(
      reference_path, window_length, stride);
  std::cout << "contig_id\treference_length\twindow_length\tstride\t"
               "number_of_windows\n"
            << reference.contig_id() << '\t' << reference.sequence().size()
            << '\t' << window_length << '\t' << stride << '\t'
            << reference.size() << '\n';
  return 0;
}

int tensor_build(int argc, char** argv) {
  tensor_index::TensorIndexConfig config;
  config.exact_vectors = true;
  std::filesystem::path out_dir;
  bool saw_ref = false;
  bool saw_window = false;
  bool saw_stride = false;
  bool saw_dimension = false;
  bool saw_seed = false;
  bool saw_hnsw_m = false;
  bool saw_hnsw_ef_construction = false;
  bool saw_hnsw_ef_search = false;
  bool saw_out_dir = false;

  for (int index = 2; index < argc; index += 2) {
    const std::string flag = argv[index];
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for flag: " + flag);
    }
    const std::string value = argv[index + 1];
    if (flag == "--ref") {
      config.reference_path = value;
      saw_ref = true;
    } else if (flag == "--window") {
      config.window_length = parse_uint32(value, flag);
      saw_window = true;
    } else if (flag == "--stride") {
      config.stride = parse_uint32(value, flag);
      saw_stride = true;
    } else if (flag == "--dimension") {
      config.dimension = parse_uint32(value, flag);
      saw_dimension = true;
    } else if (flag == "--seed") {
      config.seed = parse_uint32(value, flag);
      saw_seed = true;
    } else if (flag == "--hnsw-m") {
      config.hnsw_M = parse_uint32(value, flag);
      saw_hnsw_m = true;
    } else if (flag == "--hnsw-ef-construction") {
      config.hnsw_ef_construction = parse_uint32(value, flag);
      saw_hnsw_ef_construction = true;
    } else if (flag == "--hnsw-ef-search") {
      config.hnsw_ef_search = parse_uint32(value, flag);
      saw_hnsw_ef_search = true;
    } else if (flag == "--out-dir") {
      out_dir = value;
      saw_out_dir = true;
    } else if (flag == "--exact-vectors") {
      config.exact_vectors = parse_bool_flag(value, flag);
    } else {
      throw std::invalid_argument("unknown flag: " + flag);
    }
  }

  if (!saw_ref || !saw_window || !saw_stride || !saw_dimension || !saw_seed ||
      !saw_hnsw_m || !saw_hnsw_ef_construction || !saw_hnsw_ef_search ||
      !saw_out_dir) {
    throw std::invalid_argument("missing required tensor-build flag");
  }

  tensor_index::TensorIndex index = tensor_index::build_tensor_index(config);
  tensor_index::save_tensor_index(index, out_dir);
  return 0;
}

int tensor_query(int argc, char** argv) {
  std::filesystem::path index_dir;
  std::string query;
  std::size_t top_k = 10;
  bool saw_index_dir = false;
  bool saw_query = false;

  for (int index = 2; index < argc; index += 2) {
    const std::string flag = argv[index];
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for flag: " + flag);
    }
    const std::string value = argv[index + 1];
    if (flag == "--index-dir") {
      index_dir = value;
      saw_index_dir = true;
    } else if (flag == "--query") {
      query = value;
      saw_query = true;
    } else if (flag == "--top-k") {
      top_k = parse_uint32(value, flag);
    } else {
      throw std::invalid_argument("unknown flag: " + flag);
    }
  }

  if (!saw_index_dir || !saw_query) {
    throw std::invalid_argument("missing required tensor-query flag");
  }

  tensor_index::TensorIndex index = tensor_index::load_tensor_index(index_dir);
  const std::vector<tensor_index::QueryHit> hits =
      tensor_index::query_tensor_index(index, encode_query(query), top_k);
  std::cout << "label\tdistance\n";
  for (const auto& hit : hits) {
    std::cout << hit.label << '\t' << hit.distance << '\n';
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      print_help();
      return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "inspect-reference") {
      return inspect_reference(argc, argv);
    }
    if (argc >= 2 && std::string(argv[1]) == "tensor-build") {
      return tensor_build(argc, argv);
    }
    if (argc >= 2 && std::string(argv[1]) == "tensor-query") {
      return tensor_query(argc, argv);
    }
    if (argc < 2) {
      throw std::invalid_argument("missing command");
    }
    throw std::invalid_argument("unknown command: " + std::string(argv[1]));
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
