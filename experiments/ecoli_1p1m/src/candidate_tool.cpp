#include "candidate_indexes.hpp"
#include "reference_windows.hpp"
#include "tensor_index.hpp"

#include <cstdint>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

void print_help() {
  std::cout
      << "Usage:\n"
      << "  candidate_tool --help\n"
      << "  candidate_tool build --method contig --k N --ref PATH --window N"
         " --stride N --out-dir PATH\n"
      << "  candidate_tool query --index PATH --reads PATH --tau N --out PATH\n"
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

struct ReadRecord {
  std::string read_id;
  std::string sequence;
};

std::string trim_read_id(std::string_view header) {
  const std::size_t end = header.find_first_of(" \t\r\n");
  if (end == std::string_view::npos) {
    return std::string(header);
  }
  return std::string(header.substr(0, end));
}

void strip_trailing_carriage_return(std::string& line) {
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
}

bool contains_whitespace(std::string_view line) {
  for (unsigned char ch : line) {
    if (std::isspace(ch) != 0) {
      return true;
    }
  }
  return false;
}

std::vector<ReadRecord> read_reads(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("unable to open reads file: " + path.string());
  }

  std::vector<ReadRecord> records;
  std::string first_line;
  while (std::getline(input, first_line)) {
    strip_trailing_carriage_return(first_line);
    if (!first_line.empty()) {
      break;
    }
  }
  if (first_line.empty() && input.eof()) {
    return records;
  }

  if (first_line.front() == '@') {
    std::string header = first_line;
    while (true) {
      std::string sequence;
      std::string plus_line;
      std::string quality;
      if (!std::getline(input, sequence) || !std::getline(input, plus_line) ||
          !std::getline(input, quality)) {
        throw std::runtime_error("truncated FASTQ record in reads file");
      }
      strip_trailing_carriage_return(sequence);
      strip_trailing_carriage_return(plus_line);
      strip_trailing_carriage_return(quality);
      if (header.empty() || header.front() != '@') {
        throw std::runtime_error("invalid FASTQ header in reads file");
      }
      if (plus_line.empty() || plus_line.front() != '+') {
        throw std::runtime_error("invalid FASTQ separator line in reads file");
      }
      if (contains_whitespace(sequence)) {
        throw std::runtime_error(
            "FASTQ sequence line contains embedded whitespace in reads file");
      }
      if (sequence.size() != quality.size()) {
        throw std::runtime_error(
            "FASTQ sequence and quality lengths differ in reads file");
      }
      records.push_back({trim_read_id(header.substr(1)), sequence});
      bool found_next_header = false;
      while (std::getline(input, header)) {
        strip_trailing_carriage_return(header);
        if (!header.empty()) {
          found_next_header = true;
          break;
        }
      }
      if (!found_next_header) {
        break;
      }
      if (header.front() != '@') {
        throw std::runtime_error("invalid FASTQ header in reads file");
      }
    }
  } else if (first_line.front() == '>') {
    std::string header = first_line;
    std::string sequence;
    while (true) {
      std::string line;
      if (!std::getline(input, line)) {
        records.push_back({trim_read_id(header.substr(1)), sequence});
        break;
      }
      strip_trailing_carriage_return(line);
      if (line.empty()) {
        continue;
      }
      if (line.front() == '>') {
        records.push_back({trim_read_id(header.substr(1)), sequence});
        header = line;
        sequence.clear();
      } else {
        if (contains_whitespace(line)) {
          throw std::runtime_error(
              "FASTA sequence line contains embedded whitespace in reads file");
        }
        sequence += line;
      }
    }
  } else {
    throw std::runtime_error("reads file must be FASTA or FASTQ");
  }

  return records;
}

std::string join_window_ids(const std::vector<uint32_t>& window_ids) {
  std::string joined;
  for (std::size_t index = 0; index < window_ids.size(); ++index) {
    if (index > 0) {
      joined.push_back(',');
    }
    joined += std::to_string(window_ids[index]);
  }
  return joined;
}

int build_contiguous(int argc, char** argv) {
  ContiguousIndexConfig config;
  std::filesystem::path out_dir;
  std::string method;
  bool saw_method = false;
  bool saw_k = false;
  bool saw_ref = false;
  bool saw_window = false;
  bool saw_stride = false;
  bool saw_out_dir = false;

  for (int index = 2; index < argc; index += 2) {
    const std::string flag = argv[index];
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for flag: " + flag);
    }
    const std::string value = argv[index + 1];
    if (flag == "--method") {
      method = value;
      saw_method = true;
    } else if (flag == "--k") {
      config.k = parse_uint32(value, flag);
      saw_k = true;
    } else if (flag == "--ref") {
      config.reference_path = value;
      saw_ref = true;
    } else if (flag == "--window") {
      config.window_length = parse_uint32(value, flag);
      saw_window = true;
    } else if (flag == "--stride") {
      config.stride = parse_uint32(value, flag);
      saw_stride = true;
    } else if (flag == "--out-dir") {
      out_dir = value;
      saw_out_dir = true;
    } else {
      throw std::invalid_argument("unknown flag: " + flag);
    }
  }

  if (!saw_method || method != "contig") {
    throw std::invalid_argument("missing required flag: --method contig");
  }
  if (!saw_k || !saw_ref || !saw_window || !saw_stride || !saw_out_dir) {
    throw std::invalid_argument("missing required build flag");
  }

  const ContiguousIndex index = ContiguousIndex::build(config);
  index.save(out_dir);
  return 0;
}

int query_contiguous(int argc, char** argv) {
  std::filesystem::path index_path;
  std::filesystem::path reads_path;
  std::filesystem::path out_path;
  uint32_t tau = 0;
  bool saw_index = false;
  bool saw_reads = false;
  bool saw_tau = false;
  bool saw_out = false;

  for (int index = 2; index < argc; index += 2) {
    const std::string flag = argv[index];
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for flag: " + flag);
    }
    const std::string value = argv[index + 1];
    if (flag == "--index") {
      index_path = value;
      saw_index = true;
    } else if (flag == "--reads") {
      reads_path = value;
      saw_reads = true;
    } else if (flag == "--tau") {
      tau = parse_uint32(value, flag);
      saw_tau = true;
    } else if (flag == "--out") {
      out_path = value;
      saw_out = true;
    } else {
      throw std::invalid_argument("unknown flag: " + flag);
    }
  }

  if (!saw_index || !saw_reads || !saw_tau || !saw_out) {
    throw std::invalid_argument("missing required query flag");
  }

  const ContiguousIndex index = ContiguousIndex::load(index_path);
  const std::vector<ReadRecord> reads = read_reads(reads_path);
  if (out_path.has_parent_path()) {
    std::filesystem::create_directories(out_path.parent_path());
  }
  std::ofstream output(out_path);
  if (!output) {
    throw std::runtime_error("unable to create output file: " + out_path.string());
  }
  output << "read_id\ttau\traw_candidate_count\tcandidate_window_ids\n";
  for (const ReadRecord& read : reads) {
    const std::vector<uint32_t> candidate_window_ids = index.query(read.sequence);
    output << read.read_id << '\t' << tau << '\t' << candidate_window_ids.size()
           << '\t' << join_window_ids(candidate_window_ids) << '\n';
  }
  return 0;
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
    if (argc >= 2 && std::string(argv[1]) == "build") {
      return build_contiguous(argc, argv);
    }
    if (argc >= 2 && std::string(argv[1]) == "query") {
      return query_contiguous(argc, argv);
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
