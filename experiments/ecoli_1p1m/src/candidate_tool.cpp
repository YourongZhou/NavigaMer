#include "candidate_indexes.hpp"
#include "evaluation.hpp"
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
#include <type_traits>
#include <variant>
#include <vector>

namespace {

void print_help() {
  std::cout
      << "Usage:\n"
      << "  candidate_tool --help\n"
      << "  candidate_tool build --method contig --k N --ref PATH --window N"
         " --stride N --out-dir PATH\n"
      << "  candidate_tool build --method spaced --weight W --ref PATH"
         " --window N --stride N --out-dir PATH\n"
      << "  candidate_tool build --method randstrobe --strobe-len 15"
         " --w-min 20 --w-max 50 --seed N --ref PATH --window N --stride N"
         " --out-dir PATH\n"
      << "  candidate_tool build --method qgram-safe --q N --ref PATH"
         " --window N --stride N --out-dir PATH\n"
      << "  candidate_tool build --method pigeonhole --tau N"
         " --nominal-read-length N --ref PATH --window N --stride N"
         " --out-dir PATH\n"
      << "  candidate_tool build-matrix --ref PATH --window N --stride N"
         " --out-dir PATH [--rebuild]\n"
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

uint64_t parse_uint64(const std::string& value, const std::string& flag) {
  std::size_t parsed = 0;
  unsigned long long number = 0;
  try {
    number = std::stoull(value, &parsed);
  } catch (const std::exception&) {
    throw std::invalid_argument("invalid value for " + flag + ": " + value);
  }
  if (parsed != value.size()) {
    throw std::invalid_argument("invalid value for " + flag + ": " + value);
  }
  return static_cast<uint64_t>(number);
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

bool is_valid_dna_sequence_line(std::string_view line) {
  for (char base : line) {
    switch (base) {
      case 'A':
      case 'a':
      case 'C':
      case 'c':
      case 'G':
      case 'g':
      case 'T':
      case 't':
      case 'N':
      case 'n':
        break;
      default:
        return false;
    }
  }
  return true;
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
      std::string line;
      bool found_separator = false;
      while (std::getline(input, line)) {
        strip_trailing_carriage_return(line);
        if (line.empty()) {
          throw std::runtime_error("truncated FASTQ record in reads file");
        }
        if (line.front() == '+') {
          found_separator = true;
          break;
        }
        if (contains_whitespace(line)) {
          throw std::runtime_error(
              "FASTQ sequence line contains embedded whitespace in reads file");
        }
        if (!is_valid_dna_sequence_line(line)) {
          throw std::runtime_error(
              "FASTQ sequence line contains invalid DNA base in reads file");
        }
        sequence += line;
      }
      if (!found_separator) {
        throw std::runtime_error("truncated FASTQ record in reads file");
      }

      std::string quality;
      if (header.empty() || header.front() != '@') {
        throw std::runtime_error("invalid FASTQ header in reads file");
      }
      while (quality.size() < sequence.size()) {
        if (!std::getline(input, line)) {
          throw std::runtime_error(
              "FASTQ sequence and quality lengths differ in reads file");
        }
        strip_trailing_carriage_return(line);
        if (line.empty()) {
          throw std::runtime_error(
              "FASTQ sequence and quality lengths differ in reads file");
        }
        quality += line;
        if (quality.size() > sequence.size()) {
          throw std::runtime_error(
              "FASTQ sequence and quality lengths differ in reads file");
        }
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
        if (!is_valid_dna_sequence_line(line)) {
          throw std::runtime_error(
              "FASTA sequence line contains invalid DNA base in reads file");
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

using CandidateIndex = std::variant<ContiguousIndex, SpacedSeedIndex,
                                    RandstrobeIndex, QgramSafeIndex,
                                    PigeonholeIndex>;

CandidateIndex load_candidate_index(const std::filesystem::path& index_path) {
  const PersistedIndex loaded = read_index_file(index_path);
  if (loaded.manifest.method == "contig") {
    return ContiguousIndex::load(loaded);
  }
  if (loaded.manifest.method == "spaced") {
    return SpacedSeedIndex::load(loaded);
  }
  if (loaded.manifest.method == "randstrobe") {
    return RandstrobeIndex::load(loaded);
  }
  if (loaded.manifest.method == "qgram-safe") {
    return QgramSafeIndex::load(loaded);
  }
  if (loaded.manifest.method == "pigeonhole") {
    return PigeonholeIndex::load(loaded);
  }
  throw std::runtime_error("unsupported candidate index method: " +
                           loaded.manifest.method);
}

std::vector<uint32_t> query_candidate_index(const CandidateIndex& index,
                                            std::string_view query_sequence,
                                            uint32_t tau) {
  return std::visit(
      [&](const auto& loaded_index) {
        if constexpr (std::is_same_v<std::decay_t<decltype(loaded_index)>,
                                     QgramSafeIndex>) {
          return loaded_index.query(query_sequence, tau);
        } else if constexpr (std::is_same_v<std::decay_t<decltype(loaded_index)>,
                                            PigeonholeIndex>) {
          return loaded_index.query(query_sequence, tau);
        } else {
          (void)tau;
          return loaded_index.query(query_sequence);
        }
      },
      index);
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

int build_spaced(int argc, char** argv) {
  SpacedSeedIndexConfig config;
  std::filesystem::path out_dir;
  std::string method;
  bool saw_method = false;
  bool saw_weight = false;
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
    } else if (flag == "--weight") {
      config.weight = parse_uint32(value, flag);
      saw_weight = true;
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

  if (!saw_method || method != "spaced") {
    throw std::invalid_argument("missing required flag: --method spaced");
  }
  if (!saw_weight || !saw_ref || !saw_window || !saw_stride || !saw_out_dir) {
    throw std::invalid_argument("missing required build flag");
  }

  const SpacedSeedIndex index = SpacedSeedIndex::build(config);
  index.save(out_dir);
  return 0;
}

int build_randstrobe(int argc, char** argv) {
  RandstrobeIndexConfig config;
  std::filesystem::path out_dir;
  std::string method;
  bool saw_method = false;
  bool saw_strobe_len = false;
  bool saw_w_min = false;
  bool saw_w_max = false;
  bool saw_seed = false;
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
    } else if (flag == "--strobe-len") {
      config.strobe_length = parse_uint32(value, flag);
      saw_strobe_len = true;
    } else if (flag == "--w-min") {
      config.w_min = parse_uint32(value, flag);
      saw_w_min = true;
    } else if (flag == "--w-max") {
      config.w_max = parse_uint32(value, flag);
      saw_w_max = true;
    } else if (flag == "--seed") {
      config.seed = parse_uint64(value, flag);
      saw_seed = true;
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

  if (!saw_method || method != "randstrobe") {
    throw std::invalid_argument("missing required flag: --method randstrobe");
  }
  if (!saw_strobe_len || !saw_w_min || !saw_w_max || !saw_seed || !saw_ref ||
      !saw_window || !saw_stride || !saw_out_dir) {
    throw std::invalid_argument("missing required build flag");
  }

  const RandstrobeIndex index = RandstrobeIndex::build(config);
  index.save(out_dir);
  return 0;
}

int build_qgram_safe(int argc, char** argv) {
  QgramSafeIndexConfig config;
  std::filesystem::path out_dir;
  std::string method;
  bool saw_method = false;
  bool saw_q = false;
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
    } else if (flag == "--q") {
      config.q = parse_uint32(value, flag);
      saw_q = true;
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

  if (!saw_method || method != "qgram-safe") {
    throw std::invalid_argument("missing required flag: --method qgram-safe");
  }
  if (!saw_q || !saw_ref || !saw_window || !saw_stride || !saw_out_dir) {
    throw std::invalid_argument("missing required build flag");
  }

  const QgramSafeIndex index = QgramSafeIndex::build(config);
  index.save(out_dir);
  return 0;
}

int build_pigeonhole(int argc, char** argv) {
  PigeonholeIndexConfig config;
  std::filesystem::path out_dir;
  std::string method;
  bool saw_method = false;
  bool saw_tau = false;
  bool saw_nominal_read_length = false;
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
    } else if (flag == "--tau") {
      config.tau = parse_uint32(value, flag);
      saw_tau = true;
    } else if (flag == "--nominal-read-length") {
      config.nominal_read_length = parse_uint32(value, flag);
      saw_nominal_read_length = true;
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

  if (!saw_method || method != "pigeonhole") {
    throw std::invalid_argument(
        "missing required flag: --method pigeonhole");
  }
  if (!saw_tau || !saw_nominal_read_length || !saw_ref || !saw_window ||
      !saw_stride || !saw_out_dir) {
    throw std::invalid_argument("missing required build flag");
  }

  const PigeonholeIndex index = PigeonholeIndex::build(config);
  index.save(out_dir);
  return 0;
}

std::string build_method_from_argv(int argc, char** argv) {
  for (int index = 2; index < argc; ++index) {
    if (std::string(argv[index]) != "--method") {
      continue;
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for flag: --method");
    }
    return argv[index + 1];
  }
  throw std::invalid_argument("missing required flag: --method");
}

int build_matrix(int argc, char** argv) {
  BuildMatrixRequest request;
  bool saw_ref = false;
  bool saw_window = false;
  bool saw_stride = false;
  bool saw_out_dir = false;

  for (int index = 2; index < argc; ++index) {
    const std::string flag = argv[index];
    if (flag == "--rebuild") {
      request.rebuild = true;
      continue;
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for flag: " + flag);
    }
    const std::string value = argv[++index];
    if (flag == "--ref") {
      request.reference_path = value;
      saw_ref = true;
    } else if (flag == "--window") {
      request.window_length = parse_uint32(value, flag);
      saw_window = true;
    } else if (flag == "--stride") {
      request.stride = parse_uint32(value, flag);
      saw_stride = true;
    } else if (flag == "--out-dir") {
      request.out_dir = value;
      saw_out_dir = true;
    } else {
      throw std::invalid_argument("unknown flag: " + flag);
    }
  }

  if (!saw_ref || !saw_window || !saw_stride || !saw_out_dir) {
    throw std::invalid_argument("missing required build-matrix flag");
  }

  const std::vector<BuildSummaryRow> rows = build_candidate_matrix(request);
  (void)rows;
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

  const std::vector<ReadRecord> reads = read_reads(reads_path);
  const CandidateIndex candidate_index = load_candidate_index(index_path);
  if (out_path.has_parent_path()) {
    std::filesystem::create_directories(out_path.parent_path());
  }
  std::ofstream output(out_path);
  if (!output) {
    throw std::runtime_error("unable to create output file: " + out_path.string());
  }
  output << "read_id\ttau\traw_candidate_count\tcandidate_window_ids\n";
  for (const ReadRecord& read : reads) {
    const std::vector<uint32_t> candidate_window_ids =
        query_candidate_index(candidate_index, read.sequence, tau);
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
    if (argc >= 2 && std::string(argv[1]) == "build-matrix") {
      return build_matrix(argc, argv);
    }
    if (argc >= 2 && std::string(argv[1]) == "build") {
      const std::string method = build_method_from_argv(argc, argv);
      if (method == "contig") {
        return build_contiguous(argc, argv);
      }
      if (method == "spaced") {
        return build_spaced(argc, argv);
      }
      if (method == "randstrobe") {
        return build_randstrobe(argc, argv);
      }
      if (method == "qgram-safe") {
        return build_qgram_safe(argc, argv);
      }
      if (method == "pigeonhole") {
        return build_pigeonhole(argc, argv);
      }
      throw std::invalid_argument("unknown build method: " + method);
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
