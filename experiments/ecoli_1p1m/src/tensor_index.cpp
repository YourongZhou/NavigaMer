#include "tensor_index.hpp"

#include "reference_windows.hpp"
#include "sha256.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iterator>
#include <memory>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "hnswlib/hnswlib.h"
#include "sketch/tensor.hpp"
#include "sketch/tensor_slide.hpp"

namespace tensor_index {
namespace {

constexpr std::array<char, 8> kExactMagic = {'T', 'I', 'D', 'X', 'E', 'X', 'A', 'C'};
constexpr uint32_t kExactPayloadVersion = 1;
constexpr uint32_t kTensorSubsequenceLength = 5;
constexpr uint32_t kTopKSearchCap = 10000;

struct ParsedMeta {
  std::filesystem::path reference_path;
  std::string reference_sha256;
  uint64_t reference_length = 0;
  uint32_t window_length = 0;
  uint32_t stride = 0;
  uint64_t number_of_windows = 0;
  uint32_t dimension = 0;
  uint32_t seed = 0;
  uint32_t hnsw_M = 0;
  uint32_t hnsw_ef_construction = 0;
  uint32_t hnsw_ef_search = 0;
  bool exact_vectors = true;
  IndexManifest manifest;
};

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("unable to open file: " + path.string());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  if (!input && !input.eof()) {
    throw std::runtime_error("unable to read file: " + path.string());
  }
  return buffer.str();
}

std::string sha256_hex_of_file(const std::filesystem::path& path) {
  return sha256_hex(read_file(path));
}

std::string current_utc_timestamp() {
  std::time_t now = std::time(nullptr);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &now);
#else
  gmtime_r(&now, &tm);
#endif
  char buffer[32] = {};
  if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &tm) == 0) {
    throw std::runtime_error("unable to format timestamp");
  }
  return buffer;
}

std::string shell_quote(const std::string& value) {
  std::string quoted = "'";
  for (char character : value) {
    if (character == '\'') {
      quoted += "'\\''";
    } else {
      quoted += character;
    }
  }
  quoted += "'";
  return quoted;
}

std::string git_commit_for_root(const std::filesystem::path& root) {
  const std::string command = "git -C " + shell_quote(root.string()) +
                              " rev-parse --short=12 HEAD 2>/dev/null";
  std::array<char, 128> buffer{};
  std::string output;
  if (FILE* pipe = popen(command.c_str(), "r")) {
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) !=
           nullptr) {
      output += buffer.data();
    }
    const int status = pclose(pipe);
    if (status != 0) {
      return {};
    }
  }
  while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) {
    output.pop_back();
  }
  return output;
}

std::filesystem::path tensor_sketch_root() {
  return std::filesystem::path(NAVIGAMER_TENSOR_SKETCH_ROOT);
}

std::filesystem::path navigamer_repo_root() {
  return std::filesystem::path(NAVIGAMER_REPO_ROOT);
}

std::vector<int> encode_dna(std::string_view sequence) {
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
        throw std::runtime_error("unexpected DNA base");
    }
  }
  return encoded;
}

template <typename T>
T parse_unsigned(const std::string& value, const char* field) {
  std::size_t parsed = 0;
  unsigned long long number = 0;
  try {
    number = std::stoull(value, &parsed);
  } catch (const std::exception&) {
    throw std::runtime_error(std::string("invalid ") + field + " value: " + value);
  }
  if (parsed != value.size() || number > std::numeric_limits<T>::max()) {
    throw std::runtime_error(std::string("invalid ") + field + " value: " + value);
  }
  return static_cast<T>(number);
}

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8) {
    bytes.push_back(static_cast<uint8_t>((value >> shift) & 0xffU));
  }
}

void append_u64(std::vector<uint8_t>& bytes, uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8) {
    bytes.push_back(static_cast<uint8_t>((value >> shift) & 0xffU));
  }
}

void append_float(std::vector<uint8_t>& bytes, float value) {
  static_assert(sizeof(float) == sizeof(uint32_t), "requires 32-bit float");
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  append_u32(bytes, bits);
}

uint32_t read_u32(const std::vector<uint8_t>& bytes, std::size_t& offset) {
  if (offset + sizeof(uint32_t) > bytes.size()) {
    throw std::runtime_error("truncated exact payload");
  }
  uint32_t value = 0;
  for (unsigned shift = 0; shift < 32; shift += 8) {
    value |= static_cast<uint32_t>(bytes[offset++]) << shift;
  }
  return value;
}

uint64_t read_u64(const std::vector<uint8_t>& bytes, std::size_t& offset) {
  if (offset + sizeof(uint64_t) > bytes.size()) {
    throw std::runtime_error("truncated exact payload");
  }
  uint64_t value = 0;
  for (unsigned shift = 0; shift < 64; shift += 8) {
    value |= static_cast<uint64_t>(bytes[offset++]) << shift;
  }
  return value;
}

float read_float(const std::vector<uint8_t>& bytes, std::size_t& offset) {
  const uint32_t bits = read_u32(bytes, offset);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::vector<uint8_t> serialize_exact_payload(
    const std::vector<uint32_t>& labels,
    const std::vector<float>& exact_vectors,
    uint32_t dimension) {
  if (labels.size() * static_cast<std::size_t>(dimension) != exact_vectors.size()) {
    throw std::runtime_error("exact vector matrix is ragged");
  }
  std::vector<uint8_t> payload;
  payload.reserve(8 + 4 + 8 + 4 + labels.size() * (4 + dimension * 4));
  payload.insert(payload.end(), kExactMagic.begin(), kExactMagic.end());
  append_u32(payload, kExactPayloadVersion);
  append_u64(payload, labels.size());
  append_u32(payload, dimension);
  for (std::size_t row = 0; row < labels.size(); ++row) {
    append_u32(payload, labels[row]);
    const std::size_t offset = row * dimension;
    for (uint32_t column = 0; column < dimension; ++column) {
      append_float(payload, exact_vectors[offset + column]);
    }
  }
  return payload;
}

IndexManifest manifest_from_snapshot(const TensorIndexSnapshot& snapshot) {
  IndexManifest manifest = snapshot.manifest;
  manifest.method = "ts::Tensor";
  manifest.parameters = {
      {"algorithm", "ts::Tensor"},
      {"subsequence_length", std::to_string(kTensorSubsequenceLength)},
      {"dimension", std::to_string(snapshot.dimension)},
      {"seed", std::to_string(snapshot.seed)},
      {"metric", "L2"},
      {"hnsw_M", std::to_string(snapshot.hnsw_M)},
      {"hnsw_ef_construction", std::to_string(snapshot.hnsw_ef_construction)},
      {"hnsw_ef_search", std::to_string(snapshot.hnsw_ef_search)},
      {"dependency_source_path", tensor_sketch_root().string()},
      {"dependency_git_commit", git_commit_for_root(tensor_sketch_root())},
  };
  manifest.window_length = snapshot.manifest.window_length;
  manifest.stride = snapshot.manifest.stride;
  manifest.number_of_windows = snapshot.manifest.number_of_windows;
  manifest.format_version = 1;
  manifest.tool_version = "tensor-index/1";
  return manifest;
}

void write_meta_file(const std::filesystem::path& path,
                     const TensorIndexSnapshot& snapshot) {
  std::ofstream output(path, std::ios::trunc);
  if (!output) {
    throw std::runtime_error("unable to create tensor index metadata: " +
                             path.string());
  }
  const IndexManifest manifest = manifest_from_snapshot(snapshot);
  output << "method\t" << manifest.method << '\n';
  output << "reference_path\t" << manifest.reference_path << '\n';
  output << "reference_sha256\t" << manifest.reference_sha256 << '\n';
  output << "reference_length\t" << manifest.reference_length << '\n';
  output << "window_length\t" << manifest.window_length << '\n';
  output << "stride\t" << manifest.stride << '\n';
  output << "number_of_windows\t" << manifest.number_of_windows << '\n';
  output << "format_version\t" << manifest.format_version << '\n';
  output << "tool_version\t" << manifest.tool_version << '\n';
  output << "build_command\t" << manifest.build_command << '\n';
  output << "build_seconds\t" << manifest.build_seconds << '\n';
  output << "index_bytes\t" << manifest.index_bytes << '\n';
  output << "created_at\t" << manifest.created_at << '\n';
  output << "git_commit\t" << manifest.git_commit << '\n';
  output << "dimension\t" << snapshot.dimension << '\n';
  output << "seed\t" << snapshot.seed << '\n';
  output << "hnsw_M\t" << snapshot.hnsw_M << '\n';
  output << "hnsw_ef_construction\t" << snapshot.hnsw_ef_construction << '\n';
  output << "hnsw_ef_search\t" << snapshot.hnsw_ef_search << '\n';
  output << "exact_vectors\t" << (snapshot.persist_exact_vectors ? "true" : "false")
         << '\n';
  for (const auto& parameter : manifest.parameters) {
    output << "parameter\t" << parameter.first << '\t' << parameter.second << '\n';
  }
}

ParsedMeta parse_meta_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("unable to open tensor index metadata: " +
                             path.string());
  }

  ParsedMeta parsed;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    const std::size_t first_tab = line.find('\t');
    if (first_tab == std::string::npos) {
      throw std::runtime_error("invalid tensor index metadata line");
    }
    const std::string key = line.substr(0, first_tab);
    const std::string value = line.substr(first_tab + 1);
    if (key == "method") {
      parsed.manifest.method = value;
    } else if (key == "reference_path") {
      parsed.reference_path = value;
      parsed.manifest.reference_path = value;
    } else if (key == "reference_sha256") {
      parsed.reference_sha256 = value;
      parsed.manifest.reference_sha256 = value;
    } else if (key == "reference_length") {
      parsed.reference_length = std::stoull(value);
      parsed.manifest.reference_length = parsed.reference_length;
    } else if (key == "window_length") {
      parsed.window_length = std::stoul(value);
      parsed.manifest.window_length = parsed.window_length;
    } else if (key == "stride") {
      parsed.stride = std::stoul(value);
      parsed.manifest.stride = parsed.stride;
    } else if (key == "number_of_windows") {
      parsed.number_of_windows = std::stoull(value);
      parsed.manifest.number_of_windows = parsed.number_of_windows;
    } else if (key == "format_version") {
      parsed.manifest.format_version = std::stoul(value);
    } else if (key == "tool_version") {
      parsed.manifest.tool_version = value;
    } else if (key == "build_command") {
      parsed.manifest.build_command = value;
    } else if (key == "build_seconds") {
      parsed.manifest.build_seconds = std::stod(value);
    } else if (key == "index_bytes") {
      parsed.manifest.index_bytes = std::stoull(value);
    } else if (key == "created_at") {
      parsed.manifest.created_at = value;
    } else if (key == "git_commit") {
      parsed.manifest.git_commit = value;
    } else if (key == "dimension") {
      parsed.dimension = std::stoul(value);
    } else if (key == "seed") {
      parsed.seed = std::stoul(value);
    } else if (key == "hnsw_M") {
      parsed.hnsw_M = std::stoul(value);
    } else if (key == "hnsw_ef_construction") {
      parsed.hnsw_ef_construction = std::stoul(value);
    } else if (key == "hnsw_ef_search") {
      parsed.hnsw_ef_search = std::stoul(value);
    } else if (key == "exact_vectors") {
      parsed.exact_vectors = (value == "1" || value == "true");
    } else if (key == "parameter") {
      const std::size_t second_tab = value.find('\t');
      if (second_tab == std::string::npos) {
        throw std::runtime_error("invalid parameter metadata line");
      }
      parsed.manifest.parameters.emplace_back(value.substr(0, second_tab),
                                              value.substr(second_tab + 1));
    }
  }

  if (parsed.manifest.method.empty() || parsed.reference_sha256.empty() ||
      parsed.manifest.tool_version.empty() || parsed.window_length == 0 ||
      parsed.stride == 0 || parsed.number_of_windows == 0 ||
      parsed.dimension == 0) {
    throw std::runtime_error("tensor index metadata is incomplete");
  }

  return parsed;
}

std::vector<float> tensor_to_float_vector(const std::vector<double>& sketch) {
  std::vector<float> result;
  result.reserve(sketch.size());
  for (double value : sketch) {
    result.push_back(static_cast<float>(value));
  }
  return result;
}

}  // namespace

TensorIndex build_tensor_index(const TensorIndexConfig& config) {
  if (config.window_length == 0 || config.stride == 0 || config.dimension == 0) {
    throw std::invalid_argument("tensor index dimensions must be positive");
  }

  const ReferenceWindows reference =
      ReferenceWindows::from_fasta(config.reference_path.string(),
                                   config.window_length, config.stride);
  const std::string sequence = reference.sequence();
  const std::vector<int> encoded = encode_dna(sequence);

  TensorIndex index;
  index.snapshot.dimension = config.dimension;
  index.snapshot.seed = config.seed;
  index.snapshot.hnsw_M = config.hnsw_M;
  index.snapshot.hnsw_ef_construction = config.hnsw_ef_construction;
  index.snapshot.hnsw_ef_search = config.hnsw_ef_search;
  index.snapshot.persist_exact_vectors = config.exact_vectors;
  index.snapshot.manifest.reference_path = config.reference_path.string();
  index.snapshot.manifest.reference_sha256 = sha256_hex_of_file(config.reference_path);
  index.snapshot.manifest.reference_length = sequence.size();
  index.snapshot.manifest.window_length = config.window_length;
  index.snapshot.manifest.stride = config.stride;
  index.snapshot.manifest.number_of_windows = reference.size();
  index.snapshot.manifest.build_command = "candidate_tool tensor-build";
  index.snapshot.manifest.build_seconds = 0.0;
  index.snapshot.manifest.index_bytes = 0;
  index.snapshot.manifest.created_at = current_utc_timestamp();
  index.snapshot.manifest.git_commit = git_commit_for_root(navigamer_repo_root());

  index.snapshot.labels.reserve(reference.size());
  index.snapshot.exact_vectors.reserve(
      static_cast<std::size_t>(reference.size()) * config.dimension);

  if (config.stride == 1) {
    ts::TensorSlide<int> slide(4, config.dimension, kTensorSubsequenceLength,
                               config.window_length, 1, config.seed);
    const std::vector<std::vector<double>> sketches = slide.compute(encoded);
    if (sketches.size() != reference.size()) {
      throw std::runtime_error("TensorSlide row count does not match windows");
    }
    for (uint32_t window_id = 0; window_id < reference.size(); ++window_id) {
      index.snapshot.labels.push_back(window_id);
      const std::vector<float> row = tensor_to_float_vector(sketches[window_id]);
      if (row.size() != config.dimension) {
        throw std::runtime_error("TensorSlide dimension mismatch");
      }
      index.snapshot.exact_vectors.insert(index.snapshot.exact_vectors.end(),
                                          row.begin(), row.end());
    }
  } else {
    ts::Tensor<int> tensor(4, config.dimension, kTensorSubsequenceLength,
                           config.seed);
    for (uint32_t window_id = 0; window_id < reference.size(); ++window_id) {
      const std::vector<int> window = encode_dna(reference.window(window_id));
      const std::vector<float> row = tensor_to_float_vector(tensor.compute(window));
      if (row.size() != config.dimension) {
        throw std::runtime_error("Tensor dimension mismatch");
      }
      index.snapshot.labels.push_back(window_id);
      index.snapshot.exact_vectors.insert(index.snapshot.exact_vectors.end(),
                                          row.begin(), row.end());
    }
  }

  index.snapshot.manifest = manifest_from_snapshot(index.snapshot);

  index.space = std::make_shared<hnswlib::L2Space>(config.dimension);
  index.hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(
      index.space.get(), reference.size(), config.hnsw_M,
      config.hnsw_ef_construction);
  for (uint32_t window_id = 0; window_id < reference.size(); ++window_id) {
    const float* vector = index.snapshot.exact_vectors.data() +
                          static_cast<std::size_t>(window_id) * config.dimension;
    index.hnsw->addPoint(vector, window_id);
  }

  return index;
}

void save_tensor_index(const TensorIndex& index,
                       const std::filesystem::path& directory) {
  std::filesystem::create_directories(directory);
  const std::filesystem::path exact_path = directory / "exact.bin";
  const std::filesystem::path hnsw_path = directory / "hnsw.bin";
  const std::filesystem::path meta_path = directory / "manifest.meta";

  if (index.snapshot.persist_exact_vectors) {
    const std::vector<uint8_t> payload = serialize_exact_payload(
        index.snapshot.labels, index.snapshot.exact_vectors,
        index.snapshot.dimension);
    write_index_atomic(exact_path, index.snapshot.manifest, payload);
  }
  if (!index.hnsw) {
    throw std::runtime_error("tensor index has no HNSW payload");
  }
  index.hnsw->saveIndex(hnsw_path.string());
  write_meta_file(meta_path, index.snapshot);
}

TensorIndex load_tensor_index(const std::filesystem::path& directory) {
  const std::filesystem::path exact_path = directory / "exact.bin";
  const std::filesystem::path hnsw_path = directory / "hnsw.bin";
  const std::filesystem::path meta_path = directory / "manifest.meta";

  const ParsedMeta meta = parse_meta_file(meta_path);

  TensorIndex index;
  index.snapshot.manifest = meta.manifest;
  index.snapshot.dimension = meta.dimension;
  index.snapshot.seed = meta.seed;
  index.snapshot.hnsw_M = meta.hnsw_M;
  index.snapshot.hnsw_ef_construction = meta.hnsw_ef_construction;
  index.snapshot.hnsw_ef_search = meta.hnsw_ef_search;
  index.snapshot.persist_exact_vectors = meta.exact_vectors;
  const uint64_t row_count = meta.manifest.number_of_windows;
  const uint32_t dimension = meta.dimension;

  if (meta.exact_vectors) {
    const PersistedIndex persisted = read_index(exact_path, meta.manifest);
    if (persisted.payload.size() < kExactMagic.size() + sizeof(uint32_t) +
                                     sizeof(uint64_t) + sizeof(uint32_t)) {
      throw std::runtime_error("exact payload is truncated");
    }
    std::size_t offset = 0;
    for (char magic_byte : kExactMagic) {
      if (persisted.payload[offset++] != static_cast<uint8_t>(magic_byte)) {
        throw std::runtime_error("invalid exact payload magic");
      }
    }
    if (read_u32(persisted.payload, offset) != kExactPayloadVersion) {
      throw std::runtime_error("unsupported exact payload version");
    }
    const uint64_t exact_row_count = read_u64(persisted.payload, offset);
    const uint32_t exact_dimension = read_u32(persisted.payload, offset);
    if (exact_dimension != dimension) {
      throw std::runtime_error("exact payload dimension mismatch");
    }
    if (exact_row_count != persisted.manifest.number_of_windows) {
      throw std::runtime_error("exact payload row count mismatch");
    }

    index.snapshot.manifest = persisted.manifest;
    index.snapshot.dimension = exact_dimension;
    index.snapshot.labels.reserve(static_cast<std::size_t>(exact_row_count));
    index.snapshot.exact_vectors.reserve(static_cast<std::size_t>(exact_row_count) *
                                         exact_dimension);
    for (uint64_t row = 0; row < exact_row_count; ++row) {
      const uint32_t label = read_u32(persisted.payload, offset);
      index.snapshot.labels.push_back(label);
      for (uint32_t column = 0; column < exact_dimension; ++column) {
        index.snapshot.exact_vectors.push_back(
            read_float(persisted.payload, offset));
      }
    }
    if (offset != persisted.payload.size()) {
      throw std::runtime_error("unexpected trailing bytes in exact payload");
    }
  } else {
    index.snapshot.labels.reserve(static_cast<std::size_t>(row_count));
    for (uint64_t row = 0; row < row_count; ++row) {
      index.snapshot.labels.push_back(static_cast<uint32_t>(row));
    }
  }

  index.space = std::make_shared<hnswlib::L2Space>(dimension);
  index.hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(
      index.space.get(), hnsw_path.string(), false,
      static_cast<size_t>(row_count));
  index.hnsw_path = hnsw_path;
  index.exact_path = exact_path;
  return index;
}

std::vector<QueryHit> query_tensor_index(TensorIndex& index,
                                        const std::vector<int>& query,
                                        std::size_t top_k) {
  if (!index.hnsw) {
    throw std::runtime_error("tensor index is not loaded");
  }
  index.hnsw->setEf(index.snapshot.hnsw_ef_search);
  ts::Tensor<int> tensor(4, index.snapshot.dimension, kTensorSubsequenceLength,
                         index.snapshot.seed);
  const std::vector<float> query_vector =
      tensor_to_float_vector(tensor.compute(query));
  if (query_vector.size() != index.snapshot.dimension) {
    throw std::runtime_error("query sketch dimension mismatch");
  }

  const std::size_t request = std::min<std::size_t>(
      kTopKSearchCap, index.snapshot.labels.size());
  if (request == 0) {
    return {};
  }
  const auto candidates =
      index.hnsw->searchKnnCloserFirst(query_vector.data(), request);

  std::vector<QueryHit> hits;
  hits.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    hits.push_back(QueryHit{static_cast<uint32_t>(candidate.second),
                            static_cast<float>(candidate.first)});
  }
  std::sort(hits.begin(), hits.end(), [](const QueryHit& lhs, const QueryHit& rhs) {
    if (lhs.distance != rhs.distance) {
      return lhs.distance < rhs.distance;
    }
    return lhs.label < rhs.label;
  });
  if (hits.size() > top_k) {
    hits.resize(top_k);
  }
  return hits;
}

}  // namespace tensor_index
