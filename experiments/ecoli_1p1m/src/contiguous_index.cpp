#include "candidate_indexes.hpp"

#include "reference_windows.hpp"
#include "sha256.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "index_persistence.hpp"

namespace {

constexpr std::array<char, 8> kIndexMagic = {'E', 'C', 'O', 'L', 'I', 'B', 'L', '1'};
constexpr uint32_t kIndexFormatVersion = 1;

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

std::filesystem::path navigamer_repo_root() {
  return std::filesystem::path(NAVIGAMER_REPO_ROOT);
}

uint8_t encode_base(char base) {
  switch (base) {
    case 'A':
    case 'a':
      return 0;
    case 'C':
    case 'c':
      return 1;
    case 'G':
    case 'g':
      return 2;
    case 'T':
    case 't':
      return 3;
    default:
      return 255;
  }
}

std::vector<uint64_t> query_kmer_keys(std::string_view sequence, uint32_t k) {
  std::vector<uint64_t> keys;
  if (k == 0 || sequence.size() < k) {
    return keys;
  }
  if (k > 32) {
    throw std::invalid_argument("contiguous index k must not exceed 32 bases");
  }

  uint64_t key = 0;
  uint32_t run_length = 0;
  const uint64_t mask =
      k == 32 ? std::numeric_limits<uint64_t>::max()
              : ((uint64_t{1} << (2U * k)) - 1U);
  for (char base : sequence) {
    const uint8_t code = encode_base(base);
    if (code > 3U) {
      key = 0;
      run_length = 0;
      continue;
    }
    key = (key << 2U) | code;
    if (k < 32) {
      key &= mask;
    }
    if (run_length < k) {
      ++run_length;
    }
    if (run_length >= k) {
      keys.push_back(key);
    }
  }

  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  return keys;
}

void read_exact(std::istream& input, void* destination, std::size_t size) {
  input.read(static_cast<char*>(destination), static_cast<std::streamsize>(size));
  if (!input) {
    throw std::runtime_error("truncated index file");
  }
}

struct LoadedIndexFile {
  IndexManifest manifest;
  std::vector<uint8_t> payload;
};

LoadedIndexFile read_index_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("unable to open index file: " + path.string());
  }

  std::array<char, 8> magic{};
  read_exact(input, magic.data(), magic.size());
  if (magic != kIndexMagic) {
    throw std::runtime_error("invalid index magic");
  }

  auto read_u32_stream = [&]() {
    std::array<uint8_t, 4> bytes{};
    read_exact(input, bytes.data(), bytes.size());
    uint32_t value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8) {
      value |= static_cast<uint32_t>(bytes[shift / 8U]) << shift;
    }
    return value;
  };
  auto read_u64_stream = [&]() {
    std::array<uint8_t, 8> bytes{};
    read_exact(input, bytes.data(), bytes.size());
    uint64_t value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8) {
      value |= static_cast<uint64_t>(bytes[shift / 8U]) << shift;
    }
    return value;
  };
  auto read_double_stream = [&]() {
    const uint64_t bits = read_u64_stream();
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  };
  auto read_string_stream = [&]() {
    const uint64_t size = read_u64_stream();
    if (size > std::numeric_limits<std::size_t>::max()) {
      throw std::runtime_error("string field too large");
    }
    std::string value(static_cast<std::size_t>(size), '\0');
    read_exact(input, value.data(), static_cast<std::size_t>(size));
    return value;
  };

  if (read_u32_stream() != kIndexFormatVersion) {
    throw std::runtime_error("unsupported index format version");
  }

  LoadedIndexFile loaded;
  loaded.manifest.method = read_string_stream();
  const uint64_t parameter_count = read_u64_stream();
  loaded.manifest.parameters.reserve(static_cast<std::size_t>(parameter_count));
  for (uint64_t index = 0; index < parameter_count; ++index) {
    loaded.manifest.parameters.emplace_back(read_string_stream(),
                                            read_string_stream());
  }
  loaded.manifest.reference_path = read_string_stream();
  loaded.manifest.reference_sha256 = read_string_stream();
  loaded.manifest.reference_length = read_u64_stream();
  loaded.manifest.window_length = read_u32_stream();
  loaded.manifest.stride = read_u32_stream();
  loaded.manifest.number_of_windows = read_u64_stream();
  loaded.manifest.build_command = read_string_stream();
  loaded.manifest.build_seconds = read_double_stream();
  loaded.manifest.index_bytes = read_u64_stream();
  loaded.manifest.created_at = read_string_stream();
  loaded.manifest.git_commit = read_string_stream();
  loaded.manifest.format_version = read_u32_stream();
  loaded.manifest.tool_version = read_string_stream();

  const uint64_t payload_size = read_u64_stream();
  std::array<uint8_t, 32> digest{};
  read_exact(input, digest.data(), digest.size());
  loaded.payload.resize(static_cast<std::size_t>(payload_size));
  read_exact(input, loaded.payload.data(), loaded.payload.size());
  std::array<uint8_t, 32> actual_digest = sha256(loaded.payload);
  if (!std::equal(digest.begin(), digest.end(), actual_digest.begin())) {
    throw std::runtime_error("payload checksum mismatch");
  }
  if (loaded.manifest.index_bytes != payload_size) {
    throw std::runtime_error("manifest index bytes do not match payload length");
  }
  if (input.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error("unexpected trailing bytes after index payload");
  }
  return loaded;
}

}  // namespace

ContiguousIndex ContiguousIndex::build(const ContiguousIndexConfig& config) {
  if (config.reference_path.empty()) {
    throw std::invalid_argument("reference path must not be empty");
  }
  if (config.window_length == 0) {
    throw std::invalid_argument("window length must be greater than zero");
  }
  if (config.stride == 0) {
    throw std::invalid_argument("stride must be greater than zero");
  }
  if (config.k == 0) {
    throw std::invalid_argument("k must be greater than zero");
  }
  if (config.k > config.window_length) {
    throw std::invalid_argument("k must not exceed the window length");
  }

  const auto started = std::chrono::steady_clock::now();
  const ReferenceWindows reference =
      ReferenceWindows::from_fasta(config.reference_path.string(),
                                   config.window_length, config.stride);
  ContiguousIndex index;
  index.occurrence_index_ =
      OccurrenceIndex::build(reference.sequence(), config.k);

  index.manifest_.method = "contig";
  index.manifest_.parameters = {
      {"method", "contig"},
      {"k", std::to_string(config.k)},
  };
  index.manifest_.reference_path = config.reference_path.string();
  index.manifest_.reference_sha256 = sha256_hex_of_file(config.reference_path);
  index.manifest_.reference_length = reference.sequence().size();
  index.manifest_.window_length = config.window_length;
  index.manifest_.stride = config.stride;
  index.manifest_.number_of_windows = reference.size();
  index.manifest_.build_command = "candidate_tool build --method contig";
  const auto finished = std::chrono::steady_clock::now();
  index.manifest_.build_seconds =
      std::chrono::duration<double>(finished - started).count();
  index.manifest_.created_at = current_utc_timestamp();
  index.manifest_.git_commit = git_commit_for_root(navigamer_repo_root());
  index.manifest_.format_version = 1;
  index.manifest_.tool_version = "contig-index/1";
  return index;
}

ContiguousIndex ContiguousIndex::load(const std::filesystem::path& index_path) {
  const LoadedIndexFile loaded = read_index_file(index_path);
  if (loaded.manifest.method != "contig") {
    throw std::runtime_error("unsupported contiguous index method");
  }
  ContiguousIndex index;
  index.manifest_ = loaded.manifest;
  index.occurrence_index_ = OccurrenceIndex::deserialize(loaded.payload);
  return index;
}

void ContiguousIndex::save(const std::filesystem::path& out_dir) const {
  if (out_dir.empty()) {
    throw std::invalid_argument("output directory must not be empty");
  }
  std::filesystem::create_directories(out_dir);
  const std::filesystem::path index_path = out_dir / "index.bin";
  IndexManifest manifest = manifest_;
  const std::vector<uint8_t> payload = occurrence_index_.serialize();
  manifest.index_bytes = payload.size();
  write_index_atomic(index_path, manifest, payload);
  write_manifest_json(out_dir / "manifest.json", manifest);
}

std::vector<uint32_t> ContiguousIndex::query(
    std::string_view query_sequence) const {
  const uint32_t k = occurrence_index_.k();
  if (k == 0) {
    return {};
  }
  const ReferenceWindows reference = ReferenceWindows::from_fasta(
      manifest_.reference_path, manifest_.window_length, manifest_.stride);
  const std::vector<uint64_t> query_keys = query_kmer_keys(query_sequence, k);
  if (query_keys.empty()) {
    return {};
  }

  std::vector<uint8_t> marked(reference.size(), 0);
  for (uint64_t key : query_keys) {
    const std::vector<uint32_t> occurrences = occurrence_index_.positions_for_key(key);
    for (uint32_t occurrence_start : occurrences) {
      const std::vector<uint32_t> window_ids =
          reference.covering_window_ids(occurrence_start, k);
      for (uint32_t window_id : window_ids) {
        marked[window_id] = 1;
      }
    }
  }

  std::vector<uint32_t> window_ids;
  for (uint32_t window_id = 0; window_id < marked.size(); ++window_id) {
    if (marked[window_id]) {
      window_ids.push_back(window_id);
    }
  }
  return window_ids;
}
