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

std::vector<uint32_t> covering_window_ids_from_manifest(
    const IndexManifest& manifest, uint32_t occurrence_start, uint32_t span) {
  if (span == 0) {
    throw std::invalid_argument("occurrence span must be greater than zero");
  }
  if (manifest.window_length == 0 || manifest.stride == 0 ||
      manifest.number_of_windows == 0 || manifest.reference_length == 0) {
    throw std::runtime_error("contiguous index manifest is incomplete");
  }
  if (span > manifest.window_length) {
    throw std::invalid_argument(
        "occurrence span must not exceed the window length");
  }

  const uint64_t occurrence_end = static_cast<uint64_t>(occurrence_start) + span;
  if (occurrence_start >= manifest.reference_length ||
      occurrence_end > manifest.reference_length) {
    return {};
  }

  const uint64_t minimum_start =
      occurrence_end > manifest.window_length
          ? occurrence_end - manifest.window_length
          : 0;
  const uint64_t maximum_window_start =
      static_cast<uint64_t>(manifest.number_of_windows - 1) * manifest.stride;
  const uint64_t maximum_start =
      std::min<uint64_t>(occurrence_start, maximum_window_start);
  if (minimum_start > maximum_start) {
    return {};
  }

  const uint64_t first_id = (minimum_start + manifest.stride - 1) / manifest.stride;
  const uint64_t last_id = maximum_start / manifest.stride;
  if (first_id > last_id) {
    return {};
  }

  std::vector<uint32_t> ids;
  ids.reserve(static_cast<std::size_t>(last_id - first_id + 1));
  for (uint64_t id = first_id; id <= last_id; ++id) {
    ids.push_back(static_cast<uint32_t>(id));
  }
  return ids;
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

ContiguousIndex ContiguousIndex::load(
    const std::filesystem::path& index_path) {
  return load(read_index_file(index_path));
}

ContiguousIndex ContiguousIndex::load(const PersistedIndex& loaded_index) {
  if (loaded_index.manifest.method != "contig") {
    throw std::runtime_error("unsupported contiguous index method");
  }
  ContiguousIndex index;
  index.manifest_ = loaded_index.manifest;
  index.occurrence_index_ = OccurrenceIndex::deserialize(loaded_index.payload);
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
  const std::vector<uint64_t> query_keys = query_kmer_keys(query_sequence, k);
  if (query_keys.empty()) {
    return {};
  }

  std::vector<uint8_t> marked(
      static_cast<std::size_t>(manifest_.number_of_windows), 0);
  for (uint64_t key : query_keys) {
    const std::vector<uint32_t> occurrences =
        occurrence_index_.positions_for_key(key);
    for (uint32_t occurrence_start : occurrences) {
      const std::vector<uint32_t> window_ids =
          covering_window_ids_from_manifest(manifest_, occurrence_start, k);
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
