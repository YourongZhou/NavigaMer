#include "candidate_indexes.hpp"

#include "reference_windows.hpp"
#include "sha256.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <ctime>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "index_persistence.hpp"

namespace {

constexpr uint32_t kPayloadFormatVersion = 1;

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

void append_string(std::vector<uint8_t>& bytes, const std::string& value) {
  append_u64(bytes, value.size());
  bytes.insert(bytes.end(), value.begin(), value.end());
}

uint32_t read_u32(const std::vector<uint8_t>& bytes, std::size_t& offset) {
  if (offset + sizeof(uint32_t) > bytes.size()) {
    throw std::runtime_error("truncated pigeonhole payload");
  }
  uint32_t value = 0;
  for (unsigned shift = 0; shift < 32; shift += 8) {
    value |= static_cast<uint32_t>(bytes[offset++]) << shift;
  }
  return value;
}

uint64_t read_u64(const std::vector<uint8_t>& bytes, std::size_t& offset) {
  if (offset + sizeof(uint64_t) > bytes.size()) {
    throw std::runtime_error("truncated pigeonhole payload");
  }
  uint64_t value = 0;
  for (unsigned shift = 0; shift < 64; shift += 8) {
    value |= static_cast<uint64_t>(bytes[offset++]) << shift;
  }
  return value;
}

std::string read_string(const std::vector<uint8_t>& bytes, std::size_t& offset) {
  const uint64_t size = read_u64(bytes, offset);
  if (size > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("pigeonhole string length exceeds limits");
  }
  if (offset + size > bytes.size()) {
    throw std::runtime_error("truncated pigeonhole payload");
  }
  std::string value(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                    bytes.begin() + static_cast<std::ptrdiff_t>(offset + size));
  offset += static_cast<std::size_t>(size);
  return value;
}

struct ParsedPayload {
  uint32_t format_version = 0;
  uint32_t tau = 0;
  uint32_t nominal_read_length = 0;
  uint32_t minimum_block_length = 0;
  uint32_t supported_min_query_length = 0;
  uint32_t supported_max_query_length = 0;
  std::string reference_sequence;
  std::vector<PigeonholeIndex::PostingGroup> postings;
};

ParsedPayload parse_payload(const std::vector<uint8_t>& payload) {
  ParsedPayload parsed;
  std::size_t offset = 0;
  parsed.format_version = read_u32(payload, offset);
  if (parsed.format_version != kPayloadFormatVersion) {
    throw std::runtime_error("unsupported pigeonhole payload version");
  }
  parsed.tau = read_u32(payload, offset);
  parsed.nominal_read_length = read_u32(payload, offset);
  parsed.minimum_block_length = read_u32(payload, offset);
  parsed.supported_min_query_length = read_u32(payload, offset);
  parsed.supported_max_query_length = read_u32(payload, offset);
  parsed.reference_sequence = read_string(payload, offset);
  const uint64_t group_count = read_u64(payload, offset);
  if (group_count > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("pigeonhole posting count exceeds limits");
  }
  parsed.postings.resize(static_cast<std::size_t>(group_count));
  for (std::size_t index = 0; index < parsed.postings.size(); ++index) {
    parsed.postings[index].key = read_string(payload, offset);
    const uint64_t position_count = read_u64(payload, offset);
    if (position_count >
        static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
      throw std::runtime_error("pigeonhole posting position count exceeds limits");
    }
    parsed.postings[index].positions.resize(
        static_cast<std::size_t>(position_count));
    for (std::size_t position = 0; position < parsed.postings[index].positions.size();
         ++position) {
      parsed.postings[index].positions[position] = read_u32(payload, offset);
    }
  }
  if (offset != payload.size()) {
    throw std::runtime_error("unexpected trailing bytes in pigeonhole payload");
  }
  return parsed;
}

std::vector<uint8_t> serialize_payload(
    uint32_t tau, uint32_t nominal_read_length, uint32_t minimum_block_length,
    uint32_t supported_min_query_length, uint32_t supported_max_query_length,
    const std::string& reference_sequence,
    const std::vector<PigeonholeIndex::PostingGroup>& postings) {
  std::vector<uint8_t> payload;
  payload.reserve(24 + reference_sequence.size() +
                  postings.size() * (16 + minimum_block_length));
  append_u32(payload, kPayloadFormatVersion);
  append_u32(payload, tau);
  append_u32(payload, nominal_read_length);
  append_u32(payload, minimum_block_length);
  append_u32(payload, supported_min_query_length);
  append_u32(payload, supported_max_query_length);
  append_string(payload, reference_sequence);
  append_u64(payload, postings.size());
  for (const auto& group : postings) {
    append_string(payload, group.key);
    append_u64(payload, group.positions.size());
    for (uint32_t position : group.positions) {
      append_u32(payload, position);
    }
  }
  return payload;
}

PigeonholeIndex::PostingGroup make_posting_group(
    std::string key, std::vector<uint32_t> positions) {
  std::sort(positions.begin(), positions.end());
  return {std::move(key), std::move(positions)};
}

std::vector<PigeonholeIndex::PostingGroup> build_postings(
    const std::string& reference_sequence, uint32_t block_length) {
  std::vector<PigeonholeIndex::PostingGroup> postings;
  if (reference_sequence.size() < block_length) {
    return postings;
  }

  std::vector<std::pair<std::string, std::vector<uint32_t>>> bins;
  bins.reserve(reference_sequence.size() - block_length + 1);
  std::unordered_map<std::string, std::size_t> directory;
  directory.reserve(reference_sequence.size() - block_length + 1);

  for (uint32_t position = 0;
       position + block_length <= reference_sequence.size(); ++position) {
    const std::string key = reference_sequence.substr(position, block_length);
    const auto [it, inserted] = directory.emplace(key, bins.size());
    if (inserted) {
      bins.emplace_back(key, std::vector<uint32_t>{position});
    } else {
      bins[it->second].second.push_back(position);
    }
  }

  postings.reserve(bins.size());
  for (auto& entry : bins) {
    postings.push_back(make_posting_group(std::move(entry.first),
                                          std::move(entry.second)));
  }
  std::sort(postings.begin(), postings.end(),
            [](const PigeonholeIndex::PostingGroup& lhs,
               const PigeonholeIndex::PostingGroup& rhs) {
              if (lhs.key != rhs.key) {
                return lhs.key < rhs.key;
              }
              return lhs.positions < rhs.positions;
            });
  return postings;
}

uint32_t checked_uint32(uint64_t value, const char* description) {
  if (value > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error(std::string(description) + " exceeds uint32_t");
  }
  return static_cast<uint32_t>(value);
}

std::string manifest_build_command(uint32_t tau, uint32_t nominal_read_length) {
  return "candidate_tool build --method pigeonhole --tau " + std::to_string(tau) +
         " --nominal-read-length " + std::to_string(nominal_read_length);
}

std::vector<uint32_t> aligned_window_ids_for_start_range(
    uint64_t raw_min_start, uint64_t raw_max_start, uint32_t stride,
    uint32_t window_length, uint64_t reference_length,
    uint32_t number_of_windows) {
  if (reference_length < window_length) {
    return {};
  }
  const uint64_t maximum_window_start = reference_length - window_length;
  const uint64_t clipped_min = std::min<uint64_t>(raw_min_start, maximum_window_start);
  const uint64_t clipped_max = std::min<uint64_t>(raw_max_start, maximum_window_start);
  if (clipped_min > clipped_max) {
    return {};
  }
  const uint64_t first_start = (clipped_min + stride - 1) / stride * stride;
  const uint64_t last_start = clipped_max / stride * stride;
  if (first_start > last_start) {
    return {};
  }

  std::vector<uint32_t> window_ids;
  const uint64_t first_id = first_start / stride;
  const uint64_t last_id = last_start / stride;
  if (first_id >= number_of_windows) {
    return {};
  }
  const uint64_t clipped_last_id = std::min<uint64_t>(last_id, number_of_windows - 1);
  window_ids.reserve(checked_uint32(clipped_last_id - first_id + 1,
                                    "pigeonhole window count"));
  for (uint64_t id = first_id; id <= clipped_last_id; ++id) {
    window_ids.push_back(checked_uint32(id, "window ID"));
  }
  return window_ids;
}

}  // namespace

PigeonholeIndex PigeonholeIndex::build(const PigeonholeIndexConfig& config) {
  if (config.reference_path.empty()) {
    throw std::invalid_argument("reference path must not be empty");
  }
  if (config.window_length == 0) {
    throw std::invalid_argument("window length must be greater than zero");
  }
  if (config.stride == 0) {
    throw std::invalid_argument("stride must be greater than zero");
  }
  if (config.tau == 0) {
    throw std::invalid_argument("tau must be greater than zero");
  }
  if (config.nominal_read_length == 0) {
    throw std::invalid_argument("nominal read length must be greater than zero");
  }
  if (config.nominal_read_length < config.tau) {
    throw std::invalid_argument(
        "nominal read length must be at least the supported tau");
  }
  if (config.window_length <= config.tau) {
    throw std::invalid_argument("window length must exceed tau");
  }

  const uint32_t minimum_block_length =
      (config.window_length - config.tau) / (config.tau + 1);
  if (minimum_block_length == 0) {
    throw std::invalid_argument("minimum block length must be greater than zero");
  }

  const auto started = std::chrono::steady_clock::now();
  const ReferenceWindows reference =
      ReferenceWindows::from_fasta(config.reference_path.string(),
                                   config.window_length, config.stride);

  PigeonholeIndex index;
  index.tau_ = config.tau;
  index.nominal_read_length_ = config.nominal_read_length;
  index.minimum_block_length_ = minimum_block_length;
  index.supported_min_query_length_ = config.window_length - config.tau;
  index.supported_max_query_length_ = config.window_length + config.tau;
  index.reference_sequence_ = reference.sequence();
  index.postings_ = build_postings(index.reference_sequence_, minimum_block_length);

  index.manifest_.method = "pigeonhole";
  index.manifest_.parameters = {
      {"method", "pigeonhole"},
      {"tau", std::to_string(config.tau)},
      {"nominal-read-length", std::to_string(config.nominal_read_length)},
      {"minimum-block-length", std::to_string(minimum_block_length)},
      {"supported-query-length-min",
       std::to_string(index.supported_min_query_length_)},
      {"supported-query-length-max",
       std::to_string(index.supported_max_query_length_)},
  };
  index.manifest_.reference_path = config.reference_path.string();
  index.manifest_.reference_sha256 = sha256_hex_of_file(config.reference_path);
  index.manifest_.reference_length = reference.sequence().size();
  index.manifest_.window_length = config.window_length;
  index.manifest_.stride = config.stride;
  index.manifest_.number_of_windows = reference.size();
  index.manifest_.build_command =
      manifest_build_command(config.tau, config.nominal_read_length);
  const auto finished = std::chrono::steady_clock::now();
  index.manifest_.build_seconds =
      std::chrono::duration<double>(finished - started).count();
  index.manifest_.created_at = current_utc_timestamp();
  index.manifest_.git_commit = git_commit_for_root(navigamer_repo_root());
  index.manifest_.format_version = 1;
  index.manifest_.tool_version = "pigeonhole-index/1";
  index.payload_ = serialize_payload(index.tau_, index.nominal_read_length_,
                                     index.minimum_block_length_,
                                     index.supported_min_query_length_,
                                     index.supported_max_query_length_,
                                     index.reference_sequence_, index.postings_);
  return index;
}

PigeonholeIndex PigeonholeIndex::load(const std::filesystem::path& index_path) {
  return load(read_index_file(index_path));
}

PigeonholeIndex PigeonholeIndex::load(const PersistedIndex& loaded_index) {
  if (loaded_index.manifest.method != "pigeonhole") {
    throw std::runtime_error("unsupported pigeonhole index method");
  }

  const ParsedPayload parsed = parse_payload(loaded_index.payload);
  if (parsed.tau == 0) {
    throw std::runtime_error("invalid pigeonhole tau in payload");
  }

  PigeonholeIndex index;
  index.manifest_ = loaded_index.manifest;
  index.tau_ = parsed.tau;
  index.nominal_read_length_ = parsed.nominal_read_length;
  index.minimum_block_length_ = parsed.minimum_block_length;
  index.supported_min_query_length_ = parsed.supported_min_query_length;
  index.supported_max_query_length_ = parsed.supported_max_query_length;
  index.reference_sequence_ = parsed.reference_sequence;
  index.postings_ = std::move(parsed.postings);
  index.payload_ = loaded_index.payload;

  if (index.minimum_block_length_ == 0 || index.nominal_read_length_ == 0) {
    throw std::runtime_error("invalid pigeonhole payload dimensions");
  }
  if (index.supported_min_query_length_ > index.supported_max_query_length_) {
    throw std::runtime_error("invalid pigeonhole supported query interval");
  }
  return index;
}

void PigeonholeIndex::save(const std::filesystem::path& out_dir) const {
  if (out_dir.empty()) {
    throw std::invalid_argument("output directory must not be empty");
  }
  std::filesystem::create_directories(out_dir);
  const std::filesystem::path index_path = out_dir / "index.bin";
  IndexManifest manifest = manifest_;
  manifest.index_bytes = payload_.size();
  write_index_atomic(index_path, manifest, payload_);
  write_manifest_json(out_dir / "manifest.json", manifest);
}

std::vector<uint32_t> PigeonholeIndex::query(std::string_view query_sequence,
                                             uint32_t tau) const {
  if (tau != tau_) {
    throw std::invalid_argument("query tau does not match the pigeonhole index");
  }
  if (query_sequence.size() < supported_min_query_length_ ||
      query_sequence.size() > supported_max_query_length_) {
    throw std::invalid_argument(
        "query length is outside the supported pigeonhole interval");
  }
  if (query_sequence.size() < tau + 1) {
    throw std::invalid_argument("query is too short for the requested tau");
  }

  std::vector<uint8_t> marked(
      static_cast<std::size_t>(manifest_.number_of_windows), 0);
  const uint64_t reference_length = reference_sequence_.size();

  const uint32_t block_count = tau + 1;
  for (uint32_t block_index = 0; block_index < block_count; ++block_index) {
    const uint64_t block_start =
        (static_cast<uint64_t>(block_index) * query_sequence.size()) / block_count;
    const uint64_t block_end =
        (static_cast<uint64_t>(block_index + 1) * query_sequence.size()) /
        block_count;
    const uint64_t block_length = block_end - block_start;
    if (block_length < minimum_block_length_) {
      throw std::runtime_error("query block is shorter than the indexed block length");
    }
    if (block_length > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
      throw std::overflow_error("query block length exceeds size_t");
    }

    const std::string_view block = query_sequence.substr(
        static_cast<std::size_t>(block_start), static_cast<std::size_t>(block_length));
    const std::string_view seed = block.substr(0, minimum_block_length_);

    const auto iterator =
        std::lower_bound(postings_.begin(), postings_.end(), seed,
                         [](const PostingGroup& group, std::string_view key) {
                           return group.key < key;
                         });
    if (iterator == postings_.end() || iterator->key != seed) {
      continue;
    }

    for (uint32_t occurrence : iterator->positions) {
      if (static_cast<uint64_t>(occurrence) + block_length > reference_length) {
        continue;
      }
      if (reference_sequence_.compare(
              occurrence, static_cast<std::size_t>(block_length), block.data(),
              static_cast<std::size_t>(block_length)) != 0) {
        continue;
      }

      const int64_t aligned_start =
          static_cast<int64_t>(occurrence) - static_cast<int64_t>(block_start);
      const int64_t raw_min_start = aligned_start - static_cast<int64_t>(tau);
      const int64_t raw_max_start = aligned_start + static_cast<int64_t>(tau);
      const uint64_t clipped_min =
          raw_min_start < 0 ? 0U : static_cast<uint64_t>(raw_min_start);
      const uint64_t clipped_max = raw_max_start < 0
                                       ? 0U
                                       : static_cast<uint64_t>(raw_max_start);
      const std::vector<uint32_t> window_ids = aligned_window_ids_for_start_range(
          clipped_min, clipped_max, manifest_.stride, manifest_.window_length,
          reference_length, manifest_.number_of_windows);
      for (uint32_t window_id : window_ids) {
        if (window_id < marked.size()) {
          marked[window_id] = 1;
        }
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
