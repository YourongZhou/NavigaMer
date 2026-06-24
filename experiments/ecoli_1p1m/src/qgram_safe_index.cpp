#include "candidate_indexes.hpp"

#include "reference_windows.hpp"
#include "sha256.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "index_persistence.hpp"

namespace {

constexpr uint32_t kMaxSupportedDenseQ = 12;
constexpr uint32_t kInvalidCode = std::numeric_limits<uint32_t>::max();

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

uint32_t read_u32(const std::vector<uint8_t>& bytes, std::size_t& offset) {
  if (offset + sizeof(uint32_t) > bytes.size()) {
    throw std::runtime_error("truncated qgram-safe payload");
  }
  uint32_t value = 0;
  for (unsigned shift = 0; shift < 32; shift += 8) {
    value |= static_cast<uint32_t>(bytes[offset++]) << shift;
  }
  return value;
}

uint64_t read_u64(const std::vector<uint8_t>& bytes, std::size_t& offset) {
  if (offset + sizeof(uint64_t) > bytes.size()) {
    throw std::runtime_error("truncated qgram-safe payload");
  }
  uint64_t value = 0;
  for (unsigned shift = 0; shift < 64; shift += 8) {
    value |= static_cast<uint64_t>(bytes[offset++]) << shift;
  }
  return value;
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

std::string sha256_hex_of_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("unable to open file: " + path.string());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  if (!input && !input.eof()) {
    throw std::runtime_error("unable to read file: " + path.string());
  }
  return sha256_hex(buffer.str());
}

uint32_t encode_base(char base) {
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
      return kInvalidCode;
  }
}

bool is_supported_sequence(std::string_view sequence) {
  for (char base : sequence) {
    if (encode_base(base) == kInvalidCode) {
      return false;
    }
  }
  return true;
}

uint64_t dense_cell_count_for_q(uint32_t q) {
  if (q == 0 || q > kMaxSupportedDenseQ) {
    throw std::invalid_argument(
        "q must be between 1 and 12 for the dense q-gram vector");
  }
  uint64_t count = 1;
  for (uint32_t index = 0; index < q; ++index) {
    if (count > std::numeric_limits<uint64_t>::max() / 4U) {
      throw std::overflow_error("dense q-gram vector size overflow");
    }
    count *= 4U;
  }
  if (count > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error("dense q-gram vector size exceeds addressable memory");
  }
  return count;
}

uint32_t encode_qgram(std::string_view sequence, std::size_t start, uint32_t q) {
  uint32_t code = 0;
  for (uint32_t offset = 0; offset < q; ++offset) {
    const uint32_t base_code = encode_base(sequence[start + offset]);
    if (base_code == kInvalidCode) {
      return kInvalidCode;
    }
    code = (code << 2U) | base_code;
  }
  return code;
}

std::vector<uint32_t> encode_reference_codes(std::string_view sequence,
                                             uint32_t q) {
  if (q == 0 || sequence.size() < q) {
    return {};
  }
  std::vector<uint32_t> codes;
  codes.reserve(sequence.size() - q + 1);
  for (std::size_t start = 0; start + q <= sequence.size(); ++start) {
    codes.push_back(encode_qgram(sequence, start, q));
  }
  return codes;
}

std::vector<uint32_t> build_dense_counts_from_codes(
    const std::vector<uint32_t>& codes, std::size_t start, std::size_t count,
    std::size_t dense_cell_count) {
  std::vector<uint32_t> counts(dense_cell_count, 0);
  if (count == 0) {
    return counts;
  }
  for (std::size_t offset = 0; offset < count; ++offset) {
    const uint32_t code = codes[start + offset];
    if (code == kInvalidCode) {
      continue;
    }
    ++counts[code];
  }
  return counts;
}

std::vector<uint32_t> build_dense_counts_for_sequence(std::string_view sequence,
                                                      uint32_t q,
                                                      std::size_t dense_cell_count) {
  return build_dense_counts_from_codes(encode_reference_codes(sequence, q), 0,
                                       sequence.size() < q ? 0 : sequence.size() - q + 1,
                                       dense_cell_count);
}

std::vector<uint32_t> build_invalid_prefix(
    const std::vector<uint32_t>& codes) {
  std::vector<uint32_t> prefix(codes.size() + 1, 0);
  for (std::size_t index = 0; index < codes.size(); ++index) {
    prefix[index + 1] = prefix[index] + (codes[index] == kInvalidCode ? 1U : 0U);
  }
  return prefix;
}

struct FullSignature {
  std::unordered_map<std::string, uint32_t> counts;
};

std::string uppercase_copy(std::string_view sequence) {
  std::string value(sequence);
  for (char& base : value) {
    base = static_cast<char>(std::toupper(static_cast<unsigned char>(base)));
  }
  return value;
}

FullSignature build_full_signature(std::string_view sequence, uint32_t q) {
  FullSignature signature;
  const std::string normalized = uppercase_copy(sequence);
  if (q == 0 || normalized.size() < q) {
    return signature;
  }
  for (std::size_t offset = 0; offset + q <= normalized.size(); ++offset) {
    signature.counts[normalized.substr(offset, q)]++;
  }
  return signature;
}

uint64_t full_signature_l1(const FullSignature& lhs, const FullSignature& rhs) {
  uint64_t l1 = 0;
  for (const auto& entry : lhs.counts) {
    const auto rhs_it = rhs.counts.find(entry.first);
    const uint64_t rhs_count =
        rhs_it == rhs.counts.end() ? 0U : rhs_it->second;
    l1 += entry.second > rhs_count ? entry.second - rhs_count
                                   : rhs_count - entry.second;
  }
  for (const auto& entry : rhs.counts) {
    if (lhs.counts.find(entry.first) == lhs.counts.end()) {
      l1 += entry.second;
    }
  }
  return l1;
}

std::vector<uint8_t> serialize_qgram_payload(
    uint32_t q, const std::string& reference_sequence,
    const std::vector<uint32_t>& reference_codes,
    const std::vector<uint32_t>& first_window_counts) {
  std::vector<uint8_t> bytes;
  append_u32(bytes, q);
  append_u64(bytes, reference_sequence.size());
  bytes.insert(bytes.end(), reference_sequence.begin(), reference_sequence.end());
  append_u64(bytes, reference_codes.size());
  for (uint32_t code : reference_codes) {
    append_u32(bytes, code);
  }
  append_u64(bytes, first_window_counts.size());
  for (uint32_t count : first_window_counts) {
    append_u32(bytes, count);
  }
  return bytes;
}

struct ParsedQgramPayload {
  uint32_t q = 0;
  std::string reference_sequence;
  std::vector<uint32_t> reference_codes;
  std::vector<uint32_t> first_window_counts;
};

ParsedQgramPayload parse_qgram_payload(const std::vector<uint8_t>& payload) {
  ParsedQgramPayload parsed;
  std::size_t offset = 0;
  parsed.q = read_u32(payload, offset);
  const uint64_t reference_length = read_u64(payload, offset);
  if (reference_length >
      static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("qgram-safe reference length exceeds limits");
  }
  if (offset + static_cast<std::size_t>(reference_length) > payload.size()) {
    throw std::runtime_error("truncated qgram-safe payload");
  }
  parsed.reference_sequence.assign(
      reinterpret_cast<const char*>(payload.data() + offset),
      static_cast<std::size_t>(reference_length));
  offset += static_cast<std::size_t>(reference_length);

  const uint64_t code_count = read_u64(payload, offset);
  if (code_count >
      static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("qgram-safe code stream length exceeds limits");
  }
  parsed.reference_codes.resize(static_cast<std::size_t>(code_count));
  for (std::size_t index = 0; index < parsed.reference_codes.size(); ++index) {
    parsed.reference_codes[index] = read_u32(payload, offset);
  }

  const uint64_t dense_cell_count = read_u64(payload, offset);
  if (dense_cell_count >
      static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("qgram-safe dense vector length exceeds limits");
  }
  parsed.first_window_counts.resize(static_cast<std::size_t>(dense_cell_count));
  for (std::size_t index = 0; index < parsed.first_window_counts.size(); ++index) {
    parsed.first_window_counts[index] = read_u32(payload, offset);
  }

  if (offset != payload.size()) {
    throw std::runtime_error("unexpected trailing bytes in qgram-safe payload");
  }
  return parsed;
}

bool window_is_clean(const std::vector<uint32_t>& invalid_prefix,
                    const std::vector<uint32_t>& codes, std::size_t start,
                    std::size_t count) {
  if (count == 0) {
    return true;
  }
  if (start + count > codes.size()) {
    return false;
  }
  return invalid_prefix[start + count] == invalid_prefix[start];
}

uint64_t threshold_for(uint32_t q, uint32_t tau) {
  return 2ULL * static_cast<uint64_t>(q) * static_cast<uint64_t>(tau);
}

uint64_t dense_l1(const std::vector<uint32_t>& lhs,
                  const std::vector<uint32_t>& rhs) {
  uint64_t l1 = 0;
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    const uint32_t left = lhs[index];
    const uint32_t right = rhs[index];
    l1 += left > right ? static_cast<uint64_t>(left - right)
                       : static_cast<uint64_t>(right - left);
  }
  return l1;
}

void decrement_code(std::vector<uint32_t>& counts, const std::vector<uint32_t>& query_counts,
                    uint32_t code, uint64_t& l1) {
  if (code == kInvalidCode) {
    return;
  }
  const uint32_t query_count = query_counts[code];
  const uint32_t target_count = counts[code];
  const uint64_t before = target_count > query_count
                              ? static_cast<uint64_t>(target_count - query_count)
                              : static_cast<uint64_t>(query_count - target_count);
  --counts[code];
  const uint32_t updated = counts[code];
  const uint64_t after = updated > query_count
                             ? static_cast<uint64_t>(updated - query_count)
                             : static_cast<uint64_t>(query_count - updated);
  l1 += after - before;
}

void increment_code(std::vector<uint32_t>& counts, const std::vector<uint32_t>& query_counts,
                    uint32_t code, uint64_t& l1) {
  if (code == kInvalidCode) {
    return;
  }
  const uint32_t query_count = query_counts[code];
  const uint32_t target_count = counts[code];
  const uint64_t before = target_count > query_count
                              ? static_cast<uint64_t>(target_count - query_count)
                              : static_cast<uint64_t>(query_count - target_count);
  ++counts[code];
  const uint32_t updated = counts[code];
  const uint64_t after = updated > query_count
                             ? static_cast<uint64_t>(updated - query_count)
                             : static_cast<uint64_t>(query_count - updated);
  l1 += after - before;
}

}  // namespace

QgramSafeIndex QgramSafeIndex::build(const QgramSafeIndexConfig& config) {
  if (config.reference_path.empty()) {
    throw std::invalid_argument("reference path must not be empty");
  }
  if (config.window_length == 0) {
    throw std::invalid_argument("window length must be greater than zero");
  }
  if (config.stride == 0) {
    throw std::invalid_argument("stride must be greater than zero");
  }
  if (config.q == 0) {
    throw std::invalid_argument("q must be greater than zero");
  }
  const uint64_t dense_cell_count = dense_cell_count_for_q(config.q);

  const auto started = std::chrono::steady_clock::now();
  const ReferenceWindows reference =
      ReferenceWindows::from_fasta(config.reference_path.string(),
                                   config.window_length, config.stride);

  QgramSafeIndex index;
  index.q_ = config.q;
  index.window_length_ = config.window_length;
  index.stride_ = config.stride;
  index.reference_sequence_ = reference.sequence();
  index.reference_codes_ = encode_reference_codes(index.reference_sequence_, index.q_);
  index.invalid_prefix_ = build_invalid_prefix(index.reference_codes_);
  index.first_window_counts_ = build_dense_counts_from_codes(
      index.reference_codes_, 0,
      index.window_length_ >= index.q_ ? index.window_length_ - index.q_ + 1 : 0,
      static_cast<std::size_t>(dense_cell_count));
  index.payload_ = serialize_qgram_payload(index.q_, index.reference_sequence_,
                                           index.reference_codes_,
                                           index.first_window_counts_);

  index.manifest_.method = "qgram-safe";
  index.manifest_.parameters = {
      {"method", "qgram-safe"},
      {"q", std::to_string(config.q)},
      {"dense_cells", std::to_string(dense_cell_count)},
  };
  index.manifest_.reference_path = config.reference_path.string();
  index.manifest_.reference_sha256 = sha256_hex_of_file(config.reference_path);
  index.manifest_.reference_length = reference.sequence().size();
  index.manifest_.window_length = config.window_length;
  index.manifest_.stride = config.stride;
  index.manifest_.number_of_windows = reference.size();
  index.manifest_.build_command =
      "candidate_tool build --method qgram-safe --q " +
      std::to_string(config.q);
  const auto finished = std::chrono::steady_clock::now();
  index.manifest_.build_seconds =
      std::chrono::duration<double>(finished - started).count();
  index.manifest_.created_at = current_utc_timestamp();
  index.manifest_.git_commit = git_commit_for_root(navigamer_repo_root());
  index.manifest_.format_version = 1;
  index.manifest_.tool_version = "qgram-safe-index/1";
  return index;
}

QgramSafeIndex QgramSafeIndex::load(const std::filesystem::path& index_path) {
  return load(read_index_file(index_path));
}

QgramSafeIndex QgramSafeIndex::load(const PersistedIndex& loaded_index) {
  if (loaded_index.manifest.method != "qgram-safe") {
    throw std::runtime_error("unsupported qgram-safe index method");
  }

  QgramSafeIndex index;
  index.manifest_ = loaded_index.manifest;
  index.payload_ = loaded_index.payload;

  const ParsedQgramPayload parsed = parse_qgram_payload(index.payload_);
  if (parsed.q == 0 || parsed.q > kMaxSupportedDenseQ) {
    throw std::runtime_error("unsupported qgram-safe q value in payload");
  }
  const uint64_t dense_cell_count = dense_cell_count_for_q(parsed.q);
  if (parsed.first_window_counts.size() != dense_cell_count) {
    throw std::runtime_error("qgram-safe dense vector size mismatch");
  }
  const uint64_t expected_codes =
      parsed.reference_sequence.size() < parsed.q
          ? 0
          : parsed.reference_sequence.size() - parsed.q + 1;
  if (parsed.reference_codes.size() != expected_codes) {
    throw std::runtime_error("qgram-safe code stream length mismatch");
  }
  for (uint32_t code : parsed.reference_codes) {
    if (code != kInvalidCode && code >= dense_cell_count) {
      throw std::runtime_error("qgram-safe code stream contains an invalid code");
    }
  }

  index.q_ = parsed.q;
  index.window_length_ = loaded_index.manifest.window_length;
  index.stride_ = loaded_index.manifest.stride;
  index.reference_sequence_ = parsed.reference_sequence;
  index.reference_codes_ = parsed.reference_codes;
  index.first_window_counts_ = parsed.first_window_counts;
  index.invalid_prefix_ = build_invalid_prefix(index.reference_codes_);
  return index;
}

void QgramSafeIndex::save(const std::filesystem::path& out_dir) const {
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

std::vector<uint32_t> QgramSafeIndex::query(std::string_view query_sequence,
                                            uint32_t tau) const {
  if (q_ == 0 || payload_.empty()) {
    return {};
  }

  const uint64_t max_l1 = threshold_for(q_, tau);
  const uint64_t window_code_count =
      window_length_ >= q_ ? static_cast<uint64_t>(window_length_ - q_ + 1) : 0;
  const uint64_t max_start =
      manifest_.number_of_windows == 0
          ? 0
          : static_cast<uint64_t>(manifest_.number_of_windows - 1) * stride_;

  const bool query_supported = is_supported_sequence(query_sequence);
  const FullSignature query_full = build_full_signature(query_sequence, q_);
  std::vector<uint32_t> query_counts;
  if (query_supported) {
    query_counts =
        build_dense_counts_for_sequence(query_sequence, q_,
                                        first_window_counts_.size());
  }

  std::vector<uint32_t> target_counts = first_window_counts_;
  uint64_t current_l1 = query_supported ? dense_l1(query_counts, target_counts) : 0;
  std::vector<uint32_t> result;
  result.reserve(manifest_.number_of_windows);

  for (uint64_t start = 0; start <= max_start; ++start) {
    const bool aligned = start % stride_ == 0;
    if (aligned) {
      const uint32_t window_id = static_cast<uint32_t>(start / stride_);
      if (query_supported &&
          window_is_clean(invalid_prefix_, reference_codes_,
                          static_cast<std::size_t>(start),
                          static_cast<std::size_t>(window_code_count))) {
        if (current_l1 <= max_l1) {
          result.push_back(window_id);
        }
      } else {
        const std::string_view window =
            std::string_view(reference_sequence_).substr(
                static_cast<std::size_t>(start), window_length_);
        const FullSignature window_full =
            build_full_signature(window, q_);
        if (full_signature_l1(query_full, window_full) <= max_l1) {
          result.push_back(window_id);
        }
      }
    }

    if (start == max_start || window_code_count == 0) {
      continue;
    }

    const uint32_t outgoing =
        start < reference_codes_.size() ? reference_codes_[start] : kInvalidCode;
    const uint64_t incoming_index = start + window_code_count;
    const uint32_t incoming =
        incoming_index < reference_codes_.size()
            ? reference_codes_[static_cast<std::size_t>(incoming_index)]
            : kInvalidCode;
    if (query_supported) {
      decrement_code(target_counts, query_counts, outgoing, current_l1);
      increment_code(target_counts, query_counts, incoming, current_l1);
    } else {
      if (outgoing != kInvalidCode) {
        --target_counts[outgoing];
      }
      if (incoming != kInvalidCode) {
        ++target_counts[incoming];
      }
    }
  }

  return result;
}
