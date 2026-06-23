#include "index_persistence.hpp"

#include "sha256.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <unistd.h>

namespace {

constexpr std::array<char, 8> kMagic = {'E', 'C', 'O', 'L', 'I', 'B', 'L', '1'};
constexpr uint32_t kFormatVersion = 1;
constexpr uint64_t kMaximumStringLength = 16U * 1024U * 1024U;
constexpr uint64_t kMaximumParameterCount = 1U * 1024U * 1024U;
std::atomic<WriteIndexAtomicBeforePublishHook> g_before_publish_hook{nullptr};

void validate_digest(const std::string& digest) {
  if (digest.size() != 64) {
    throw std::invalid_argument("reference SHA-256 must contain 64 hex digits");
  }
  for (char byte : digest) {
    const bool digit = byte >= '0' && byte <= '9';
    const bool lower = byte >= 'a' && byte <= 'f';
    const bool upper = byte >= 'A' && byte <= 'F';
    if (!digit && !lower && !upper) {
      throw std::invalid_argument(
          "reference SHA-256 must contain only hex digits");
    }
  }
}

void validate_manifest_semantics(const IndexManifest& manifest) {
  validate_digest(manifest.reference_sha256);
  if (manifest.method.empty()) {
    throw std::invalid_argument("index method must not be empty");
  }
  if (manifest.format_version != kFormatVersion) {
    throw std::invalid_argument("unsupported index format version");
  }
  if (manifest.tool_version.empty()) {
    throw std::invalid_argument("tool version must not be empty");
  }
  if (manifest.reference_length == 0 || manifest.window_length == 0 ||
      manifest.stride == 0 || manifest.number_of_windows == 0) {
    throw std::invalid_argument("reference and window dimensions must be positive");
  }
  if (manifest.parameters.size() > kMaximumParameterCount) {
    throw std::invalid_argument("too many manifest parameters");
  }
}

void validate_manifest(const IndexManifest& manifest) {
  validate_manifest_semantics(manifest);
  if (!std::isfinite(manifest.build_seconds) || manifest.build_seconds < 0.0) {
    throw std::invalid_argument("build seconds must be finite and non-negative");
  }
}

void write_exact(std::ostream& output, const void* data, std::size_t size) {
  output.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
  if (!output) {
    throw std::runtime_error("unable to write index file");
  }
}

void write_u32(std::ostream& output, uint32_t value) {
  std::array<uint8_t, 4> bytes{};
  for (unsigned index = 0; index < bytes.size(); ++index) {
    bytes[index] = static_cast<uint8_t>(value >> (index * 8U));
  }
  write_exact(output, bytes.data(), bytes.size());
}

void write_u64(std::ostream& output, uint64_t value) {
  std::array<uint8_t, 8> bytes{};
  for (unsigned index = 0; index < bytes.size(); ++index) {
    bytes[index] = static_cast<uint8_t>(value >> (index * 8U));
  }
  write_exact(output, bytes.data(), bytes.size());
}

void write_double(std::ostream& output, double value) {
  static_assert(sizeof(double) == sizeof(uint64_t),
                "binary format requires 64-bit double");
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  write_u64(output, bits);
}

void write_string(std::ostream& output, const std::string& value) {
  if (value.size() > kMaximumStringLength) {
    throw std::invalid_argument("manifest string length exceeds format limit");
  }
  write_u64(output, value.size());
  write_exact(output, value.data(), value.size());
}

class CheckedReader {
 public:
  explicit CheckedReader(const std::filesystem::path& path)
      : input_(path, std::ios::binary), remaining_(file_size(path)) {
    if (!input_) {
      throw std::runtime_error("unable to open index file: " + path.string());
    }
  }

  void read_exact(void* destination, uint64_t size, std::string_view field) {
    if (size > remaining_) {
      throw std::runtime_error("truncated index while reading " +
                               std::string(field));
    }
    input_.read(static_cast<char*>(destination),
                static_cast<std::streamsize>(size));
    if (!input_) {
      throw std::runtime_error("truncated index while reading " +
                               std::string(field));
    }
    remaining_ -= size;
  }

  uint32_t read_u32(std::string_view field) {
    std::array<uint8_t, 4> bytes{};
    read_exact(bytes.data(), bytes.size(), field);
    uint32_t value = 0;
    for (unsigned index = 0; index < bytes.size(); ++index) {
      value |= static_cast<uint32_t>(bytes[index]) << (index * 8U);
    }
    return value;
  }

  uint64_t read_u64(std::string_view field) {
    std::array<uint8_t, 8> bytes{};
    read_exact(bytes.data(), bytes.size(), field);
    uint64_t value = 0;
    for (unsigned index = 0; index < bytes.size(); ++index) {
      value |= static_cast<uint64_t>(bytes[index]) << (index * 8U);
    }
    return value;
  }

  double read_double(std::string_view field) {
    const uint64_t bits = read_u64(field);
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }

  std::string read_string(std::string_view field) {
    const uint64_t size = read_u64(field);
    if (size > kMaximumStringLength || size > remaining_) {
      throw std::runtime_error("invalid " + std::string(field) + " length");
    }
    std::string value(static_cast<std::size_t>(size), '\0');
    read_exact(value.data(), size, field);
    return value;
  }

  uint64_t remaining() const { return remaining_; }

 private:
  static uint64_t file_size(const std::filesystem::path& path) {
    std::error_code error;
    const uintmax_t size = std::filesystem::file_size(path, error);
    if (error || size > std::numeric_limits<uint64_t>::max()) {
      throw std::runtime_error("unable to determine index file size: " +
                               path.string());
    }
    return static_cast<uint64_t>(size);
  }

  std::ifstream input_;
  uint64_t remaining_;
};

void write_manifest_binary(std::ostream& output, const IndexManifest& manifest) {
  write_string(output, manifest.method);
  write_u64(output, manifest.parameters.size());
  for (const auto& parameter : manifest.parameters) {
    write_string(output, parameter.first);
    write_string(output, parameter.second);
  }
  write_string(output, manifest.reference_path);
  write_string(output, manifest.reference_sha256);
  write_u64(output, manifest.reference_length);
  write_u32(output, manifest.window_length);
  write_u32(output, manifest.stride);
  write_u64(output, manifest.number_of_windows);
  write_string(output, manifest.build_command);
  write_double(output, manifest.build_seconds);
  write_u64(output, manifest.index_bytes);
  write_string(output, manifest.created_at);
  write_string(output, manifest.git_commit);
  write_u32(output, manifest.format_version);
  write_string(output, manifest.tool_version);
}

IndexManifest read_manifest_binary(CheckedReader& input) {
  IndexManifest manifest;
  manifest.method = input.read_string("method");
  const uint64_t parameter_count = input.read_u64("parameter count");
  if (parameter_count > kMaximumParameterCount) {
    throw std::runtime_error("invalid parameter count length");
  }
  manifest.parameters.reserve(static_cast<std::size_t>(parameter_count));
  for (uint64_t index = 0; index < parameter_count; ++index) {
    std::string key = input.read_string("parameter key");
    std::string value = input.read_string("parameter value");
    manifest.parameters.emplace_back(std::move(key), std::move(value));
  }
  manifest.reference_path = input.read_string("reference path");
  manifest.reference_sha256 = input.read_string("reference SHA-256");
  manifest.reference_length = input.read_u64("reference length");
  manifest.window_length = input.read_u32("window length");
  manifest.stride = input.read_u32("stride");
  manifest.number_of_windows = input.read_u64("number of windows");
  manifest.build_command = input.read_string("build command");
  manifest.build_seconds = input.read_double("build seconds");
  manifest.index_bytes = input.read_u64("index bytes");
  manifest.created_at = input.read_string("creation time");
  manifest.git_commit = input.read_string("git commit");
  manifest.format_version = input.read_u32("manifest format version");
  manifest.tool_version = input.read_string("tool version");
  return manifest;
}

std::string json_escape(std::string_view value) {
  std::ostringstream output;
  for (unsigned char byte : value) {
    switch (byte) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (byte < 0x20U) {
          output << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                 << static_cast<unsigned>(byte) << std::dec;
        } else {
          output << static_cast<char>(byte);
        }
    }
  }
  return output.str();
}

void write_json_string(std::ostream& output, std::string_view value) {
  output << '"' << json_escape(value) << '"';
}

std::filesystem::path temporary_path_for(const std::filesystem::path& path) {
  static std::atomic<uint64_t> counter{0};
  std::random_device random;
  const uint64_t timestamp = static_cast<uint64_t>(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const std::string suffix = std::to_string(static_cast<unsigned long>(getpid())) +
                             "." + std::to_string(timestamp) + "." +
                             std::to_string(random()) + "." +
                             std::to_string(counter.fetch_add(1));
  return path.parent_path() / (path.filename().string() + ".tmp." + suffix);
}

void run_before_publish_hook_for_testing() {
  if (const auto hook = g_before_publish_hook.load(std::memory_order_acquire)) {
    hook();
  }
}

void publish_without_overwrite(const std::filesystem::path& temporary,
                               const std::filesystem::path& final_path) {
  std::error_code error;
  std::filesystem::create_hard_link(temporary, final_path, error);
  if (error) {
    if (error == std::errc::file_exists) {
      throw std::runtime_error(
          "final index appeared during write; refusing to overwrite it");
    }
    throw std::runtime_error("unable to publish index atomically: " +
                             error.message());
  }

  std::filesystem::remove(temporary, error);
  if (error) {
    throw std::runtime_error("published index but failed to remove temporary file: " +
                             error.message());
  }
}

}  // namespace

bool IndexManifest::operator==(const IndexManifest& other) const {
  return method == other.method && parameters == other.parameters &&
         reference_path == other.reference_path &&
         reference_sha256 == other.reference_sha256 &&
         reference_length == other.reference_length &&
         window_length == other.window_length && stride == other.stride &&
         number_of_windows == other.number_of_windows &&
         build_command == other.build_command &&
         build_seconds == other.build_seconds && index_bytes == other.index_bytes &&
         created_at == other.created_at && git_commit == other.git_commit &&
         format_version == other.format_version &&
         tool_version == other.tool_version;
}

bool semantically_compatible(const IndexManifest& stored,
                             const IndexManifest& expected) {
  return stored.method == expected.method &&
         stored.parameters == expected.parameters &&
         stored.reference_sha256 == expected.reference_sha256 &&
         stored.reference_length == expected.reference_length &&
         stored.window_length == expected.window_length &&
         stored.stride == expected.stride &&
         stored.number_of_windows == expected.number_of_windows &&
         stored.format_version == expected.format_version &&
         stored.tool_version == expected.tool_version;
}

void write_index_atomic(const std::filesystem::path& path,
                        const IndexManifest& manifest,
                        const std::vector<uint8_t>& payload) {
  validate_manifest(manifest);
  if (path.empty() || path.filename().empty()) {
    throw std::invalid_argument("index path must name a file");
  }

  if (std::filesystem::exists(path)) {
    try {
      read_index(path, manifest);
      return;
    } catch (const std::exception& error) {
      throw std::runtime_error("incompatible existing index; remove or rebuild " +
                               path.string() + ": " + error.what());
    }
  }

  IndexManifest stored_manifest = manifest;
  stored_manifest.index_bytes = payload.size();

  const std::filesystem::path temporary = temporary_path_for(path);
  try {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
      throw std::runtime_error("unable to create temporary index: " +
                               temporary.string());
    }
    write_exact(output, kMagic.data(), kMagic.size());
    write_u32(output, kFormatVersion);
    write_manifest_binary(output, stored_manifest);
    write_u64(output, payload.size());
    const Sha256Digest digest = sha256(payload);
    write_exact(output, digest.data(), digest.size());
    write_exact(output, payload.data(), payload.size());
    output.flush();
    if (!output) {
      throw std::runtime_error("unable to flush temporary index");
    }
    output.close();
    if (!output) {
      throw std::runtime_error("unable to close temporary index");
    }

    const PersistedIndex checked = read_index(temporary, stored_manifest);
    if (checked.payload != payload) {
      throw std::runtime_error("temporary index validation payload mismatch");
    }
    run_before_publish_hook_for_testing();
    publish_without_overwrite(temporary, path);
  } catch (...) {
    std::error_code cleanup_error;
    std::filesystem::remove(temporary, cleanup_error);
    throw;
  }
}

PersistedIndex read_index(const std::filesystem::path& path,
                          const IndexManifest& expected_semantics) {
  validate_manifest_semantics(expected_semantics);
  CheckedReader input(path);
  std::array<char, 8> magic{};
  input.read_exact(magic.data(), magic.size(), "magic");
  if (magic != kMagic) {
    throw std::runtime_error("invalid index magic");
  }
  if (input.read_u32("format version") != kFormatVersion) {
    throw std::runtime_error("unsupported index format version");
  }

  PersistedIndex index;
  index.manifest = read_manifest_binary(input);
  try {
    validate_manifest(index.manifest);
  } catch (const std::exception& error) {
    throw std::runtime_error("invalid stored manifest: " +
                             std::string(error.what()));
  }
  if (!semantically_compatible(index.manifest, expected_semantics)) {
    throw std::runtime_error("incompatible index manifest semantics");
  }

  const uint64_t payload_size = input.read_u64("payload length");
  if (payload_size > std::numeric_limits<std::size_t>::max() ||
      payload_size >
          static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()) ||
      payload_size > input.remaining() || input.remaining() - payload_size != 32) {
    throw std::runtime_error("invalid or truncated payload length");
  }
  Sha256Digest stored_digest{};
  input.read_exact(stored_digest.data(), stored_digest.size(), "payload checksum");
  index.payload.resize(static_cast<std::size_t>(payload_size));
  input.read_exact(index.payload.data(), payload_size, "payload");
  if (input.remaining() != 0) {
    throw std::runtime_error("unexpected trailing bytes after index payload");
  }
  if (index.manifest.index_bytes != payload_size) {
    throw std::runtime_error("manifest index bytes do not match payload length");
  }
  if (sha256(index.payload) != stored_digest) {
    throw std::runtime_error("payload checksum mismatch");
  }
  return index;
}

void write_manifest_json(const std::filesystem::path& path,
                         const IndexManifest& manifest) {
  validate_manifest(manifest);
  std::ofstream output(path, std::ios::trunc);
  if (!output) {
    throw std::runtime_error("unable to create manifest JSON: " + path.string());
  }
  output << "{\n  \"method\": ";
  write_json_string(output, manifest.method);
  output << ",\n  \"parameters\": [";
  for (std::size_t index = 0; index < manifest.parameters.size(); ++index) {
    output << (index == 0 ? "\n    {" : ",\n    {");
    output << "\"key\": ";
    write_json_string(output, manifest.parameters[index].first);
    output << ", \"value\": ";
    write_json_string(output, manifest.parameters[index].second);
    output << "}";
  }
  if (!manifest.parameters.empty()) {
    output << '\n';
  }
  output << "  ],\n  \"reference_path\": ";
  write_json_string(output, manifest.reference_path);
  output << ",\n  \"reference_sha256\": ";
  write_json_string(output, manifest.reference_sha256);
  output << ",\n  \"reference_length\": " << manifest.reference_length
         << ",\n  \"window_length\": " << manifest.window_length
         << ",\n  \"stride\": " << manifest.stride
         << ",\n  \"number_of_windows\": " << manifest.number_of_windows
         << ",\n  \"build_command\": ";
  write_json_string(output, manifest.build_command);
  output << ",\n  \"build_seconds\": " << std::setprecision(17)
         << manifest.build_seconds << ",\n  \"index_bytes\": " << manifest.index_bytes
         << ",\n  \"created_at\": ";
  write_json_string(output, manifest.created_at);
  output << ",\n  \"git_commit\": ";
  write_json_string(output, manifest.git_commit);
  output << ",\n  \"format_version\": " << manifest.format_version
         << ",\n  \"tool_version\": ";
  write_json_string(output, manifest.tool_version);
  output << "\n}\n";
  output.flush();
  if (!output) {
    throw std::runtime_error("unable to write manifest JSON: " + path.string());
  }
}

void set_write_index_atomic_before_publish_hook_for_testing(
    WriteIndexAtomicBeforePublishHook hook) {
  g_before_publish_hook.store(hook, std::memory_order_release);
}
