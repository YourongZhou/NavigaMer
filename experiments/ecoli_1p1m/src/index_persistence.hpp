#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

struct IndexManifest {
  std::string method;
  std::vector<std::pair<std::string, std::string>> parameters;
  std::string reference_path;
  std::string reference_sha256;
  uint64_t reference_length = 0;
  uint32_t window_length = 0;
  uint32_t stride = 0;
  uint64_t number_of_windows = 0;
  std::string build_command;
  double build_seconds = 0.0;
  uint64_t index_bytes = 0;
  std::string created_at;
  std::string git_commit;
  uint32_t format_version = 1;
  std::string tool_version;

  bool operator==(const IndexManifest& other) const;
};

struct PersistedIndex {
  IndexManifest manifest;
  std::vector<uint8_t> payload;
};

using WriteIndexAtomicBeforePublishHook = void (*)();

// Compares only index semantics. Paths, commands, timings, byte counts,
// timestamps, and source-control provenance are intentionally excluded.
bool semantically_compatible(const IndexManifest& stored,
                             const IndexManifest& expected);

void write_index_atomic(const std::filesystem::path& path,
                        const IndexManifest& manifest,
                        const std::vector<uint8_t>& payload);
PersistedIndex read_index_file(const std::filesystem::path& path);
PersistedIndex read_index(const std::filesystem::path& path,
                          const IndexManifest& expected_semantics);
void write_manifest_json(const std::filesystem::path& path,
                         const IndexManifest& manifest);
void set_write_index_atomic_before_publish_hook_for_testing(
    WriteIndexAtomicBeforePublishHook hook);
