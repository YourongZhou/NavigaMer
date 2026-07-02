#include "index_persistence.hpp"
#include "sha256.hpp"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

class TempDirectory {
 public:
  TempDirectory() {
    std::random_device random;
    for (int attempt = 0; attempt < 64; ++attempt) {
      path_ = std::filesystem::temp_directory_path() /
              ("candidate_persistence_" + std::to_string(random()) + "_" +
               std::to_string(random()));
      std::error_code error;
      if (std::filesystem::create_directory(path_, error)) {
        return;
      }
      if (error) {
        throw std::runtime_error("unable to create temporary directory: " +
                                 error.message());
      }
    }
    throw std::runtime_error("unable to allocate temporary directory");
  }

  ~TempDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  std::filesystem::path file(std::string_view name) const {
    return path_ / name;
  }

 private:
  std::filesystem::path path_;
};

class ScopedBeforePublishHook {
 public:
  explicit ScopedBeforePublishHook(WriteIndexAtomicBeforePublishHook hook) {
    set_write_index_atomic_before_publish_hook_for_testing(hook);
  }

  ~ScopedBeforePublishHook() {
    set_write_index_atomic_before_publish_hook_for_testing(nullptr);
  }

  ScopedBeforePublishHook(const ScopedBeforePublishHook&) = delete;
  ScopedBeforePublishHook& operator=(const ScopedBeforePublishHook&) = delete;
};

template <typename Function>
void assert_throws(Function&& function, std::string_view message_substring) {
  try {
    function();
  } catch (const std::exception& error) {
    assert(std::string_view(error.what()).find(message_substring) !=
           std::string_view::npos);
    return;
  }
  throw std::runtime_error("expected exception was not thrown");
}

IndexManifest sample_manifest() {
  return IndexManifest{
      "tensorsketch",
      {{"dimension", "1024"}, {"seed", "42"}},
      "/data/ecoli\"reference.fa",
      sha256_hex("ACGTACGT"),
      8,
      4,
      1,
      5,
      "candidate_tool build --note 'line1\\nline2'",
      1.25,
      6,
      "2026-06-23T12:00:00Z",
      "abc123",
      1,
      "candidate-tool/1",
  };
}

const std::filesystem::path* g_race_target_path = nullptr;
const IndexManifest* g_race_manifest = nullptr;
const std::vector<uint8_t>* g_race_payload = nullptr;

void publish_competing_index_in_race_window() {
  if (g_race_target_path == nullptr || g_race_manifest == nullptr ||
      g_race_payload == nullptr) {
    throw std::runtime_error("race hook not configured");
  }
  set_write_index_atomic_before_publish_hook_for_testing(nullptr);
  write_index_atomic(*g_race_target_path, *g_race_manifest, *g_race_payload);
}

void test_sha256_known_vectors() {
  assert(sha256_hex("") ==
         "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  assert(sha256_hex("abc") ==
         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

void test_typed_binary_round_trip_and_semantic_metadata_distinction() {
  TempDirectory temp;
  const auto path = temp.file("index.bin");
  const IndexManifest manifest = sample_manifest();
  const std::vector<uint8_t> payload = {0, 1, 2, 127, 128, 255};

  write_index_atomic(path, manifest, payload);
  const PersistedIndex loaded = read_index(path, manifest);
  assert(loaded.manifest == manifest);
  assert(loaded.payload == payload);

  IndexManifest same_semantics = manifest;
  same_semantics.reference_path = "/mounted/elsewhere/ecoli.fa";
  same_semantics.build_command = "different command spelling";
  same_semantics.build_seconds = 99.0;
  same_semantics.index_bytes = 999;
  same_semantics.created_at = "later";
  same_semantics.git_commit = "def456";
  assert(semantically_compatible(manifest, same_semantics));
  assert(read_index(path, same_semantics).payload == payload);
}

void test_payload_checksum_corruption_is_rejected() {
  TempDirectory temp;
  const auto path = temp.file("index.bin");
  const IndexManifest manifest = sample_manifest();
  write_index_atomic(path, manifest, {10, 20, 30});

  std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
  file.seekg(-1, std::ios::end);
  char byte = 0;
  file.read(&byte, 1);
  byte ^= 0x55;
  file.seekp(-1, std::ios::end);
  file.write(&byte, 1);
  file.close();

  assert_throws([&] { read_index(path, manifest); }, "checksum");
}

void test_incompatible_expected_manifest_is_rejected() {
  TempDirectory temp;
  const auto path = temp.file("index.bin");
  const IndexManifest manifest = sample_manifest();
  write_index_atomic(path, manifest, {1, 2, 3, 4, 5, 6});

  IndexManifest incompatible = manifest;
  incompatible.parameters[0].second = "2048";
  assert(!semantically_compatible(manifest, incompatible));
  assert_throws([&] { read_index(path, incompatible); }, "incompatible");
}

void write_u32_little_endian(std::ofstream& output, uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8) {
    output.put(static_cast<char>((value >> shift) & 0xffU));
  }
}

void write_u64_little_endian(std::ofstream& output, uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8) {
    output.put(static_cast<char>((value >> shift) & 0xffU));
  }
}

void test_truncated_and_malformed_lengths_are_rejected_before_allocation() {
  TempDirectory temp;
  const IndexManifest manifest = sample_manifest();

  const auto truncated = temp.file("truncated.bin");
  write_index_atomic(truncated, manifest, {1, 2, 3, 4, 5, 6});
  std::filesystem::resize_file(truncated,
                               std::filesystem::file_size(truncated) - 2);
  assert_throws([&] { read_index(truncated, manifest); }, "truncated");

  const auto malformed = temp.file("malformed.bin");
  std::ofstream output(malformed, std::ios::binary);
  output.write("ECOLIBL1", 8);
  write_u32_little_endian(output, 1);
  write_u64_little_endian(output, UINT64_MAX);
  output.close();
  assert_throws([&] { read_index(malformed, manifest); }, "length");
}

std::size_t count_temp_files_for(const std::filesystem::path& final_path) {
  const std::string prefix = final_path.filename().string() + ".tmp.";
  std::size_t count = 0;
  for (const auto& entry :
       std::filesystem::directory_iterator(final_path.parent_path())) {
    if (entry.path().filename().string().rfind(prefix, 0) == 0) {
      ++count;
    }
  }
  return count;
}

std::size_t count_occurrences(std::string_view haystack,
                              std::string_view needle) {
  std::size_t count = 0;
  std::size_t position = 0;
  while ((position = haystack.find(needle, position)) != std::string::npos) {
    ++count;
    position += needle.size();
  }
  return count;
}

void test_atomic_write_cleanup_and_existing_index_protection() {
  TempDirectory temp;
  const auto path = temp.file("index.bin");
  IndexManifest invalid = sample_manifest();
  invalid.reference_sha256 = "not-a-digest";
  assert_throws([&] { write_index_atomic(path, invalid, {1}); }, "SHA-256");
  assert(!std::filesystem::exists(path));
  assert(count_temp_files_for(path) == 0);

  const IndexManifest manifest = sample_manifest();
  write_index_atomic(path, manifest, {1, 2, 3, 4, 5, 6});
  assert(std::filesystem::exists(path));
  assert(count_temp_files_for(path) == 0);

  IndexManifest incompatible = manifest;
  incompatible.stride = 2;
  assert_throws([&] { write_index_atomic(path, incompatible, {9}); },
                "incompatible existing index");
  assert(read_index(path, manifest).payload ==
         std::vector<uint8_t>({1, 2, 3, 4, 5, 6}));
}

void test_atomic_write_refuses_publish_if_final_index_appears_after_check() {
  TempDirectory temp;
  const auto path = temp.file("index.bin");
  const IndexManifest manifest = sample_manifest();
  IndexManifest competing_manifest = manifest;
  competing_manifest.stride = 2;
  const std::vector<uint8_t> first_payload = {1, 2, 3};
  const std::vector<uint8_t> second_payload = {9, 8, 7, 6};

  g_race_target_path = &path;
  g_race_manifest = &competing_manifest;
  g_race_payload = &second_payload;
  {
    const ScopedBeforePublishHook hook(&publish_competing_index_in_race_window);
    assert_throws(
        [&] { write_index_atomic(path, manifest, first_payload); },
        "incompatible existing index");
  }
  g_race_target_path = nullptr;
  g_race_manifest = nullptr;
  g_race_payload = nullptr;

  const PersistedIndex loaded = read_index(path, competing_manifest);
  assert(loaded.payload == second_payload);
  assert(count_temp_files_for(path) == 0);
}

void test_atomic_write_reuses_matching_index_if_final_appears_during_publish() {
  TempDirectory temp;
  const auto path = temp.file("index.bin");
  const IndexManifest manifest = sample_manifest();
  const std::vector<uint8_t> payload = {4, 5, 6, 7};

  g_race_target_path = &path;
  g_race_manifest = &manifest;
  g_race_payload = &payload;
  {
    const ScopedBeforePublishHook hook(&publish_competing_index_in_race_window);
    write_index_atomic(path, manifest, payload);
  }
  g_race_target_path = nullptr;
  g_race_manifest = nullptr;
  g_race_payload = nullptr;

  const PersistedIndex loaded = read_index(path, manifest);
  assert(loaded.payload == payload);
  assert(count_temp_files_for(path) == 0);
}

std::string shell_quote(const std::filesystem::path& path) {
  std::string quoted = "'";
  for (char byte : path.string()) {
    if (byte == '\'') {
      quoted += "'\\''";
    } else {
      quoted += byte;
    }
  }
  return quoted + "'";
}

void test_manifest_json_is_escaped_and_structured() {
  TempDirectory temp;
  const auto path = temp.file("manifest.json");
  IndexManifest manifest = sample_manifest();
  manifest.parameters = {{"dup", "first"},
                         {"quoted\"key", "line1\nline2\\tail"},
                         {"dup", "third"}};
  write_manifest_json(path, manifest);

  std::ifstream input(path);
  const std::string json((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
  assert(json.find("\"parameters\": [") != std::string::npos);
  assert(count_occurrences(json, "\"key\": \"dup\"") == 2);
  assert(json.find("\"value\": \"line1\\nline2\\\\tail\"") !=
         std::string::npos);
  const std::size_t first_dup =
      json.find("{\"key\": \"dup\", \"value\": \"first\"}");
  const std::size_t quoted =
      json.find("{\"key\": \"quoted\\\"key\", \"value\": \"line1\\nline2\\\\tail\"}");
  const std::size_t second_dup =
      json.find("{\"key\": \"dup\", \"value\": \"third\"}");
  assert(first_dup != std::string::npos);
  assert(quoted != std::string::npos);
  assert(second_dup != std::string::npos);
  assert(first_dup < quoted);
  assert(quoted < second_dup);

  const std::string command =
      "python3 -m json.tool " + shell_quote(path) + " >/dev/null";
  assert(std::system(command.c_str()) == 0);
}

}  // namespace

int main() {
  test_sha256_known_vectors();
  test_typed_binary_round_trip_and_semantic_metadata_distinction();
  test_payload_checksum_corruption_is_rejected();
  test_incompatible_expected_manifest_is_rejected();
  test_truncated_and_malformed_lengths_are_rejected_before_allocation();
  test_atomic_write_cleanup_and_existing_index_protection();
  test_atomic_write_refuses_publish_if_final_index_appears_after_check();
  test_atomic_write_reuses_matching_index_if_final_appears_during_publish();
  test_manifest_json_is_escaped_and_structured();
  std::cout << "candidate persistence tests passed\n";
  return 0;
}
