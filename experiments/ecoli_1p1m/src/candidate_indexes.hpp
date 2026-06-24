#pragma once

#include "index_persistence.hpp"
#include "occurrence_index.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

struct SpacedMask {
  uint32_t span = 0;
  std::vector<uint8_t> bits;
};

class ContiguousIndexConfig {
 public:
  std::filesystem::path reference_path;
  uint32_t window_length = 0;
  uint32_t stride = 0;
  uint32_t k = 0;
};

class SpacedSeedIndexConfig {
 public:
  std::filesystem::path reference_path;
  uint32_t window_length = 0;
  uint32_t stride = 0;
  uint32_t weight = 0;
};

class RandstrobeIndexConfig {
 public:
  std::filesystem::path reference_path;
  uint32_t window_length = 0;
  uint32_t stride = 0;
  uint32_t strobe_length = 15;
  uint32_t w_min = 20;
  uint32_t w_max = 50;
  uint64_t seed = 0;
};

std::vector<SpacedMask> make_spaced_masks(uint32_t weight);
std::vector<uint64_t> randstrobe_composite_keys(std::string_view sequence,
                                                uint32_t strobe_length,
                                                uint32_t w_min,
                                                uint32_t w_max,
                                                uint64_t seed);

class ContiguousIndex {
 public:
  static ContiguousIndex build(const ContiguousIndexConfig& config);
  static ContiguousIndex load(const std::filesystem::path& index_path);

  void save(const std::filesystem::path& out_dir) const;
  std::vector<uint32_t> query(std::string_view query_sequence) const;

 private:
  IndexManifest manifest_;
  OccurrenceIndex occurrence_index_;
};

class SpacedSeedIndex {
 public:
  static SpacedSeedIndex build(const SpacedSeedIndexConfig& config);
  static SpacedSeedIndex load(const std::filesystem::path& index_path);

  void save(const std::filesystem::path& out_dir) const;
  std::vector<uint32_t> query(std::string_view query_sequence) const;

 private:
  IndexManifest manifest_;
  std::vector<SpacedMask> masks_;
  uint32_t weight_ = 0;
  std::vector<uint8_t> payload_;
};

class RandstrobeIndex {
 public:
  static RandstrobeIndex build(const RandstrobeIndexConfig& config);
  static RandstrobeIndex load(const std::filesystem::path& index_path);

  void save(const std::filesystem::path& out_dir) const;
  std::vector<uint32_t> query(std::string_view query_sequence) const;

 private:
  IndexManifest manifest_;
  uint32_t strobe_length_ = 15;
  uint32_t w_min_ = 20;
  uint32_t w_max_ = 50;
  uint64_t seed_ = 0;
  std::vector<uint8_t> payload_;
};
