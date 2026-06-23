#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

class OccurrenceIndex {
 public:
  struct Posting {
    uint64_t key = 0;
    uint32_t position = 0;
  };

  struct DirectoryEntry {
    uint64_t key = 0;
    uint32_t begin = 0;
    uint32_t end = 0;
  };

  static OccurrenceIndex build(std::string_view sequence, uint32_t k);
  static OccurrenceIndex deserialize(const std::vector<uint8_t>& payload);

  uint32_t k() const;
  const std::vector<Posting>& postings() const;
  const std::vector<DirectoryEntry>& directory() const;
  std::vector<uint32_t> positions_for_key(uint64_t key) const;
  std::vector<uint8_t> serialize() const;

 private:
  uint32_t k_ = 0;
  std::vector<Posting> postings_;
  std::vector<DirectoryEntry> directory_;
};
