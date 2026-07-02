#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

class ReferenceWindows {
 public:
  static ReferenceWindows from_fasta(const std::string& path,
                                     uint32_t window_length,
                                     uint32_t stride);

  uint32_t size() const;
  const std::string& contig_id() const;
  const std::string& sequence() const;
  std::string_view window(uint32_t id) const;
  uint32_t start(uint32_t id) const;
  uint32_t window_id_for_start(uint32_t start) const;
  std::vector<uint32_t> covering_window_ids(uint32_t occurrence_start,
                                            uint32_t span) const;

 private:
  ReferenceWindows(std::string contig_id, std::string sequence,
                   uint32_t window_length, uint32_t stride,
                   uint32_t window_count);

  std::string contig_id_;
  std::string sequence_;
  uint32_t window_length_;
  uint32_t stride_;
  uint32_t window_count_;
};
