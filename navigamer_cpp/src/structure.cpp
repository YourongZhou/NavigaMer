#include "structure.hpp"
#include <sstream>
#include <random>
#include <chrono>

namespace navigamer {

void BioSequence::add_occurrence(const std::string& ref_id, int start, int end,
                                 const std::string& strand) {
  ref_positions.push_back({ref_id, start, end, strand});
}

void BioSequence::set_bwt_interval(int64_t bwt_start, int64_t bwt_end) {
  bwt_interval.start = bwt_start;
  bwt_interval.end = bwt_end;
}

static std::string make_node_id_extended(int extended_tier) {
  static const char* names[] = {"LW", "I1", "MW", "I2", "SW"};
  const char* layer_name =
      (extended_tier >= 0 && extended_tier <= 4) ? names[extended_tier] : "UNK";
  thread_local std::mt19937 gen(std::random_device{}());
  std::uniform_int_distribution<> dis(0, 15);
  const char hex[] = "0123456789abcdef";
  std::string suffix(8, '0');
  for (int i = 0; i < 8; ++i) suffix[i] = hex[dis(gen)];
  return std::string(layer_name) + "_" + suffix;
}

WorldNode::WorldNode(std::shared_ptr<BioSequence> center, int r, int extended_tier)
    : node_id(make_node_id_extended(extended_tier)),
      center_ptr(std::move(center)),
      radius(r),
      layer(extended_tier) {}

std::string WorldNode::get_center_sequence() const {
  return center_ptr ? center_ptr->seq : std::string();
}

}  // namespace navigamer
