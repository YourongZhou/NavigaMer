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

static std::string make_node_id(int layer_level) {
  static const char* names[] = {"", "SW", "MW", "LW"};
  const char* layer_name = (layer_level >= 1 && layer_level <= 3) ? names[layer_level] : "UNK";
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 15);
  const char hex[] = "0123456789abcdef";
  std::string suffix(8, '0');
  for (int i = 0; i < 8; ++i) suffix[i] = hex[dis(gen)];
  return std::string(layer_name) + "_" + suffix;
}

WorldNode::WorldNode(std::shared_ptr<BioSequence> center, int r, int layer_level)
    : node_id(make_node_id(layer_level)),
      center_ptr(std::move(center)),
      radius(r),
      layer(layer_level) {}

std::string WorldNode::get_center_sequence() const {
  return center_ptr ? center_ptr->seq : std::string();
}

}  // namespace navigamer
