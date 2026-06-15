#include "structure.hpp"
#include <sstream>
#include <random>
#include <chrono>

namespace navigamer {

void BioSequence::add_occurrence(const std::string& ref_id, int start, int end,
                                 const std::string& strand) {
  ref_positions.push_back({ref_id, start, end, strand});
}

void BioSequence::set_sa_interval(int64_t sa_start, int64_t sa_end) {
  bwt_interval.start = sa_start;
  bwt_interval.end = sa_end;
}

void BioSequence::set_bwt_interval(int64_t bwt_start, int64_t bwt_end) {
  set_sa_interval(bwt_start, bwt_end);
}

static std::string make_node_id_extended(int expanded_layer_idx) {
  thread_local std::mt19937 gen(std::random_device{}());
  std::uniform_int_distribution<> dis(0, 15);
  const char hex[] = "0123456789abcdef";
  std::string suffix(8, '0');
  for (int i = 0; i < 8; ++i) suffix[i] = hex[dis(gen)];
  std::ostringstream os;
  os << "L" << expanded_layer_idx << "_" << suffix;
  return os.str();
}

WorldNode::WorldNode(std::shared_ptr<BioSequence> center, int r, int expanded_layer_idx)
    : node_id(make_node_id_extended(expanded_layer_idx)),
      center_ptr(std::move(center)),
      radius(r),
      expanded_layer_index(expanded_layer_idx) {}

std::string WorldNode::get_center_sequence() const {
  return center_ptr ? center_ptr->seq : std::string();
}

}  // namespace navigamer
