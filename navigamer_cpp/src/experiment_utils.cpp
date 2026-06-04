#include "experiment_utils.hpp"
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace navigamer {

std::vector<int> generate_geometric_radius_schedule(int num_primary_layers,
                                                    int r_leaf,
                                                    double alpha) {
  if (num_primary_layers < 2) {
    throw std::invalid_argument("num_primary_layers must be at least 2");
  }
  if (r_leaf <= 0) {
    throw std::invalid_argument("r_leaf must be positive");
  }
  if (!(alpha > 0.0 && alpha < 1.0)) {
    throw std::invalid_argument("alpha must be in (0, 1)");
  }

  std::vector<int> radii(static_cast<size_t>(num_primary_layers), r_leaf);
  for (int layer_idx = 0; layer_idx < num_primary_layers; ++layer_idx) {
    int exponent = num_primary_layers - 1 - layer_idx;
    int radius = static_cast<int>(std::llround(
        static_cast<double>(r_leaf) / std::pow(alpha, exponent)));
    radii[static_cast<size_t>(layer_idx)] = radius;
  }
  radii.back() = r_leaf;

  for (size_t i = 0; i + 1 < radii.size(); ++i) {
    if (radii[i] <= radii[i + 1]) {
      radii[i] = radii[i + 1] + 1;
    }
  }
  return radii;
}

std::string join_radius_schedule(const std::vector<int>& radii) {
  std::ostringstream os;
  for (size_t i = 0; i < radii.size(); ++i) {
    if (i) os << "|";
    os << radii[i];
  }
  return os.str();
}

}  // namespace navigamer
