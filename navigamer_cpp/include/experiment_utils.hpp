#ifndef NAVIGAMER_EXPERIMENT_UTILS_HPP
#define NAVIGAMER_EXPERIMENT_UTILS_HPP

#include <string>
#include <vector>

namespace navigamer {

std::vector<int> generate_geometric_radius_schedule(int num_primary_layers,
                                                    int r_leaf,
                                                    double alpha);

std::string join_radius_schedule(const std::vector<int>& radii);

}  // namespace navigamer

#endif  // NAVIGAMER_EXPERIMENT_UTILS_HPP
