#include "query_benchmark.hpp"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

int main() {
  using navigamer::compare_result_ids;
  using navigamer::nearest_rank_percentile;

  assert(nearest_rank_percentile({1.0, 2.0, 3.0, 4.0}, 0.50) == 2.0);
  assert(nearest_rank_percentile({1.0, 2.0, 3.0, 4.0}, 0.95) == 4.0);

  auto equal = compare_result_ids({"a", "b"}, {"b", "a"}, {"a", "b"});
  assert(equal.baseline_equals_optimized);
  assert(equal.baseline_no_fn);
  assert(equal.optimized_no_fn);

  auto mismatch = compare_result_ids({"a"}, {"b"}, {"a", "b"});
  assert(!mismatch.baseline_equals_optimized);
  assert((mismatch.baseline_only == std::vector<std::string>{"a"}));
  assert((mismatch.optimized_only == std::vector<std::string>{"b"}));
  assert((mismatch.brute_force_missing_from_baseline ==
          std::vector<std::string>{"b"}));
  assert((mismatch.brute_force_missing_from_optimized ==
          std::vector<std::string>{"a"}));

  std::cout << "query benchmark gate tests passed\n";
  return 0;
}
