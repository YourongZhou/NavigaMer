#include "query_benchmark.hpp"
#include <cassert>
#include <iostream>
#include <memory>
#include <set>
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

  std::vector<std::shared_ptr<navigamer::BioSequence>> index_sequences = {
      std::make_shared<navigamer::BioSequence>("unique_a", "ACGTGCACTGAT"),
      std::make_shared<navigamer::BioSequence>("unique_b", "TGCATAGCTACG"),
      std::make_shared<navigamer::BioSequence>("low", "AAAAAAAAAAAA"),
      std::make_shared<navigamer::BioSequence>("repeat_a", "CCCCGGGGCCCC"),
      std::make_shared<navigamer::BioSequence>("repeat_b", "CCCCGGGGCCCC"),
  };
  auto first = navigamer::generate_benchmark_queries(
      index_sequences, index_sequences, 12, 1, 1234, 1);
  auto second = navigamer::generate_benchmark_queries(
      index_sequences, index_sequences, 12, 1, 1234, 1);
  assert(first.size() == 6);
  assert(first.size() == second.size());
  std::set<std::string> class_names;
  for (size_t i = 0; i < first.size(); ++i) {
    assert(first[i].query_class == second[i].query_class);
    assert(first[i].query.seq == second[i].query.seq);
    assert(first[i].brute_force_ids == second[i].brute_force_ids);
    class_names.insert(navigamer::query_class_name(first[i].query_class));
    if (first[i].query_class == navigamer::QueryClass::NoHit) {
      assert(first[i].brute_force_ids.empty());
    } else if (first[i].query_class == navigamer::QueryClass::SingleHit) {
      assert(first[i].brute_force_ids.size() == 1);
    } else if (first[i].query_class == navigamer::QueryClass::MultiHit) {
      assert(first[i].brute_force_ids.size() >= 2);
    }
  }
  assert(class_names.size() == 6);

  std::cout << "query benchmark gate tests passed\n";
  return 0;
}
