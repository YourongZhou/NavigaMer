/**
 * test_distance_bound.cpp
 *
 * Verify that search methods only return candidates whose edit distance to
 * the query is within the requested tolerance.
 *
 * Covered modes:
 *   - adaptive: exact leaf verification after hierarchical pruning
 *   - exhaustive: exact leaf verification on all admissible paths
 *   - brute_force: linear scan baseline
 *   - greedy: single-path descent followed by the same leaf verification
 */

#include "structure.hpp"
#include "index_builder.hpp"
#include "search_engine.hpp"
#include "tools.hpp"
#include <iostream>
#include <random>
#include <algorithm>
#include <cassert>
#include <string>
#include <vector>

namespace {

std::string random_dna(size_t len, std::mt19937& gen) {
  static const char bases[] = "ATCG";
  std::uniform_int_distribution<> dis(0, 3);
  std::string s;
  s.reserve(len);
  for (size_t i = 0; i < len; ++i) s += bases[dis(gen)];
  return s;
}

std::string mutate(const std::string& seq, int num_mutations, std::mt19937& gen) {
  std::string out = seq;
  if (out.empty()) return out;
  std::uniform_int_distribution<size_t> pos_dis(0, out.size() - 1);
  std::uniform_int_distribution<> base_dis(0, 3);
  static const char bases[] = "ATCG";
  for (int i = 0; i < num_mutations; ++i) {
    size_t pos = pos_dis(gen);
    char orig = out[pos];
    char c;
    do { c = bases[base_dis(gen)]; } while (c == orig);
    out[pos] = c;
  }
  return out;
}

struct TestConfig {
  std::vector<int> primary_radii;
  int r_sw;
  int r_mw;
  int r_lw;
  int tolerance;
  size_t num_seqs;
  size_t seq_len;
  size_t num_queries;
  int query_mutations;
  unsigned seed;
};

std::string config_label(const TestConfig& cfg) {
  if (!cfg.primary_radii.empty()) {
    std::string label = "primary=";
    for (size_t i = 0; i < cfg.primary_radii.size(); ++i) {
      if (i) label += ",";
      label += std::to_string(cfg.primary_radii[i]);
    }
    return label;
  }
  return "R_SW=" + std::to_string(cfg.r_sw) +
         " R_MW=" + std::to_string(cfg.r_mw) +
         " R_LW=" + std::to_string(cfg.r_lw);
}

struct ViolationInfo {
  std::string query_id;
  std::string candidate_id;
  int distance;
  int tolerance;
};

struct TestResult {
  bool passed;
  size_t total_queries;
  size_t total_candidates_checked;
  size_t violations_adaptive;
  size_t violations_exhaustive;
  size_t violations_brute_force;
  size_t violations_greedy;
  std::vector<ViolationInfo> first_violations;
};

TestResult run_test(const TestConfig& cfg) {
  std::mt19937 gen(cfg.seed);

  std::string ref = random_dna(cfg.seq_len * cfg.num_seqs, gen);
  std::vector<std::shared_ptr<navigamer::BioSequence>> seqs;
  for (size_t i = 0; i < cfg.num_seqs; ++i) {
    size_t start = i * cfg.seq_len;
    if (start + cfg.seq_len > ref.size()) break;
    std::string frag = ref.substr(start, cfg.seq_len);
    seqs.push_back(std::make_shared<navigamer::BioSequence>(
        "seq_" + std::to_string(i), frag));
  }
  for (size_t i = 0; i < cfg.num_seqs / 5; ++i) {
    seqs.push_back(std::make_shared<navigamer::BioSequence>(
        "rand_" + std::to_string(i), random_dna(cfg.seq_len, gen)));
  }

  navigamer::BioGeometryIndexBuilder builder =
      cfg.primary_radii.empty()
          ? navigamer::BioGeometryIndexBuilder(cfg.r_sw, cfg.r_mw, cfg.r_lw)
          : navigamer::BioGeometryIndexBuilder(navigamer::HierarchyConfig(cfg.primary_radii));
  builder.build(seqs);

  navigamer::BioGeometrySearchEngine engine(builder);

  std::vector<std::shared_ptr<navigamer::BioSequence>> unique_list;
  for (const auto& p : builder.unique_sequences) unique_list.push_back(p.second);

  std::vector<navigamer::BioSequence> queries;
  std::uniform_int_distribution<size_t> seq_pick(0, seqs.size() - 1);
  for (size_t i = 0; i < cfg.num_queries; ++i) {
    std::string base_seq = seqs[seq_pick(gen)]->seq;
    std::string q = mutate(base_seq, cfg.query_mutations, gen);
    queries.emplace_back("query_" + std::to_string(i), q);
  }
  for (size_t i = 0; i < cfg.num_queries / 5; ++i) {
    queries.emplace_back("qrand_" + std::to_string(i), random_dna(cfg.seq_len, gen));
  }

  TestResult result{};
  result.total_queries = queries.size();

  auto check_results = [&](const std::vector<std::shared_ptr<navigamer::BioSequence>>& candidates,
                           const navigamer::BioSequence& query, int tolerance,
                           size_t& violation_count, const std::string& method_name) {
    for (const auto& cand : candidates) {
      result.total_candidates_checked++;
      int d = navigamer::compute_distance(query.seq, cand->seq);
      if (d > tolerance) {
        violation_count++;
        if (result.first_violations.size() < 10) {
          result.first_violations.push_back({
            query.id + " [" + method_name + "]",
            cand->id, d, tolerance
          });
        }
      }
    }
  };

  for (const auto& q : queries) {
    auto [ad_res, ad_st] = engine.search_adaptive(q, cfg.tolerance);
    check_results(ad_res, q, cfg.tolerance, result.violations_adaptive, "adaptive");

    auto [ex_res, ex_st] = engine.search_exhaustive(q, cfg.tolerance);
    check_results(ex_res, q, cfg.tolerance, result.violations_exhaustive, "exhaustive");

    auto [bf_res, bf_st] = engine.search_brute_force(q, cfg.tolerance, unique_list);
    check_results(bf_res, q, cfg.tolerance, result.violations_brute_force, "brute_force");

    auto [gr_res, gr_st] = engine.search_greedy(q, cfg.tolerance);
    check_results(gr_res, q, cfg.tolerance, result.violations_greedy, "greedy");
  }

  result.passed = (result.violations_adaptive == 0 &&
                   result.violations_exhaustive == 0 &&
                   result.violations_brute_force == 0 &&
                   result.violations_greedy == 0);
  return result;
}

}  // namespace

int main() {
  std::cerr << "=== NavigaMer Distance Bound Test ===\n";
  std::cerr << "Check: every returned candidate has distance <= tolerance\n\n";

  std::vector<TestConfig> configs = {
    // Exact matching.
    {{}, 5, 15, 30,  0,  100, 20,  50, 0, 42},
    // tolerance=1
    {{}, 5, 15, 30,  1,  100, 20,  50, 1, 123},
    // tolerance=2
    {{}, 5, 15, 30,  2,  150, 20,  80, 2, 456},
    // tolerance=3
    {{}, 5, 15, 30,  3,  200, 20, 100, 3, 789},
    // Boundary case: tolerance is still below R_SW.
    {{}, 5, 15, 30,  4,  200, 20, 100, 4, 1001},
    // tolerance=5 (== R_SW)
    {{}, 5, 15, 30,  5,  150, 20,  80, 5, 2001},
    // tolerance > R_SW
    {{}, 5, 15, 30,  7,  100, 20,  50, 5, 3001},
    // Longer reads.
    {{}, 5, 15, 30,  2,  100, 50,  50, 2, 4001},
    // Larger index.
    {{}, 5, 15, 30,  2,  500, 20, 200, 2, 5001},
    // Alternate radius configuration.
    {{}, 3, 10, 20,  2,  200, 20, 100, 2, 6001},
    // Highly mutated random queries may have no results.
    {{}, 5, 15, 30,  1,  100, 20,  50, 10, 7001},
    // Two primary layers.
    {{20, 5}, 0, 0, 0,  2,  200, 20, 100, 2, 8001},
    // Four primary layers.
    {{40, 28, 18, 8}, 0, 0, 0,  2,  200, 20, 100, 2, 9001},
    // Five primary layers.
    {{50, 35, 24, 16, 8}, 0, 0, 0,  2,  200, 20, 100, 2, 10001},
  };

  int pass_count = 0;
  int fail_count = 0;

  for (size_t i = 0; i < configs.size(); ++i) {
    const auto& c = configs[i];
    std::cerr << "Test " << (i + 1) << "/" << configs.size()
              << ": " << config_label(c)
              << " tol=" << c.tolerance
              << " seqs=" << c.num_seqs << " len=" << c.seq_len
              << " queries=" << c.num_queries
              << " mutations=" << c.query_mutations
              << " ... ";

    auto result = run_test(c);

    if (result.passed) {
      std::cerr << "PASS";
      pass_count++;
    } else {
      std::cerr << "FAIL";
      fail_count++;
    }
    std::cerr << " (checked=" << result.total_candidates_checked
              << " violations: adaptive=" << result.violations_adaptive
              << " exhaustive=" << result.violations_exhaustive
              << " brute_force=" << result.violations_brute_force
              << " greedy=" << result.violations_greedy
              << ")\n";

    if (!result.first_violations.empty()) {
      std::cerr << "  First violations:\n";
      for (const auto& v : result.first_violations) {
        std::cerr << "    " << v.query_id
                  << " -> " << v.candidate_id
                  << ": dist=" << v.distance
                  << " > tolerance=" << v.tolerance << "\n";
      }
    }
  }

  std::cerr << "\n=== Summary: " << pass_count << " passed, "
            << fail_count << " failed ===\n";

  if (fail_count > 0) {
    std::cout << "FAILED\n";
    return 1;
  }
  std::cout << "ALL PASSED\n";
  return 0;
}
