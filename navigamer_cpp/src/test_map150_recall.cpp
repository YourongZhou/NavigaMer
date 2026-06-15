#include "map150.hpp"
#include "tools.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const navigamer::HierarchyConfig kMapperHierarchy{{30, 15, 5}};

std::string repeated_pattern(size_t len) {
  static const std::string pattern = "ACGTGCAATGTC";
  std::string out;
  out.reserve(len);
  for (size_t i = 0; i < len; ++i) out.push_back(pattern[i % pattern.size()]);
  return out;
}

std::string mutate_base(std::string seq, size_t pos) {
  assert(pos < seq.size());
  switch (seq[pos]) {
    case 'A': seq[pos] = 'C'; break;
    case 'C': seq[pos] = 'G'; break;
    case 'G': seq[pos] = 'T'; break;
    default: seq[pos] = 'A'; break;
  }
  return seq;
}

std::vector<navigamer::Map150Result> map_one(
    const std::string& ref,
    const std::string& read,
    int tolerance = 1) {
  std::vector<std::shared_ptr<navigamer::BioSequence>> reads = {
      std::make_shared<navigamer::BioSequence>("read0", read)};
  return navigamer::map150_reads_refpos(
      "ref", ref, reads, tolerance, "adaptive", kMapperHierarchy);
}

bool has_hit(const std::vector<navigamer::Map150Result>& hits,
             int start, int end, const std::string& strand, int ed) {
  return std::any_of(hits.begin(), hits.end(), [&](const auto& hit) {
    return hit.ref_id == "ref" &&
           hit.reference_start == start &&
           hit.reference_end == end &&
           hit.strand == strand &&
           hit.edit_distance == ed;
  });
}

int min_window_distance_near(const std::string& ref,
                             const std::string& query,
                             int true_start,
                             int true_end,
                             int tolerance) {
  int best = 1000000;
  int first = std::max(0, true_start - tolerance);
  int last = std::min(static_cast<int>(ref.size()) - navigamer::MAP150_READ_LEN,
                      true_end + tolerance - navigamer::MAP150_READ_LEN);
  for (int start = first; start <= last; ++start) {
    std::string window = ref.substr(static_cast<size_t>(start),
                                    navigamer::MAP150_READ_LEN);
    best = std::min(best, navigamer::compute_distance(query, window));
  }
  return best;
}

void test_candidate_tolerance_is_twice_result_tolerance() {
  assert(navigamer::map150_candidate_tolerance(0) == 0);
  assert(navigamer::map150_candidate_tolerance(1) == 2);
  assert(navigamer::map150_candidate_tolerance(4) == 8);
}

void test_deletion_span_149_is_recovered_even_when_150mer_distance_is_two() {
  std::string query = repeated_pattern(navigamer::MAP150_READ_LEN);
  std::string span = query;
  span.erase(span.begin() + 75);
  std::string ref = "T" + span + "A";
  int true_start = 1;
  int true_end = true_start + static_cast<int>(span.size());

  assert(static_cast<int>(span.size()) == navigamer::MAP150_READ_LEN - 1);
  assert(navigamer::compute_distance(query, span) == 1);
  assert(min_window_distance_near(ref, query, true_start, true_end, 1) == 2);

  auto hits = map_one(ref, query);
  assert(has_hit(hits, true_start, true_end, "+", 1));
}

void test_insertion_span_151_is_recovered_even_when_150mer_distance_is_two() {
  std::string query = repeated_pattern(navigamer::MAP150_READ_LEN);
  std::string span = query;
  span.insert(span.begin() + 75, 'A');
  std::string ref = span;
  int true_start = 0;
  int true_end = true_start + static_cast<int>(span.size());

  assert(static_cast<int>(span.size()) == navigamer::MAP150_READ_LEN + 1);
  assert(navigamer::compute_distance(query, span) == 1);
  assert(min_window_distance_near(ref, query, true_start, true_end, 1) == 2);

  auto hits = map_one(ref, query);
  assert(has_hit(hits, true_start, true_end, "+", 1));
}

void test_substitution_duplicate_reverse_nohit_and_false_positive_filtering() {
  std::string query = repeated_pattern(navigamer::MAP150_READ_LEN);
  std::string one_sub = mutate_base(query, 42);
  std::string ref = one_sub + "GGGGGGGGGG" + one_sub;

  auto substitution_hits = map_one(ref, query);
  assert(has_hit(substitution_hits, 0, navigamer::MAP150_READ_LEN, "+", 1));
  assert(has_hit(substitution_hits, navigamer::MAP150_READ_LEN + 10,
                 2 * navigamer::MAP150_READ_LEN + 10, "+", 1));

  std::string rc_read = navigamer::reverse_complement(query);
  auto reverse_hits = map_one(query, rc_read);
  assert(has_hit(reverse_hits, 0, navigamer::MAP150_READ_LEN, "-", 0));

  std::string distant = repeated_pattern(navigamer::MAP150_READ_LEN);
  for (size_t i = 0; i < distant.size(); i += 3) distant[i] = 'T';
  assert(map_one(query, distant).empty());

  std::string two_sub = mutate_base(mutate_base(query, 50), 100);
  assert(navigamer::compute_distance(query, two_sub) == 2);
  assert(map_one(two_sub, query).empty());
}

void test_validation_rejects_wrong_read_length_and_non_acgt() {
  std::vector<std::shared_ptr<navigamer::BioSequence>> short_read = {
      std::make_shared<navigamer::BioSequence>("short", repeated_pattern(149))};
  bool saw_short = false;
  try {
    (void)navigamer::map150_reads_refpos(
        "ref", repeated_pattern(151), short_read, 1, "adaptive", kMapperHierarchy);
  } catch (const std::runtime_error&) {
    saw_short = true;
  }
  assert(saw_short);

  std::vector<std::shared_ptr<navigamer::BioSequence>> read = {
      std::make_shared<navigamer::BioSequence>("read0", repeated_pattern(150))};
  bool saw_non_acgt = false;
  try {
    (void)navigamer::map150_reads_refpos(
        "ref", repeated_pattern(150) + "N", read, 1, "adaptive", kMapperHierarchy);
  } catch (const std::runtime_error&) {
    saw_non_acgt = true;
  }
  assert(saw_non_acgt);
}

#ifdef NAVIGAMER_WITH_SEQAN3
std::vector<std::shared_ptr<navigamer::BioSequence>> build_windows_and_index(
    const std::string& ref,
    navigamer::BioGeometryIndexBuilder& builder) {
  auto windows = navigamer::build_map150_reference_windows("ref", ref);
  builder.build(windows);
  return windows;
}

std::shared_ptr<navigamer::BioSequence> find_sequence(
    const navigamer::BioGeometryIndexBuilder& builder,
    const std::string& seq) {
  for (const auto& entry : builder.unique_sequences) {
    if (entry.second && entry.second->seq == seq) return entry.second;
  }
  return nullptr;
}

void test_seqan_locator_uses_sa_interval_not_hit_sequence() {
  std::string query = repeated_pattern(navigamer::MAP150_READ_LEN);
  std::string ref = query + "GGGGGGGGGG" + query;
  navigamer::BioGeometryIndexBuilder builder(kMapperHierarchy);
  (void)build_windows_and_index(ref, builder);
  auto locator = navigamer::make_seqan_fm_locator("ref", ref, builder);

  auto indexed = find_sequence(builder, query);
  assert(indexed);
  assert(indexed->bwt_interval.valid());

  navigamer::BioSequence fake("fake", repeated_pattern(navigamer::MAP150_READ_LEN));
  fake.seq[0] = fake.seq[0] == 'A' ? 'C' : 'A';
  assert(fake.seq != query);
  fake.set_sa_interval(indexed->bwt_interval.start, indexed->bwt_interval.end);

  auto positions = locator->locate(fake);
  std::vector<int> starts;
  for (const auto& pos : positions) starts.push_back(pos.start);
  std::sort(starts.begin(), starts.end());

  std::vector<int> expected = {0, navigamer::MAP150_READ_LEN + 10};
  assert(starts == expected);
}

void test_seqan_locator_rejects_invalid_interval() {
  std::string query = repeated_pattern(navigamer::MAP150_READ_LEN);
  navigamer::BioGeometryIndexBuilder builder(kMapperHierarchy);
  (void)build_windows_and_index(query, builder);
  auto locator = navigamer::make_seqan_fm_locator("ref", query, builder);

  bool saw_invalid = false;
  try {
    navigamer::BioSequence invalid("invalid", query);
    (void)locator->locate(invalid);
  } catch (const std::runtime_error&) {
    saw_invalid = true;
  }
  assert(saw_invalid);
}
#endif

}  // namespace

int main() {
  test_candidate_tolerance_is_twice_result_tolerance();
  test_deletion_span_149_is_recovered_even_when_150mer_distance_is_two();
  test_insertion_span_151_is_recovered_even_when_150mer_distance_is_two();
  test_substitution_duplicate_reverse_nohit_and_false_positive_filtering();
  test_validation_rejects_wrong_read_length_and_non_acgt();
#ifdef NAVIGAMER_WITH_SEQAN3
  test_seqan_locator_uses_sa_interval_not_hit_sequence();
  test_seqan_locator_rejects_invalid_interval();
#endif
  std::cerr << "ALL PASSED\n";
  return 0;
}
