#ifndef NAVIGAMER_MAP150_HPP
#define NAVIGAMER_MAP150_HPP

#include "index_builder.hpp"
#include "search_engine.hpp"
#include "structure.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace navigamer {

constexpr int MAP150_READ_LEN = 150;

struct Map150Result {
  std::string query_id;
  std::string hit_id;
  std::string ref_id;
  std::string strand = "+";
  int query_start = 0;
  int reference_start = 0;
  int reference_end = 0;
  int aligned_length = 0;
  int score = 0;
  int edit_distance = 0;
  std::string query_fragment;
  std::string reference_fragment;
  BwtInterval sa_interval;
  SearchStats stats;
};

class OccurrenceLocator {
 public:
  virtual ~OccurrenceLocator() = default;
  virtual std::vector<RefPosition> locate(const BioSequence& hit) const = 0;
  virtual std::string name() const = 0;
};

class RefPositionLocator final : public OccurrenceLocator {
 public:
  std::vector<RefPosition> locate(const BioSequence& hit) const override;
  std::string name() const override;
};

int map150_candidate_tolerance(int result_tolerance);
std::string normalize_acgt_sequence(const std::string& seq, const std::string& label);
std::string reverse_complement(const std::string& seq);

std::vector<std::shared_ptr<BioSequence>> build_map150_reference_windows(
    const std::string& ref_id,
    const std::string& ref_seq);

std::vector<Map150Result> map150_reads_with_locator(
    const std::string& ref_id,
    std::string_view ref_seq,
    const std::vector<std::shared_ptr<BioSequence>>& reads,
    int tolerance,
    const std::string& mode,
    const HierarchyConfig& config,
    const OccurrenceLocator& locator,
    const BioGeometryIndexBuilder& builder,
    const SearchConfig& search_config = SearchConfig{});

std::vector<Map150Result> map150_reads_refpos(
    const std::string& ref_id,
    const std::string& ref_seq,
    const std::vector<std::shared_ptr<BioSequence>>& reads,
    int tolerance,
    const std::string& mode,
    const HierarchyConfig& config);

std::vector<std::string> map150_tsv_columns();
std::vector<std::vector<std::string>> map150_results_to_tsv_rows(
    const std::vector<Map150Result>& results);

std::unique_ptr<OccurrenceLocator> make_seqan_fm_locator(
    const std::string& ref_id,
    const std::string& ref_seq,
    BioGeometryIndexBuilder& builder);

}  // namespace navigamer

#endif  // NAVIGAMER_MAP150_HPP
