#include "map150.hpp"

#include "tools.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#ifdef NAVIGAMER_WITH_SEQAN3
#include <seqan3/alphabet/nucleotide/dna4.hpp>
#include <seqan3/search/fm_index/all.hpp>
#endif

namespace navigamer {

namespace {

std::string make_dedup_key(const std::string& query_id,
                           const std::string& ref_id,
                           const std::string& strand,
                           int start,
                           int end) {
  return query_id + "\t" + ref_id + "\t" + strand + "\t" +
         std::to_string(start) + "\t" + std::to_string(end);
}

void validate_mode(const std::string& mode) {
  if (mode != "adaptive") {
    throw std::runtime_error("map150 currently supports only --mode adaptive");
  }
}

void validate_mapper_config(const HierarchyConfig& config, int tolerance) {
  if (tolerance < 0) throw std::runtime_error("map150 --tolerance must be non-negative");
  if (config.primary_radii.empty()) {
    throw std::runtime_error("map150 requires at least one primary radius");
  }
  int finest_radius = config.primary_radii.back();
  if (finest_radius <= map150_candidate_tolerance(tolerance)) {
    throw std::runtime_error(
        "map150 requires finest primary radius > 2 * tolerance for recall-safe candidate search");
  }
}

void keep_best(std::unordered_map<std::string, Map150Result>& unique,
               Map150Result result) {
  std::string key = make_dedup_key(result.query_id, result.ref_id, result.strand,
                                   result.reference_start, result.reference_end);
  auto it = unique.find(key);
  if (it == unique.end() ||
      result.edit_distance < it->second.edit_distance ||
      (result.edit_distance == it->second.edit_distance && result.hit_id < it->second.hit_id)) {
    unique[std::move(key)] = std::move(result);
  }
}

void verify_occurrence(const std::string& read_id,
                       const std::string& original_query,
                       const std::string& oriented_query,
                       const std::string& ref_seq,
                       const BioSequence& hit,
                       const RefPosition& occurrence,
                       const std::string& strand,
                       int tolerance,
                       const SearchStats& stats,
                       std::unordered_map<std::string, Map150Result>& unique_results) {
  int ref_len = static_cast<int>(ref_seq.size());
  int pad_start = std::max(0, occurrence.start - tolerance);
  int pad_end = std::min(ref_len, occurrence.start + MAP150_READ_LEN + tolerance);
  int min_len = std::max(0, MAP150_READ_LEN - tolerance);
  int max_len = MAP150_READ_LEN + tolerance;

  for (int start = pad_start; start < pad_end; ++start) {
    for (int len = min_len; len <= max_len; ++len) {
      int end = start + len;
      if (end > pad_end) continue;
      std::string fragment =
          ref_seq.substr(static_cast<size_t>(start), static_cast<size_t>(len));
      int ed = compute_distance(oriented_query, fragment);
      if (ed > tolerance) continue;

      Map150Result result;
      result.query_id = read_id;
      result.hit_id = hit.id;
      result.ref_id = occurrence.ref_id;
      result.strand = strand;
      result.query_start = 0;
      result.reference_start = start;
      result.reference_end = end;
      result.aligned_length = len;
      result.score = len - ed;
      result.edit_distance = ed;
      result.query_fragment = original_query;
      result.reference_fragment = fragment;
      result.sa_interval = hit.bwt_interval;
      result.stats = stats;
      keep_best(unique_results, std::move(result));
    }
  }
}

SearchResult search_candidates(
    BioGeometrySearchEngine& engine,
    const BioSequence& query,
    int candidate_tolerance,
    SearchStats& stats) {
  auto [results, st] = engine.search_adaptive(query, candidate_tolerance);
  stats = st;
  return results;
}

void map_orientation(const BioSequence& read,
                     const std::string& original_query,
                     const std::string& oriented_query,
                     const std::string& strand,
                     const std::string& ref_seq,
                     int tolerance,
                     BioGeometrySearchEngine& engine,
                     const OccurrenceLocator& locator,
                     std::unordered_map<std::string, Map150Result>& unique_results) {
  BioSequence query(read.id + "/" + strand, oriented_query);
  SearchStats stats;
  auto candidates = search_candidates(
      engine, query, map150_candidate_tolerance(tolerance), stats);

  for (const auto& candidate : candidates) {
    for (const auto& occurrence : locator.locate(*candidate)) {
      verify_occurrence(read.id, original_query, oriented_query, ref_seq, *candidate,
                        occurrence, strand, tolerance, stats, unique_results);
    }
  }
}

std::string interval_start(const BwtInterval& interval) {
  return interval.valid() ? std::to_string(interval.start) : "-1";
}

std::string interval_end(const BwtInterval& interval) {
  return interval.valid() ? std::to_string(interval.end) : "-1";
}

#ifdef NAVIGAMER_WITH_SEQAN3
std::vector<seqan3::dna4> to_dna4_vector(const std::string& seq) {
  std::vector<seqan3::dna4> out;
  out.reserve(seq.size());
  for (char c : seq) out.push_back(seqan3::assign_char_to(c, seqan3::dna4{}));
  return out;
}

struct IntervalKey {
  int64_t start = -1;
  int64_t end = -1;

  bool operator==(const IntervalKey& other) const {
    return start == other.start && end == other.end;
  }
};

struct IntervalKeyHash {
  size_t operator()(const IntervalKey& key) const {
    size_t h1 = std::hash<int64_t>{}(key.start);
    size_t h2 = std::hash<int64_t>{}(key.end);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

IntervalKey interval_key(const BwtInterval& interval) {
  return {interval.start, interval.end};
}

void sort_and_deduplicate_positions(std::vector<RefPosition>& positions) {
  std::sort(positions.begin(), positions.end(), [](const auto& a, const auto& b) {
    if (a.ref_id != b.ref_id) return a.ref_id < b.ref_id;
    if (a.start != b.start) return a.start < b.start;
    if (a.end != b.end) return a.end < b.end;
    return a.strand < b.strand;
  });
  positions.erase(std::unique(positions.begin(), positions.end(), [](const auto& a, const auto& b) {
                    return a.ref_id == b.ref_id &&
                           a.start == b.start &&
                           a.end == b.end &&
                           a.strand == b.strand;
                  }),
                  positions.end());
}

class SeqanFmLocator final : public OccurrenceLocator {
 public:
  SeqanFmLocator(std::string ref_id, const std::string& ref_seq,
                 BioGeometryIndexBuilder& builder)
      : ref_id_(std::move(ref_id)),
        genome_(to_dna4_vector(ref_seq)),
        index_(genome_) {
    for (auto& seq : builder.sequence_store().records) {
      auto cursor = index_.cursor();
      if (!cursor.extend_right(to_dna4_vector(seq.seq))) {
        throw std::runtime_error("SeqAn FM-index could not locate indexed 150-mer: " + seq.id);
      }

      auto interval = cursor.suffix_array_interval();
      seq.set_sa_interval(static_cast<int64_t>(interval.begin_position),
                          static_cast<int64_t>(interval.end_position));

      std::vector<RefPosition> positions;
      for (auto const& pos : cursor.locate()) {
        if (pos.first != 0) continue;
        int start = static_cast<int>(pos.second);
        positions.push_back(
            {ref_id_, start, start + static_cast<int>(seq.seq.size()), "+"});
      }
      sort_and_deduplicate_positions(positions);
      interval_occurrences_[interval_key(seq.bwt_interval)] =
          std::move(positions);
    }
  }

  std::vector<RefPosition> locate(const BioSequence& hit) const override {
    if (!hit.bwt_interval.valid() || hit.bwt_interval.start == hit.bwt_interval.end) {
      throw std::runtime_error("SeqAn locator hit has no valid SA interval: " + hit.id);
    }
    auto it = interval_occurrences_.find(interval_key(hit.bwt_interval));
    if (it == interval_occurrences_.end()) {
      throw std::runtime_error("SeqAn locator has no cached occurrences for SA interval on hit: " +
                               hit.id);
    }
    return it->second;
  }

  std::string name() const override { return "seqan"; }

 private:
  std::string ref_id_;
  std::vector<seqan3::dna4> genome_;
  seqan3::fm_index<seqan3::dna4, seqan3::text_layout::single> index_;
  std::unordered_map<IntervalKey, std::vector<RefPosition>, IntervalKeyHash>
      interval_occurrences_;
};
#endif

}  // namespace

std::vector<RefPosition> RefPositionLocator::locate(const BioSequence& hit) const {
  return hit.ref_positions;
}

std::string RefPositionLocator::name() const {
  return "refpos";
}

int map150_candidate_tolerance(int result_tolerance) {
  if (result_tolerance < 0) {
    throw std::runtime_error("map150 tolerance must be non-negative");
  }
  return result_tolerance * 2;
}

std::string normalize_acgt_sequence(const std::string& seq, const std::string& label) {
  std::string out;
  out.reserve(seq.size());
  for (char c : seq) {
    char upper = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (upper != 'A' && upper != 'C' && upper != 'G' && upper != 'T') {
      throw std::runtime_error(label + " contains non-ACGT base");
    }
    out.push_back(upper);
  }
  return out;
}

std::string reverse_complement(const std::string& seq) {
  std::string normalized = normalize_acgt_sequence(seq, "read");
  std::string out;
  out.reserve(normalized.size());
  for (auto it = normalized.rbegin(); it != normalized.rend(); ++it) {
    switch (*it) {
      case 'A': out.push_back('T'); break;
      case 'C': out.push_back('G'); break;
      case 'G': out.push_back('C'); break;
      case 'T': out.push_back('A'); break;
    }
  }
  return out;
}

std::vector<std::shared_ptr<BioSequence>> build_map150_reference_windows(
    const std::string& ref_id,
    const std::string& ref_seq) {
  std::string normalized_ref = normalize_acgt_sequence(ref_seq, "reference");
  if (normalized_ref.size() < static_cast<size_t>(MAP150_READ_LEN)) {
    throw std::runtime_error("map150 reference must be at least 150 bp");
  }

  std::vector<std::shared_ptr<BioSequence>> windows;
  int ref_len = static_cast<int>(normalized_ref.size());
  windows.reserve(static_cast<size_t>(ref_len - MAP150_READ_LEN + 1));
  for (int start = 0; start + MAP150_READ_LEN <= ref_len; ++start) {
    std::string fragment = normalized_ref.substr(static_cast<size_t>(start), MAP150_READ_LEN);
    auto seq = std::make_shared<BioSequence>("ref_" + std::to_string(start), fragment);
    seq->add_occurrence(ref_id, start, start + MAP150_READ_LEN, "+");
    windows.push_back(seq);
  }
  return windows;
}

std::vector<Map150Result> map150_reads_with_locator(
    const std::string& ref_id,
    const std::string& ref_seq,
    const std::vector<std::shared_ptr<BioSequence>>& reads,
    int tolerance,
    const std::string& mode,
    const HierarchyConfig& config,
    const OccurrenceLocator& locator,
    const BioGeometryIndexBuilder& builder,
    const SearchConfig& search_config) {
  (void)ref_id;
  validate_mode(mode);
  validate_mapper_config(config, tolerance);
  std::string normalized_ref = normalize_acgt_sequence(ref_seq, "reference");
  if (normalized_ref.size() < static_cast<size_t>(MAP150_READ_LEN)) {
    throw std::runtime_error("map150 reference must be at least 150 bp");
  }

  BioGeometrySearchEngine engine(builder, search_config);
  std::unordered_map<std::string, Map150Result> unique_results;

  for (const auto& read : reads) {
    if (!read) continue;
    std::string normalized_read = normalize_acgt_sequence(read->seq, read->id);
    if (normalized_read.size() != static_cast<size_t>(MAP150_READ_LEN)) {
      throw std::runtime_error("map150 supports only 150 bp reads");
    }
    map_orientation(*read, normalized_read, normalized_read, "+", normalized_ref,
                    tolerance, engine, locator, unique_results);
    map_orientation(*read, normalized_read, reverse_complement(normalized_read), "-",
                    normalized_ref, tolerance, engine, locator, unique_results);
  }

  std::vector<Map150Result> out;
  out.reserve(unique_results.size());
  for (auto& entry : unique_results) out.push_back(std::move(entry.second));
  std::sort(out.begin(), out.end(), [](const Map150Result& a, const Map150Result& b) {
    if (a.query_id != b.query_id) return a.query_id < b.query_id;
    if (a.ref_id != b.ref_id) return a.ref_id < b.ref_id;
    if (a.reference_start != b.reference_start) return a.reference_start < b.reference_start;
    if (a.reference_end != b.reference_end) return a.reference_end < b.reference_end;
    return a.strand < b.strand;
  });
  return out;
}

std::vector<Map150Result> map150_reads_refpos(
    const std::string& ref_id,
    const std::string& ref_seq,
    const std::vector<std::shared_ptr<BioSequence>>& reads,
    int tolerance,
    const std::string& mode,
    const HierarchyConfig& config) {
  auto windows = build_map150_reference_windows(ref_id, ref_seq);
  BioGeometryIndexBuilder builder(config);
  builder.build(std::move(windows));
  RefPositionLocator locator;
  return map150_reads_with_locator(ref_id, ref_seq, reads, tolerance, mode,
                                   config, locator, builder);
}

std::vector<std::string> map150_tsv_columns() {
  return {
      "query_id", "hit_id", "distance", "ref_id", "strand", "query_start",
      "reference_start", "aligned_length", "score", "edit_distance",
      "query_fragment", "reference_fragment", "bwt_start", "bwt_end",
      "dist_calcs", "leaf_verify_count", "candidate_count_for_prune",
      "beacon_prune_count"};
}

std::vector<std::vector<std::string>> map150_results_to_tsv_rows(
    const std::vector<Map150Result>& results) {
  std::vector<std::vector<std::string>> rows;
  rows.reserve(results.size());
  for (const auto& r : results) {
    rows.push_back({
        r.query_id,
        r.hit_id,
        std::to_string(r.edit_distance),
        r.ref_id,
        r.strand,
        std::to_string(r.query_start),
        std::to_string(r.reference_start),
        std::to_string(r.aligned_length),
        std::to_string(r.score),
        std::to_string(r.edit_distance),
        r.query_fragment,
        r.reference_fragment,
        interval_start(r.sa_interval),
        interval_end(r.sa_interval),
        std::to_string(r.stats.dist_calc_count),
        std::to_string(r.stats.leaf_verify_count),
        std::to_string(r.stats.candidate_count_for_prune),
        std::to_string(r.stats.beacon_prune_count)});
  }
  return rows;
}

std::unique_ptr<OccurrenceLocator> make_seqan_fm_locator(
    const std::string& ref_id,
    const std::string& ref_seq,
    BioGeometryIndexBuilder& builder) {
#ifdef NAVIGAMER_WITH_SEQAN3
  return std::make_unique<SeqanFmLocator>(ref_id, ref_seq, builder);
#else
  (void)ref_id;
  (void)ref_seq;
  (void)builder;
  throw std::runtime_error("map150 --locator seqan requires building with SeqAn3 support");
#endif
}

}  // namespace navigamer
