#include "search_engine.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <omp.h>

// Keep the three dominant array-query entry points on stable instruction-cache
// boundaries even when construction-only object code changes size.
#if defined(__GNUC__) || defined(__clang__)
#define NAVIGAMER_QUERY_HOT_ALIGN __attribute__((hot, aligned(256)))
#else
#define NAVIGAMER_QUERY_HOT_ALIGN
#endif

namespace navigamer {
namespace {

constexpr size_t kMinRouterHintCandidateCount = 5;

struct ActiveMyersQueryContext {
  const std::string* query = nullptr;
  PreparedMyersPattern pattern;
};

thread_local ActiveMyersQueryContext* g_active_myers_query_context = nullptr;

class ScopedActiveMyersQueryContext {
 public:
  explicit ScopedActiveMyersQueryContext(ActiveMyersQueryContext* context)
      : previous_(g_active_myers_query_context) {
    g_active_myers_query_context = context;
  }

  ~ScopedActiveMyersQueryContext() {
    g_active_myers_query_context = previous_;
  }

 private:
  ActiveMyersQueryContext* previous_ = nullptr;
};

int compute_query_distance_with_mode(const std::string& lhs,
                                     std::string_view rhs,
                                     int tau,
                                     DistanceMode mode) {
  const auto* context = g_active_myers_query_context;
  if ((mode == DistanceMode::Myers || mode == DistanceMode::Auto) &&
      context && context->query == &lhs && context->pattern.supported) {
    return compute_distance_bounded_myers_prepared(context->pattern, rhs, tau);
  }
  return compute_distance_bounded_with_mode(lhs, rhs, tau, mode);
}

int compute_exact_distance_with_mode(const std::string& lhs,
                                     std::string_view rhs,
                                     DistanceMode mode) {
  const size_t max_length = std::max(lhs.size(), rhs.size());
  if (max_length > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error("sequence length exceeds distance backend range");
  }
  return compute_query_distance_with_mode(
      lhs, rhs, static_cast<int>(max_length), mode);
}

// One compact entry records an exact distance for the current query. A
// fixed-size, thread-local table keeps working memory independent of index
// size. Excessive collisions safely bypass the cache, and values outside the
// compact distance range do the same.
class QueryDistanceCache {
 public:
  void begin_query() {
    if (++epoch_ == 0) {
      for (auto& entry : entries_) entry.epoch = 0;
      epoch_ = 1;
    }
  }

  bool lookup(LeafId sequence_id, int* distance) const {
    if (!distance) return false;
    size_t slot = hash_slot(sequence_id);
    for (size_t probe = 0; probe < kMaxProbe; ++probe) {
      const auto& entry = entries_[slot];
      if (entry.epoch != epoch_) return false;
      if (entry.sequence_id == sequence_id) {
        const uint16_t code = entry.code;
        if (code >= kExactBase && code <= kExactLimit) {
          *distance = static_cast<int>(code - kExactBase);
          return true;
        }
        return false;
      }
      slot = (slot + 1) & kSlotMask;
    }
    return false;
  }

  void store(LeafId sequence_id, int distance) {
    if (distance < 0 || distance > 255) return;
    const uint16_t code = static_cast<uint16_t>(kExactBase + distance);

    size_t slot = hash_slot(sequence_id);
    for (size_t probe = 0; probe < kMaxProbe; ++probe) {
      auto& entry = entries_[slot];
      if (entry.epoch != epoch_) {
        entry = {sequence_id, code, epoch_};
        return;
      }
      if (entry.sequence_id == sequence_id) {
        entry.code = code;
        return;
      }
      slot = (slot + 1) & kSlotMask;
    }
  }

 private:
  struct Entry {
    LeafId sequence_id = INVALID_LEAF_ID;
    uint16_t code = 0;
    uint16_t epoch = 0;
  };
  static_assert(sizeof(Entry) == 8,
                "query distance cache entry must remain 8 bytes");
  static constexpr size_t kSlotCount = 2048;
  static constexpr size_t kSlotMask = kSlotCount - 1;
  static constexpr size_t kMaxProbe = 8;
  static constexpr uint16_t kExactBase = 1;
  static constexpr uint16_t kExactLimit = kExactBase + 255;

  static size_t hash_slot(LeafId sequence_id) {
    return (static_cast<uint64_t>(sequence_id) * 11400714819323198485ull) &
           kSlotMask;
  }

  std::array<Entry, kSlotCount> entries_{};
  uint16_t epoch_ = 0;
};

thread_local QueryDistanceCache g_query_distance_cache;

int compute_indexed_query_distance(
    const std::string& lhs,
    std::string_view rhs,
    LeafId sequence_id,
    int tau,
    DistanceMode mode,
    bool require_exact,
    bool* cache_hit = nullptr) {
  int distance = 0;
  if (sequence_id != INVALID_LEAF_ID &&
      g_query_distance_cache.lookup(
          sequence_id, &distance)) {
    if (cache_hit) *cache_hit = true;
    return distance;
  }
  if (cache_hit) *cache_hit = false;
  distance = require_exact
                 ? compute_exact_distance_with_mode(lhs, rhs, mode)
                 : compute_query_distance_with_mode(lhs, rhs, tau, mode);
  if (sequence_id != INVALID_LEAF_ID &&
      (require_exact || distance <= tau)) {
    g_query_distance_cache.store(sequence_id, distance);
  }
  return distance;
}

inline void prefetch_read(const void* ptr) {
  if (!ptr) return;
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(ptr, 0, 1);
#else
  (void)ptr;
#endif
}

template <typename T>
void prefetch_vector_data(const std::vector<T>& values) {
  if (!values.empty()) prefetch_read(values.data());
}

void prefetch_sequence(const std::shared_ptr<BioSequence>& sequence) {
  if (!sequence) return;
  prefetch_read(sequence.get());
  if (!sequence->seq.empty()) prefetch_read(sequence->seq.data());
}

void prefetch_world_node(const std::shared_ptr<WorldNode>& node) {
  if (!node) return;
  prefetch_read(node.get());
  prefetch_sequence(node->center_ptr);
  prefetch_vector_data(node->child_nodes);
  prefetch_vector_data(node->child_leaves);
  prefetch_vector_data(node->child_beacon_mbbs);
  prefetch_vector_data(node->leaf_beacon_dists);
}

struct ScopedSearchTimer {
  using Clock = std::chrono::steady_clock;

  ScopedSearchTimer(bool enabled, double* target)
      : enabled_(enabled && target != nullptr),
        target_(target),
        start_(enabled_ ? Clock::now() : Clock::time_point()) {}

  ~ScopedSearchTimer() {
    if (!enabled_) return;
    *target_ +=
        std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
  }

 private:
  bool enabled_ = false;
  double* target_ = nullptr;
  Clock::time_point start_;
};

void update_frontier_stats(SearchStats& stats, size_t frontier_size) {
  stats.frontier_total_pushed += frontier_size;
  stats.frontier_max_size = std::max(stats.frontier_max_size, frontier_size);
}

size_t max_primary_child_fanout(const BioGeometryIndexBuilder& index) {
  size_t max_fanout = 0;
  const auto& view = index.search_graph_view();
  const NodeId finest_begin =
      view.layer_begin.empty()
          ? static_cast<NodeId>(view.node_records.size())
          : view.layer_begin.back();
  for (NodeId node_id = 0; node_id < finest_begin; ++node_id) {
    max_fanout =
        std::max(max_fanout, static_cast<size_t>(view.child_count(node_id)));
  }
  return max_fanout;
}

std::string node_key(NodeId node_id) {
  return std::to_string(node_id);
}

RangeCandidateMode parse_safe_child_router_mode(const std::string& mode) {
  if (mode == "auto") return RangeCandidateMode::Auto;
  if (mode == "pigeonhole") return RangeCandidateMode::PigeonholeOnly;
  if (mode == "qgram") return RangeCandidateMode::QGramOnly;
  if (mode == "mbb") return RangeCandidateMode::FullScan;
  if (mode == "full-fallback") return RangeCandidateMode::FullScan;
  throw std::invalid_argument(
      "safe child router mode must be auto, pigeonhole, qgram, mbb, or full-fallback");
}

bool safe_child_router_mode_is_mbb(const std::string& mode) {
  return mode == "mbb";
}

double local_router_mbb_score(const std::vector<MBB>& row,
                              const std::vector<int>& query_beacon_dists,
                              size_t max_anchors) {
  const size_t dims =
      std::min({row.size(), query_beacon_dists.size(), max_anchors});
  double score = 0.0;
  for (size_t dim = 0; dim < dims; ++dim) {
    const int q = query_beacon_dists[dim];
    if (q < row[dim].min_dist) {
      score += static_cast<double>(row[dim].min_dist - q) * 1024.0;
    } else if (q > row[dim].max_dist) {
      score += static_cast<double>(q - row[dim].max_dist) * 1024.0;
    }
    score += static_cast<double>(row[dim].max_dist - row[dim].min_dist);
  }
  return score;
}

double child_world_mbb_lower_bound(const std::vector<MBB>& row,
                                   const std::vector<int>& query_beacon_dists) {
  const size_t dims = std::min(row.size(), query_beacon_dists.size());
  double lower_bound = 0.0;
  for (size_t dim = 0; dim < dims; ++dim) {
    const int q = query_beacon_dists[dim];
    if (q < row[dim].min_dist) {
      lower_bound = std::max(
          lower_bound, static_cast<double>(row[dim].min_dist - q));
    } else if (q > row[dim].max_dist) {
      lower_bound = std::max(
          lower_bound, static_cast<double>(q - row[dim].max_dist));
    }
  }
  return lower_bound;
}

double child_world_mbb_span(const std::vector<MBB>& row) {
  double span = 0.0;
  for (const auto& mbb : row) {
    span += static_cast<double>(mbb.max_dist - mbb.min_dist);
  }
  return span;
}

int reconstructed_mbb_lo(uint8_t center_dist, int child_radius) {
  return std::max(0, static_cast<int>(center_dist) - child_radius);
}

int reconstructed_mbb_hi(uint8_t center_dist, int child_radius) {
  return std::min(
      static_cast<int>(std::numeric_limits<uint8_t>::max()),
      static_cast<int>(center_dist) + child_radius);
}

struct MinimizerSignature {
  bool usable = false;
  std::vector<uint64_t> codes;
};

bool encode_dna_base(char base, uint8_t* encoded) {
  switch (base) {
    case 'A': *encoded = 0; return true;
    case 'C': *encoded = 1; return true;
    case 'G': *encoded = 2; return true;
    case 'T': *encoded = 3; return true;
    default: return false;
  }
}

MinimizerSignature compute_minimizer_signature(
    std::string_view sequence, int k, int window) {
  MinimizerSignature signature;
  if (k <= 0 || window < k || k > 32 ||
      sequence.size() < static_cast<size_t>(k)) {
    return signature;
  }

  std::vector<uint8_t> bases;
  bases.reserve(sequence.size());
  for (char base : sequence) {
    uint8_t encoded = 0;
    if (!encode_dna_base(base, &encoded)) return signature;
    bases.push_back(encoded);
  }

  const size_t k_size = static_cast<size_t>(k);
  const size_t window_size = static_cast<size_t>(window);
  const uint64_t mask =
      k == 32 ? std::numeric_limits<uint64_t>::max()
              : (uint64_t{1} << (2 * k)) - 1;
  std::vector<uint64_t> kmers;
  kmers.reserve(bases.size() - k_size + 1);
  uint64_t code = 0;
  for (size_t i = 0; i < bases.size(); ++i) {
    code = ((code << 2) | bases[i]) & mask;
    if (i + 1 >= k_size) kmers.push_back(code);
  }
  if (kmers.empty()) return signature;

  std::vector<uint64_t> minimizers;
  const size_t kmer_window =
      window_size > k_size ? window_size - k_size + 1 : size_t{1};
  if (kmers.size() <= kmer_window) {
    minimizers.push_back(*std::min_element(kmers.begin(), kmers.end()));
  } else {
    minimizers.reserve(kmers.size() - kmer_window + 1);
    for (size_t begin = 0; begin + kmer_window <= kmers.size(); ++begin) {
      const auto window_begin = kmers.begin() + static_cast<long>(begin);
      const auto window_end = window_begin + static_cast<long>(kmer_window);
      minimizers.push_back(*std::min_element(window_begin, window_end));
    }
  }
  std::sort(minimizers.begin(), minimizers.end());
  minimizers.erase(std::unique(minimizers.begin(), minimizers.end()),
                   minimizers.end());
  signature.usable = !minimizers.empty();
  signature.codes = std::move(minimizers);
  return signature;
}

size_t minimizer_overlap_count(const std::vector<uint64_t>& lhs,
                               const std::vector<uint64_t>& rhs) {
  size_t overlap = 0;
  size_t lhs_idx = 0;
  size_t rhs_idx = 0;
  while (lhs_idx < lhs.size() && rhs_idx < rhs.size()) {
    if (lhs[lhs_idx] == rhs[rhs_idx]) {
      overlap++;
      lhs_idx++;
      rhs_idx++;
    } else if (lhs[lhs_idx] < rhs[rhs_idx]) {
      lhs_idx++;
    } else {
      rhs_idx++;
    }
  }
  return overlap;
}

const QGramSignature* lookup_qgram_signature(
    const std::unordered_map<int, std::unordered_map<std::string, QGramSignature>>&
        signatures_by_q,
    int q,
    const std::string& node_id) {
  auto q_it = signatures_by_q.find(q);
  if (q_it == signatures_by_q.end()) return nullptr;
  auto sig_it = q_it->second.find(node_id);
  if (sig_it == q_it->second.end()) return nullptr;
  return &sig_it->second;
}

struct ActiveRouterHintQueryContext {
  bool enabled = false;
  const std::string* query_sequence = nullptr;
  int shared_qgram_q = 0;
  const QGramSignature* shared_qgram_signature = nullptr;
  bool router_qgram_signature_ready = false;
  QGramSignature router_qgram_signature;
  bool router_minimizer_signature_ready = false;
  MinimizerSignature router_minimizer_signature;
};

thread_local std::unordered_map<const BioGeometrySearchEngine*,
                                ActiveRouterHintQueryContext*>
    g_active_router_hint_context_by_engine;

class ScopedActiveRouterHintQueryContext {
 public:
  ScopedActiveRouterHintQueryContext(const BioGeometrySearchEngine* engine,
                                     ActiveRouterHintQueryContext* context)
      : engine_(engine) {
    if (!engine_ || !context) return;
    auto it = g_active_router_hint_context_by_engine.find(engine_);
    if (it != g_active_router_hint_context_by_engine.end()) {
      previous_ = it->second;
      had_previous_ = true;
    }
    g_active_router_hint_context_by_engine[engine_] = context;
  }

  ~ScopedActiveRouterHintQueryContext() {
    if (!engine_) return;
    if (had_previous_) {
      g_active_router_hint_context_by_engine[engine_] = previous_;
    } else {
      g_active_router_hint_context_by_engine.erase(engine_);
    }
  }

 private:
  const BioGeometrySearchEngine* engine_ = nullptr;
  ActiveRouterHintQueryContext* previous_ = nullptr;
  bool had_previous_ = false;
};

ActiveRouterHintQueryContext* active_router_hint_query_context(
    const BioGeometrySearchEngine* engine) {
  auto it = g_active_router_hint_context_by_engine.find(engine);
  return it == g_active_router_hint_context_by_engine.end() ? nullptr
                                                            : it->second;
}

const QGramSignature* get_router_qgram_signature(
    const BioGeometrySearchEngine* engine,
    const SearchConfig& config,
    SearchStats& stats) {
  auto* context = active_router_hint_query_context(engine);
  if (!context || !context->enabled || !context->query_sequence ||
      config.router_hint_qgram_q <= 0) {
    return nullptr;
  }
  if (context->shared_qgram_signature &&
      context->shared_qgram_q == config.router_hint_qgram_q) {
    return context->shared_qgram_signature;
  }
  if (!context->router_qgram_signature_ready) {
    context->router_qgram_signature =
        compute_qgram_signature(*context->query_sequence, config.router_hint_qgram_q);
    context->router_qgram_signature_ready = true;
    stats.router_qgram_signature_build_count++;
  }
  return &context->router_qgram_signature;
}

const std::vector<uint64_t>* get_router_minimizers(
    const BioGeometrySearchEngine* engine,
    const SearchConfig& config,
    SearchStats& stats) {
  auto* context = active_router_hint_query_context(engine);
  if (!context || !context->enabled || !context->query_sequence) return nullptr;
  if (!context->router_minimizer_signature_ready) {
    context->router_minimizer_signature = compute_minimizer_signature(
        *context->query_sequence, config.router_hint_minimizer_k,
        config.router_hint_minimizer_w);
    context->router_minimizer_signature_ready = true;
    stats.router_minimizer_signature_build_count++;
  }
  return context->router_minimizer_signature.usable
             ? &context->router_minimizer_signature.codes
             : nullptr;
}

struct PathReuseCacheEntry {
  std::string exact_query;
  std::string fingerprint;
  int tolerance = -1;
  bool exact_query_completed = false;
	  std::vector<LeafId> exact_verified_hits;
	  std::unordered_map<std::string, std::vector<int>> anchor_dists_by_node;
	  std::unordered_map<std::string, int> center_dist_by_node;
	  std::unordered_map<std::string, int> leaf_dist_by_id;
	  std::unordered_map<LeafId, int> leaf_dist_by_sequence_id;
	  std::unordered_map<std::string, std::vector<std::string>>
	      child_orders_by_parent;
	  std::unordered_map<std::string, std::string> contained_child_by_parent;
	};

struct ActivePathReuseContext {
  bool enabled = false;
  bool previous_exact_query_match = false;
  bool previous_fingerprint_match = false;
  bool previous_near_query_match = false;
  int previous_neighbor_edit_distance = 0;
  const PathReuseCacheEntry* previous = nullptr;
  PathReuseCacheEntry next;
};

thread_local std::unordered_map<const BioGeometrySearchEngine*, PathReuseCacheEntry>
    g_path_reuse_cache_by_engine;
thread_local std::unordered_map<const BioGeometrySearchEngine*, ActivePathReuseContext*>
    g_active_path_reuse_context_by_engine;

class ScopedActivePathReuseContext {
 public:
  ScopedActivePathReuseContext(const BioGeometrySearchEngine* engine,
                               ActivePathReuseContext* context)
      : engine_(engine) {
    if (!engine_ || !context) return;
    auto it = g_active_path_reuse_context_by_engine.find(engine_);
    if (it != g_active_path_reuse_context_by_engine.end()) {
      previous_ = it->second;
      had_previous_ = true;
    }
    g_active_path_reuse_context_by_engine[engine_] = context;
  }

  ~ScopedActivePathReuseContext() {
    if (!engine_) return;
    if (had_previous_) {
      g_active_path_reuse_context_by_engine[engine_] = previous_;
    } else {
      g_active_path_reuse_context_by_engine.erase(engine_);
    }
  }

 private:
  const BioGeometrySearchEngine* engine_ = nullptr;
  ActivePathReuseContext* previous_ = nullptr;
  bool had_previous_ = false;
};

ActivePathReuseContext* active_path_reuse_context(
    const BioGeometrySearchEngine* engine) {
  auto it = g_active_path_reuse_context_by_engine.find(engine);
  return it == g_active_path_reuse_context_by_engine.end() ? nullptr : it->second;
}

std::string build_path_reuse_fingerprint(const std::string& sequence) {
  auto signature = compute_minimizer_signature(sequence, 4, 8);
  if (!signature.usable || signature.codes.empty()) {
    return "seq:" + sequence;
  }
  std::ostringstream out;
  out << "len:" << sequence.size();
  const size_t limit = std::min<size_t>(8, signature.codes.size());
  for (size_t i = 0; i < limit; ++i) {
    out << "|" << signature.codes[i];
  }
  return out.str();
}

std::string contained_root_parent_key(int layer_id) {
  return "__root_layer:" + std::to_string(layer_id);
}

std::unordered_set<std::string> near_query_qgram_set(const std::string& sequence,
                                                     size_t q) {
  std::unordered_set<std::string> qgrams;
  if (q == 0 || sequence.size() < q) {
    qgrams.insert(sequence);
    return qgrams;
  }
  qgrams.reserve(sequence.size() - q + 1);
  for (size_t i = 0; i + q <= sequence.size(); ++i) {
    qgrams.insert(sequence.substr(i, q));
  }
  return qgrams;
}

double near_query_qgram_jaccard(const std::string& left,
                                const std::string& right) {
  constexpr size_t kNearQueryJaccardQ = 5;
  const auto left_qgrams = near_query_qgram_set(left, kNearQueryJaccardQ);
  const auto right_qgrams = near_query_qgram_set(right, kNearQueryJaccardQ);
  size_t intersection = 0;
  for (const auto& qgram : left_qgrams) {
    if (right_qgrams.find(qgram) != right_qgrams.end()) intersection++;
  }
  const size_t union_size =
      left_qgrams.size() + right_qgrams.size() - intersection;
  if (union_size == 0) return 1.0;
  return static_cast<double>(intersection) / static_cast<double>(union_size);
}

template <class Candidate, class IdOf>
bool apply_cached_rank_order(const std::vector<Candidate>& candidates,
                             const std::vector<std::string>& cached_ids,
                             IdOf id_of,
                             std::vector<Candidate>* reordered,
                             size_t* matched_count) {
  reordered->clear();
  reordered->reserve(candidates.size());
  std::unordered_map<std::string, size_t> cached_rank;
  cached_rank.reserve(cached_ids.size());
  for (size_t i = 0; i < cached_ids.size(); ++i) {
    cached_rank.emplace(cached_ids[i], i);
  }

  struct RankedCandidate {
    Candidate candidate;
    size_t cached_rank = std::numeric_limits<size_t>::max();
    size_t original_rank = 0;
  };

  std::vector<RankedCandidate> ranked;
  ranked.reserve(candidates.size());
  size_t matched = 0;
  for (size_t i = 0; i < candidates.size(); ++i) {
    RankedCandidate entry;
    entry.candidate = candidates[i];
    entry.original_rank = i;
    auto rank_it = cached_rank.find(id_of(candidates[i]));
    if (rank_it != cached_rank.end()) {
      entry.cached_rank = rank_it->second;
      matched++;
    }
    ranked.push_back(std::move(entry));
  }
  if (matched == 0) {
    *matched_count = 0;
    return false;
  }

  std::stable_sort(ranked.begin(), ranked.end(),
                   [](const RankedCandidate& left,
                      const RankedCandidate& right) {
                     if (left.cached_rank != right.cached_rank) {
                       return left.cached_rank < right.cached_rank;
                     }
                     return left.original_rank < right.original_rank;
                   });

  bool changed = false;
  for (size_t i = 0; i < ranked.size(); ++i) {
    reordered->push_back(ranked[i].candidate);
    if (!changed && id_of(candidates[i]) != id_of(ranked[i].candidate)) {
      changed = true;
    }
  }
  *matched_count = matched;
  return changed;
}

}  // namespace

const char* mbb_filter_mode_name(MBBFilterMode mode) {
  return mode == MBBFilterMode::Scan ? "scan" : "rect";
}

MBBFilterMode parse_mbb_filter_mode(const std::string& value) {
  if (value == "scan") return MBBFilterMode::Scan;
  if (value == "rect") return MBBFilterMode::RectIndex;
  throw std::invalid_argument("MBB filter mode must be scan or rect");
}

const char* visited_mode_name(VisitedMode mode) {
  return mode == VisitedMode::StringSet ? "string" : "epoch";
}

VisitedMode parse_visited_mode(const std::string& value) {
  if (value == "string") return VisitedMode::StringSet;
  if (value == "epoch") return VisitedMode::Epoch;
  throw std::invalid_argument("visited mode must be string or epoch");
}

const char* graph_view_mode_name(GraphViewMode mode) {
  return mode == GraphViewMode::Original ? "original" : "flat";
}

GraphViewMode parse_graph_view_mode(const std::string& value) {
  if (value == "original") return GraphViewMode::Original;
  if (value == "flat") return GraphViewMode::Flat;
  throw std::invalid_argument("graph view mode must be original or flat");
}

void SearchScratch::begin_query(size_t node_count) {
  if (visited_epoch.size() < node_count) {
    visited_epoch.assign(node_count, 0);
    current_epoch = 0;
  }
  if (current_epoch == std::numeric_limits<uint32_t>::max()) {
    std::fill(visited_epoch.begin(), visited_epoch.end(), 0);
    current_epoch = 1;
  } else {
    current_epoch++;
  }
  frontier.clear();
  next_frontier.clear();
  mbb_candidates.clear();
  verified_children.clear();
}

bool SearchScratch::mark_visited(NodeId id) {
  if (id >= visited_epoch.size()) {
    throw std::out_of_range("visited node id is outside scratch epoch array");
  }
  if (visited_epoch[id] == current_epoch) return false;
  visited_epoch[id] = current_epoch;
  return true;
}

bool SearchScratch::is_visited(NodeId id) const {
  if (id >= visited_epoch.size()) {
    throw std::out_of_range("visited node id is outside scratch epoch array");
  }
  return visited_epoch[id] == current_epoch;
}

BioGeometrySearchEngine::BioGeometrySearchEngine(
    const BioGeometryIndexBuilder& index, const SearchConfig& config)
    : index_(index), config_(config) {
  g_path_reuse_cache_by_engine.erase(this);
  g_active_path_reuse_context_by_engine.erase(this);

  std::vector<int> q_values;
  if (config_.search_qgram_prefilter && config_.search_qgram_q > 0) {
    q_values.push_back(config_.search_qgram_q);
  }
  if (config_.router_hint_enabled && config_.router_hint_qgram_q > 0 &&
      std::find(q_values.begin(), q_values.end(),
                config_.router_hint_qgram_q) == q_values.end()) {
    q_values.push_back(config_.router_hint_qgram_q);
  }
  if (q_values.empty() &&
      !config_.router_hint_enabled &&
      !config_.safe_child_router_enabled) {
    return;
  }

  const auto& view = index_.search_graph_view();
  const NodeId finest_begin =
      view.layer_begin.empty()
          ? static_cast<NodeId>(view.node_records.size())
          : view.layer_begin.back();
  for (NodeId node_id = 0; node_id < view.node_records.size(); ++node_id) {
    const LeafId center_id = view.center_sequence_id(node_id);
    if (center_id >= view.sequences.size()) continue;
    const std::string key = node_key(node_id);
    const std::string_view center =
        view.sequences.sequence(center_id);
    for (int q : q_values) {
      auto& signatures = world_qgram_signatures_by_q_[q];
      if (!signatures.count(key)) {
        signatures.emplace(key, compute_qgram_signature(center, q));
      }
    }
    if (config_.router_hint_enabled &&
        !world_minimizer_signatures_.count(key)) {
      auto signature = compute_minimizer_signature(
          center, config_.router_hint_minimizer_k,
          config_.router_hint_minimizer_w);
      world_minimizer_signatures_.emplace(key, std::move(signature.codes));
    }
  }

  if (config_.safe_child_router_enabled &&
      !safe_child_router_mode_is_mbb(config_.safe_child_router_mode)) {
    const auto build_start = std::chrono::steady_clock::now();
    RangeJoinConfig safe_child_config;
    safe_child_config.min_seed_len =
        std::max(1, config_.safe_child_router_min_seed_len);
    safe_child_config.max_seed_len = 20;
    safe_child_config.qgram_q =
        std::max(1, config_.router_hint_qgram_q > 0
                         ? config_.router_hint_qgram_q
                         : config_.search_qgram_q);
    safe_child_config.candidate_mode =
        parse_safe_child_router_mode(config_.safe_child_router_mode);
    safe_child_config.auto_pigeonhole_max_candidates =
        config_.safe_child_router_max_candidates;
    safe_child_config.auto_hybrid_on_large_candidates = true;

    std::vector<int> seed_lengths;
    for (int seed = safe_child_config.min_seed_len;
         seed <= safe_child_config.max_seed_len; ++seed) {
      seed_lengths.push_back(seed);
    }

    const auto& primary_radii =
        index_.hierarchy_config().primary_radii;
    for (size_t parent_layer = 0;
         parent_layer + 1 < view.layer_begin.size();
         ++parent_layer) {
      const int child_radius = primary_radii[parent_layer + 1];
      for (NodeId node_id = view.layer_begin[parent_layer];
           node_id < view.layer_end[parent_layer]; ++node_id) {
        if (view.child_count(node_id) <
            config_.safe_child_router_min_fanout) {
          continue;
        }
        ParentSafeChildRouterIndex parent_index;
        parent_index.child_count = view.child_count(node_id);
        parent_index.max_child_radius = child_radius;
        const bool uniform_sequences =
            view.sequences.fixed_sequence_length != 0;
        std::vector<RangeJoinItemView> items;
        std::vector<const char*> uniform_sequence_data;
        if (uniform_sequences) {
          uniform_sequence_data.reserve(view.child_count(node_id));
        } else {
          items.reserve(view.child_count(node_id));
        }
        const auto children = view.child_ids_for(node_id);
        const auto child_at = [&](size_t offset) {
          return children.at(static_cast<uint32_t>(offset));
        };
        bool usable = true;
        for (size_t child_idx = 0;
             child_idx < view.child_count(node_id); ++child_idx) {
          const NodeId child_id = child_at(child_idx);
          if (child_id >= view.node_records.size()) {
            usable = false;
            break;
          }
          const LeafId child_center_id =
              view.center_sequence_id(child_id);
          if (child_center_id >= view.sequences.size()) {
            usable = false;
            break;
          }
          const std::string_view sequence =
              view.sequences.sequence(child_center_id);
          if (uniform_sequences) {
            uniform_sequence_data.push_back(sequence.data());
          } else {
            items.push_back({child_idx, sequence});
          }
        }
        if (!usable ||
            (uniform_sequences
                 ? uniform_sequence_data.empty()
                 : items.empty())) {
          continue;
        }
        ParentSafeChildRouterIndex::RadiusBucket bucket;
        bucket.radius = child_radius;
        bucket.range_index = ExactRangeJoinIndex(safe_child_config);
        if (uniform_sequences) {
          bucket.range_index.build_uniform_identity_views(
              std::move(uniform_sequence_data),
              view.sequences.fixed_sequence_length);
        } else {
          bucket.range_index.build_views(std::move(items));
        }
        bucket.range_index.prepare_qgram();
        if (safe_child_config.candidate_mode != RangeCandidateMode::FullScan) {
          bucket.range_index.prepare_seed_lengths(seed_lengths);
        }
        parent_index.radius_buckets.push_back(std::move(bucket));
        parent_safe_child_router_indexes_.emplace(
            node_key(node_id), std::move(parent_index));
      }
    }
    safe_child_router_build_ms_ =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - build_start)
            .count();
  }

  if (!config_.router_hint_enabled) return;

  RangeJoinConfig router_index_config;
  router_index_config.min_seed_len = 4;
  router_index_config.max_seed_len = 12;
  router_index_config.qgram_q =
      std::max(1, config_.router_hint_qgram_q);
  router_index_config.candidate_mode = RangeCandidateMode::Auto;
  router_index_config.auto_pigeonhole_max_candidates = 256;
  router_index_config.auto_hybrid_on_large_candidates = true;

  std::vector<int> seed_lengths;
  for (int seed = router_index_config.min_seed_len;
       seed <= router_index_config.max_seed_len; ++seed) {
    seed_lengths.push_back(seed);
  }

  for (NodeId node_id = 0; node_id < finest_begin; ++node_id) {
    if (view.child_count(node_id) < 2) continue;
    ParentRouterHintIndex parent_index{
        ExactRangeJoinIndex(router_index_config), {}};
    const bool uniform_sequences =
        view.sequences.fixed_sequence_length != 0;
    std::vector<RangeJoinItemView> items;
    std::vector<const char*> uniform_sequence_data;
    if (uniform_sequences) {
      uniform_sequence_data.reserve(view.child_count(node_id));
    } else {
      items.reserve(view.child_count(node_id));
    }
    const auto children = view.child_ids_for(node_id);
    const auto child_at = [&](size_t offset) {
      return children.at(static_cast<uint32_t>(offset));
    };
    bool usable = true;
    for (size_t child_idx = 0;
         child_idx < view.child_count(node_id); ++child_idx) {
      const NodeId child_id = child_at(child_idx);
      if (child_id >= view.node_records.size()) {
        usable = false;
        break;
      }
      const LeafId child_center_id =
          view.center_sequence_id(child_id);
      if (child_center_id >= view.sequences.size()) {
        usable = false;
        break;
      }
      parent_index.child_item_ids_by_node_id.emplace(
          node_key(child_id), child_idx);
      const std::string_view sequence =
          view.sequences.sequence(child_center_id);
      if (uniform_sequences) {
        uniform_sequence_data.push_back(sequence.data());
      } else {
        items.push_back({child_idx, sequence});
      }
    }
    const size_t item_count = uniform_sequences
                                  ? uniform_sequence_data.size()
                                  : items.size();
    if (!usable || item_count < 2) continue;
    if (uniform_sequences) {
      parent_index.range_index.build_uniform_identity_views(
          std::move(uniform_sequence_data),
          view.sequences.fixed_sequence_length);
    } else {
      parent_index.range_index.build_views(std::move(items));
    }
    parent_index.range_index.prepare_qgram();
    parent_index.range_index.prepare_seed_lengths(seed_lengths);
    parent_router_hint_indexes_.emplace(
        node_key(node_id), std::move(parent_index));
  }
}

bool BioGeometrySearchEngine::mbb_prunable_row(const std::vector<MBB>& row,
                                               const std::vector<int>& V_Q,
                                               int tolerance) const {
  if (row.size() != V_Q.size()) return false;
  for (size_t i = 0; i < V_Q.size(); ++i) {
    int q_b = V_Q[i];
    if (q_b < row[i].min_dist - tolerance || q_b > row[i].max_dist + tolerance) {
      return true;
    }
  }
  return false;
}

bool BioGeometrySearchEngine::leaf_beacon_prunable_row(const std::vector<int>& row,
                                                       const std::vector<int>& V_Q,
                                                       int tolerance) const {
  if (row.size() != V_Q.size()) return false;
  for (size_t i = 0; i < V_Q.size(); ++i) {
    if (std::abs(V_Q[i] - row[i]) > tolerance) return true;
  }
  return false;
}

std::vector<int> BioGeometrySearchEngine::compute_query_beacon_distances(
    const std::shared_ptr<WorldNode>& node,
    const BioSequence& query_seq,
    SearchStats& stats) const {
  if (config_.proximal_oracle_enabled && node) {
    stats.proximal_actual_anchor_node_ids.push_back(node->node_id);
  }
  if (config_.path_reuse_enabled) {
    if (auto* reuse = active_path_reuse_context(this);
        reuse && reuse->enabled && reuse->previous_exact_query_match &&
        reuse->previous) {
      ScopedSearchTimer timer(stats.query_profile_enabled, &stats.path_reuse_ms);
      stats.path_reuse_attempt_count++;
      auto cached_it = reuse->previous->anchor_dists_by_node.find(node->node_id);
      if (cached_it != reuse->previous->anchor_dists_by_node.end() &&
          cached_it->second.size() == node->beacons.size()) {
        stats.path_reuse_hit_count++;
        stats.anchor_cache_hit_count++;
        reuse->next.anchor_dists_by_node[node->node_id] = cached_it->second;
        return cached_it->second;
      }
    }
  }
  ScopedSearchTimer timer(stats.query_profile_enabled, &stats.anchor_distance_ms);
  std::vector<int> dists;
  dists.reserve(node->beacons.size());
  for (const auto& beacon : node->beacons) {
    if (!beacon) {
      dists.push_back(0);
      continue;
    }
    dists.push_back(compute_exact_distance_with_mode(
        query_seq.seq, beacon->seq, config_.distance_mode));
    stats.anchor_distance_count++;
    stats.dist_calc_count++;
  }
  if (config_.path_reuse_enabled) {
    if (auto* reuse = active_path_reuse_context(this); reuse && reuse->enabled) {
      reuse->next.anchor_dists_by_node[node->node_id] = dists;
    }
  }
  return dists;
}

NAVIGAMER_QUERY_HOT_ALIGN
std::vector<int> BioGeometrySearchEngine::compute_query_beacon_distances_view(
    NodeId node_id,
    const BioSequence& query_seq,
    SearchStats& stats) const {
  const auto& view = index_.search_graph_view();
  if (node_id >= view.node_records.size()) {
    throw std::out_of_range("array node id is outside index");
  }
  const auto node = view.node_records[node_id];
  const uint32_t beacon_count = view.beacon_count(node);
  std::string key;
  if (config_.proximal_oracle_enabled || config_.path_reuse_enabled) {
    key = node_key(node_id);
  }
  if (config_.proximal_oracle_enabled) {
    stats.proximal_actual_anchor_node_ids.push_back(key);
  }
  if (config_.path_reuse_enabled) {
    if (auto* reuse = active_path_reuse_context(this);
        reuse && reuse->enabled && reuse->previous_exact_query_match &&
        reuse->previous) {
      ScopedSearchTimer timer(stats.query_profile_enabled, &stats.path_reuse_ms);
      stats.path_reuse_attempt_count++;
      auto cached_it = reuse->previous->anchor_dists_by_node.find(key);
      if (cached_it != reuse->previous->anchor_dists_by_node.end() &&
          cached_it->second.size() == beacon_count) {
        stats.path_reuse_hit_count++;
        stats.anchor_cache_hit_count++;
        reuse->next.anchor_dists_by_node[key] = cached_it->second;
        return cached_it->second;
      }
    }
  }

  ScopedSearchTimer timer(stats.query_profile_enabled, &stats.anchor_distance_ms);
  std::vector<int> dists;
  dists.reserve(beacon_count);
  const auto measure_beacon = [&](LeafId beacon_id) {
    if (beacon_id >= view.sequences.size()) {
      throw std::runtime_error("array beacon id has no sequence");
    }
    bool cache_hit = false;
    dists.push_back(compute_indexed_query_distance(
        query_seq.seq, view.sequences.sequence(beacon_id), beacon_id,
        0,
        config_.distance_mode, true, &cache_hit));
    stats.anchor_distance_count++;
    if (!cache_hit) {
      stats.dist_calc_count++;
    }
  };
  const uint32_t beacon_begin =
      node.beacon_storage() ==
              WorldNodeRecord::BeaconStorage::ImplicitCenter
          ? 0
          : view.beacon_begin(node_id);
  const LeafId center_id = view.center_sequence_id(node_id);
  switch (node.beacon_storage()) {
    case WorldNodeRecord::BeaconStorage::Delta8: {
      const int8_t* deltas =
          view.beacon_deltas8.data() + beacon_begin;
      for (uint32_t offset = 0; offset < beacon_count; ++offset) {
        measure_beacon(static_cast<LeafId>(
            static_cast<int64_t>(center_id) +
            deltas[offset]));
      }
      break;
    }
    case WorldNodeRecord::BeaconStorage::Delta16: {
      const int16_t* deltas =
          view.beacon_deltas16.data() + beacon_begin;
      for (uint32_t offset = 0; offset < beacon_count; ++offset) {
        measure_beacon(static_cast<LeafId>(
            static_cast<int64_t>(center_id) +
            deltas[offset]));
      }
      break;
    }
    case WorldNodeRecord::BeaconStorage::Absolute32: {
      const LeafId* beacon_ids =
          view.beacon_ids32.data() + beacon_begin;
      for (uint32_t offset = 0; offset < beacon_count; ++offset) {
        measure_beacon(beacon_ids[offset]);
      }
      break;
    }
    case WorldNodeRecord::BeaconStorage::ImplicitCenter:
      measure_beacon(center_id);
      break;
  }
  if (config_.path_reuse_enabled) {
    if (auto* reuse = active_path_reuse_context(this); reuse && reuse->enabled) {
      reuse->next.anchor_dists_by_node[key] = dists;
    }
  }
  return dists;
}

std::vector<std::shared_ptr<WorldNode>>
BioGeometrySearchEngine::scan_mbb_surviving_children(
    const std::shared_ptr<WorldNode>& node,
    const std::vector<int>& query_beacon_dists,
    int tolerance,
    SearchStats& stats) const {
  ScopedSearchTimer timer(stats.query_profile_enabled, &stats.mbb_filter_ms);
  std::vector<std::shared_ptr<WorldNode>> surviving;
  const auto& children = node->child_nodes;
  stats.child_edge_considered_count += children.size();
  bool mbb_ok =
      !query_beacon_dists.empty() &&
      query_beacon_dists.size() == node->beacons.size() &&
      node->child_beacon_mbbs.size() == children.size();
  if (mbb_ok) {
    for (const auto& row : node->child_beacon_mbbs) {
      if (row.size() != query_beacon_dists.size()) {
        mbb_ok = false;
        break;
      }
    }
  }

  if (!mbb_ok) {
    surviving = children;
	  } else {
	    surviving.reserve(children.size());
	    for (size_t child_idx = 0; child_idx < children.size(); ++child_idx) {
	      if (config_.search_prefetch && child_idx + 1 < children.size()) {
	        prefetch_world_node(children[child_idx + 1]);
	        prefetch_vector_data(node->child_beacon_mbbs[child_idx + 1]);
	      }
	      stats.edge_access_count++;
	      stats.mbb_check_count++;
      stats.candidate_count_for_prune++;
      stats.bound_check_count++;
      stats.mbb_scan_child_checks++;
      stats.mbb_scalar_checks++;
      if (mbb_prunable_row(
              node->child_beacon_mbbs[child_idx], query_beacon_dists, tolerance)) {
        stats.beacon_prune_count++;
        stats.child_mbb_pruned_count++;
        continue;
      }
      surviving.push_back(children[child_idx]);
    }
  }
  stats.mbb_surviving_child_count += surviving.size();
  return surviving;
}

std::vector<std::shared_ptr<WorldNode>>
BioGeometrySearchEngine::get_mbb_surviving_children(
    const std::shared_ptr<WorldNode>& node,
    const std::vector<int>& query_beacon_dists,
    int tolerance,
    SearchStats& stats) const {
  stats.mbb_filter_parent_count++;
  if (config_.mbb_filter_mode == MBBFilterMode::Scan) {
    return scan_mbb_surviving_children(node, query_beacon_dists, tolerance, stats);
  }

  const auto& children = node->child_nodes;
  bool index_ok =
      node->mbb_rect_index &&
      node->mbb_rect_index->size() == children.size() &&
      node->mbb_rect_index->dim() == query_beacon_dists.size() &&
      !query_beacon_dists.empty() &&
      node->child_beacon_mbbs.size() == children.size();
  if (index_ok) {
    for (const auto& row : node->child_beacon_mbbs) {
      if (row.size() != query_beacon_dists.size()) {
        index_ok = false;
        break;
      }
    }
  }
  if (!index_ok) {
    stats.mbb_rect_fallback_count++;
    return scan_mbb_surviving_children(node, query_beacon_dists, tolerance, stats);
  }

  try {
    ScopedSearchTimer timer(stats.query_profile_enabled, &stats.mbb_filter_ms);
    std::vector<int> q_lo;
    std::vector<int> q_hi;
    q_lo.reserve(query_beacon_dists.size());
    q_hi.reserve(query_beacon_dists.size());
    for (int distance : query_beacon_dists) {
      q_lo.push_back(distance - tolerance);
      q_hi.push_back(distance + tolerance);
    }

    stats.mbb_rect_index_queries++;
    auto child_ids = node->mbb_rect_index->query_intersect(q_lo, q_hi);
    std::vector<bool> seen(children.size(), false);
    std::vector<std::shared_ptr<WorldNode>> surviving;
    surviving.reserve(child_ids.size());
    for (uint32_t child_id : child_ids) {
      if (child_id >= children.size() || seen[child_id]) {
        stats.mbb_rect_fallback_count++;
        return scan_mbb_surviving_children(node, query_beacon_dists, tolerance, stats);
      }
      seen[child_id] = true;
      surviving.push_back(children[child_id]);
    }

    stats.edge_access_count += children.size();
    stats.mbb_check_count += children.size();
    stats.candidate_count_for_prune += children.size();
    stats.bound_check_count += children.size();
    stats.beacon_prune_count += children.size() - surviving.size();
    stats.child_edge_considered_count += children.size();
    stats.child_mbb_pruned_count += children.size() - surviving.size();
    stats.mbb_rect_candidate_children += surviving.size();
    stats.mbb_surviving_child_count += surviving.size();
    return surviving;
  } catch (...) {
    stats.mbb_rect_fallback_count++;
    return scan_mbb_surviving_children(node, query_beacon_dists, tolerance, stats);
  }
}

std::vector<uint32_t> BioGeometrySearchEngine::safe_child_router_candidate_indices(
    const std::shared_ptr<WorldNode>& node,
    const BioSequence& query_seq,
    const std::vector<int>& query_beacon_dists,
    int tolerance,
    SearchStats& stats,
    bool* used_router) const {
  if (used_router) *used_router = false;
  if (stats.planner_disable_router_stack) return {};
  if (!config_.safe_child_router_enabled || !node) return {};
  const size_t child_count = node->child_nodes.size();
  if (child_count < config_.safe_child_router_min_fanout) {
    stats.safe_child_router_skipped_low_fanout_count++;
    return {};
  }

  auto mbb_router_ready = [&]() -> bool {
    if (!node || query_beacon_dists.empty() ||
        query_beacon_dists.size() != node->beacons.size() ||
        node->child_beacon_mbbs.size() != child_count) {
      return false;
    }
    for (const auto& row : node->child_beacon_mbbs) {
      if (row.size() != query_beacon_dists.size()) return false;
    }
    return true;
  };

  auto accept_mbb_candidates = [&]() -> std::vector<uint32_t> {
    if (!mbb_router_ready()) {
      stats.children_actually_processed += child_count;
      stats.safe_child_router_fallback_count++;
      return {};
    }

    std::vector<uint32_t> candidates;
    candidates.reserve(child_count);
    bool used_rect_index = false;
    const bool rect_index_ready =
        node->mbb_rect_index && node->mbb_rect_index->size() == child_count &&
        node->mbb_rect_index->dim() == query_beacon_dists.size();
    if (rect_index_ready) {
      try {
        std::vector<int> q_lo;
        std::vector<int> q_hi;
        q_lo.reserve(query_beacon_dists.size());
        q_hi.reserve(query_beacon_dists.size());
        for (int distance : query_beacon_dists) {
          q_lo.push_back(distance - tolerance);
          q_hi.push_back(distance + tolerance);
        }
        auto rect_candidates = node->mbb_rect_index->query_intersect(q_lo, q_hi);
        std::vector<uint8_t> seen(child_count, 0);
        candidates.reserve(rect_candidates.size());
        for (uint32_t child_idx : rect_candidates) {
          if (child_idx >= child_count || seen[child_idx]) {
            candidates.clear();
            break;
          }
          seen[child_idx] = 1;
          candidates.push_back(child_idx);
        }
        used_rect_index = !candidates.empty() || rect_candidates.empty();
      } catch (...) {
        candidates.clear();
      }
    }
    if (!used_rect_index) {
      candidates.clear();
      for (size_t child_idx = 0; child_idx < child_count; ++child_idx) {
        const auto& row = node->child_beacon_mbbs[child_idx];
        if (!mbb_prunable_row(row, query_beacon_dists, tolerance)) {
          candidates.push_back(static_cast<uint32_t>(child_idx));
        }
      }
    }

    const double ratio = child_count == 0
                             ? 1.0
                             : static_cast<double>(candidates.size()) /
                                   static_cast<double>(child_count);
    if (candidates.size() > config_.safe_child_router_max_candidates ||
        ratio > config_.safe_child_router_max_ratio) {
      stats.children_actually_processed += child_count;
      stats.safe_child_router_fallback_count++;
      return {};
    }

    if (used_router) *used_router = true;
    stats.safe_child_router_candidate_count += candidates.size();
    stats.safe_router_candidate_count += candidates.size();
    stats.safe_child_router_candidate_ratio_sum += ratio;
    stats.candidate_ratio_to_all_children += ratio;
    stats.safe_child_router_pruned_by_not_candidate_count +=
        child_count - candidates.size();
    stats.children_actually_processed += candidates.size();
    stats.center_checks_saved += child_count - candidates.size();
    return candidates;
  };

  if (safe_child_router_mode_is_mbb(config_.safe_child_router_mode)) {
    ScopedSearchTimer timer(stats.query_profile_enabled,
                            &stats.safe_child_router_query_ms);
    stats.safe_child_router_invoked_count++;
    stats.child_count_before_router += child_count;
    return accept_mbb_candidates();
  }

  if (config_.safe_child_router_mode == "auto" && mbb_router_ready()) {
    ScopedSearchTimer timer(stats.query_profile_enabled,
                            &stats.safe_child_router_query_ms);
    stats.safe_child_router_invoked_count++;
    stats.child_count_before_router += child_count;
    return accept_mbb_candidates();
  }

  auto index_it = parent_safe_child_router_indexes_.find(node->node_id);
  if (index_it == parent_safe_child_router_indexes_.end() ||
      index_it->second.child_count != child_count ||
      index_it->second.radius_buckets.empty() ||
      config_.safe_child_router_mode == "full-fallback") {
    stats.child_count_before_router += child_count;
    stats.children_actually_processed += child_count;
    stats.safe_child_router_fallback_count++;
    return {};
  }

  ScopedSearchTimer timer(stats.query_profile_enabled,
                          &stats.safe_child_router_query_ms);
  stats.safe_child_router_invoked_count++;
  stats.child_count_before_router += child_count;
  RangeJoinQueryWorkspace workspace;
  std::vector<uint32_t> candidates;
  for (const auto& bucket : index_it->second.radius_buckets) {
    const int tau = tolerance + bucket.radius;
    auto result = bucket.range_index.query(query_seq.seq, tau, &workspace);
    if (result.used_full_scan) {
      return accept_mbb_candidates();
    }
    candidates.insert(candidates.end(), result.candidate_item_ids.begin(),
                      result.candidate_item_ids.end());
  }
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  candidates.erase(
      std::remove_if(candidates.begin(), candidates.end(),
                     [child_count](uint32_t idx) { return idx >= child_count; }),
      candidates.end());

  const double ratio = child_count == 0
                           ? 1.0
                           : static_cast<double>(candidates.size()) /
                                 static_cast<double>(child_count);
  if (candidates.size() > config_.safe_child_router_max_candidates ||
      ratio > config_.safe_child_router_max_ratio) {
    return accept_mbb_candidates();
  }

  std::vector<uint8_t> in_candidate(child_count, 0);
  for (size_t child_idx : candidates) in_candidate[child_idx] = 1;
  if (config_.safe_child_router_validate) {
    for (size_t child_idx = 0; child_idx < child_count; ++child_idx) {
      const auto& child = node->child_nodes[child_idx];
      if (!child || !child->center_ptr) {
        stats.children_actually_processed += child_count;
        stats.safe_child_router_fallback_count++;
        return {};
      }
      const int child_tau = tolerance + child->radius;
      stats.safe_child_router_exact_verify_count++;
      const int dist = compute_query_distance_with_mode(
          query_seq.seq, child->center_ptr->seq, child_tau,
          config_.distance_mode);
      if (dist <= child_tau && !in_candidate[child_idx]) {
        throw std::runtime_error(
            "safe child router missed a possible child candidate");
      }
    }
  }

  if (used_router) *used_router = true;
  stats.safe_child_router_candidate_count += candidates.size();
  stats.safe_router_candidate_count += candidates.size();
  stats.safe_child_router_candidate_ratio_sum += ratio;
  stats.candidate_ratio_to_all_children += ratio;
  stats.safe_child_router_pruned_by_not_candidate_count +=
      child_count - candidates.size();
  stats.children_actually_processed += candidates.size();
  stats.center_checks_saved += child_count - candidates.size();
  return candidates;
}

std::vector<uint32_t>
BioGeometrySearchEngine::safe_child_router_candidate_indices_view(
    NodeId node_id,
    const BioSequence& query_seq,
    const std::vector<int>& query_beacon_dists,
    int child_radius,
    int tolerance,
    SearchStats& stats,
    bool* used_router) const {
  if (used_router) *used_router = false;
  if (stats.planner_disable_router_stack ||
      !config_.safe_child_router_enabled) {
    return {};
  }

  const auto& view = index_.search_graph_view();
  if (node_id >= view.node_records.size()) {
    throw std::out_of_range("array node id is outside index");
  }
  const auto node = view.node_records[node_id];
  const uint32_t child_count = view.link_count(node);
  const uint32_t beacon_count = view.beacon_count(node);
  const auto children = view.child_ids_for(node_id, node);
  const auto child_at = [&](uint32_t offset) {
    return children.at(offset);
  };
  if (child_count < config_.safe_child_router_min_fanout) {
    stats.safe_child_router_skipped_low_fanout_count++;
    return {};
  }

  const bool mbb_ready =
      !query_beacon_dists.empty() &&
      query_beacon_dists.size() == beacon_count &&
      view.child_mbb_range_valid(
          node_id, node, child_count, beacon_count);

  auto accept_mbb_candidates = [&]() -> std::vector<uint32_t> {
    if (!mbb_ready) {
      stats.children_actually_processed += child_count;
      stats.safe_child_router_fallback_count++;
      return {};
    }
    std::vector<uint32_t> candidates;
    candidates.reserve(child_count);
    const uint32_t mbb_bits = view.child_mbb_bits(node_id, node);
    const uint32_t mbb_begin = node.child_mbb_begin();
    const uint32_t quantization_bin_width =
        view.child_mbb_bin_width(node_id);
    const uint32_t quantization_error =
        SearchGraphView::child_mbb_quantization_error(
            quantization_bin_width);
    for (size_t child_idx = 0; child_idx < child_count; ++child_idx) {
      bool prunable = false;
      for (size_t dim = 0; dim < beacon_count; ++dim) {
        const size_t cell = dim * child_count + child_idx;
        const int q = query_beacon_dists[dim];
        const int center_dist =
            view.child_beacon_distance_unchecked(
                mbb_begin, child_count, beacon_count, cell, mbb_bits,
                quantization_bin_width);
        if (std::abs(
                static_cast<int64_t>(q) - center_dist) >
            static_cast<int64_t>(child_radius) + tolerance +
                quantization_error) {
          prunable = true;
          break;
        }
      }
      if (!prunable) {
        candidates.push_back(static_cast<uint32_t>(child_idx));
      }
    }

    const double ratio =
        child_count == 0
            ? 1.0
            : static_cast<double>(candidates.size()) /
                  static_cast<double>(child_count);
    if (candidates.size() > config_.safe_child_router_max_candidates ||
        ratio > config_.safe_child_router_max_ratio) {
      stats.children_actually_processed += child_count;
      stats.safe_child_router_fallback_count++;
      return {};
    }
    if (used_router) *used_router = true;
    stats.safe_child_router_candidate_count += candidates.size();
    stats.safe_router_candidate_count += candidates.size();
    stats.safe_child_router_candidate_ratio_sum += ratio;
    stats.candidate_ratio_to_all_children += ratio;
    stats.safe_child_router_pruned_by_not_candidate_count +=
        child_count - candidates.size();
    stats.children_actually_processed += candidates.size();
    stats.center_checks_saved += child_count - candidates.size();
    return candidates;
  };

  if (safe_child_router_mode_is_mbb(config_.safe_child_router_mode) ||
      (config_.safe_child_router_mode == "auto" && mbb_ready)) {
    ScopedSearchTimer timer(stats.query_profile_enabled,
                            &stats.safe_child_router_query_ms);
    stats.safe_child_router_invoked_count++;
    stats.child_count_before_router += child_count;
    return accept_mbb_candidates();
  }

  auto index_it =
      parent_safe_child_router_indexes_.find(node_key(node_id));
  if (index_it == parent_safe_child_router_indexes_.end() ||
      index_it->second.child_count != child_count ||
      index_it->second.radius_buckets.empty() ||
      config_.safe_child_router_mode == "full-fallback") {
    stats.child_count_before_router += child_count;
    stats.children_actually_processed += child_count;
    stats.safe_child_router_fallback_count++;
    return {};
  }

  ScopedSearchTimer timer(stats.query_profile_enabled,
                          &stats.safe_child_router_query_ms);
  stats.safe_child_router_invoked_count++;
  stats.child_count_before_router += child_count;
  RangeJoinQueryWorkspace workspace;
  std::vector<uint32_t> candidates;
  for (const auto& bucket : index_it->second.radius_buckets) {
    const int tau = tolerance + bucket.radius;
    auto result = bucket.range_index.query(query_seq.seq, tau, &workspace);
    if (result.used_full_scan) return accept_mbb_candidates();
    candidates.insert(candidates.end(), result.candidate_item_ids.begin(),
                      result.candidate_item_ids.end());
  }
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  candidates.erase(
      std::remove_if(candidates.begin(), candidates.end(),
                     [child_count](uint32_t idx) {
                       return idx >= child_count;
                     }),
      candidates.end());

  const double ratio =
      child_count == 0
          ? 1.0
          : static_cast<double>(candidates.size()) /
                static_cast<double>(child_count);
  if (candidates.size() > config_.safe_child_router_max_candidates ||
      ratio > config_.safe_child_router_max_ratio) {
    return accept_mbb_candidates();
  }

  if (config_.safe_child_router_validate) {
    std::vector<uint8_t> in_candidate(child_count, 0);
    for (size_t child_idx : candidates) in_candidate[child_idx] = 1;
    for (size_t child_idx = 0; child_idx < child_count; ++child_idx) {
      const NodeId child_id =
          child_at(static_cast<uint32_t>(child_idx));
      if (child_id >= view.node_records.size()) {
        stats.children_actually_processed += child_count;
        stats.safe_child_router_fallback_count++;
        return {};
      }
      const LeafId child_center_id =
          view.center_sequence_id(child_id);
      if (child_center_id >= view.sequences.size()) {
        stats.children_actually_processed += child_count;
        stats.safe_child_router_fallback_count++;
        return {};
      }
      const int child_tau = tolerance + child_radius;
      stats.safe_child_router_exact_verify_count++;
      const int dist = compute_indexed_query_distance(
          query_seq.seq,
          view.sequences.sequence(child_center_id),
          child_center_id, child_tau,
          config_.distance_mode, false);
      if (dist <= child_tau && !in_candidate[child_idx]) {
        throw std::runtime_error(
            "safe child router missed a possible child candidate");
      }
    }
  }

  if (used_router) *used_router = true;
  stats.safe_child_router_candidate_count += candidates.size();
  stats.safe_router_candidate_count += candidates.size();
  stats.safe_child_router_candidate_ratio_sum += ratio;
  stats.candidate_ratio_to_all_children += ratio;
  stats.safe_child_router_pruned_by_not_candidate_count +=
      child_count - candidates.size();
  stats.children_actually_processed += candidates.size();
  stats.center_checks_saved += child_count - candidates.size();
  return candidates;
}

std::vector<std::shared_ptr<WorldNode>>
BioGeometrySearchEngine::scan_mbb_surviving_child_indices(
    const std::shared_ptr<WorldNode>& node,
    const std::vector<uint32_t>& child_indices,
    const std::vector<int>& query_beacon_dists,
    int tolerance,
    SearchStats& stats) const {
  ScopedSearchTimer timer(stats.query_profile_enabled, &stats.mbb_filter_ms);
  std::vector<std::shared_ptr<WorldNode>> surviving;
  if (!node) return surviving;
  const auto& children = node->child_nodes;
  stats.child_edge_considered_count += child_indices.size();
  bool mbb_ok =
      !query_beacon_dists.empty() &&
      query_beacon_dists.size() == node->beacons.size() &&
      node->child_beacon_mbbs.size() == children.size();
  if (mbb_ok) {
    for (size_t child_idx : child_indices) {
      if (child_idx >= children.size() ||
          node->child_beacon_mbbs[child_idx].size() != query_beacon_dists.size()) {
        mbb_ok = false;
        break;
      }
    }
  }

  surviving.reserve(child_indices.size());
  if (!mbb_ok) {
    for (size_t child_idx : child_indices) {
      if (child_idx < children.size()) surviving.push_back(children[child_idx]);
    }
  } else {
    for (size_t child_idx : child_indices) {
      stats.edge_access_count++;
      stats.mbb_check_count++;
      stats.candidate_count_for_prune++;
      stats.bound_check_count++;
      stats.mbb_scan_child_checks++;
      stats.mbb_scalar_checks++;
      if (mbb_prunable_row(
              node->child_beacon_mbbs[child_idx], query_beacon_dists, tolerance)) {
        stats.beacon_prune_count++;
        stats.child_mbb_pruned_count++;
        continue;
      }
      surviving.push_back(children[child_idx]);
    }
  }
  stats.mbb_surviving_child_count += surviving.size();
  stats.post_mbb_survivor_count += surviving.size();
  if (!surviving.empty()) {
    stats.candidate_ratio_to_post_mbb_survivors +=
        static_cast<double>(child_indices.size()) /
        static_cast<double>(surviving.size());
  } else if (!child_indices.empty()) {
    stats.candidate_ratio_to_post_mbb_survivors +=
        static_cast<double>(child_indices.size());
  }
  return surviving;
}

std::vector<std::shared_ptr<WorldNode>>
BioGeometrySearchEngine::rank_children_with_router_hints(
    const std::shared_ptr<WorldNode>& node,
    const std::vector<std::shared_ptr<WorldNode>>& candidates,
    const BioSequence& query_seq,
    int tolerance,
    SearchStats& stats,
    const QGramSignature* router_qgram_signature,
    const std::vector<uint64_t>* router_minimizers) const {
  if (!config_.router_hint_enabled) return candidates;
  if (stats.planner_disable_router_stack) return candidates;
  if (candidates.size() < kMinRouterHintCandidateCount) return candidates;
  ScopedSearchTimer timer(stats.query_profile_enabled, &stats.router_lookup_ms);
  stats.router_hint_invoked_count++;
  stats.router_candidate_count += candidates.size();
  if (!node || candidates.empty()) {
    stats.router_fallback_count++;
    return candidates;
  }

  const auto parent_it = parent_router_hint_indexes_.find(node->node_id);
  if (parent_it == parent_router_hint_indexes_.end()) {
    stats.router_fallback_count++;
    return candidates;
  }

  if (!router_qgram_signature) {
    router_qgram_signature = get_router_qgram_signature(this, config_, stats);
  }
  if (!router_minimizers) {
    router_minimizers = get_router_minimizers(this, config_, stats);
  }

  RangeJoinQueryWorkspace workspace;
  auto predicted = parent_it->second.range_index.query(
      query_seq.seq, tolerance, &workspace);
  stats.router_pigeonhole_query_count++;
  std::unordered_map<size_t, size_t> predicted_rank_by_item;
  predicted_rank_by_item.reserve(predicted.candidate_item_ids.size());
  for (size_t rank = 0; rank < predicted.candidate_item_ids.size(); ++rank) {
    predicted_rank_by_item.emplace(predicted.candidate_item_ids[rank], rank);
  }

  struct RankedChild {
    std::shared_ptr<WorldNode> child;
    size_t predicted_rank = std::numeric_limits<size_t>::max();
    size_t qgram_distance = std::numeric_limits<size_t>::max();
    size_t minimizer_distance = std::numeric_limits<size_t>::max();
    size_t original_rank = 0;
  };

  std::vector<RankedChild> ranked;
  ranked.reserve(candidates.size());
  size_t predicted_hits = 0;
  for (size_t candidate_idx = 0; candidate_idx < candidates.size(); ++candidate_idx) {
    const auto& candidate = candidates[candidate_idx];
    RankedChild entry;
    entry.child = candidate;
    entry.original_rank = candidate_idx;
    if (!candidate) {
      stats.unsafe_hint_ignored_count++;
      ranked.push_back(entry);
      continue;
    }

    auto item_it =
        parent_it->second.child_item_ids_by_node_id.find(candidate->node_id);
    if (item_it != parent_it->second.child_item_ids_by_node_id.end()) {
      auto predicted_it = predicted_rank_by_item.find(item_it->second);
      if (predicted_it != predicted_rank_by_item.end()) {
        entry.predicted_rank = predicted_it->second;
        predicted_hits++;
      }
    } else {
      stats.unsafe_hint_ignored_count++;
    }

    if (router_qgram_signature && router_qgram_signature->safe_for_pruning) {
      const auto* candidate_signature = lookup_qgram_signature(
          world_qgram_signatures_by_q_, config_.router_hint_qgram_q,
          candidate->node_id);
      if (candidate_signature && candidate_signature->safe_for_pruning) {
        entry.qgram_distance = qgram_l1_distance(
            *router_qgram_signature, *candidate_signature);
        stats.router_qgram_ranked_count++;
      } else {
        stats.unsafe_hint_ignored_count++;
      }
    }

    if (router_minimizers) {
      auto minimizer_it = world_minimizer_signatures_.find(candidate->node_id);
      if (minimizer_it != world_minimizer_signatures_.end() &&
          !minimizer_it->second.empty()) {
        const size_t overlap = minimizer_overlap_count(
            *router_minimizers, minimizer_it->second);
        entry.minimizer_distance =
            router_minimizers->size() + minimizer_it->second.size() - 2 * overlap;
        stats.router_minimizer_ranked_count++;
      } else {
        stats.unsafe_hint_ignored_count++;
      }
    }

    ranked.push_back(entry);
  }

  stats.router_candidate_hit_count += predicted_hits;
  if (predicted_hits < candidates.size()) stats.router_fallback_count++;

  std::stable_sort(ranked.begin(), ranked.end(),
                   [](const RankedChild& left, const RankedChild& right) {
                     if (left.predicted_rank != right.predicted_rank) {
                       return left.predicted_rank < right.predicted_rank;
                     }
                     if (left.qgram_distance != right.qgram_distance) {
                       return left.qgram_distance < right.qgram_distance;
                     }
                     if (left.minimizer_distance != right.minimizer_distance) {
                       return left.minimizer_distance < right.minimizer_distance;
                     }
                     return left.original_rank < right.original_rank;
                   });

  std::vector<std::shared_ptr<WorldNode>> ordered;
  ordered.reserve(ranked.size());
  for (const auto& entry : ranked) ordered.push_back(entry.child);
  return ordered;
}

std::vector<std::string> BioGeometrySearchEngine::debug_safe_child_router_candidate_ids(
    const std::string& parent_node_id,
    const BioSequence& query_seq,
    int tolerance,
    bool* used_router) const {
  g_query_distance_cache.begin_query();
  if (used_router) *used_router = false;
  size_t parsed = 0;
  unsigned long raw_id = 0;
  try {
    raw_id = std::stoul(parent_node_id, &parsed);
  } catch (const std::exception&) {
    throw std::invalid_argument("debug safe child router parent not found");
  }
  const auto& view = index_.search_graph_view();
  if (parsed != parent_node_id.size() ||
      raw_id >= view.node_records.size()) {
    throw std::invalid_argument("debug safe child router parent not found");
  }
  const NodeId parent_id = static_cast<NodeId>(raw_id);

  SearchStats stats(static_cast<size_t>(index_.num_primary_layers()));
  std::vector<int> V_Q;
  if (view.beacon_count(parent_id) > 0) {
    V_Q = compute_query_beacon_distances_view(parent_id, query_seq, stats);
  }
  bool routed = false;
  const auto layer_it = std::upper_bound(
      view.layer_end.begin(), view.layer_end.end(), parent_id);
  if (layer_it == view.layer_end.end()) {
    throw std::invalid_argument(
        "debug safe child router parent has no layer");
  }
  const size_t parent_layer =
      static_cast<size_t>(
          std::distance(view.layer_end.begin(), layer_it));
  const auto& primary_radii =
      index_.hierarchy_config().primary_radii;
  const int child_radius =
      parent_layer + 1 < primary_radii.size()
          ? primary_radii[parent_layer + 1]
          : 0;
  auto child_indices = safe_child_router_candidate_indices_view(
      parent_id, query_seq, V_Q, child_radius, tolerance,
      stats, &routed);
  std::vector<std::string> out;
  if (!routed) {
    child_indices.clear();
    child_indices.reserve(view.child_count(parent_id));
    for (size_t i = 0; i < view.child_count(parent_id); ++i) {
      child_indices.push_back(static_cast<uint32_t>(i));
    }
  }
  out.reserve(child_indices.size());
  for (size_t child_idx : child_indices) {
    if (child_idx < view.child_count(parent_id)) {
      out.push_back(
          node_key(view.child_id(
              parent_id, static_cast<uint32_t>(child_idx))));
    }
  }
  if (used_router) *used_router = routed;
  return out;
}

std::vector<std::shared_ptr<WorldNode>>
BioGeometrySearchEngine::rank_children_with_local_router(
    const std::shared_ptr<WorldNode>& node,
    const std::vector<std::shared_ptr<WorldNode>>& candidates,
    const std::vector<int>& query_beacon_dists,
  SearchStats& stats) const {
  if (!config_.local_router_enabled) return candidates;
  if (stats.planner_disable_router_stack) return candidates;
  ScopedSearchTimer timer(stats.query_profile_enabled, &stats.router_lookup_ms);
  stats.local_router_invoked_count++;
  stats.local_router_enabled_count++;
  if (config_.local_router_score_mode != "anchor-envelope") {
    stats.unsafe_hint_ignored_count++;
    stats.local_router_empty_count++;
    stats.local_router_fallback_count++;
    return candidates;
  }

  const size_t max_anchors =
      std::min(config_.local_router_max_anchors, query_beacon_dists.size());
  const bool router_ready =
      node && max_anchors > 0 && !candidates.empty() &&
      node->child_beacon_mbbs.size() == node->child_nodes.size();
  if (!router_ready) {
    stats.local_router_empty_count++;
    stats.local_router_fallback_count++;
    return candidates;
  }
  struct RankedChild {
    std::shared_ptr<WorldNode> child;
    double score = 0.0;
    size_t original_rank = 0;
  };
  std::vector<RankedChild> ranked;
  ranked.reserve(candidates.size());
  for (size_t candidate_idx = 0; candidate_idx < candidates.size(); ++candidate_idx) {
    const auto& candidate = candidates[candidate_idx];
    size_t child_idx = node->child_nodes.size();
    for (size_t i = 0; i < node->child_nodes.size(); ++i) {
      if (node->child_nodes[i] == candidate) {
        child_idx = i;
        break;
      }
    }
    if (child_idx == node->child_nodes.size() ||
        child_idx >= node->child_beacon_mbbs.size() ||
        node->child_beacon_mbbs[child_idx].size() < max_anchors) {
      stats.unsafe_hint_ignored_count++;
      ranked.push_back({candidate, std::numeric_limits<double>::max(),
                        candidate_idx});
      continue;
    }
    ranked.push_back({candidate,
                      local_router_mbb_score(node->child_beacon_mbbs[child_idx],
                                             query_beacon_dists, max_anchors),
                      candidate_idx});
  }

  std::stable_sort(ranked.begin(), ranked.end(),
                   [](const RankedChild& left, const RankedChild& right) {
                     if (left.score != right.score) return left.score < right.score;
                     return left.original_rank < right.original_rank;
                   });

  std::vector<std::shared_ptr<WorldNode>> ordered;
  ordered.reserve(ranked.size());
  for (const auto& entry : ranked) ordered.push_back(entry.child);

  const size_t shortlist =
      config_.local_router_max_children == 0
          ? ordered.size()
          : std::min(config_.local_router_max_children, ordered.size());
  stats.router_candidate_count += ordered.size();
  stats.local_router_shortlist_child_count += shortlist;
  stats.local_router_remaining_child_count += ordered.size() - shortlist;
  if (shortlist < ordered.size()) stats.local_router_fallback_count++;
  return ordered;
}

std::vector<NodeId> BioGeometrySearchEngine::rank_child_ids_with_local_router_view(
    NodeId node_id,
    const std::vector<NodeId>& candidates,
    const std::vector<int>& query_beacon_dists,
    int child_radius,
  SearchStats& stats) const {
  if (!config_.local_router_enabled) return candidates;
  if (stats.planner_disable_router_stack) return candidates;
  const auto& view = index_.search_graph_view();
  if (node_id >= view.node_records.size()) {
    throw std::out_of_range("view node id is outside search graph view");
  }
  const auto node = view.node_records[node_id];
  ScopedSearchTimer timer(stats.query_profile_enabled, &stats.router_lookup_ms);
  stats.local_router_invoked_count++;
  stats.local_router_enabled_count++;
  if (config_.local_router_score_mode != "anchor-envelope") {
    stats.unsafe_hint_ignored_count++;
    stats.local_router_empty_count++;
    stats.local_router_fallback_count++;
    return candidates;
  }

  const uint32_t child_count = view.link_count(node);
  const uint32_t beacon_count = view.beacon_count(node);
  const auto children = view.child_ids_for(node_id, node);
  const size_t max_anchors =
      std::min(config_.local_router_max_anchors,
               query_beacon_dists.size());
  const bool router_ready =
      max_anchors > 0 && !candidates.empty() &&
      beacon_count >= max_anchors &&
      view.child_mbb_range_valid(
          node_id, node, child_count, beacon_count);
  if (!router_ready) {
    stats.local_router_empty_count++;
    stats.local_router_fallback_count++;
    return candidates;
  }
  const uint32_t mbb_bits = view.child_mbb_bits(node_id, node);
  const uint32_t mbb_begin = node.child_mbb_begin();
  const uint32_t quantization_bin_width =
      view.child_mbb_bin_width(node_id);
  const uint32_t quantization_error =
      SearchGraphView::child_mbb_quantization_error(
          quantization_bin_width);

  struct RankedChild {
    NodeId child_id = INVALID_NODE_ID;
    double score = 0.0;
    size_t original_rank = 0;
  };
  std::vector<RankedChild> ranked;
  ranked.reserve(candidates.size());
  for (size_t candidate_idx = 0; candidate_idx < candidates.size();
       ++candidate_idx) {
    const NodeId candidate = candidates[candidate_idx];
    size_t child_idx = child_count;
    for (size_t idx = 0; idx < child_count; ++idx) {
      if (children.at(static_cast<uint32_t>(idx)) == candidate) {
        child_idx = idx;
        break;
      }
    }
    if (child_idx == child_count) {
      stats.unsafe_hint_ignored_count++;
      ranked.push_back({candidate, std::numeric_limits<double>::max(),
                        candidate_idx});
      continue;
    }
    double score = 0.0;
    for (size_t dim = 0; dim < max_anchors; ++dim) {
      const size_t cell = dim * child_count + child_idx;
      const uint8_t center_dist =
          view.child_beacon_distance_unchecked(
              mbb_begin, child_count, beacon_count, cell, mbb_bits,
              quantization_bin_width);
      const int lo =
          reconstructed_mbb_lo(
              center_dist,
              child_radius + quantization_error);
      const int hi =
          reconstructed_mbb_hi(
              center_dist,
              child_radius + quantization_error);
      const int q = query_beacon_dists[dim];
      if (q < lo) {
        score += static_cast<double>(lo - q) * 1024.0;
      } else if (q > hi) {
        score += static_cast<double>(q - hi) * 1024.0;
      }
      score += static_cast<double>(hi - lo);
    }
    ranked.push_back({candidate, score, candidate_idx});
  }
  std::stable_sort(ranked.begin(), ranked.end(),
                   [](const RankedChild& left,
                      const RankedChild& right) {
                     if (left.score != right.score) {
                       return left.score < right.score;
                     }
                     return left.original_rank < right.original_rank;
                   });
  std::vector<NodeId> ordered;
  ordered.reserve(ranked.size());
  for (const auto& entry : ranked) ordered.push_back(entry.child_id);
  const size_t shortlist =
      config_.local_router_max_children == 0
          ? ordered.size()
          : std::min(config_.local_router_max_children, ordered.size());
  stats.router_candidate_count += ordered.size();
  stats.local_router_shortlist_child_count += shortlist;
  stats.local_router_remaining_child_count += ordered.size() - shortlist;
  if (shortlist < ordered.size()) stats.local_router_fallback_count++;
  return ordered;
}

std::vector<NodeId>
BioGeometrySearchEngine::rank_child_ids_with_router_hints_view(
    NodeId node_id,
    const std::vector<NodeId>& candidates,
    const BioSequence& query_seq,
    int tolerance,
    SearchStats& stats,
    const QGramSignature* router_qgram_signature,
  const std::vector<uint64_t>* router_minimizers) const {
  if (!config_.router_hint_enabled) return candidates;
  if (stats.planner_disable_router_stack) return candidates;
  if (candidates.size() < kMinRouterHintCandidateCount) return candidates;
  const auto& view = index_.search_graph_view();
  if (node_id >= view.node_records.size()) {
    throw std::out_of_range("view node id is outside search graph view");
  }
  ScopedSearchTimer timer(stats.query_profile_enabled, &stats.router_lookup_ms);
  stats.router_hint_invoked_count++;
  stats.router_candidate_count += candidates.size();
  const auto parent_it =
      parent_router_hint_indexes_.find(node_key(node_id));
  if (parent_it == parent_router_hint_indexes_.end()) {
    stats.router_fallback_count++;
    return candidates;
  }

  if (!router_qgram_signature) {
    router_qgram_signature = get_router_qgram_signature(this, config_, stats);
  }
  if (!router_minimizers) {
    router_minimizers = get_router_minimizers(this, config_, stats);
  }
  RangeJoinQueryWorkspace workspace;
  auto predicted =
      parent_it->second.range_index.query(query_seq.seq, tolerance, &workspace);
  stats.router_pigeonhole_query_count++;
  std::unordered_map<size_t, size_t> predicted_rank_by_item;
  predicted_rank_by_item.reserve(predicted.candidate_item_ids.size());
  for (size_t rank = 0; rank < predicted.candidate_item_ids.size(); ++rank) {
    predicted_rank_by_item.emplace(predicted.candidate_item_ids[rank], rank);
  }

  struct RankedChild {
    NodeId child_id = INVALID_NODE_ID;
    size_t predicted_rank = std::numeric_limits<size_t>::max();
    size_t qgram_distance = std::numeric_limits<size_t>::max();
    size_t minimizer_distance = std::numeric_limits<size_t>::max();
    size_t original_rank = 0;
  };
  std::vector<RankedChild> ranked;
  ranked.reserve(candidates.size());
  size_t predicted_hits = 0;
  for (size_t candidate_idx = 0; candidate_idx < candidates.size();
       ++candidate_idx) {
    const NodeId candidate_id = candidates[candidate_idx];
    if (candidate_id >= view.node_records.size()) {
      stats.unsafe_hint_ignored_count++;
      continue;
    }
    RankedChild entry;
    entry.child_id = candidate_id;
    entry.original_rank = candidate_idx;
    const std::string candidate_key = node_key(candidate_id);
    auto item_it =
        parent_it->second.child_item_ids_by_node_id.find(candidate_key);
    if (item_it != parent_it->second.child_item_ids_by_node_id.end()) {
      auto predicted_it = predicted_rank_by_item.find(item_it->second);
      if (predicted_it != predicted_rank_by_item.end()) {
        entry.predicted_rank = predicted_it->second;
        predicted_hits++;
      }
    } else {
      stats.unsafe_hint_ignored_count++;
    }
    if (router_qgram_signature && router_qgram_signature->safe_for_pruning) {
      const auto* candidate_signature = lookup_qgram_signature(
          world_qgram_signatures_by_q_, config_.router_hint_qgram_q,
          candidate_key);
      if (candidate_signature && candidate_signature->safe_for_pruning) {
        entry.qgram_distance = qgram_l1_distance(
            *router_qgram_signature, *candidate_signature);
        stats.router_qgram_ranked_count++;
      } else {
        stats.unsafe_hint_ignored_count++;
      }
    }
    if (router_minimizers) {
      auto minimizer_it =
          world_minimizer_signatures_.find(candidate_key);
      if (minimizer_it != world_minimizer_signatures_.end() &&
          !minimizer_it->second.empty()) {
        const size_t overlap =
            minimizer_overlap_count(*router_minimizers, minimizer_it->second);
        entry.minimizer_distance =
            router_minimizers->size() + minimizer_it->second.size() -
            2 * overlap;
        stats.router_minimizer_ranked_count++;
      } else {
        stats.unsafe_hint_ignored_count++;
      }
    }
    ranked.push_back(entry);
  }
  stats.router_candidate_hit_count += predicted_hits;
  if (predicted_hits < candidates.size()) stats.router_fallback_count++;
  std::stable_sort(ranked.begin(), ranked.end(),
                   [](const RankedChild& left,
                      const RankedChild& right) {
                     if (left.predicted_rank != right.predicted_rank) {
                       return left.predicted_rank < right.predicted_rank;
                     }
                     if (left.qgram_distance != right.qgram_distance) {
                       return left.qgram_distance < right.qgram_distance;
                     }
                     if (left.minimizer_distance != right.minimizer_distance) {
                       return left.minimizer_distance <
                              right.minimizer_distance;
                     }
                     return left.original_rank < right.original_rank;
                   });
  std::vector<NodeId> ordered;
  ordered.reserve(ranked.size());
  for (const auto& entry : ranked) ordered.push_back(entry.child_id);
  return ordered;
}

std::vector<std::shared_ptr<WorldNode>> BioGeometrySearchEngine::apply_path_reuse_order(
    const std::shared_ptr<WorldNode>& parent,
    const std::vector<std::shared_ptr<WorldNode>>& candidates,
    SearchStats& stats) const {
  if (!config_.path_reuse_enabled || !parent || candidates.size() < 2) {
    return candidates;
  }
  if (stats.planner_disable_router_stack) return candidates;

  auto* reuse = active_path_reuse_context(this);
  if (!reuse || !reuse->enabled) return candidates;

  std::vector<std::string> current_order;
  current_order.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    if (!candidate) return candidates;
    current_order.push_back(candidate->node_id);
  }
  reuse->next.child_orders_by_parent[parent->node_id] = current_order;

  if (!reuse->previous_fingerprint_match || !reuse->previous) return candidates;
  auto cached_it = reuse->previous->child_orders_by_parent.find(parent->node_id);
  if (cached_it == reuse->previous->child_orders_by_parent.end()) return candidates;

  ScopedSearchTimer timer(stats.query_profile_enabled, &stats.path_reuse_ms);
  stats.path_reuse_attempt_count++;
  if (cached_it->second == current_order) {
    stats.path_reuse_hit_count++;
    stats.child_shortlist_reuse_hit_count += candidates.size();
    stats.child_shortlist_cache_hit_count += candidates.size();
    return candidates;
  }
  std::vector<std::shared_ptr<WorldNode>> reordered;
  size_t matched = 0;
  const bool changed = apply_cached_rank_order(
      candidates, cached_it->second,
      [](const std::shared_ptr<WorldNode>& candidate) {
        return candidate ? candidate->node_id : std::string();
      },
      &reordered, &matched);
  if (matched == 0) return candidates;

  stats.path_reuse_hit_count++;
  stats.child_shortlist_reuse_hit_count += matched;
  stats.child_shortlist_cache_hit_count += matched;
  return changed ? reordered : candidates;
}

std::vector<std::shared_ptr<WorldNode>>
BioGeometrySearchEngine::rank_children_with_best_first(
    const std::shared_ptr<WorldNode>& node,
    const std::vector<std::shared_ptr<WorldNode>>& candidates,
    const std::vector<int>& query_beacon_dists,
    int tolerance,
  SearchStats& stats) const {
  if (!config_.best_first_enabled) return candidates;
  if (stats.planner_disable_router_stack) return candidates;
  ScopedSearchTimer timer(stats.query_profile_enabled, &stats.best_first_queue_ms);
  stats.best_first_enabled_count++;
  stats.best_first_invoked_count++;
  if (!node || candidates.size() < 2 || query_beacon_dists.empty() ||
      node->child_beacon_mbbs.size() != node->child_nodes.size()) {
    return candidates;
  }

  struct RankedChild {
    std::shared_ptr<WorldNode> child;
    double lower_bound = 0.0;
    double span = 0.0;
    size_t original_rank = 0;
  };

  std::vector<RankedChild> ranked;
  ranked.reserve(candidates.size());
  for (size_t candidate_idx = 0; candidate_idx < candidates.size(); ++candidate_idx) {
    const auto& candidate = candidates[candidate_idx];
    size_t child_idx = node->child_nodes.size();
    for (size_t i = 0; i < node->child_nodes.size(); ++i) {
      if (node->child_nodes[i] == candidate) {
        child_idx = i;
        break;
      }
    }
    if (child_idx == node->child_nodes.size() ||
        child_idx >= node->child_beacon_mbbs.size() ||
        node->child_beacon_mbbs[child_idx].size() != query_beacon_dists.size()) {
      ranked.push_back(
          {candidate, std::numeric_limits<double>::max(),
           std::numeric_limits<double>::max(), candidate_idx});
      continue;
    }

    stats.best_first_bound_candidate_count++;
    const auto& row = node->child_beacon_mbbs[child_idx];
    const double lower_bound = child_world_mbb_lower_bound(
        row, query_beacon_dists);
    if (lower_bound > static_cast<double>(tolerance)) {
      stats.child_safe_bound_pruned_count++;
      continue;
    }
    ranked.push_back(
        {candidate, lower_bound, child_world_mbb_span(row), candidate_idx});
  }

  std::stable_sort(ranked.begin(), ranked.end(),
                   [](const RankedChild& left, const RankedChild& right) {
                     if (left.lower_bound != right.lower_bound) {
                       return left.lower_bound < right.lower_bound;
                     }
                     if (left.span != right.span) return left.span < right.span;
                     return left.original_rank < right.original_rank;
                   });

  std::vector<std::shared_ptr<WorldNode>> ordered;
  ordered.reserve(ranked.size());
  bool reordered = ranked.size() != candidates.size();
  for (size_t i = 0; i < ranked.size(); ++i) {
    ordered.push_back(ranked[i].child);
    if (!reordered && ordered[i] != candidates[i]) reordered = true;
  }
  if (reordered) stats.best_first_reordered_count++;
  return ordered;
}

std::vector<NodeId> BioGeometrySearchEngine::rank_child_ids_with_best_first_view(
    NodeId node_id,
    const std::vector<NodeId>& candidates,
    const std::vector<int>& query_beacon_dists,
    int child_radius,
    int tolerance,
  SearchStats& stats) const {
  if (!config_.best_first_enabled) return candidates;
  if (stats.planner_disable_router_stack) return candidates;
  const auto& view = index_.search_graph_view();
  if (node_id >= view.node_records.size()) {
    throw std::out_of_range("view node id is outside search graph view");
  }
  const auto node = view.node_records[node_id];
  ScopedSearchTimer timer(stats.query_profile_enabled,
                          &stats.best_first_queue_ms);
  stats.best_first_enabled_count++;
  stats.best_first_invoked_count++;
  const uint32_t child_count = view.link_count(node);
  const uint32_t beacon_count = view.beacon_count(node);
  const auto children = view.child_ids_for(node_id, node);
  const bool mbb_ready =
      !query_beacon_dists.empty() &&
      query_beacon_dists.size() == beacon_count &&
      view.child_mbb_range_valid(
          node_id, node, child_count, beacon_count);
  if (candidates.size() < 2 || !mbb_ready) return candidates;
  const uint32_t mbb_bits = view.child_mbb_bits(node_id, node);
  const uint32_t mbb_begin = node.child_mbb_begin();
  const uint32_t quantization_bin_width =
      view.child_mbb_bin_width(node_id);
  const uint32_t quantization_error =
      SearchGraphView::child_mbb_quantization_error(
          quantization_bin_width);

  struct RankedChild {
    NodeId child_id = INVALID_NODE_ID;
    double lower_bound = 0.0;
    double span = 0.0;
    size_t original_rank = 0;
  };
  std::vector<RankedChild> ranked;
  ranked.reserve(candidates.size());
  for (size_t candidate_idx = 0; candidate_idx < candidates.size();
       ++candidate_idx) {
    const NodeId candidate = candidates[candidate_idx];
    size_t child_idx = child_count;
    for (size_t idx = 0; idx < child_count; ++idx) {
      if (children.at(static_cast<uint32_t>(idx)) == candidate) {
        child_idx = idx;
        break;
      }
    }
    if (child_idx == child_count) {
      ranked.push_back(
          {candidate, std::numeric_limits<double>::max(),
           std::numeric_limits<double>::max(), candidate_idx});
      continue;
    }
    double lower_bound = 0.0;
    double span = 0.0;
    for (size_t dim = 0; dim < beacon_count; ++dim) {
      const size_t cell = dim * child_count + child_idx;
      const uint8_t center_dist =
          view.child_beacon_distance_unchecked(
              mbb_begin, child_count, beacon_count, cell, mbb_bits,
              quantization_bin_width);
      const int lo =
          reconstructed_mbb_lo(
              center_dist,
              child_radius + quantization_error);
      const int hi =
          reconstructed_mbb_hi(
              center_dist,
              child_radius + quantization_error);
      const int q = query_beacon_dists[dim];
      if (q < lo) {
        lower_bound =
            std::max(lower_bound, static_cast<double>(lo - q));
      } else if (q > hi) {
        lower_bound =
            std::max(lower_bound, static_cast<double>(q - hi));
      }
      span += static_cast<double>(hi - lo);
    }
    stats.best_first_bound_candidate_count++;
    if (lower_bound > static_cast<double>(tolerance)) {
      stats.child_safe_bound_pruned_count++;
      continue;
    }
    ranked.push_back({candidate, lower_bound, span, candidate_idx});
  }
  std::stable_sort(ranked.begin(), ranked.end(),
                   [](const RankedChild& left,
                      const RankedChild& right) {
                     if (left.lower_bound != right.lower_bound) {
                       return left.lower_bound < right.lower_bound;
                     }
                     if (left.span != right.span) {
                       return left.span < right.span;
                     }
                     return left.original_rank < right.original_rank;
                   });
  std::vector<NodeId> ordered;
  ordered.reserve(ranked.size());
  bool reordered = ranked.size() != candidates.size();
  for (size_t idx = 0; idx < ranked.size(); ++idx) {
    ordered.push_back(ranked[idx].child_id);
    if (!reordered && ordered[idx] != candidates[idx]) {
      reordered = true;
    }
  }
  if (reordered) stats.best_first_reordered_count++;
  return ordered;
}

std::vector<NodeId> BioGeometrySearchEngine::apply_path_reuse_order_view(
    NodeId node_id,
    const std::vector<NodeId>& candidates,
    SearchStats& stats) const {
  if (!config_.path_reuse_enabled || candidates.size() < 2) return candidates;
  if (stats.planner_disable_router_stack) return candidates;
  const auto& view = index_.search_graph_view();
  if (node_id >= view.node_records.size()) {
    throw std::out_of_range("view node id is outside search graph view");
  }

  auto* reuse = active_path_reuse_context(this);
  if (!reuse || !reuse->enabled) return candidates;

  std::vector<std::string> current_order;
  current_order.reserve(candidates.size());
  for (NodeId child_id : candidates) {
    if (child_id >= view.node_records.size()) {
      throw std::out_of_range("view child id is outside search graph view");
    }
    current_order.push_back(node_key(child_id));
  }
  const std::string parent_key = node_key(node_id);
  reuse->next.child_orders_by_parent[parent_key] = current_order;

  if (!reuse->previous_fingerprint_match || !reuse->previous) return candidates;
  auto cached_it =
      reuse->previous->child_orders_by_parent.find(parent_key);
  if (cached_it == reuse->previous->child_orders_by_parent.end()) return candidates;

  ScopedSearchTimer timer(stats.query_profile_enabled, &stats.path_reuse_ms);
  stats.path_reuse_attempt_count++;
  if (cached_it->second == current_order) {
    stats.path_reuse_hit_count++;
    stats.child_shortlist_reuse_hit_count += candidates.size();
    stats.child_shortlist_cache_hit_count += candidates.size();
    return candidates;
  }
  std::vector<NodeId> reordered;
  size_t matched = 0;
  const bool changed = apply_cached_rank_order(
      candidates, cached_it->second,
      [](NodeId child_id) { return node_key(child_id); },
      &reordered, &matched);
  if (matched == 0) return candidates;

  stats.path_reuse_hit_count++;
  stats.child_shortlist_reuse_hit_count += matched;
  stats.child_shortlist_cache_hit_count += matched;
  return changed ? reordered : candidates;
}

bool BioGeometrySearchEngine::near_query_triangle_prunes_center(
    const std::string& node_id,
    int tau,
    SearchStats& stats) const {
  auto* reuse = active_path_reuse_context(this);
  if (!reuse || !reuse->enabled || !reuse->previous_near_query_match ||
      !reuse->previous) {
    return false;
  }
  auto dist_it = reuse->previous->center_dist_by_node.find(node_id);
  if (dist_it == reuse->previous->center_dist_by_node.end()) {
    stats.near_query_bound_fallback_count++;
    return false;
  }
  const int lower_bound =
      std::max(0, dist_it->second - reuse->previous_neighbor_edit_distance);
  if (lower_bound > tau) {
    stats.near_query_triangle_pruned_count++;
    stats.near_query_center_distance_reused_count++;
    stats.center_checks_saved++;
    stats.child_safe_bound_pruned_count++;
    return true;
  }
  stats.near_query_bound_fallback_count++;
  return false;
}

bool near_query_triangle_prunes_leaf(const BioGeometrySearchEngine* engine,
                                     const std::string& leaf_id,
                                     int tolerance,
                                     SearchStats& stats) {
  auto* reuse = active_path_reuse_context(engine);
  if (!reuse || !reuse->enabled || !reuse->previous_near_query_match ||
      !reuse->previous) {
    return false;
  }
  auto dist_it = reuse->previous->leaf_dist_by_id.find(leaf_id);
  if (dist_it == reuse->previous->leaf_dist_by_id.end()) {
    stats.near_query_leaf_bound_fallback_count++;
    return false;
  }
  const int lower_bound =
      std::max(0, dist_it->second - reuse->previous_neighbor_edit_distance);
  if (lower_bound > tolerance) {
    stats.near_query_leaf_triangle_pruned_count++;
    stats.near_query_leaf_distance_reused_count++;
    stats.child_safe_bound_pruned_count++;
    return true;
  }
  stats.near_query_leaf_bound_fallback_count++;
  return false;
}

bool near_query_triangle_prunes_leaf(
    const BioGeometrySearchEngine* engine,
    LeafId sequence_id,
    int tolerance,
    SearchStats& stats) {
  auto* reuse = active_path_reuse_context(engine);
  if (!reuse || !reuse->enabled || !reuse->previous_near_query_match ||
      !reuse->previous) {
    return false;
  }
  const auto dist_it =
      reuse->previous->leaf_dist_by_sequence_id.find(sequence_id);
  if (dist_it == reuse->previous->leaf_dist_by_sequence_id.end()) {
    stats.near_query_leaf_bound_fallback_count++;
    return false;
  }
  const int lower_bound =
      std::max(0, dist_it->second - reuse->previous_neighbor_edit_distance);
  if (lower_bound > tolerance) {
    stats.near_query_leaf_triangle_pruned_count++;
    stats.near_query_leaf_distance_reused_count++;
    stats.child_safe_bound_pruned_count++;
    return true;
  }
  stats.near_query_leaf_bound_fallback_count++;
  return false;
}

int leaf_distance_cache_bound(const SearchConfig& config,
                              int tolerance,
                              bool path_reuse_enabled) {
  if (!path_reuse_enabled) return tolerance;
  const int max_neighbor_dist = std::min(
      tolerance, std::max(0, config.near_query_max_neighbor_edit_distance));
  return tolerance + max_neighbor_dist;
}

int BioGeometrySearchEngine::compute_center_distance_for_search(
    const BioSequence& query_seq,
    const std::string& node_id,
    LeafId center_sequence_id,
    std::string_view center_sequence,
    int tau,
    bool after_mbb_filter,
    bool* cache_hit) const {
  (void)after_mbb_filter;
  auto* reuse = active_path_reuse_context(this);
  const bool record_exact_center_distance =
      reuse && reuse->enabled && config_.path_reuse_enabled;
  const int dist = compute_indexed_query_distance(
      query_seq.seq, center_sequence, center_sequence_id, tau,
      config_.distance_mode, record_exact_center_distance, cache_hit);
  if (record_exact_center_distance) {
    reuse->next.center_dist_by_node[node_id] = dist;
  }
  return dist;
}

void BioGeometrySearchEngine::verify_leaf_candidates(
    const std::shared_ptr<WorldNode>& node,
    const BioSequence& query_seq,
    int tolerance,
    std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
    SearchStats& stats) const {
  ScopedSearchTimer collect_timer(stats.query_profile_enabled, &stats.leaf_collect_ms);
  stats.leaf_world_count++;
  std::vector<int> V_Q;
  const bool has_leaf_sieve =
      !node->beacons.empty() &&
      node->leaf_beacon_dists.size() == node->child_leaves.size();
  if (has_leaf_sieve) V_Q = compute_query_beacon_distances(node, query_seq, stats);

  for (size_t leaf_idx = 0; leaf_idx < node->child_leaves.size(); ++leaf_idx) {
    stats.node_access_count++;
    if (has_leaf_sieve && node->leaf_beacon_dists[leaf_idx].size() == V_Q.size()) {
      ScopedSearchTimer leaf_filter_timer(stats.query_profile_enabled,
                                          &stats.leaf_mbb_filter_ms);
      stats.leaf_beacon_check_count++;
      stats.candidate_count_for_prune++;
      stats.bound_check_count++;
      stats.leaf_beacon_scalar_checks++;
      if (leaf_beacon_prunable_row(node->leaf_beacon_dists[leaf_idx], V_Q, tolerance)) {
        stats.beacon_prune_count++;
        continue;
      }
    }

	    const auto& child = node->child_leaves[leaf_idx];
	    if (!child) continue;
	    if (near_query_triangle_prunes_leaf(this, child->id, tolerance, stats)) {
	      continue;
	    }
	    ScopedSearchTimer leaf_verify_timer(stats.query_profile_enabled,
	                                        &stats.leaf_verify_ms);
	    stats.candidate_count++;
	    stats.raw_candidate_count++;
	    stats.candidate_verify_count++;
	    stats.leaf_exact_distance_call_count++;
	    const bool cache_leaf_distance =
	        active_path_reuse_context(this) &&
	        active_path_reuse_context(this)->enabled;
	    const int distance_bound =
	        leaf_distance_cache_bound(config_, tolerance, cache_leaf_distance);
	    int leaf_dist = compute_query_distance_with_mode(
	        query_seq.seq, child->seq, distance_bound, config_.distance_mode);
	    if (auto* reuse = active_path_reuse_context(this);
	        reuse && reuse->enabled) {
	      reuse->next.leaf_dist_by_id[child->id] = leaf_dist;
	    }
	    stats.dist_calc_count++;
	    stats.leaf_verify_count++;
	    if (config_.trace_paths && child->sequence_id != INVALID_LEAF_ID) {
	      stats.leaf_trace.push_back(child->sequence_id);
	    }
	    if (leaf_dist <= tolerance) unique_results[child->id] = child;
  }
}

NAVIGAMER_QUERY_HOT_ALIGN
void BioGeometrySearchEngine::verify_leaf_candidates_view(
    NodeId node_id,
    const BioSequence& query_seq,
    int tolerance,
    std::unordered_set<LeafId>& unique_results,
    SearchStats& stats) const {
  ScopedSearchTimer collect_timer(stats.query_profile_enabled, &stats.leaf_collect_ms);
  stats.leaf_world_count++;
  const auto& view = index_.search_graph_view();
  if (node_id >= view.node_records.size()) {
    throw std::out_of_range("view node id is outside search graph view");
  }
  const auto node = view.node_records[node_id];

  const uint32_t leaf_begin = node.leaf_begin();
  const uint32_t leaf_count = view.link_count(node);
  const uint32_t beacon_count = view.beacon_count(node);
  const auto prefetch_leaf_code = [&](size_t leaf_offset) {
    switch (node.link_storage()) {
      case WorldNodeRecord::LinkStorage::Delta8:
        prefetch_read(
            view.leaf_id_deltas8.data() + leaf_begin + leaf_offset);
        break;
      case WorldNodeRecord::LinkStorage::Delta16:
        prefetch_read(
            view.leaf_id_deltas16.data() + leaf_begin + leaf_offset);
        break;
      case WorldNodeRecord::LinkStorage::Absolute32:
        prefetch_read(view.leaf_ids.data() + leaf_begin + leaf_offset);
        break;
      case WorldNodeRecord::LinkStorage::PackedDelta: {
        const size_t byte_offset =
            static_cast<size_t>(leaf_offset) *
            node.packed_leaf_bits() >> 3;
        prefetch_read(
            view.leaf_id_deltas8.data() + leaf_begin + byte_offset);
        break;
      }
    }
  };

  std::vector<int> V_Q;
  const bool has_leaf_sieve =
      beacon_count > 0 &&
      view.leaf_mbb_range_valid(
          node_id, node, leaf_count, beacon_count);
  if (has_leaf_sieve) {
    V_Q = compute_query_beacon_distances_view(node_id, query_seq, stats);
  }

  std::vector<uint32_t> survivor_offsets;
  const bool leaf_sieve_ready =
      has_leaf_sieve && beacon_count == V_Q.size();
  if (leaf_sieve_ready) {
    ScopedSearchTimer leaf_filter_timer(stats.query_profile_enabled,
                                        &stats.leaf_mbb_filter_ms);
    LeafBeaconFilterSimdStats simd_stats;
    const uint32_t offset = node.leaf_mbb_begin();
    if (config_.search_prefetch) {
      prefetch_read(view.leaf_beacon_dists.data() + offset);
      if (leaf_count != 0) prefetch_leaf_code(0);
    }
    survivor_offsets = filter_leaf_beacon_survivors(
        view.leaf_beacon_dists.data() + offset,
        leaf_count,
        beacon_count,
        V_Q.data(),
        static_cast<int32_t>(tolerance),
        config_.simd_mode,
        &simd_stats,
        view.leaf_mbb_bits(node_id, node));

    stats.node_access_count += leaf_count;
    stats.leaf_beacon_check_count += leaf_count;
    stats.candidate_count_for_prune += leaf_count;
    stats.bound_check_count += leaf_count;
    stats.beacon_prune_count += leaf_count - survivor_offsets.size();
    stats.leaf_beacon_scalar_checks += simd_stats.scalar_checks;
    stats.leaf_beacon_simd_batches += simd_stats.simd_batches;
    stats.leaf_beacon_simd_fallbacks += simd_stats.simd_fallbacks;
  } else {
    survivor_offsets.reserve(leaf_count);
    for (size_t leaf_idx = 0; leaf_idx < leaf_count; ++leaf_idx) {
      if (config_.search_prefetch && leaf_idx + 1 < leaf_count) {
        prefetch_leaf_code(leaf_idx + 1);
      }
      survivor_offsets.push_back(static_cast<uint32_t>(leaf_idx));
    }
  }

  const auto verify_survivors = [&](const auto& leaf_id_at) {
    for (size_t survivor_idx = 0;
         survivor_idx < survivor_offsets.size(); ++survivor_idx) {
      const uint32_t leaf_offset = survivor_offsets[survivor_idx];
      if (leaf_offset >= leaf_count) {
        throw std::runtime_error(
            "SIMD leaf beacon filter returned offset out of range");
      }
      if (config_.search_prefetch &&
          survivor_idx + 1 < survivor_offsets.size()) {
        const uint32_t next_offset = survivor_offsets[survivor_idx + 1];
        if (next_offset < leaf_count) {
          const LeafId next_leaf_id = leaf_id_at(next_offset);
          if (next_leaf_id < view.sequences.size()) {
            const auto& next_sequence =
                view.sequences.sequence(next_leaf_id);
            if (!next_sequence.empty()) {
              prefetch_read(next_sequence.data());
            }
          }
        }
      }
      if (!leaf_sieve_ready) stats.node_access_count++;
      const LeafId leaf_id = leaf_id_at(leaf_offset);
      if (leaf_id >= view.sequences.size()) {
        throw std::runtime_error("view leaf id has no sequence record");
      }
      if (near_query_triangle_prunes_leaf(
              this, leaf_id, tolerance, stats)) {
        continue;
      }
      ScopedSearchTimer leaf_verify_timer(
          stats.query_profile_enabled, &stats.leaf_verify_ms);
      stats.candidate_count++;
      stats.raw_candidate_count++;
      stats.candidate_verify_count++;
      const bool cache_leaf_distance =
          active_path_reuse_context(this) &&
          active_path_reuse_context(this)->enabled;
      const int distance_bound = leaf_distance_cache_bound(
          config_, tolerance, cache_leaf_distance);
      bool cache_hit = false;
      int leaf_dist = compute_indexed_query_distance(
          query_seq.seq, view.sequences.sequence(leaf_id), leaf_id,
          distance_bound, config_.distance_mode, false, &cache_hit);
      if (auto* reuse = active_path_reuse_context(this);
          reuse && reuse->enabled) {
        reuse->next.leaf_dist_by_sequence_id[leaf_id] = leaf_dist;
      }
      if (!cache_hit) {
        stats.leaf_exact_distance_call_count++;
        stats.dist_calc_count++;
      }
      stats.leaf_verify_count++;
      if (config_.trace_paths) {
        stats.leaf_trace.push_back(leaf_id);
      }
      if (leaf_dist <= tolerance) unique_results.insert(leaf_id);
    }
  };

  switch (node.link_storage()) {
    case WorldNodeRecord::LinkStorage::Delta8: {
      const int8_t* deltas =
          view.leaf_id_deltas8.data() + leaf_begin;
      const LeafId center_id = view.center_sequence_id(node_id);
      verify_survivors([&](uint32_t offset) {
        return center_id + deltas[offset];
      });
      break;
    }
    case WorldNodeRecord::LinkStorage::Delta16: {
      const int16_t* deltas =
          view.leaf_id_deltas16.data() + leaf_begin;
      const LeafId center_id = view.center_sequence_id(node_id);
      verify_survivors([&](uint32_t offset) {
        return center_id + deltas[offset];
      });
      break;
    }
    case WorldNodeRecord::LinkStorage::Absolute32: {
      const LeafId* leaf_ids = view.leaf_ids.data() + leaf_begin;
      verify_survivors(
          [&](uint32_t offset) { return leaf_ids[offset]; });
      break;
    }
    case WorldNodeRecord::LinkStorage::PackedDelta:
      verify_survivors([&](uint32_t offset) {
        return view.packed_leaf_id(node_id, offset);
      });
      break;
    default:
      throw std::runtime_error("view leaf ID encoding is invalid");
  }
}

void BioGeometrySearchEngine::process_node_adaptive(
    const std::shared_ptr<WorldNode>& node, int current_layer,
    const BioSequence& query_seq, int tolerance,
    std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
    std::unordered_set<std::string>& visited_nodes,
    SearchStats& stats,
    const QGramSignature* query_qgram_signature,
    const QGramSignature* router_qgram_signature,
    const std::vector<uint64_t>* router_minimizers) const {
  if (current_layer == index_.finest_primary_layer_index()) {
    verify_leaf_candidates(node, query_seq, tolerance, unique_results, stats);
    return;
  }

  if (node->child_nodes.empty()) return;

  int child_layer = current_layer + 1;
  std::vector<int> V_Q;
  if (!node->beacons.empty()) {
    V_Q = compute_query_beacon_distances(node, query_seq, stats);
  }
  bool used_safe_child_router = false;
  auto child_indices = safe_child_router_candidate_indices(
      node, query_seq, V_Q, tolerance, stats, &used_safe_child_router);
  auto surviving =
      used_safe_child_router
          ? scan_mbb_surviving_child_indices(node, child_indices, V_Q,
                                             tolerance, stats)
          : get_mbb_surviving_children(node, V_Q, tolerance, stats);
  auto ordered = rank_children_with_router_hints(
      node, surviving, query_seq, tolerance, stats, router_qgram_signature,
      router_minimizers);
  if (!used_safe_child_router) {
    ordered = rank_children_with_local_router(node, ordered, V_Q, stats);
    ordered = rank_children_with_best_first(node, ordered, V_Q, tolerance, stats);
  }
  ordered = apply_path_reuse_order(node, ordered, stats);

  search_layer_adaptive(ordered, child_layer, query_seq, tolerance,
                        unique_results, visited_nodes, stats, true,
                        query_qgram_signature, router_qgram_signature,
                        router_minimizers, node->node_id);
}

void BioGeometrySearchEngine::search_layer_adaptive(
    const std::vector<std::shared_ptr<WorldNode>>& candidates, int layer_id,
    const BioSequence& query_seq, int tolerance,
    std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
    std::unordered_set<std::string>& visited_nodes,
    SearchStats& stats,
    bool after_mbb_filter,
    const QGramSignature* query_qgram_signature,
    const QGramSignature* router_qgram_signature,
    const std::vector<uint64_t>* router_minimizers,
    const std::string& contained_parent_key) const {
  std::shared_ptr<WorldNode> contained_node;
  std::vector<std::shared_ptr<WorldNode>> overlap_nodes;
  update_frontier_stats(stats, candidates.size());

  if (auto* reuse = active_path_reuse_context(this);
      reuse && reuse->enabled && reuse->previous_near_query_match &&
      reuse->previous) {
    auto cached_it =
        reuse->previous->contained_child_by_parent.find(contained_parent_key);
    if (cached_it != reuse->previous->contained_child_by_parent.end()) {
      stats.contained_path_reuse_attempt_count++;
      for (const auto& node : candidates) {
        if (!node || node->node_id != cached_it->second) continue;
        stats.visited_check_count++;
        if (visited_nodes.count(node->node_id)) {
          stats.visited_hit_count++;
          break;
        }
        const int tau = node->radius + tolerance;
        stats.center_exact_distance_call_count++;
        stats.center_distance_count++;
        ScopedSearchTimer center_timer(stats.query_profile_enabled,
                                       &stats.center_distance_ms);
        const int dist = compute_center_distance_for_search(
            query_seq, node->node_id, INVALID_LEAF_ID,
            node->get_center_sequence(), tau,
            after_mbb_filter);
        stats.dist_calc_count++;
        stats.world_access_count++;
        if (config_.trace_paths && node->integer_id != INVALID_NODE_ID) {
          stats.world_trace.push_back(node->integer_id);
        }
        if (layer_id >= 0 &&
            static_cast<size_t>(layer_id) < stats.layer_breakdown.size()) {
          stats.layer_breakdown[static_cast<size_t>(layer_id)]++;
        }
        if (dist <= tau && dist + tolerance <= node->radius) {
          stats.contained_fastpath_count++;
          stats.contained_path_reuse_hit_count++;
          stats.path_reuse_hit_count++;
          stats.record_path_step(0, true);
          reuse->next.contained_child_by_parent[contained_parent_key] =
              node->node_id;
          visited_nodes.insert(node->node_id);
          process_node_adaptive(node, layer_id, query_seq, tolerance,
                                unique_results, visited_nodes, stats,
                                query_qgram_signature, router_qgram_signature,
                                router_minimizers);
          return;
        }
        break;
      }
    }
  }

  for (size_t candidate_idx = 0; candidate_idx < candidates.size();
       ++candidate_idx) {
    const auto& node = candidates[candidate_idx];
    if (config_.search_prefetch && candidate_idx + 1 < candidates.size()) {
      prefetch_world_node(candidates[candidate_idx + 1]);
    }
    stats.visited_check_count++;
    if (visited_nodes.count(node->node_id)) {
      stats.visited_hit_count++;
      continue;
    }

    const int tau = node->radius + tolerance;
    if (after_mbb_filter) {
      stats.center_distance_calls_after_mbb++;
      stats.center_distance_calls_before_qgram++;
      if (stats.search_qgram_prefilter_enabled) {
        const auto* candidate_signature = lookup_qgram_signature(
            world_qgram_signatures_by_q_, stats.search_qgram_q, node->node_id);
        if (!query_qgram_signature ||
            !query_qgram_signature->safe_for_pruning ||
            !candidate_signature ||
            !candidate_signature->safe_for_pruning) {
          stats.search_qgram_signature_missing_count++;
        } else {
          stats.search_qgram_checks++;
          if (qgram_can_prune_edit_distance(
                  *query_qgram_signature, *candidate_signature, tau)) {
            stats.search_qgram_pruned_children++;
            continue;
          }
          stats.search_qgram_passed_children++;
        }
      }
      stats.center_distance_calls_after_qgram++;
    }
    if (near_query_triangle_prunes_center(node->node_id, tau, stats)) {
      continue;
    }
    stats.center_exact_distance_call_count++;
    stats.center_distance_count++;
    ScopedSearchTimer center_timer(stats.query_profile_enabled,
                                   &stats.center_distance_ms);
    int dist = compute_center_distance_for_search(
        query_seq, node->node_id, INVALID_LEAF_ID,
        node->get_center_sequence(), tau,
        after_mbb_filter);
    stats.dist_calc_count++;
    stats.world_access_count++;
    if (config_.trace_paths && node->integer_id != INVALID_NODE_ID) {
      stats.world_trace.push_back(node->integer_id);
    }
    if (layer_id >= 0 && static_cast<size_t>(layer_id) < stats.layer_breakdown.size()) {
      stats.layer_breakdown[static_cast<size_t>(layer_id)]++;
    }

    if (dist > tau) continue;
    if (config_.proximal_oracle_enabled) {
      stats.proximal_frontier_node_ids.push_back(node->node_id);
    }

    if (dist + tolerance <= node->radius) {
      stats.contained_fastpath_count++;
      contained_node = node;
      break;
    }
    overlap_nodes.push_back(node);
  }

  stats.record_path_step(overlap_nodes.size(), static_cast<bool>(contained_node));
  if (contained_node) {
    if (auto* reuse = active_path_reuse_context(this);
        reuse && reuse->enabled) {
      reuse->next.contained_child_by_parent[contained_parent_key] =
          contained_node->node_id;
    }
    visited_nodes.insert(contained_node->node_id);
    process_node_adaptive(contained_node, layer_id, query_seq, tolerance,
                          unique_results, visited_nodes, stats,
                          query_qgram_signature, router_qgram_signature,
                          router_minimizers);
  } else {
    if (!overlap_nodes.empty()) stats.overlap_fallback_count++;
    for (const auto& node : overlap_nodes) {
      stats.visited_check_count++;
      if (visited_nodes.count(node->node_id)) {
        stats.visited_hit_count++;
        continue;
      }
      visited_nodes.insert(node->node_id);
      process_node_adaptive(node, layer_id, query_seq, tolerance,
                            unique_results, visited_nodes, stats,
                            query_qgram_signature, router_qgram_signature,
                            router_minimizers);
    }
  }
}

void BioGeometrySearchEngine::process_node_adaptive_epoch(
    const std::shared_ptr<WorldNode>& node, int current_layer,
    const BioSequence& query_seq, int tolerance,
    std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
    SearchScratch& scratch,
    SearchStats& stats,
    const QGramSignature* query_qgram_signature,
    const QGramSignature* router_qgram_signature,
    const std::vector<uint64_t>* router_minimizers) const {
  if (current_layer == index_.finest_primary_layer_index()) {
    verify_leaf_candidates(node, query_seq, tolerance, unique_results, stats);
    return;
  }

  if (node->child_nodes.empty()) return;

  int child_layer = current_layer + 1;
  std::vector<int> V_Q;
  if (!node->beacons.empty()) {
    V_Q = compute_query_beacon_distances(node, query_seq, stats);
  }
  bool used_safe_child_router = false;
  auto child_indices = safe_child_router_candidate_indices(
      node, query_seq, V_Q, tolerance, stats, &used_safe_child_router);
  auto surviving =
      used_safe_child_router
          ? scan_mbb_surviving_child_indices(node, child_indices, V_Q,
                                             tolerance, stats)
          : get_mbb_surviving_children(node, V_Q, tolerance, stats);
  auto ordered = rank_children_with_router_hints(
      node, surviving, query_seq, tolerance, stats, router_qgram_signature,
      router_minimizers);
  if (!used_safe_child_router) {
    ordered = rank_children_with_local_router(node, ordered, V_Q, stats);
    ordered = rank_children_with_best_first(node, ordered, V_Q, tolerance, stats);
  }
  ordered = apply_path_reuse_order(node, ordered, stats);

  search_layer_adaptive_epoch(ordered, child_layer, query_seq, tolerance,
                              unique_results, scratch, stats, true,
                              query_qgram_signature, router_qgram_signature,
                              router_minimizers, node->node_id);
}

void BioGeometrySearchEngine::search_layer_adaptive_epoch(
    const std::vector<std::shared_ptr<WorldNode>>& candidates, int layer_id,
    const BioSequence& query_seq, int tolerance,
    std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
    SearchScratch& scratch,
    SearchStats& stats,
    bool after_mbb_filter,
    const QGramSignature* query_qgram_signature,
    const QGramSignature* router_qgram_signature,
    const std::vector<uint64_t>* router_minimizers,
    const std::string& contained_parent_key) const {
  std::shared_ptr<WorldNode> contained_node;
  std::vector<std::shared_ptr<WorldNode>> overlap_nodes;
  update_frontier_stats(stats, candidates.size());

  if (auto* reuse = active_path_reuse_context(this);
      reuse && reuse->enabled && reuse->previous_near_query_match &&
      reuse->previous) {
    auto cached_it =
        reuse->previous->contained_child_by_parent.find(contained_parent_key);
    if (cached_it != reuse->previous->contained_child_by_parent.end()) {
      stats.contained_path_reuse_attempt_count++;
      for (const auto& node : candidates) {
        if (!node || node->node_id != cached_it->second) continue;
        stats.visited_check_count++;
        if (scratch.is_visited(node->integer_id)) {
          stats.visited_hit_count++;
          break;
        }
        const int tau = node->radius + tolerance;
        stats.center_exact_distance_call_count++;
        stats.center_distance_count++;
        ScopedSearchTimer center_timer(stats.query_profile_enabled,
                                       &stats.center_distance_ms);
        const int dist = compute_center_distance_for_search(
            query_seq, node->node_id, INVALID_LEAF_ID,
            node->get_center_sequence(), tau,
            after_mbb_filter);
        stats.dist_calc_count++;
        stats.world_access_count++;
        if (config_.trace_paths && node->integer_id != INVALID_NODE_ID) {
          stats.world_trace.push_back(node->integer_id);
        }
        if (layer_id >= 0 &&
            static_cast<size_t>(layer_id) < stats.layer_breakdown.size()) {
          stats.layer_breakdown[static_cast<size_t>(layer_id)]++;
        }
        if (dist <= tau && dist + tolerance <= node->radius) {
          stats.contained_fastpath_count++;
          stats.contained_path_reuse_hit_count++;
          stats.path_reuse_hit_count++;
          stats.record_path_step(0, true);
          reuse->next.contained_child_by_parent[contained_parent_key] =
              node->node_id;
          scratch.mark_visited(node->integer_id);
          process_node_adaptive_epoch(node, layer_id, query_seq, tolerance,
                                      unique_results, scratch, stats,
                                      query_qgram_signature,
                                      router_qgram_signature,
                                      router_minimizers);
          return;
        }
        break;
      }
    }
  }

  for (size_t candidate_idx = 0; candidate_idx < candidates.size();
       ++candidate_idx) {
    const auto& node = candidates[candidate_idx];
    if (config_.search_prefetch && candidate_idx + 1 < candidates.size()) {
      prefetch_world_node(candidates[candidate_idx + 1]);
    }
    stats.visited_check_count++;
    if (scratch.is_visited(node->integer_id)) {
      stats.visited_hit_count++;
      continue;
    }

    const int tau = node->radius + tolerance;
    if (after_mbb_filter) {
      stats.center_distance_calls_after_mbb++;
      stats.center_distance_calls_before_qgram++;
      if (stats.search_qgram_prefilter_enabled) {
        const auto* candidate_signature = lookup_qgram_signature(
            world_qgram_signatures_by_q_, stats.search_qgram_q, node->node_id);
        if (!query_qgram_signature ||
            !query_qgram_signature->safe_for_pruning ||
            !candidate_signature ||
            !candidate_signature->safe_for_pruning) {
          stats.search_qgram_signature_missing_count++;
        } else {
          stats.search_qgram_checks++;
          if (qgram_can_prune_edit_distance(
                  *query_qgram_signature, *candidate_signature, tau)) {
            stats.search_qgram_pruned_children++;
            continue;
          }
          stats.search_qgram_passed_children++;
        }
      }
      stats.center_distance_calls_after_qgram++;
    }
    if (near_query_triangle_prunes_center(node->node_id, tau, stats)) {
      continue;
    }
    stats.center_exact_distance_call_count++;
    stats.center_distance_count++;
    ScopedSearchTimer center_timer(stats.query_profile_enabled,
                                   &stats.center_distance_ms);
    int dist = compute_center_distance_for_search(
        query_seq, node->node_id, INVALID_LEAF_ID,
        node->get_center_sequence(), tau,
        after_mbb_filter);
    stats.dist_calc_count++;
    stats.world_access_count++;
    if (config_.trace_paths && node->integer_id != INVALID_NODE_ID) {
      stats.world_trace.push_back(node->integer_id);
    }
    if (layer_id >= 0 && static_cast<size_t>(layer_id) < stats.layer_breakdown.size()) {
      stats.layer_breakdown[static_cast<size_t>(layer_id)]++;
    }

    if (dist > tau) continue;
    if (config_.proximal_oracle_enabled) {
      stats.proximal_frontier_node_ids.push_back(node->node_id);
    }

    if (dist + tolerance <= node->radius) {
      stats.contained_fastpath_count++;
      contained_node = node;
      break;
    }
    overlap_nodes.push_back(node);
  }

  stats.record_path_step(overlap_nodes.size(), static_cast<bool>(contained_node));
  if (contained_node) {
    if (auto* reuse = active_path_reuse_context(this);
        reuse && reuse->enabled) {
      reuse->next.contained_child_by_parent[contained_parent_key] =
          contained_node->node_id;
    }
    scratch.mark_visited(contained_node->integer_id);
    process_node_adaptive_epoch(contained_node, layer_id, query_seq, tolerance,
                                unique_results, scratch, stats,
                                query_qgram_signature, router_qgram_signature,
                                router_minimizers);
  } else {
    if (!overlap_nodes.empty()) stats.overlap_fallback_count++;
    for (const auto& node : overlap_nodes) {
      stats.visited_check_count++;
      if (!scratch.mark_visited(node->integer_id)) {
        stats.visited_hit_count++;
        continue;
      }
      process_node_adaptive_epoch(node, layer_id, query_seq, tolerance,
                                  unique_results, scratch, stats,
                                  query_qgram_signature, router_qgram_signature,
                                  router_minimizers);
    }
  }
}

bool BioGeometrySearchEngine::flat_is_visited(
    NodeId node_id,
    const std::unordered_set<std::string>* visited_nodes,
    const SearchScratch* scratch) const {
  if (scratch) return scratch->is_visited(node_id);
  const auto& view = index_.search_graph_view();
  if (!visited_nodes || node_id >= view.node_records.size()) {
    throw std::runtime_error("invalid flat visited state");
  }
  return visited_nodes->count(node_key(node_id)) != 0;
}

bool BioGeometrySearchEngine::flat_mark_visited(
    NodeId node_id,
    std::unordered_set<std::string>* visited_nodes,
    SearchScratch* scratch) const {
  if (scratch) return scratch->mark_visited(node_id);
  const auto& view = index_.search_graph_view();
  if (!visited_nodes || node_id >= view.node_records.size()) {
    throw std::runtime_error("invalid flat visited state");
  }
  return visited_nodes->insert(node_key(node_id)).second;
}

NAVIGAMER_QUERY_HOT_ALIGN
std::vector<NodeId> BioGeometrySearchEngine::get_mbb_surviving_child_ids_view(
    NodeId node_id,
    const std::vector<int>& query_beacon_dists,
    int child_radius,
    int tolerance,
  SearchStats& stats) const {
  const auto& view = index_.search_graph_view();
  if (node_id >= view.node_records.size()) {
    throw std::out_of_range("view node id is outside search graph view");
  }
  const auto node = view.node_records[node_id];
  stats.mbb_filter_parent_count++;
  const uint32_t child_count = view.link_count(node);
  const uint32_t dim = view.beacon_count(node);

  const bool rect_requested =
      config_.mbb_filter_mode == MBBFilterMode::RectIndex;
  const bool rect_available =
      rect_requested &&
      child_count >= index_.build_range_config().min_rect_index_fanout;
  if (rect_requested) {
    if (rect_available) {
      stats.mbb_rect_index_queries++;
    } else {
      stats.mbb_rect_fallback_count++;
    }
  }

  const auto children = view.child_ids_for(node_id, node);
  const auto child_at = [&](uint32_t offset) {
    return children.at(offset);
  };
  const auto child_address = [&](uint32_t offset) -> const void* {
    return children.address(offset);
  };
  stats.child_edge_considered_count += child_count;
  const size_t mbb_begin = node.child_mbb_begin();
  const bool mbb_ok =
      !query_beacon_dists.empty() &&
      query_beacon_dists.size() == dim &&
      view.child_mbb_range_valid(
          node_id, node, child_count, dim);

  std::vector<NodeId> surviving;
  if (!mbb_ok) {
    surviving.reserve(child_count);
    for (size_t child_idx = 0; child_idx < child_count; ++child_idx) {
      if (config_.search_prefetch && child_idx + 1 < child_count) {
        prefetch_read(child_address(
            static_cast<uint32_t>(child_idx + 1)));
      }
      surviving.push_back(
          child_at(static_cast<uint32_t>(child_idx)));
    }
  } else {
    ScopedSearchTimer timer(stats.query_profile_enabled, &stats.mbb_filter_ms);
    const uint32_t mbb_bits = view.child_mbb_bits(node_id, node);
    const uint32_t quantization_bin_width =
        view.child_mbb_bin_width(node_id);
    MBBFilterSimdStats simd_stats;
    if (config_.search_prefetch) {
      prefetch_read(
          view.child_beacon_dists.data() + mbb_begin);
      prefetch_read(child_address(0));
    }
    auto survivor_offsets = filter_mbb_survivors(
        view.child_beacon_dists.data() + mbb_begin,
        child_count,
        dim,
        query_beacon_dists.data(),
        child_radius,
        static_cast<int32_t>(tolerance),
        config_.simd_mode,
        &simd_stats,
        mbb_bits,
        quantization_bin_width);

    stats.edge_access_count += child_count;
    stats.mbb_check_count += child_count;
    stats.candidate_count_for_prune += child_count;
    stats.bound_check_count += child_count;
    stats.mbb_scan_child_checks += child_count;
    stats.beacon_prune_count += child_count - survivor_offsets.size();
    stats.child_mbb_pruned_count += child_count - survivor_offsets.size();
    stats.mbb_scalar_checks += simd_stats.scalar_checks;
    stats.mbb_simd_batches += simd_stats.simd_batches;
    stats.mbb_simd_fallbacks += simd_stats.simd_fallbacks;

    surviving.reserve(survivor_offsets.size());
    for (size_t survivor_idx = 0; survivor_idx < survivor_offsets.size();
         ++survivor_idx) {
      const uint32_t child_offset = survivor_offsets[survivor_idx];
      if (child_offset >= child_count) {
        throw std::runtime_error("SIMD MBB filter returned child offset out of range");
      }
      if (config_.search_prefetch && survivor_idx + 1 < survivor_offsets.size()) {
        const uint32_t next_offset = survivor_offsets[survivor_idx + 1];
        if (next_offset < child_count)
          prefetch_read(view.node_records.record_data(
              child_at(next_offset)));
      }
      surviving.push_back(child_at(child_offset));
    }
  }
  if (rect_available) {
    stats.mbb_rect_candidate_children += surviving.size();
  }
  stats.mbb_surviving_child_count += surviving.size();
  return surviving;
}

std::vector<NodeId> BioGeometrySearchEngine::scan_mbb_surviving_child_ids_view(
    NodeId node_id,
    const std::vector<uint32_t>& child_offsets,
    const std::vector<int>& query_beacon_dists,
    int child_radius,
    int tolerance,
  SearchStats& stats) const {
  const auto& view = index_.search_graph_view();
  if (node_id >= view.node_records.size()) {
    throw std::out_of_range("view node id is outside search graph view");
  }
  const auto node = view.node_records[node_id];
  const uint32_t child_count = view.link_count(node);
  const uint32_t dim = view.beacon_count(node);
  const auto children = view.child_ids_for(node_id, node);
  const auto child_at = [&](size_t offset) {
    return children.at(static_cast<uint32_t>(offset));
  };
  stats.child_edge_considered_count += child_offsets.size();

  bool mbb_ok =
      !query_beacon_dists.empty() &&
      query_beacon_dists.size() == dim &&
      view.child_mbb_range_valid(
          node_id, node, child_count, dim);
  for (size_t child_offset : child_offsets) {
    if (child_offset >= child_count) {
      throw std::runtime_error("safe child router returned child offset out of range");
    }
  }

  std::vector<NodeId> surviving;
  surviving.reserve(child_offsets.size());
  if (!mbb_ok) {
    for (size_t child_offset : child_offsets) {
      surviving.push_back(child_at(child_offset));
    }
  } else {
    ScopedSearchTimer timer(stats.query_profile_enabled, &stats.mbb_filter_ms);
    const uint32_t mbb_bits = view.child_mbb_bits(node_id, node);
    const uint32_t mbb_begin = node.child_mbb_begin();
    const uint32_t quantization_bin_width =
        view.child_mbb_bin_width(node_id);
    const uint32_t quantization_error =
        SearchGraphView::child_mbb_quantization_error(
            quantization_bin_width);
    for (size_t child_offset : child_offsets) {
      stats.edge_access_count++;
      stats.mbb_check_count++;
      stats.candidate_count_for_prune++;
      stats.bound_check_count++;
      stats.mbb_scan_child_checks++;
      stats.mbb_scalar_checks++;
      bool prunable = false;
      for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
        const size_t cell = dim_idx * child_count + child_offset;
        const int q_b = query_beacon_dists[dim_idx];
        const int center_dist =
            view.child_beacon_distance_unchecked(
                mbb_begin, child_count, dim, cell, mbb_bits,
                quantization_bin_width);
        if (std::abs(
                static_cast<int64_t>(q_b) - center_dist) >
            static_cast<int64_t>(child_radius) + tolerance +
                quantization_error) {
          prunable = true;
          break;
        }
      }
      if (prunable) {
        stats.beacon_prune_count++;
        stats.child_mbb_pruned_count++;
        continue;
      }
      surviving.push_back(child_at(child_offset));
    }
  }
  stats.mbb_surviving_child_count += surviving.size();
  stats.post_mbb_survivor_count += surviving.size();
  if (!surviving.empty()) {
    stats.candidate_ratio_to_post_mbb_survivors +=
        static_cast<double>(child_offsets.size()) /
        static_cast<double>(surviving.size());
  } else if (!child_offsets.empty()) {
    stats.candidate_ratio_to_post_mbb_survivors +=
        static_cast<double>(child_offsets.size());
  }
  return surviving;
}

NAVIGAMER_QUERY_HOT_ALIGN
void BioGeometrySearchEngine::process_node_adaptive_view(
    NodeId node_id, int current_layer,
    const BioSequence& query_seq, int tolerance,
    std::unordered_set<LeafId>& unique_results,
    std::unordered_set<std::string>* visited_nodes,
    SearchScratch* scratch,
    SearchStats& stats,
    const QGramSignature* query_qgram_signature,
    const QGramSignature* router_qgram_signature,
  const std::vector<uint64_t>* router_minimizers) const {
  const auto& view = index_.search_graph_view();
  if (node_id >= view.node_records.size()) {
    throw std::out_of_range("view node id is outside search graph view");
  }
  if (current_layer == index_.finest_primary_layer_index()) {
    verify_leaf_candidates_view(node_id, query_seq, tolerance, unique_results, stats);
    return;
  }

  if (view.child_count(node_id) == 0) return;

  int child_layer = current_layer + 1;
  const int child_radius =
      index_.hierarchy_config().primary_radii.at(
          static_cast<size_t>(child_layer));
  std::vector<int> V_Q;
  if (view.beacon_count(node_id) > 0) {
    V_Q = compute_query_beacon_distances_view(node_id, query_seq, stats);
  }
  bool used_safe_child_router = false;
  auto child_offsets = safe_child_router_candidate_indices_view(
      node_id, query_seq, V_Q, child_radius, tolerance,
      stats, &used_safe_child_router);
  auto surviving =
      used_safe_child_router
          ? scan_mbb_surviving_child_ids_view(
                node_id, child_offsets, V_Q, child_radius,
                tolerance, stats)
          : get_mbb_surviving_child_ids_view(
                node_id, V_Q, child_radius, tolerance, stats);
  auto ordered = rank_child_ids_with_router_hints_view(
      node_id, surviving, query_seq, tolerance, stats, router_qgram_signature,
      router_minimizers);
  if (!used_safe_child_router) {
    ordered = rank_child_ids_with_local_router_view(
        node_id, ordered, V_Q, child_radius, stats);
    ordered = rank_child_ids_with_best_first_view(
        node_id, ordered, V_Q, child_radius, tolerance, stats);
  }
  ordered = apply_path_reuse_order_view(node_id, ordered, stats);

  std::string parent_key;
  if (config_.path_reuse_enabled) {
    parent_key = node_key(node_id);
  }
  search_layer_adaptive_view(ordered, child_layer, query_seq, tolerance,
                             unique_results, visited_nodes, scratch, stats, true,
                             query_qgram_signature, router_qgram_signature,
                             router_minimizers, parent_key);
}

void BioGeometrySearchEngine::search_layer_adaptive_view(
    const std::vector<NodeId>& candidates, int layer_id,
    const BioSequence& query_seq, int tolerance,
    std::unordered_set<LeafId>& unique_results,
    std::unordered_set<std::string>* visited_nodes,
    SearchScratch* scratch,
    SearchStats& stats,
    bool after_mbb_filter,
    const QGramSignature* query_qgram_signature,
    const QGramSignature* router_qgram_signature,
    const std::vector<uint64_t>* router_minimizers,
    const std::string& contained_parent_key) const {
  const auto& view = index_.search_graph_view();
  const int layer_radius =
      index_.hierarchy_config().primary_radii.at(
          static_cast<size_t>(layer_id));
  NodeId contained_node = INVALID_NODE_ID;
  std::vector<NodeId> overlap_nodes;
  update_frontier_stats(stats, candidates.size());

  if (auto* reuse = active_path_reuse_context(this);
      reuse && reuse->enabled && reuse->previous_near_query_match &&
      reuse->previous) {
    auto cached_it =
        reuse->previous->contained_child_by_parent.find(contained_parent_key);
    if (cached_it != reuse->previous->contained_child_by_parent.end()) {
      stats.contained_path_reuse_attempt_count++;
      for (NodeId node_id : candidates) {
        if (node_id >= view.node_records.size()) {
          throw std::out_of_range("view node id is outside search graph view");
        }
        const std::string key = node_key(node_id);
        if (key != cached_it->second) continue;
        const LeafId center_id = view.center_sequence_id(node_id);
        if (center_id >= view.sequences.size()) {
          throw std::runtime_error("array node center id is invalid");
        }
        stats.visited_check_count++;
        if (flat_is_visited(node_id, visited_nodes, scratch)) {
          stats.visited_hit_count++;
          break;
        }
        const int tau = layer_radius + tolerance;
        stats.center_distance_count++;
        ScopedSearchTimer center_timer(stats.query_profile_enabled,
                                       &stats.center_distance_ms);
        bool cache_hit = false;
        const int dist = compute_center_distance_for_search(
            query_seq, key, center_id,
            view.sequences.sequence(center_id), tau,
            after_mbb_filter, &cache_hit);
        if (!cache_hit) {
          stats.center_exact_distance_call_count++;
          stats.dist_calc_count++;
        }
        stats.world_access_count++;
        if (config_.trace_paths) {
          stats.world_trace.push_back(node_id);
        }
        if (layer_id >= 0 &&
            static_cast<size_t>(layer_id) < stats.layer_breakdown.size()) {
          stats.layer_breakdown[static_cast<size_t>(layer_id)]++;
        }
        if (dist <= tau && dist + tolerance <= layer_radius) {
          stats.contained_fastpath_count++;
          stats.contained_path_reuse_hit_count++;
          stats.path_reuse_hit_count++;
          stats.record_path_step(0, true);
          reuse->next.contained_child_by_parent[contained_parent_key] =
              key;
          flat_mark_visited(node_id, visited_nodes, scratch);
          process_node_adaptive_view(node_id, layer_id, query_seq, tolerance,
                                     unique_results, visited_nodes, scratch,
                                     stats, query_qgram_signature,
                                     router_qgram_signature,
                                     router_minimizers);
          return;
        }
        break;
      }
    }
  }

  for (size_t candidate_idx = 0; candidate_idx < candidates.size();
       ++candidate_idx) {
    const NodeId node_id = candidates[candidate_idx];
    if (node_id >= view.node_records.size()) {
      throw std::out_of_range("view node id is outside search graph view");
    }
    if (config_.search_prefetch && candidate_idx + 1 < candidates.size()) {
      const NodeId next_id = candidates[candidate_idx + 1];
      if (next_id < view.node_records.size()) {
        prefetch_read(view.node_records.record_data(next_id));
      }
    }
    const LeafId center_id = view.center_sequence_id(node_id);
    if (center_id >= view.sequences.size()) {
      throw std::runtime_error("array node center id is invalid");
    }
    std::string key;
    if (config_.path_reuse_enabled || config_.proximal_oracle_enabled ||
        stats.search_qgram_prefilter_enabled) {
      key = node_key(node_id);
    }
    stats.visited_check_count++;
    if (flat_is_visited(node_id, visited_nodes, scratch)) {
      stats.visited_hit_count++;
      continue;
    }

    const int tau = layer_radius + tolerance;
    if (after_mbb_filter) {
      stats.center_distance_calls_after_mbb++;
      stats.center_distance_calls_before_qgram++;
      if (stats.search_qgram_prefilter_enabled) {
        const auto* candidate_signature = lookup_qgram_signature(
            world_qgram_signatures_by_q_, stats.search_qgram_q, key);
        if (!query_qgram_signature ||
            !query_qgram_signature->safe_for_pruning ||
            !candidate_signature ||
            !candidate_signature->safe_for_pruning) {
          stats.search_qgram_signature_missing_count++;
        } else {
          stats.search_qgram_checks++;
          if (qgram_can_prune_edit_distance(
                  *query_qgram_signature, *candidate_signature, tau)) {
            stats.search_qgram_pruned_children++;
            continue;
          }
          stats.search_qgram_passed_children++;
        }
      }
      stats.center_distance_calls_after_qgram++;
    }
    if (config_.path_reuse_enabled &&
        near_query_triangle_prunes_center(key, tau, stats)) {
      continue;
    }
    stats.center_distance_count++;
    ScopedSearchTimer center_timer(stats.query_profile_enabled,
                                   &stats.center_distance_ms);
    bool cache_hit = false;
    int dist = compute_center_distance_for_search(
        query_seq, key, center_id,
        view.sequences.sequence(center_id), tau,
        after_mbb_filter, &cache_hit);
    if (!cache_hit) {
      stats.center_exact_distance_call_count++;
      stats.dist_calc_count++;
    }
    stats.world_access_count++;
    if (config_.trace_paths) {
      stats.world_trace.push_back(node_id);
    }
    if (layer_id >= 0 && static_cast<size_t>(layer_id) < stats.layer_breakdown.size()) {
      stats.layer_breakdown[static_cast<size_t>(layer_id)]++;
    }

    if (dist > tau) continue;
    if (config_.proximal_oracle_enabled) {
      stats.proximal_frontier_node_ids.push_back(key);
    }

    if (dist + tolerance <= layer_radius) {
      stats.contained_fastpath_count++;
      contained_node = node_id;
      break;
    }
    overlap_nodes.push_back(node_id);
  }

  stats.record_path_step(overlap_nodes.size(),
                         contained_node != INVALID_NODE_ID);
  if (contained_node != INVALID_NODE_ID) {
    if (auto* reuse = active_path_reuse_context(this);
        reuse && reuse->enabled) {
      reuse->next.contained_child_by_parent[contained_parent_key] =
          node_key(contained_node);
    }
    flat_mark_visited(contained_node, visited_nodes, scratch);
    process_node_adaptive_view(contained_node, layer_id, query_seq, tolerance,
                               unique_results, visited_nodes, scratch, stats,
                               query_qgram_signature, router_qgram_signature,
                               router_minimizers);
  } else {
    if (!overlap_nodes.empty()) stats.overlap_fallback_count++;
    for (NodeId node_id : overlap_nodes) {
      stats.visited_check_count++;
      if (!flat_mark_visited(node_id, visited_nodes, scratch)) {
        stats.visited_hit_count++;
        continue;
      }
      process_node_adaptive_view(node_id, layer_id, query_seq, tolerance,
                                 unique_results, visited_nodes, scratch, stats,
                                 query_qgram_signature, router_qgram_signature,
                                 router_minimizers);
    }
  }
}

NAVIGAMER_QUERY_HOT_ALIGN
std::pair<SearchResult, SearchStats>
BioGeometrySearchEngine::search_adaptive(const BioSequence& query_seq, int tolerance) {
  ActiveMyersQueryContext myers_context;
  ActiveMyersQueryContext* active_myers_context = nullptr;
  if (config_.distance_mode == DistanceMode::Myers ||
      config_.distance_mode == DistanceMode::Auto) {
    myers_context.query = &query_seq.seq;
    myers_context.pattern = prepare_myers_pattern(query_seq.seq);
    if (myers_context.pattern.supported) {
      active_myers_context = &myers_context;
    }
  }
  ScopedActiveMyersQueryContext scoped_myers(active_myers_context);
  g_query_distance_cache.begin_query();

  SearchStats stats(static_cast<size_t>(index_.num_primary_layers()));
  stats.query_profile_enabled = config_.query_profile;
  stats.query_count = 1;
  stats.search_prefetch_enabled = config_.search_prefetch;
  stats.trace_paths_enabled = config_.trace_paths;
  stats.safe_child_router_build_ms = safe_child_router_build_ms_;
  stats.query_similarity_schedule_enabled = config_.path_reuse_enabled;
  stats.query_similarity_cluster_count = config_.path_reuse_enabled ? 1 : 0;
  if (config_.query_planner_enabled) {
    ScopedSearchTimer timer(stats.query_profile_enabled,
                            &stats.planner_decision_ms);
    stats.planner_invoked_count = 1;
    const size_t max_fanout = max_primary_child_fanout(index_);
    const bool low_fanout =
        max_fanout < config_.planner_router_min_fanout &&
        max_fanout < config_.planner_safe_child_router_min_fanout;
    if (low_fanout) {
      stats.planner_strategy_baseline_count = 1;
      stats.planner_disable_router_stack = true;
    } else {
      stats.planner_strategy_router_count = 1;
      if (config_.safe_child_router_enabled &&
          max_fanout >= config_.planner_safe_child_router_min_fanout) {
        stats.planner_strategy_safe_child_router_count = 1;
      }
      if (config_.path_reuse_enabled) {
        stats.planner_strategy_path_reuse_count = 1;
      }
    }
  }
  ActiveRouterHintQueryContext router_hint_context;
  router_hint_context.enabled =
      config_.router_hint_enabled && !stats.planner_disable_router_stack;
  router_hint_context.query_sequence = &query_seq.seq;
  ActivePathReuseContext reuse_context;
  reuse_context.enabled =
      config_.path_reuse_enabled && !stats.planner_disable_router_stack;
  if (reuse_context.enabled) {
    auto& previous_cache = g_path_reuse_cache_by_engine[this];
    reuse_context.previous = &previous_cache;
    reuse_context.previous_exact_query_match =
        previous_cache.exact_query == query_seq.seq;
    const std::string fingerprint = build_path_reuse_fingerprint(query_seq.seq);
    reuse_context.previous_fingerprint_match =
        previous_cache.fingerprint == fingerprint &&
        !previous_cache.child_orders_by_parent.empty();
    const bool has_previous_query = !previous_cache.exact_query.empty();
    if (has_previous_query && !reuse_context.previous_exact_query_match) {
      stats.near_query_reuse_attempt_count++;
      const double jaccard =
          near_query_qgram_jaccard(previous_cache.exact_query, query_seq.seq);
      if (jaccard >= config_.near_query_min_qgram_jaccard) {
        const int neighbor_dist =
            compute_distance(previous_cache.exact_query, query_seq.seq);
        stats.dist_calc_count++;
        stats.query_similarity_mean_neighbor_distance =
            static_cast<double>(neighbor_dist);
        const int max_neighbor_dist = std::min(
            tolerance, std::max(0, config_.near_query_max_neighbor_edit_distance));
        if (neighbor_dist <= max_neighbor_dist) {
          reuse_context.previous_near_query_match = true;
          reuse_context.previous_neighbor_edit_distance = neighbor_dist;
          stats.near_query_reuse_hit_count++;
          stats.path_reuse_hit_count++;
        }
      }
    }
    reuse_context.next.exact_query = query_seq.seq;
    reuse_context.next.fingerprint = fingerprint;
    if (config_.query_planner_enabled) {
      if (reuse_context.previous_near_query_match) {
        stats.planner_near_reuse_enabled_count = 1;
      } else {
        stats.planner_near_reuse_disabled_count = 1;
      }
    }
  }
  ScopedActiveRouterHintQueryContext scoped_router_hint(
      this, router_hint_context.enabled ? &router_hint_context : nullptr);
  ScopedActivePathReuseContext scoped_reuse(this,
                                            reuse_context.enabled ? &reuse_context
                                                                  : nullptr);
  const auto query_start = std::chrono::steady_clock::now();
  if (reuse_context.enabled && reuse_context.previous_exact_query_match &&
      reuse_context.previous && reuse_context.previous->exact_query_completed &&
      reuse_context.previous->tolerance == tolerance) {
    ScopedSearchTimer timer(stats.query_profile_enabled, &stats.path_reuse_ms);
    stats.path_reuse_attempt_count++;
    const auto& view = index_.search_graph_view();
    SearchResult cached_hits;
    cached_hits.reserve(reuse_context.previous->exact_verified_hits.size());
    bool cache_valid = true;
    for (LeafId hit_id : reuse_context.previous->exact_verified_hits) {
      if (hit_id >= view.sequences.size()) {
        cache_valid = false;
        break;
      }
      const int distance_bound =
          leaf_distance_cache_bound(config_, tolerance, true);
      bool cache_hit = false;
      const int dist = compute_indexed_query_distance(
          query_seq.seq, view.sequences.sequence(hit_id), hit_id,
          distance_bound, config_.distance_mode, false, &cache_hit);
      reuse_context.next.leaf_dist_by_sequence_id[hit_id] = dist;
      if (!cache_hit) {
        stats.dist_calc_count++;
        stats.leaf_exact_distance_call_count++;
      }
      stats.candidate_verify_count++;
      stats.leaf_verify_count++;
      if (dist <= tolerance) {
        cached_hits.push_back(hit_id);
      } else {
        cache_valid = false;
        break;
      }
    }
    if (cache_valid) {
      stats.path_reuse_hit_count++;
      stats.productive_world_reuse_hit_count++;
      stats.anchor_cache_hit_count +=
          reuse_context.previous->anchor_dists_by_node.size();
      stats.result_count = cached_hits.size();
      if (stats.query_profile_enabled) {
        stats.query_total_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - query_start)
                .count();
      }
      reuse_context.next.tolerance = tolerance;
      reuse_context.next.exact_query_completed = true;
      reuse_context.next.exact_verified_hits.clear();
      reuse_context.next.exact_verified_hits.reserve(cached_hits.size());
      reuse_context.next.exact_verified_hits = cached_hits;
      g_path_reuse_cache_by_engine[this] = std::move(reuse_context.next);
      return {cached_hits, stats};
    }
  }
  stats.search_qgram_prefilter_enabled =
      config_.search_qgram_prefilter && !stats.planner_disable_router_stack &&
      config_.search_qgram_q > 0;
  stats.search_qgram_q =
      stats.search_qgram_prefilter_enabled ? config_.search_qgram_q : 0;
  stats.search_qgram_signature_build_count = 0;
  if (stats.search_qgram_prefilter_enabled) {
    auto search_q_it = world_qgram_signatures_by_q_.find(stats.search_qgram_q);
    if (search_q_it != world_qgram_signatures_by_q_.end()) {
      stats.search_qgram_signature_build_count = search_q_it->second.size();
    }
  }
  std::unordered_set<LeafId> unique_results;
  std::unordered_set<std::string> visited_nodes;
  QGramSignature query_qgram_signature;
  const QGramSignature* query_qgram_signature_ptr = nullptr;
  if (stats.search_qgram_prefilter_enabled) {
    query_qgram_signature =
        compute_qgram_signature(query_seq.seq, config_.search_qgram_q);
    query_qgram_signature_ptr = &query_qgram_signature;
  }
  const QGramSignature* router_qgram_signature_ptr = nullptr;
  if (config_.router_hint_enabled && config_.router_hint_qgram_q > 0) {
    router_hint_context.shared_qgram_q = config_.search_qgram_q;
    router_hint_context.shared_qgram_signature = query_qgram_signature_ptr;
    if (query_qgram_signature_ptr &&
        config_.router_hint_qgram_q == config_.search_qgram_q) {
      router_qgram_signature_ptr = query_qgram_signature_ptr;
    }
  }
  const std::vector<uint64_t>* router_minimizers_ptr = nullptr;

  if (reuse_context.enabled && reuse_context.previous_near_query_match &&
      reuse_context.previous && reuse_context.previous->exact_query_completed &&
      !reuse_context.previous->exact_verified_hits.empty()) {
    ScopedSearchTimer timer(stats.query_profile_enabled, &stats.path_reuse_ms);
    size_t direct_verified_hits = 0;
    const auto& view = index_.search_graph_view();
    for (LeafId hit_id : reuse_context.previous->exact_verified_hits) {
      if (hit_id >= view.sequences.size()) continue;
      stats.near_query_direct_verify_count++;
      stats.candidate_verify_count++;
      stats.leaf_verify_count++;
      const int distance_bound =
          leaf_distance_cache_bound(config_, tolerance, true);
      bool cache_hit = false;
      const int dist = compute_indexed_query_distance(
          query_seq.seq, view.sequences.sequence(hit_id), hit_id,
          distance_bound, config_.distance_mode, false, &cache_hit);
      if (!cache_hit) {
        stats.dist_calc_count++;
        stats.leaf_exact_distance_call_count++;
      }
      reuse_context.next.leaf_dist_by_sequence_id[hit_id] = dist;
      if (dist <= tolerance && unique_results.insert(hit_id).second) {
        direct_verified_hits++;
      }
    }
    if (direct_verified_hits > 0) {
      stats.productive_world_reuse_hit_count++;
    }
  }

  // "original" remains a CLI compatibility alias. Both values use the
  // canonical array representation.
  const auto& view = index_.search_graph_view();
  const size_t top_layer_idx =
      static_cast<size_t>(index_.coarsest_primary_layer_index());
  if (top_layer_idx >= view.layer_begin.size() ||
      top_layer_idx >= view.layer_end.size()) {
    throw std::runtime_error("array index has no top-layer range");
  }
  std::vector<NodeId> top_candidates;
  top_candidates.reserve(view.layer_end[top_layer_idx] -
                         view.layer_begin[top_layer_idx]);
  for (uint32_t node_id = view.layer_begin[top_layer_idx];
       node_id < view.layer_end[top_layer_idx]; ++node_id) {
    top_candidates.push_back(node_id);
  }

  const std::string root_key =
      contained_root_parent_key(index_.coarsest_primary_layer_index());
  if (config_.visited_mode == VisitedMode::StringSet) {
    search_layer_adaptive_view(
        top_candidates, index_.coarsest_primary_layer_index(), query_seq,
        tolerance, unique_results, &visited_nodes, nullptr, stats, false,
        query_qgram_signature_ptr, router_qgram_signature_ptr,
        router_minimizers_ptr, root_key);
  } else {
    thread_local SearchScratch scratch;
    scratch.begin_query(index_.num_world_nodes());
    search_layer_adaptive_view(
        top_candidates, index_.coarsest_primary_layer_index(), query_seq,
        tolerance, unique_results, nullptr, &scratch, stats, false,
        query_qgram_signature_ptr, router_qgram_signature_ptr,
        router_minimizers_ptr, root_key);
  }

  SearchResult out;
  {
    ScopedSearchTimer dedup_timer(stats.query_profile_enabled, &stats.result_dedup_ms);
    out.insert(out.end(), unique_results.begin(), unique_results.end());
  }
  stats.result_count = out.size();
  if (reuse_context.enabled) {
    reuse_context.next.tolerance = tolerance;
    reuse_context.next.exact_query_completed = true;
    reuse_context.next.exact_verified_hits.clear();
    reuse_context.next.exact_verified_hits.reserve(out.size());
    reuse_context.next.exact_verified_hits = out;
  }
  if (stats.query_profile_enabled) {
    stats.query_total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - query_start)
            .count();
  }
  if (reuse_context.enabled) {
    g_path_reuse_cache_by_engine[this] = std::move(reuse_context.next);
  }
  return {out, stats};
}

std::pair<SearchResult, SearchStats>
BioGeometrySearchEngine::search_greedy(const BioSequence& query_seq, int tolerance) {
  g_query_distance_cache.begin_query();
  SearchStats stats(static_cast<size_t>(index_.num_primary_layers()));
  const auto& view = index_.search_graph_view();
  const size_t top_layer =
      static_cast<size_t>(index_.coarsest_primary_layer_index());
  std::vector<NodeId> current;
  for (uint32_t node_id = view.layer_begin.at(top_layer);
       node_id < view.layer_end.at(top_layer); ++node_id) {
    current.push_back(node_id);
  }

  for (int layer_id = index_.coarsest_primary_layer_index();
       layer_id <= index_.finest_primary_layer_index(); ++layer_id) {
    const int layer_radius =
        index_.hierarchy_config().primary_radii.at(
            static_cast<size_t>(layer_id));
    NodeId best_node = INVALID_NODE_ID;
    int min_dist = std::numeric_limits<int>::max();

    for (NodeId node_id : current) {
      if (node_id >= view.node_records.size()) {
        throw std::out_of_range("greedy node id is outside array index");
      }
      const LeafId center_id = view.center_sequence_id(node_id);
      if (center_id >= view.sequences.size()) {
        throw std::runtime_error("greedy node center id is invalid");
      }
      int dist = compute_distance(
          query_seq.seq,
          view.sequences.sequence(center_id));
      stats.dist_calc_count++;
      stats.world_access_count++;
      if (layer_id >= 0 && static_cast<size_t>(layer_id) < stats.layer_breakdown.size()) {
        stats.layer_breakdown[static_cast<size_t>(layer_id)]++;
      }
      if (dist <= layer_radius + tolerance && dist < min_dist) {
        min_dist = dist;
        best_node = node_id;
      }
    }

    if (best_node == INVALID_NODE_ID) return {{}, stats};

    if (layer_id == index_.finest_primary_layer_index()) {
      std::unordered_set<LeafId> unique_results;
      verify_leaf_candidates_view(
          best_node, query_seq, tolerance, unique_results, stats);

      SearchResult results;
      results.insert(results.end(), unique_results.begin(), unique_results.end());
      return {results, stats};
    }

    std::vector<int> V_Q =
        compute_query_beacon_distances_view(best_node, query_seq, stats);
    current =
        get_mbb_surviving_child_ids_view(
            best_node, V_Q,
            index_.hierarchy_config().primary_radii.at(
                static_cast<size_t>(layer_id + 1)),
            tolerance, stats);
  }

  return {{}, stats};
}

void BioGeometrySearchEngine::traverse_exhaustive(
    const std::shared_ptr<WorldNode>& node, int current_layer,
    const BioSequence& query_seq, int tolerance,
    std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
    std::unordered_set<std::string>& visited_nodes,
    SearchStats& stats) const {
  if (visited_nodes.count(node->node_id)) return;
  visited_nodes.insert(node->node_id);

  int dist = compute_distance(query_seq.seq, node->get_center_sequence());
  stats.dist_calc_count++;
  stats.world_access_count++;
  if (current_layer >= 0 && static_cast<size_t>(current_layer) < stats.layer_breakdown.size()) {
    stats.layer_breakdown[static_cast<size_t>(current_layer)]++;
  }

  if (dist > node->radius + tolerance) return;

  for (const auto& child : node->child_nodes) {
    stats.edge_access_count++;
    traverse_exhaustive(child, current_layer + 1, query_seq, tolerance,
                        unique_results, visited_nodes, stats);
  }

  if (current_layer == index_.finest_primary_layer_index()) {
    verify_leaf_candidates(node, query_seq, tolerance, unique_results, stats);
  }
}

void BioGeometrySearchEngine::traverse_exhaustive_view(
    NodeId node_id, int current_layer,
    const BioSequence& query_seq, int tolerance,
    std::unordered_set<LeafId>& unique_results,
    std::vector<uint8_t>& visited_nodes,
    SearchStats& stats) const {
  const auto& view = index_.search_graph_view();
  if (node_id >= view.node_records.size() ||
      node_id >= visited_nodes.size()) {
    throw std::out_of_range("exhaustive node id is outside array index");
  }
  if (visited_nodes[node_id]) return;
  visited_nodes[node_id] = 1;
  const LeafId center_id = view.center_sequence_id(node_id);
  if (center_id >= view.sequences.size()) {
    throw std::runtime_error("exhaustive node center id is invalid");
  }
  const int dist = compute_distance(
      query_seq.seq,
      view.sequences.sequence(center_id));
  stats.dist_calc_count++;
  stats.world_access_count++;
  if (current_layer >= 0 &&
      static_cast<size_t>(current_layer) < stats.layer_breakdown.size()) {
    stats.layer_breakdown[static_cast<size_t>(current_layer)]++;
  }
  const int layer_radius =
      index_.hierarchy_config().primary_radii.at(
          static_cast<size_t>(current_layer));
  if (dist > layer_radius + tolerance) return;

  if (current_layer == index_.finest_primary_layer_index()) {
    verify_leaf_candidates_view(
        node_id, query_seq, tolerance, unique_results, stats);
    return;
  }
  const uint32_t child_count = view.child_count(node_id);
  for (uint32_t child_idx = 0; child_idx < child_count; ++child_idx) {
    stats.edge_access_count++;
    traverse_exhaustive_view(
        view.child_id(node_id, child_idx), current_layer + 1,
        query_seq, tolerance, unique_results, visited_nodes, stats);
  }
}

std::pair<SearchResult, SearchStats>
BioGeometrySearchEngine::search_exhaustive(const BioSequence& query_seq, int tolerance) {
  g_query_distance_cache.begin_query();
  SearchStats stats(static_cast<size_t>(index_.num_primary_layers()));
  std::unordered_set<LeafId> unique_results;
  const auto& view = index_.search_graph_view();
  std::vector<uint8_t> visited_nodes(view.node_records.size(), 0);
  const size_t top_layer =
      static_cast<size_t>(index_.coarsest_primary_layer_index());
  for (uint32_t node_id = view.layer_begin.at(top_layer);
       node_id < view.layer_end.at(top_layer); ++node_id) {
    traverse_exhaustive_view(
        node_id, index_.coarsest_primary_layer_index(), query_seq, tolerance,
        unique_results, visited_nodes, stats);
  }

  SearchResult out;
  out.insert(out.end(), unique_results.begin(), unique_results.end());
  return {out, stats};
}

std::pair<SearchResult, SearchStats>
BioGeometrySearchEngine::search_brute_force(
    const BioSequence& query_seq, int tolerance) {
  SearchStats stats(static_cast<size_t>(index_.num_primary_layers()));
  const auto& sequence_store = index_.sequence_store();
  std::vector<SearchResult> thread_results;

  #pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    #pragma omp single
    thread_results.resize(static_cast<size_t>(nthreads));

    int tid = omp_get_thread_num();
    #pragma omp for schedule(dynamic, 64)
    for (size_t i = 0; i < sequence_store.size(); ++i) {
      int dist = compute_distance(
          query_seq.seq,
          sequence_store.sequence(static_cast<LeafId>(i)));
      if (dist <= tolerance) {
        thread_results[static_cast<size_t>(tid)].push_back(
            static_cast<LeafId>(i));
      }
    }
  }

  SearchResult results;
  for (auto& thread_vec : thread_results) {
    for (auto& result : thread_vec) results.push_back(std::move(result));
  }

  stats.dist_calc_count = sequence_store.size();
  stats.leaf_verify_count = sequence_store.size();
  return {results, stats};
}

}  // namespace navigamer

#undef NAVIGAMER_QUERY_HOT_ALIGN
