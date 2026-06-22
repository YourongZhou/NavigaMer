#include "index_persistence.hpp"

#include "mbb_rect_index.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace navigamer {

namespace {

constexpr std::array<char, 8> kMagic = {'N', 'G', 'I', 'D', 'X', '0', '0', '1'};

template <typename T>
void write_pod(std::ostream& out, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
  if (!out) throw std::runtime_error("failed to write index file");
}

template <typename T>
T read_pod(std::istream& in, const char* field) {
  T value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!in) throw std::runtime_error(std::string("failed to read index field: ") + field);
  return value;
}

void write_size(std::ostream& out, size_t value) {
  if (value > std::numeric_limits<uint64_t>::max()) {
    throw std::runtime_error("index value exceeds 64-bit storage range");
  }
  write_pod<uint64_t>(out, static_cast<uint64_t>(value));
}

size_t read_size(std::istream& in, const char* field) {
  uint64_t value = read_pod<uint64_t>(in, field);
  if (value > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw std::runtime_error(std::string("index field exceeds size_t range: ") + field);
  }
  return static_cast<size_t>(value);
}

void write_bool(std::ostream& out, bool value) {
  write_pod<uint8_t>(out, value ? 1 : 0);
}

bool read_bool(std::istream& in, const char* field) {
  uint8_t value = read_pod<uint8_t>(in, field);
  if (value > 1) throw std::runtime_error(std::string("invalid bool field: ") + field);
  return value != 0;
}

void write_string(std::ostream& out, const std::string& value) {
  write_size(out, value.size());
  out.write(value.data(), static_cast<std::streamsize>(value.size()));
  if (!out) throw std::runtime_error("failed to write string field");
}

std::string read_string(std::istream& in, const char* field) {
  size_t size = read_size(in, field);
  std::string value(size, '\0');
  if (size > 0) {
    in.read(&value[0], static_cast<std::streamsize>(size));
    if (!in) throw std::runtime_error(std::string("failed to read string field: ") + field);
  }
  return value;
}

void write_int_vector(std::ostream& out, const std::vector<int>& values) {
  write_size(out, values.size());
  for (int value : values) write_pod<int32_t>(out, static_cast<int32_t>(value));
}

std::vector<int> read_int_vector(std::istream& in, const char* field) {
  size_t count = read_size(in, field);
  std::vector<int> values;
  values.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    values.push_back(static_cast<int>(read_pod<int32_t>(in, field)));
  }
  return values;
}

uint64_t fnv1a_update(uint64_t hash, const char* data, size_t size) {
  constexpr uint64_t prime = 1099511628211ULL;
  for (size_t i = 0; i < size; ++i) {
    hash ^= static_cast<unsigned char>(data[i]);
    hash *= prime;
  }
  return hash;
}

std::string hex64(uint64_t value) {
  std::ostringstream os;
  os << std::hex << std::setw(16) << std::setfill('0') << value;
  return os.str();
}

std::string hash_string(const std::string& value) {
  uint64_t hash = 1469598103934665603ULL;
  hash = fnv1a_update(hash, value.data(), value.size());
  return hex64(hash);
}

bool file_exists(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return in.good();
}

std::string fingerprint_input(const std::string& value) {
  if (!file_exists(value)) {
    return "literal:" + std::to_string(value.size()) + ":" + hash_string(value);
  }

  std::ifstream in(value, std::ios::binary);
  if (!in) throw std::runtime_error("unable to fingerprint input file: " + value);
  uint64_t hash = 1469598103934665603ULL;
  uint64_t size = 0;
  std::array<char, 8192> buffer{};
  while (in) {
    in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    std::streamsize got = in.gcount();
    if (got > 0) {
      hash = fnv1a_update(hash, buffer.data(), static_cast<size_t>(got));
      size += static_cast<uint64_t>(got);
    }
  }
  return "file:" + value + ":" + std::to_string(size) + ":" + hex64(hash);
}

std::string manifest_signature_payload(const IndexBuildManifest& manifest) {
  std::ostringstream os;
  os << "format=" << manifest.format_version << '\n';
  os << "ref=" << manifest.ref_fingerprint << '\n';
  os << "reads=" << manifest.reads_fingerprint << '\n';
  auto emit_ints = [&](const char* name, const std::vector<int>& values) {
    os << name << '=';
    for (size_t i = 0; i < values.size(); ++i) {
      if (i) os << ',';
      os << values[i];
    }
    os << '\n';
  };
  emit_ints("primary", manifest.primary_radii);
  emit_ints("auxiliary", manifest.auxiliary_radii);
  os << "link=" << manifest.link_mode << '\n';
  os << "leaf_mode=" << manifest.leaf_attach_mode << '\n';
  os << "leaf_direction=" << manifest.leaf_attach_direction << '\n';
  os << "build_distance=" << manifest.build_distance_mode << '\n';
  os << "phase1=" << manifest.phase1_candidate_mode << '\n';
  os << "range_mode=" << manifest.range_candidate_mode << '\n';
  os << "min_seed=" << manifest.range_min_seed_len << '\n';
  os << "max_seed=" << manifest.range_max_seed_len << '\n';
  os << "qgram=" << manifest.qgram_q << '\n';
  os << "auto_max_candidates=" << manifest.auto_pigeonhole_max_candidates << '\n';
  os << "auto_ratio=" << std::setprecision(17)
     << manifest.auto_pigeonhole_max_ratio << '\n';
  os << "auto_hybrid=" << manifest.auto_hybrid_on_large_candidates << '\n';
  os << "rect_fanout=" << manifest.min_rect_index_fanout << '\n';
  os << "phase1_metric_min=" << manifest.phase1_metric_min_fanout << '\n';
  os << "phase1_qgram_min=" << manifest.phase1_qgram_min_fanout << '\n';
  os << "phase1_qgram_max_touched=" << manifest.phase1_qgram_max_touched << '\n';
  os << "phase2_qgram_postfilter=" << manifest.phase2_qgram_postfilter << '\n';
  if (manifest.format_version >= 2) {
    os << "leaf_qgram_postfilter=" << manifest.leaf_qgram_postfilter << '\n';
  }
  return os.str();
}

void refresh_signature(IndexBuildManifest& manifest) {
  manifest.signature = hash_string(manifest_signature_payload(manifest));
}

BuildRangeConfig build_config_from_manifest(const IndexBuildManifest& manifest) {
  BuildRangeConfig config;
  config.link_mode = parse_build_range_mode(manifest.link_mode);
  config.leaf_attach_mode = parse_build_range_mode(manifest.leaf_attach_mode);
  config.leaf_attach_direction =
      parse_leaf_attach_direction(manifest.leaf_attach_direction);
  config.distance_mode = parse_build_distance_mode(manifest.build_distance_mode);
  config.phase1_candidate_mode =
      parse_phase1_candidate_mode(manifest.phase1_candidate_mode);
  config.range_join.min_seed_len = manifest.range_min_seed_len;
  config.range_join.max_seed_len = manifest.range_max_seed_len;
  config.range_join.qgram_q = manifest.qgram_q;
  config.range_join.candidate_mode =
      parse_range_candidate_mode(manifest.range_candidate_mode);
  config.range_join.auto_pigeonhole_max_candidates =
      manifest.auto_pigeonhole_max_candidates;
  config.range_join.auto_pigeonhole_max_ratio =
      manifest.auto_pigeonhole_max_ratio;
  config.range_join.auto_hybrid_on_large_candidates =
      manifest.auto_hybrid_on_large_candidates;
  config.min_rect_index_fanout = manifest.min_rect_index_fanout;
  config.phase1_metric_min_fanout = manifest.phase1_metric_min_fanout;
  config.phase1_qgram_min_fanout = manifest.phase1_qgram_min_fanout;
  config.phase1_qgram_max_touched = manifest.phase1_qgram_max_touched;
  config.phase2_qgram_postfilter = manifest.phase2_qgram_postfilter;
  config.leaf_qgram_postfilter = manifest.leaf_qgram_postfilter;
  return config;
}

void write_manifest(std::ostream& out, const IndexBuildManifest& manifest) {
  write_pod<uint32_t>(out, manifest.format_version);
  write_string(out, manifest.signature);
  write_string(out, manifest.ref_input);
  write_string(out, manifest.reads_input);
  write_string(out, manifest.ref_fingerprint);
  write_string(out, manifest.reads_fingerprint);
  write_int_vector(out, manifest.primary_radii);
  write_int_vector(out, manifest.auxiliary_radii);
  write_string(out, manifest.link_mode);
  write_string(out, manifest.leaf_attach_mode);
  write_string(out, manifest.leaf_attach_direction);
  write_string(out, manifest.build_distance_mode);
  write_string(out, manifest.phase1_candidate_mode);
  write_string(out, manifest.range_candidate_mode);
  write_pod<int32_t>(out, manifest.range_min_seed_len);
  write_pod<int32_t>(out, manifest.range_max_seed_len);
  write_pod<int32_t>(out, manifest.qgram_q);
  write_size(out, manifest.auto_pigeonhole_max_candidates);
  write_pod<double>(out, manifest.auto_pigeonhole_max_ratio);
  write_bool(out, manifest.auto_hybrid_on_large_candidates);
  write_size(out, manifest.min_rect_index_fanout);
  write_size(out, manifest.phase1_metric_min_fanout);
  write_size(out, manifest.phase1_qgram_min_fanout);
  write_size(out, manifest.phase1_qgram_max_touched);
  write_bool(out, manifest.phase2_qgram_postfilter);
  write_bool(out, manifest.leaf_qgram_postfilter);
  write_size(out, manifest.sequence_count);
  write_size(out, manifest.world_node_count);
  write_size(out, manifest.edge_count);
  write_size(out, manifest.leaf_link_count);
}

IndexBuildManifest read_manifest(std::istream& in) {
  IndexBuildManifest manifest;
  manifest.format_version = read_pod<uint32_t>(in, "format_version");
  if (manifest.format_version < 1 || manifest.format_version > 2) {
    throw std::runtime_error("unsupported NavigaMer index format version");
  }
  manifest.signature = read_string(in, "signature");
  manifest.ref_input = read_string(in, "ref_input");
  manifest.reads_input = read_string(in, "reads_input");
  manifest.ref_fingerprint = read_string(in, "ref_fingerprint");
  manifest.reads_fingerprint = read_string(in, "reads_fingerprint");
  manifest.primary_radii = read_int_vector(in, "primary_radii");
  manifest.auxiliary_radii = read_int_vector(in, "auxiliary_radii");
  manifest.link_mode = read_string(in, "link_mode");
  manifest.leaf_attach_mode = read_string(in, "leaf_attach_mode");
  manifest.leaf_attach_direction = read_string(in, "leaf_attach_direction");
  manifest.build_distance_mode = read_string(in, "build_distance_mode");
  manifest.phase1_candidate_mode = read_string(in, "phase1_candidate_mode");
  manifest.range_candidate_mode = read_string(in, "range_candidate_mode");
  manifest.range_min_seed_len = static_cast<int>(read_pod<int32_t>(in, "range_min_seed_len"));
  manifest.range_max_seed_len = static_cast<int>(read_pod<int32_t>(in, "range_max_seed_len"));
  manifest.qgram_q = static_cast<int>(read_pod<int32_t>(in, "qgram_q"));
  manifest.auto_pigeonhole_max_candidates =
      read_size(in, "auto_pigeonhole_max_candidates");
  manifest.auto_pigeonhole_max_ratio =
      read_pod<double>(in, "auto_pigeonhole_max_ratio");
  manifest.auto_hybrid_on_large_candidates =
      read_bool(in, "auto_hybrid_on_large_candidates");
  manifest.min_rect_index_fanout = read_size(in, "min_rect_index_fanout");
  manifest.phase1_metric_min_fanout = read_size(in, "phase1_metric_min_fanout");
  manifest.phase1_qgram_min_fanout = read_size(in, "phase1_qgram_min_fanout");
  manifest.phase1_qgram_max_touched = read_size(in, "phase1_qgram_max_touched");
  manifest.phase2_qgram_postfilter = read_bool(in, "phase2_qgram_postfilter");
  if (manifest.format_version >= 2) {
    manifest.leaf_qgram_postfilter = read_bool(in, "leaf_qgram_postfilter");
  } else {
    manifest.leaf_qgram_postfilter = false;
  }
  manifest.sequence_count = read_size(in, "sequence_count");
  manifest.world_node_count = read_size(in, "world_node_count");
  manifest.edge_count = read_size(in, "edge_count");
  manifest.leaf_link_count = read_size(in, "leaf_link_count");
  return manifest;
}

void write_magic(std::ostream& out) {
  out.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
  if (!out) throw std::runtime_error("failed to write index magic");
}

void read_magic(std::istream& in) {
  std::array<char, 8> magic{};
  in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!in) throw std::runtime_error("failed to read index magic");
  if (magic != kMagic) throw std::runtime_error("not a NavigaMer index file");
}

std::vector<std::shared_ptr<BioSequence>> sequences_by_id(
    const BioGeometryIndexBuilder& builder) {
  std::vector<std::shared_ptr<BioSequence>> sequences(builder.num_sequences());
  for (const auto& entry : builder.unique_sequences) {
    const auto& sequence = entry.second;
    if (!sequence || sequence->sequence_id >= sequences.size()) {
      throw std::runtime_error("cannot persist index with invalid sequence ids");
    }
    sequences[sequence->sequence_id] = sequence;
  }
  for (const auto& sequence : sequences) {
    if (!sequence) throw std::runtime_error("cannot persist sparse sequence id table");
  }
  return sequences;
}

std::vector<std::shared_ptr<WorldNode>> nodes_by_id(
    const std::vector<std::vector<std::shared_ptr<WorldNode>>>& layers,
    size_t node_count) {
  std::vector<std::shared_ptr<WorldNode>> nodes(node_count);
  for (const auto& layer : layers) {
    for (const auto& node : layer) {
      if (!node || node->integer_id >= node_count) {
        throw std::runtime_error("cannot persist index with invalid node ids");
      }
      nodes[node->integer_id] = node;
    }
  }
  for (const auto& node : nodes) {
    if (!node) throw std::runtime_error("cannot persist sparse node id table");
  }
  return nodes;
}

uint32_t checked_node_id(const std::shared_ptr<WorldNode>& node) {
  if (!node || node->integer_id == INVALID_NODE_ID) {
    throw std::runtime_error("cannot persist invalid node reference");
  }
  return node->integer_id;
}

uint32_t checked_leaf_id(const std::shared_ptr<BioSequence>& sequence) {
  if (!sequence || sequence->sequence_id == INVALID_LEAF_ID) {
    throw std::runtime_error("cannot persist invalid leaf reference");
  }
  return sequence->sequence_id;
}

void write_u32_vector(std::ostream& out, const std::vector<uint32_t>& values) {
  write_size(out, values.size());
  for (uint32_t value : values) write_pod<uint32_t>(out, value);
}

std::vector<uint32_t> read_u32_vector(std::istream& in, const char* field) {
  size_t count = read_size(in, field);
  std::vector<uint32_t> values;
  values.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    values.push_back(read_pod<uint32_t>(in, field));
  }
  return values;
}

size_t count_edges(
    const std::vector<std::vector<std::shared_ptr<WorldNode>>>& layers) {
  size_t total = 0;
  for (const auto& layer : layers)
    for (const auto& node : layer)
      if (node) total += node->child_nodes.size();
  return total;
}

size_t count_leaf_links(
    const std::vector<std::vector<std::shared_ptr<WorldNode>>>& layers) {
  size_t total = 0;
  for (const auto& layer : layers)
    for (const auto& node : layer)
      if (node) total += node->child_leaves.size();
  return total;
}

void rebuild_rect_indexes(BioGeometryIndexBuilder& builder);

}  // namespace

class IndexPersistenceAccess {
 public:
  static const std::vector<std::vector<std::shared_ptr<WorldNode>>>&
  primary_layers(const BioGeometryIndexBuilder& builder) {
    return builder.primary_layers_;
  }

  static void reset_loaded_state(
      BioGeometryIndexBuilder& builder,
      std::unordered_map<std::string, std::shared_ptr<BioSequence>> unique_sequences,
      std::vector<std::vector<std::shared_ptr<WorldNode>>> primary_layers,
      size_t world_node_count,
      size_t sequence_count,
      const IndexBuildManifest& manifest) {
    builder.stats_ = BioGeometryIndexBuilder::Statistics{};
    builder.stats_.added_sequences = sequence_count;
    builder.stats_.unique_sequences = sequence_count;
    builder.stats_.created_primary_nodes.assign(primary_layers.size(), 0);
    for (size_t i = 0; i < primary_layers.size(); ++i) {
      builder.stats_.created_primary_nodes[i] = primary_layers[i].size();
    }
    builder.stats_.phase2_edges_added = manifest.edge_count;
    builder.stats_.leaf_attachments_added = manifest.leaf_link_count;
    builder.unique_sequences = std::move(unique_sequences);
    builder.world_node_count_ = world_node_count;
    builder.sequence_count_ = sequence_count;
    builder.primary_layers_ = std::move(primary_layers);
    builder.extended_layers_.clear();
    builder.search_graph_view_ = SearchGraphView{};
    rebuild_rect_indexes(builder);
    builder.build_search_graph_view();
  }

  static BuildRangeConfig& range_config(BioGeometryIndexBuilder& builder) {
    return builder.range_config_;
  }
};

namespace {

void rebuild_rect_indexes(BioGeometryIndexBuilder& builder) {
  BuildRangeConfig& config = IndexPersistenceAccess::range_config(builder);
  for (const auto& layer : IndexPersistenceAccess::primary_layers(builder)) {
    for (const auto& node : layer) {
      if (!node) continue;
      node->mbb_rect_index.reset();
      if (node->child_nodes.size() < config.min_rect_index_fanout ||
          node->child_nodes.size() > std::numeric_limits<uint32_t>::max() ||
          node->beacons.empty() ||
          node->child_beacon_mbbs.size() != node->child_nodes.size()) {
        continue;
      }
      bool valid = true;
      std::vector<MBBRectIndex::Rect> rects;
      rects.reserve(node->child_nodes.size());
      for (size_t child_idx = 0; child_idx < node->child_nodes.size(); ++child_idx) {
        const auto& row = node->child_beacon_mbbs[child_idx];
        if (row.size() != node->beacons.size()) {
          valid = false;
          break;
        }
        MBBRectIndex::Rect rect;
        rect.child_id = static_cast<uint32_t>(child_idx);
        rect.lo.reserve(row.size());
        rect.hi.reserve(row.size());
        for (const auto& mbb : row) {
          rect.lo.push_back(mbb.min_dist);
          rect.hi.push_back(mbb.max_dist);
        }
        rects.push_back(std::move(rect));
      }
      if (!valid) continue;
      auto rect_index = std::make_shared<MBBRectIndex>();
      rect_index->build(rects);
      if (rect_index->size() == node->child_nodes.size() &&
          rect_index->dim() == node->beacons.size()) {
        node->mbb_rect_index = std::move(rect_index);
      }
    }
  }
}

void write_sequences(std::ostream& out,
                     const std::vector<std::shared_ptr<BioSequence>>& sequences) {
  write_size(out, sequences.size());
  for (const auto& sequence : sequences) {
    write_string(out, sequence->id);
    write_string(out, sequence->seq);
    write_pod<uint32_t>(out, sequence->sequence_id);
    write_pod<int64_t>(out, sequence->bwt_interval.start);
    write_pod<int64_t>(out, sequence->bwt_interval.end);
    write_size(out, sequence->ref_positions.size());
    for (const auto& pos : sequence->ref_positions) {
      write_string(out, pos.ref_id);
      write_pod<int32_t>(out, static_cast<int32_t>(pos.start));
      write_pod<int32_t>(out, static_cast<int32_t>(pos.end));
      write_string(out, pos.strand);
    }
  }
}

std::vector<std::shared_ptr<BioSequence>> read_sequences(std::istream& in) {
  size_t count = read_size(in, "sequence_count");
  std::vector<std::shared_ptr<BioSequence>> sequences(count);
  for (size_t i = 0; i < count; ++i) {
    std::string id = read_string(in, "sequence.id");
    std::string seq = read_string(in, "sequence.seq");
    uint32_t sequence_id = read_pod<uint32_t>(in, "sequence.sequence_id");
    if (sequence_id >= count) throw std::runtime_error("sequence id out of range");
    auto sequence = std::make_shared<BioSequence>(id, seq);
    sequence->sequence_id = sequence_id;
    sequence->bwt_interval.start = read_pod<int64_t>(in, "sequence.bwt_start");
    sequence->bwt_interval.end = read_pod<int64_t>(in, "sequence.bwt_end");
    size_t pos_count = read_size(in, "sequence.ref_position_count");
    sequence->ref_positions.reserve(pos_count);
    for (size_t pos_idx = 0; pos_idx < pos_count; ++pos_idx) {
      RefPosition pos;
      pos.ref_id = read_string(in, "ref_position.ref_id");
      pos.start = static_cast<int>(read_pod<int32_t>(in, "ref_position.start"));
      pos.end = static_cast<int>(read_pod<int32_t>(in, "ref_position.end"));
      pos.strand = read_string(in, "ref_position.strand");
      sequence->ref_positions.push_back(std::move(pos));
    }
    sequences[sequence_id] = std::move(sequence);
  }
  for (const auto& sequence : sequences) {
    if (!sequence) throw std::runtime_error("sparse sequence table in index file");
  }
  return sequences;
}

struct PendingNode {
  std::shared_ptr<WorldNode> node;
  std::vector<uint32_t> child_ids;
  std::vector<uint32_t> leaf_ids;
  std::vector<uint32_t> beacon_ids;
  std::vector<std::vector<MBB>> child_mbbs;
  std::vector<std::vector<int>> leaf_beacon_dists;
};

void write_nodes(
    std::ostream& out,
    const std::vector<std::shared_ptr<WorldNode>>& nodes) {
  write_size(out, nodes.size());
  for (const auto& node : nodes) {
    write_string(out, node->node_id);
    write_pod<uint32_t>(out, checked_leaf_id(node->center_ptr));
    write_pod<uint32_t>(out, node->integer_id);
    write_pod<int32_t>(out, static_cast<int32_t>(node->radius));
    write_pod<int32_t>(out, static_cast<int32_t>(node->expanded_layer_index));
    write_pod<int32_t>(out, static_cast<int32_t>(node->primary_layer_index));
    write_bool(out, node->is_primary);
    write_pod<int32_t>(out, static_cast<int32_t>(node->data_count));

    std::vector<uint32_t> child_ids;
    child_ids.reserve(node->child_nodes.size());
    for (const auto& child : node->child_nodes) child_ids.push_back(checked_node_id(child));
    write_u32_vector(out, child_ids);

    std::vector<uint32_t> leaf_ids;
    leaf_ids.reserve(node->child_leaves.size());
    for (const auto& leaf : node->child_leaves) leaf_ids.push_back(checked_leaf_id(leaf));
    write_u32_vector(out, leaf_ids);

    std::vector<uint32_t> beacon_ids;
    beacon_ids.reserve(node->beacons.size());
    for (const auto& beacon : node->beacons) beacon_ids.push_back(checked_leaf_id(beacon));
    write_u32_vector(out, beacon_ids);

    write_size(out, node->child_beacon_mbbs.size());
    for (const auto& row : node->child_beacon_mbbs) {
      write_size(out, row.size());
      for (const auto& mbb : row) {
        write_pod<int32_t>(out, static_cast<int32_t>(mbb.min_dist));
        write_pod<int32_t>(out, static_cast<int32_t>(mbb.max_dist));
      }
    }

    write_size(out, node->leaf_beacon_dists.size());
    for (const auto& row : node->leaf_beacon_dists) {
      write_size(out, row.size());
      for (int dist : row) write_pod<int32_t>(out, static_cast<int32_t>(dist));
    }
  }
}

std::vector<PendingNode> read_nodes(
    std::istream& in,
    const std::vector<std::shared_ptr<BioSequence>>& sequences) {
  size_t count = read_size(in, "node_count");
  std::vector<PendingNode> pending(count);
  for (size_t i = 0; i < count; ++i) {
    std::string node_id = read_string(in, "node.node_id");
    uint32_t center_id = read_pod<uint32_t>(in, "node.center_id");
    if (center_id >= sequences.size()) {
      throw std::runtime_error("node center sequence id out of range");
    }
    uint32_t integer_id = read_pod<uint32_t>(in, "node.integer_id");
    if (integer_id >= count) throw std::runtime_error("node id out of range");
    int radius = static_cast<int>(read_pod<int32_t>(in, "node.radius"));
    int expanded_layer_index =
        static_cast<int>(read_pod<int32_t>(in, "node.expanded_layer_index"));
    int primary_layer_index =
        static_cast<int>(read_pod<int32_t>(in, "node.primary_layer_index"));
    bool is_primary = read_bool(in, "node.is_primary");
    int data_count = static_cast<int>(read_pod<int32_t>(in, "node.data_count"));

    auto node = std::make_shared<WorldNode>(
        sequences[center_id], radius, expanded_layer_index);
    node->node_id = std::move(node_id);
    node->integer_id = integer_id;
    node->expanded_layer_index = expanded_layer_index;
    node->primary_layer_index = primary_layer_index;
    node->is_primary = is_primary;
    node->data_count = data_count;

    PendingNode record;
    record.node = std::move(node);
    record.child_ids = read_u32_vector(in, "node.child_ids");
    record.leaf_ids = read_u32_vector(in, "node.leaf_ids");
    record.beacon_ids = read_u32_vector(in, "node.beacon_ids");

    size_t mbb_rows = read_size(in, "node.child_mbb_rows");
    record.child_mbbs.reserve(mbb_rows);
    for (size_t row_idx = 0; row_idx < mbb_rows; ++row_idx) {
      size_t dim = read_size(in, "node.child_mbb_dim");
      std::vector<MBB> row;
      row.reserve(dim);
      for (size_t d = 0; d < dim; ++d) {
        MBB mbb;
        mbb.min_dist = static_cast<int>(read_pod<int32_t>(in, "mbb.min"));
        mbb.max_dist = static_cast<int>(read_pod<int32_t>(in, "mbb.max"));
        row.push_back(mbb);
      }
      record.child_mbbs.push_back(std::move(row));
    }

    size_t leaf_rows = read_size(in, "node.leaf_beacon_rows");
    record.leaf_beacon_dists.reserve(leaf_rows);
    for (size_t row_idx = 0; row_idx < leaf_rows; ++row_idx) {
      size_t dim = read_size(in, "node.leaf_beacon_dim");
      std::vector<int> row;
      row.reserve(dim);
      for (size_t d = 0; d < dim; ++d) {
        row.push_back(static_cast<int>(read_pod<int32_t>(in, "leaf_beacon_dist")));
      }
      record.leaf_beacon_dists.push_back(std::move(row));
    }

    pending[integer_id] = std::move(record);
  }

  for (const auto& record : pending) {
    if (!record.node) throw std::runtime_error("sparse node table in index file");
  }
  return pending;
}

void connect_nodes(
    std::vector<PendingNode>& pending,
    const std::vector<std::shared_ptr<BioSequence>>& sequences) {
  std::vector<std::shared_ptr<WorldNode>> nodes(pending.size());
  for (const auto& record : pending) nodes[record.node->integer_id] = record.node;

  for (auto& record : pending) {
    auto& node = record.node;
    node->child_nodes.reserve(record.child_ids.size());
    for (uint32_t child_id : record.child_ids) {
      if (child_id >= nodes.size()) throw std::runtime_error("child node id out of range");
      node->child_nodes.push_back(nodes[child_id]);
    }
    node->child_leaves.reserve(record.leaf_ids.size());
    for (uint32_t leaf_id : record.leaf_ids) {
      if (leaf_id >= sequences.size()) throw std::runtime_error("leaf id out of range");
      node->child_leaves.push_back(sequences[leaf_id]);
    }
    node->beacons.reserve(record.beacon_ids.size());
    for (uint32_t beacon_id : record.beacon_ids) {
      if (beacon_id >= sequences.size()) throw std::runtime_error("beacon id out of range");
      node->beacons.push_back(sequences[beacon_id]);
    }
    node->child_beacon_mbbs = std::move(record.child_mbbs);
    node->leaf_beacon_dists = std::move(record.leaf_beacon_dists);
  }
}

void write_primary_layers(
    std::ostream& out,
    const std::vector<std::vector<std::shared_ptr<WorldNode>>>& layers) {
  write_size(out, layers.size());
  for (const auto& layer : layers) {
    write_size(out, layer.size());
    for (const auto& node : layer) write_pod<uint32_t>(out, checked_node_id(node));
  }
}

std::vector<std::vector<std::shared_ptr<WorldNode>>> read_primary_layers(
    std::istream& in,
    const std::vector<PendingNode>& pending) {
  size_t layer_count = read_size(in, "primary_layer_count");
  std::vector<std::vector<std::shared_ptr<WorldNode>>> layers(layer_count);
  for (size_t layer_idx = 0; layer_idx < layer_count; ++layer_idx) {
    size_t node_count = read_size(in, "primary_layer_size");
    layers[layer_idx].reserve(node_count);
    for (size_t node_idx = 0; node_idx < node_count; ++node_idx) {
      uint32_t node_id = read_pod<uint32_t>(in, "primary_layer_node_id");
      if (node_id >= pending.size()) {
        throw std::runtime_error("primary layer node id out of range");
      }
      layers[layer_idx].push_back(pending[node_id].node);
    }
  }
  return layers;
}

}  // namespace

IndexBuildManifest make_index_manifest(
    const std::string& ref_input,
    const std::string& reads_input,
    const HierarchyConfig& hierarchy,
    const BuildRangeConfig& range_config) {
  IndexBuildManifest manifest;
  manifest.ref_input = ref_input;
  manifest.reads_input = reads_input;
  manifest.ref_fingerprint = fingerprint_input(ref_input);
  manifest.reads_fingerprint = fingerprint_input(reads_input);
  manifest.primary_radii = hierarchy.primary_radii;
  manifest.auxiliary_radii = hierarchy.auxiliary_radii;
  manifest.link_mode = build_range_mode_name(range_config.link_mode);
  manifest.leaf_attach_mode = build_range_mode_name(range_config.leaf_attach_mode);
  manifest.leaf_attach_direction =
      leaf_attach_direction_name(range_config.leaf_attach_direction);
  manifest.build_distance_mode = build_distance_mode_name(range_config.distance_mode);
  manifest.phase1_candidate_mode =
      phase1_candidate_mode_name(range_config.phase1_candidate_mode);
  manifest.range_candidate_mode =
      range_candidate_mode_name(range_config.range_join.candidate_mode);
  manifest.range_min_seed_len = range_config.range_join.min_seed_len;
  manifest.range_max_seed_len = range_config.range_join.max_seed_len;
  manifest.qgram_q = range_config.range_join.qgram_q;
  manifest.auto_pigeonhole_max_candidates =
      range_config.range_join.auto_pigeonhole_max_candidates;
  manifest.auto_pigeonhole_max_ratio =
      range_config.range_join.auto_pigeonhole_max_ratio;
  manifest.auto_hybrid_on_large_candidates =
      range_config.range_join.auto_hybrid_on_large_candidates;
  manifest.min_rect_index_fanout = range_config.min_rect_index_fanout;
  manifest.phase1_metric_min_fanout = range_config.phase1_metric_min_fanout;
  manifest.phase1_qgram_min_fanout = range_config.phase1_qgram_min_fanout;
  manifest.phase1_qgram_max_touched = range_config.phase1_qgram_max_touched;
  manifest.phase2_qgram_postfilter = range_config.phase2_qgram_postfilter;
  manifest.leaf_qgram_postfilter = range_config.leaf_qgram_postfilter;
  refresh_signature(manifest);
  return manifest;
}

IndexBuildManifest make_reference_window_index_manifest(
    const std::string& ref_input,
    size_t actual_prefix_length,
    int window_size,
    int stride,
    const HierarchyConfig& hierarchy,
    const BuildRangeConfig& range_config) {
  std::ostringstream descriptor;
  descriptor << "reference-windows:v1"
             << ";prefix=" << actual_prefix_length
             << ";window=" << window_size
             << ";stride=" << stride;
  return make_index_manifest(ref_input, descriptor.str(), hierarchy,
                             range_config);
}

IndexBuildManifest read_index_manifest(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("unable to open index file: " + path);
  read_magic(in);
  IndexBuildManifest manifest = read_manifest(in);
  IndexBuildManifest signature_check = manifest;
  refresh_signature(signature_check);
  if (signature_check.signature != manifest.signature) {
    throw std::runtime_error("index manifest signature is inconsistent");
  }
  return manifest;
}

bool index_matches_manifest(
    const std::string& path,
    const IndexBuildManifest& expected,
    IndexBuildManifest* stored,
    std::string* reason) {
  if (reason) reason->clear();
  try {
    IndexBuildManifest current = read_index_manifest(path);
    if (stored) *stored = current;
    if (current.signature != expected.signature) {
      if (reason) {
        *reason = "stored index signature differs from requested build parameters";
      }
      return false;
    }
    return true;
  } catch (const std::exception& ex) {
    if (reason) *reason = ex.what();
    return false;
  }
}

void save_index(const std::string& path,
                const BioGeometryIndexBuilder& builder,
                const IndexBuildManifest& manifest) {
  if (!builder.validate_integer_ids() || !builder.validate_search_graph_view()) {
    throw std::runtime_error("cannot persist invalid NavigaMer index");
  }

  const auto& layers = IndexPersistenceAccess::primary_layers(builder);
  std::vector<std::shared_ptr<BioSequence>> sequences = sequences_by_id(builder);
  std::vector<std::shared_ptr<WorldNode>> nodes =
      nodes_by_id(layers, builder.num_world_nodes());

  IndexBuildManifest stored = manifest;
  stored.sequence_count = builder.num_sequences();
  stored.world_node_count = builder.num_world_nodes();
  stored.edge_count = count_edges(layers);
  stored.leaf_link_count = count_leaf_links(layers);

  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("unable to open index output: " + path);
  write_magic(out);
  write_manifest(out, stored);
  write_sequences(out, sequences);
  write_nodes(out, nodes);
  write_primary_layers(out, layers);
  out.close();
  if (!out) throw std::runtime_error("failed to write index output: " + path);
}

LoadedIndex load_index(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("unable to open index file: " + path);
  read_magic(in);
  IndexBuildManifest manifest = read_manifest(in);
  IndexBuildManifest signature_check = manifest;
  refresh_signature(signature_check);
  if (signature_check.signature != manifest.signature) {
    throw std::runtime_error("index manifest signature is inconsistent");
  }

  BuildRangeConfig range_config = build_config_from_manifest(manifest);
  BioGeometryIndexBuilder builder(
      HierarchyConfig(manifest.primary_radii, manifest.auxiliary_radii),
      range_config);

  std::vector<std::shared_ptr<BioSequence>> sequences = read_sequences(in);
  std::vector<PendingNode> pending = read_nodes(in, sequences);
  connect_nodes(pending, sequences);
  std::vector<std::vector<std::shared_ptr<WorldNode>>> layers =
      read_primary_layers(in, pending);

  std::unordered_map<std::string, std::shared_ptr<BioSequence>> unique_sequences;
  unique_sequences.reserve(sequences.size());
  for (const auto& sequence : sequences) {
    unique_sequences[sequence->seq] = sequence;
  }

  IndexPersistenceAccess::reset_loaded_state(
      builder, std::move(unique_sequences), std::move(layers),
      pending.size(), sequences.size(), manifest);

  if (!builder.validate_integer_ids() || !builder.validate_search_graph_view()) {
    throw std::runtime_error("loaded NavigaMer index failed validation");
  }

  return {std::move(builder), std::move(manifest)};
}

}  // namespace navigamer
