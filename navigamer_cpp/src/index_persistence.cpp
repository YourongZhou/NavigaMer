#include "index_persistence.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace navigamer {

namespace {

constexpr std::array<char, 8> kMagic = {'N', 'G', 'I', 'D', 'X', '0', '0', '3'};

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
  if (manifest.format_version != 3) {
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
  if (magic != kMagic) {
    throw std::runtime_error(
        "unsupported NavigaMer index format; rebuild the array index");
  }
}

void write_u32_vector(std::ostream& out, const std::vector<uint32_t>& values) {
  write_size(out, values.size());
  if (values.empty()) return;
  const size_t byte_count = values.size() * sizeof(uint32_t);
  if (byte_count > static_cast<size_t>(
                       std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error("u32 vector exceeds stream size range");
  }
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(byte_count));
  if (!out) throw std::runtime_error("failed to write u32 vector");
}

std::vector<uint32_t> read_u32_vector(std::istream& in, const char* field) {
  const size_t count = read_size(in, field);
  if (count > static_cast<size_t>(
                  std::numeric_limits<std::streamsize>::max()) /
                  sizeof(uint32_t)) {
    throw std::runtime_error(std::string(field) +
                             " exceeds stream size range");
  }
  std::vector<uint32_t> values(count);
  if (values.empty()) return values;
  const size_t byte_count = count * sizeof(uint32_t);
  in.read(reinterpret_cast<char*>(values.data()),
          static_cast<std::streamsize>(byte_count));
  if (!in) {
    throw std::runtime_error(std::string("failed to read index field: ") +
                             field);
  }
  return values;
}

}  // namespace

class IndexPersistenceAccess {
 public:
  static void reset_loaded_array_state(
      BioGeometryIndexBuilder& builder,
      SearchGraphView view,
      const IndexBuildManifest& manifest) {
    builder.stats_ = BioGeometryIndexBuilder::Statistics{};
    builder.stats_.added_sequences = view.sequences.size();
    builder.stats_.unique_sequences = view.sequences.size();
    builder.stats_.created_primary_nodes.assign(
        view.layer_begin.size(), 0);
    for (size_t layer = 0; layer < view.layer_begin.size(); ++layer) {
      builder.stats_.created_primary_nodes[layer] =
          view.layer_end[layer] - view.layer_begin[layer];
    }
    builder.stats_.phase2_edges_added = manifest.edge_count;
    builder.stats_.leaf_attachments_added = manifest.leaf_link_count;
    builder.world_node_count_ = view.node_records.size();
    builder.sequence_count_ = view.sequences.size();
    builder.build_nodes_.clear();
    builder.final_node_ids_.clear();
    builder.primary_layers_.clear();
    builder.extended_layers_.clear();
    builder.search_graph_view_ = std::move(view);
  }

};

namespace {

void write_i32_vector(std::ostream& out,
                      const std::vector<int32_t>& values) {
  write_size(out, values.size());
  if (values.empty()) return;
  const size_t byte_count = values.size() * sizeof(int32_t);
  if (byte_count > static_cast<size_t>(
                       std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error("i32 vector exceeds stream size range");
  }
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(byte_count));
  if (!out) throw std::runtime_error("failed to write i32 vector");
}

std::vector<int32_t> read_i32_vector(std::istream& in, const char* field) {
  const size_t count = read_size(in, field);
  if (count > static_cast<size_t>(
                  std::numeric_limits<std::streamsize>::max()) /
                  sizeof(int32_t)) {
    throw std::runtime_error(std::string(field) +
                             " exceeds stream size range");
  }
  std::vector<int32_t> values(count);
  if (values.empty()) return values;
  const size_t byte_count = count * sizeof(int32_t);
  in.read(reinterpret_cast<char*>(values.data()),
          static_cast<std::streamsize>(byte_count));
  if (!in) {
    throw std::runtime_error(std::string("failed to read index field: ") +
                             field);
  }
  return values;
}

void write_sequence_store(std::ostream& out, const SequenceStore& store) {
  write_size(out, store.size());
  for (const auto& sequence : store.records) {
    write_string(out, sequence.id);
    write_string(out, sequence.seq);
    write_pod<uint32_t>(out, sequence.sequence_id);
    write_bool(out, sequence.has_source_pos);
    write_size(out, sequence.source_pos);
    write_pod<int64_t>(out, sequence.bwt_interval.start);
    write_pod<int64_t>(out, sequence.bwt_interval.end);
    write_size(out, sequence.ref_positions.size());
    for (const auto& pos : sequence.ref_positions) {
      write_string(out, pos.ref_id);
      write_pod<int32_t>(out, static_cast<int32_t>(pos.start));
      write_pod<int32_t>(out, static_cast<int32_t>(pos.end));
      write_string(out, pos.strand);
    }
  }
}

SequenceStore read_sequence_store(std::istream& in) {
  SequenceStore store;
  const size_t count = read_size(in, "sequence_store.count");
  store.records.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    std::string id = read_string(in, "sequence.id");
    std::string seq = read_string(in, "sequence.seq");
    BioSequence sequence(std::move(id), std::move(seq));
    sequence.sequence_id =
        read_pod<uint32_t>(in, "sequence.sequence_id");
    if (sequence.sequence_id != i) {
      throw std::runtime_error("array index sequence ids are not contiguous");
    }
    sequence.has_source_pos =
        read_bool(in, "sequence.has_source_pos");
    sequence.source_pos = read_size(in, "sequence.source_pos");
    sequence.bwt_interval.start =
        read_pod<int64_t>(in, "sequence.bwt_start");
    sequence.bwt_interval.end =
        read_pod<int64_t>(in, "sequence.bwt_end");
    const size_t position_count =
        read_size(in, "sequence.ref_position_count");
    sequence.ref_positions.reserve(position_count);
    for (size_t pos_idx = 0; pos_idx < position_count; ++pos_idx) {
      RefPosition pos;
      pos.ref_id = read_string(in, "ref_position.ref_id");
      pos.start =
          static_cast<int>(read_pod<int32_t>(in, "ref_position.start"));
      pos.end =
          static_cast<int>(read_pod<int32_t>(in, "ref_position.end"));
      pos.strand = read_string(in, "ref_position.strand");
      sequence.ref_positions.push_back(std::move(pos));
    }
    store.records.push_back(std::move(sequence));
  }
  return store;
}

void write_node_records(
    std::ostream& out,
    const std::vector<WorldNodeRecord>& records) {
  write_size(out, records.size());
  for (const auto& record : records) {
    write_pod<uint32_t>(out, record.center_sequence_id);
    write_pod<int32_t>(out, static_cast<int32_t>(record.radius));
    write_pod<int32_t>(
        out, static_cast<int32_t>(record.expanded_layer_index));
    write_pod<int32_t>(
        out, static_cast<int32_t>(record.primary_layer_index));
    write_pod<uint32_t>(out, record.child_begin);
    write_pod<uint32_t>(out, record.child_count);
    write_pod<uint32_t>(out, record.leaf_begin);
    write_pod<uint32_t>(out, record.leaf_count);
    write_pod<uint32_t>(out, record.beacon_begin);
    write_pod<uint32_t>(out, record.beacon_count);
    write_pod<uint32_t>(out, record.mbb_begin);
    write_pod<uint32_t>(out, record.leaf_beacon_begin);
  }
}

std::vector<WorldNodeRecord> read_node_records(std::istream& in) {
  const size_t count = read_size(in, "node_records.count");
  std::vector<WorldNodeRecord> records(count);
  for (auto& record : records) {
    record.center_sequence_id =
        read_pod<uint32_t>(in, "node.center_sequence_id");
    record.radius = static_cast<int>(read_pod<int32_t>(in, "node.radius"));
    record.expanded_layer_index =
        static_cast<int>(read_pod<int32_t>(
            in, "node.expanded_layer_index"));
    record.primary_layer_index =
        static_cast<int>(read_pod<int32_t>(
            in, "node.primary_layer_index"));
    record.child_begin = read_pod<uint32_t>(in, "node.child_begin");
    record.child_count = read_pod<uint32_t>(in, "node.child_count");
    record.leaf_begin = read_pod<uint32_t>(in, "node.leaf_begin");
    record.leaf_count = read_pod<uint32_t>(in, "node.leaf_count");
    record.beacon_begin = read_pod<uint32_t>(in, "node.beacon_begin");
    record.beacon_count = read_pod<uint32_t>(in, "node.beacon_count");
    record.mbb_begin = read_pod<uint32_t>(in, "node.mbb_begin");
    record.leaf_beacon_begin =
        read_pod<uint32_t>(in, "node.leaf_beacon_begin");
  }
  return records;
}

void write_search_graph_view(std::ostream& out,
                             const SearchGraphView& view) {
  write_sequence_store(out, view.sequences);
  write_node_records(out, view.node_records);
  write_u32_vector(out, view.layer_begin);
  write_u32_vector(out, view.layer_end);
  write_u32_vector(out, view.child_ids);
  write_u32_vector(out, view.leaf_ids);
  write_u32_vector(out, view.beacon_ids);
  write_i32_vector(out, view.mbb_lo);
  write_i32_vector(out, view.mbb_hi);
  write_i32_vector(out, view.leaf_beacon_dists);
}

SearchGraphView read_search_graph_view(std::istream& in) {
  SearchGraphView view;
  view.sequences = read_sequence_store(in);
  view.node_records = read_node_records(in);
  view.layer_begin = read_u32_vector(in, "layer_begin");
  view.layer_end = read_u32_vector(in, "layer_end");
  view.child_ids = read_u32_vector(in, "child_ids");
  view.leaf_ids = read_u32_vector(in, "leaf_ids");
  view.beacon_ids = read_u32_vector(in, "beacon_ids");
  view.mbb_lo = read_i32_vector(in, "mbb_lo");
  view.mbb_hi = read_i32_vector(in, "mbb_hi");
  view.leaf_beacon_dists =
      read_i32_vector(in, "leaf_beacon_dists");
  return view;
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
  if (manifest.format_version != 3) {
    throw std::runtime_error(
        "unsupported NavigaMer index version; rebuild the array index");
  }
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

  const auto& view = builder.search_graph_view();

  IndexBuildManifest stored = manifest;
  stored.format_version = 3;
  stored.sequence_count = builder.num_sequences();
  stored.world_node_count = builder.num_world_nodes();
  stored.edge_count = view.child_ids.size();
  stored.leaf_link_count = view.leaf_ids.size();
  refresh_signature(stored);

  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("unable to open index output: " + path);
  write_magic(out);
  write_manifest(out, stored);
  write_search_graph_view(out, view);
  out.close();
  if (!out) throw std::runtime_error("failed to write index output: " + path);
}

LoadedIndex load_index(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("unable to open index file: " + path);
  read_magic(in);
  IndexBuildManifest manifest = read_manifest(in);
  if (manifest.format_version != 3) {
    throw std::runtime_error(
        "unsupported NavigaMer index version; rebuild the array index");
  }
  IndexBuildManifest signature_check = manifest;
  refresh_signature(signature_check);
  if (signature_check.signature != manifest.signature) {
    throw std::runtime_error("index manifest signature is inconsistent");
  }

  BuildRangeConfig range_config = build_config_from_manifest(manifest);
  BioGeometryIndexBuilder builder(
      HierarchyConfig(manifest.primary_radii, manifest.auxiliary_radii),
      range_config);

  SearchGraphView view = read_search_graph_view(in);
  if (view.sequences.size() != manifest.sequence_count ||
      view.node_records.size() != manifest.world_node_count ||
      view.child_ids.size() != manifest.edge_count ||
      view.leaf_ids.size() != manifest.leaf_link_count) {
    throw std::runtime_error(
        "array index manifest counts do not match stored arrays");
  }
  IndexPersistenceAccess::reset_loaded_array_state(
      builder, std::move(view), manifest);

  if (!builder.validate_integer_ids() || !builder.validate_search_graph_view()) {
    throw std::runtime_error("loaded NavigaMer index failed validation");
  }

  return {std::move(builder), std::move(manifest)};
}

}  // namespace navigamer
