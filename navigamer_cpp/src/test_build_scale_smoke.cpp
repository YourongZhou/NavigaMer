#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> values;
  std::stringstream ss(line);
  std::string token;
  while (std::getline(ss, token, ',')) values.push_back(token);
  return values;
}

std::vector<std::string> split_tsv_line(const std::string& line) {
  std::vector<std::string> values;
  std::stringstream ss(line);
  std::string token;
  while (std::getline(ss, token, '\t')) values.push_back(token);
  return values;
}

}  // namespace

int main() {
  const std::string out_path = "/tmp/navigamer_build_scale_smoke.csv";
  const std::string index_path =
      "/tmp/navigamer_build_scale_smoke.navidx";
  const std::string persisted_csv =
      "/tmp/navigamer_build_scale_persisted.csv";
  const std::string rejected_index =
      "/tmp/navigamer_build_scale_rejected.navidx";
  std::remove(out_path.c_str());
  std::remove(index_path.c_str());
  std::remove(persisted_csv.c_str());
  std::remove(rejected_index.c_str());

  const std::string command =
      "./navigamer build-scale "
      "--ref ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT "
      "--window 12 --stride 4 --prefix-lengths 24,36 "
      "--primary-radii 12,6,2 --leaf-attach-direction world-to-seq "
      "--phase2-qgram-postfilter off --progress-interval-seconds 0 --out " +
      out_path + " >/tmp/navigamer_build_scale_smoke.stdout "
      "2>/tmp/navigamer_build_scale_smoke.stderr";
  const int rc = std::system(command.c_str());
  assert(rc == 0);

  std::ifstream in(out_path);
  assert(in.good());

  std::string header_line;
  std::getline(in, header_line);
  const auto header = split_csv_line(header_line);
  std::map<std::string, size_t> column;
  for (size_t i = 0; i < header.size(); ++i) column[header[i]] = i;

  for (const std::string& required : {
           "prefix_len",
           "invalid_window_count",
           "total_build_ms",
           "phase0_dedup_ms",
           "phase2_candidate_query_ms",
           "phase2_candidate_query_worker_ms",
           "phase2_exact_verify_worker_ms",
           "phase3_child_mbb_distance_ms",
           "leaf_beacon_distance_ms",
           "leaf_attach_direction_used",
           "range_candidate_mode",
           "qgram_q",
       }) {
    assert(column.count(required) == 1);
  }

  size_t row_count = 0;
  std::string row_line;
  while (std::getline(in, row_line)) {
    if (row_line.empty()) continue;
    const auto row = split_csv_line(row_line);
    assert(row.size() == header.size());
    const double total_ms =
        std::stod(row[column.at("total_build_ms")]);
    const double phase0_ms =
        std::stod(row[column.at("phase0_dedup_ms")]);
    const double phase2_query_ms =
        std::stod(row[column.at("phase2_candidate_query_ms")]);
    const double phase2_query_worker_ms =
        std::stod(row[column.at("phase2_candidate_query_worker_ms")]);
    const double phase2_verify_worker_ms =
        std::stod(row[column.at("phase2_exact_verify_worker_ms")]);
    const double phase3_distance_ms =
        std::stod(row[column.at("phase3_child_mbb_distance_ms")]);
    const double leaf_beacon_ms =
        std::stod(row[column.at("leaf_beacon_distance_ms")]);
    assert(total_ms > 0.0);
    assert(phase0_ms >= 0.0);
    assert(phase2_query_ms >= 0.0);
    assert(phase2_query_worker_ms >= 0.0);
    assert(phase2_verify_worker_ms >= 0.0);
    assert(phase3_distance_ms >= 0.0);
    assert(leaf_beacon_ms >= 0.0);
    assert(row[column.at("invalid_window_count")] == "0");
    assert(row[column.at("leaf_attach_direction_used")] == "world_to_seq");
    row_count++;
  }
  assert(row_count == 2);

  std::ifstream stderr_in("/tmp/navigamer_build_scale_smoke.stderr");
  assert(stderr_in.good());
  const std::string stderr_text(
      (std::istreambuf_iterator<char>(stderr_in)),
      std::istreambuf_iterator<char>());
  assert(stderr_text.find("phase2_qgram_postfilter=off") !=
         std::string::npos);
  assert(stderr_text.find("Build progress: timestamp=") !=
         std::string::npos);
  assert(stderr_text.find("event=start phase=phase1_sketch") !=
         std::string::npos);
  assert(stderr_text.find("event=finish phase=phase4_attach") !=
         std::string::npos);

  const int invalid_progress_rc = std::system(
      "./navigamer build-scale --ref ACGTACGTACGTACGT "
      "--window 8 --stride 2 --prefix-lengths 16 "
      "--progress-interval-seconds -1 "
      "--out /tmp/navigamer_invalid_progress.csv "
      ">/tmp/navigamer_invalid_progress.stdout "
      "2>/tmp/navigamer_invalid_progress.stderr");
  assert(invalid_progress_rc != 0);

  const int removed_backend_rc = std::system(
      "./navigamer build-scale --ref ACGTACGTACGTACGT "
      "--window 8 --stride 2 --prefix-lengths 16 "
      "--phase2-distance-backend cpu "
      "--out /tmp/navigamer_removed_backend.csv "
      ">/tmp/navigamer_removed_backend.stdout "
      "2>/tmp/navigamer_removed_backend.stderr");
  assert(removed_backend_rc != 0);

  const std::string persist_command =
      "./navigamer build-scale "
      "--ref ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT "
      "--window 12 --stride 4 --prefix-lengths 36 "
      "--primary-radii 12,6,2 --progress-interval-seconds 0 "
      "--index " + index_path + " --out " + persisted_csv +
      " >/tmp/navigamer_build_scale_persist.stdout "
      "2>/tmp/navigamer_build_scale_persist.stderr";
  assert(std::system(persist_command.c_str()) == 0);
  {
    std::ifstream index_in(index_path, std::ios::binary | std::ios::ate);
    assert(index_in.good());
    assert(index_in.tellg() > 0);
  }

  const std::string query_command =
      "./navigamer query-index --index " + index_path +
      " --query ACGTACGTACGT --tolerance 0 "
      ">/tmp/navigamer_build_scale_query.stdout "
      "2>/tmp/navigamer_build_scale_query.stderr";
  assert(std::system(query_command.c_str()) == 0);

  const std::string batch_reads = "/tmp/navigamer_query_index_batch_reads.fq";
  {
    std::ofstream reads_out(batch_reads);
    assert(reads_out.good());
    reads_out << "@read0\nACGTACGTACGT\n+\nIIIIIIIIIIII\n";
    reads_out << "@read1\nTTTTTTTTTTTT\n+\nIIIIIIIIIIII\n";
  }
  const std::string batch_out = "/tmp/navigamer_query_index_batch.tsv";
  const std::string batch_trace =
      "/tmp/navigamer_query_index_batch_trace.tsv";
  std::remove(batch_out.c_str());
  std::remove(batch_trace.c_str());
  const std::string batch_command =
      "./navigamer query-index-batch --index " + index_path +
      " --reads " + batch_reads +
      " --tolerance 0 --search-prefetch on --path-trace-out " +
      batch_trace + " --out " + batch_out +
      " >/tmp/navigamer_query_index_batch.stdout "
      "2>/tmp/navigamer_query_index_batch.stderr";
  assert(std::system(batch_command.c_str()) == 0);
  std::ifstream batch_in(batch_out);
  assert(batch_in.good());
  std::string batch_header;
  std::getline(batch_in, batch_header);
  assert(batch_header.find("query_id\thit_id") == 0);
  assert(batch_header.find("search_prefetch_enabled") != std::string::npos);
  assert(batch_header.find("query_path_class") != std::string::npos);
  assert(batch_header.find("path_contained_step_count") != std::string::npos);
  assert(batch_header.find("path_overlap_step_count") != std::string::npos);
  assert(batch_header.find("path_uncovered_step_count") != std::string::npos);
  std::string batch_row;
  bool saw_read0 = false;
  bool saw_read1 = false;
  size_t read0_rows = 0;
  while (std::getline(batch_in, batch_row)) {
    if (batch_row.find("read0\t") == 0) {
      saw_read0 = true;
      read0_rows++;
    }
    if (batch_row.find("read1\t") == 0) saw_read1 = true;
    assert(batch_row.find("\ttrue\t") != std::string::npos);
  }
  assert(saw_read0);
  assert(saw_read1);
  assert(read0_rows == 7);

  std::ifstream trace_in(batch_trace);
  assert(trace_in.good());
  std::string trace_header_line;
  std::getline(trace_in, trace_header_line);
  const auto trace_header = split_tsv_line(trace_header_line);
  std::map<std::string, size_t> trace_column;
  for (size_t i = 0; i < trace_header.size(); ++i) {
    trace_column[trace_header[i]] = i;
  }
  for (const std::string& required : {
           "query_id",
           "query_ordinal",
           "world_visit_count",
           "leaf_visit_count",
           "prev_world_jaccard",
           "prev_leaf_jaccard",
           "world_path",
           "leaf_path",
       }) {
    assert(trace_column.count(required) == 1);
  }
  std::string trace_row0;
  std::string trace_row1;
  assert(static_cast<bool>(std::getline(trace_in, trace_row0)));
  assert(static_cast<bool>(std::getline(trace_in, trace_row1)));
  const auto trace0 = split_tsv_line(trace_row0);
  const auto trace1 = split_tsv_line(trace_row1);
  assert(trace0[trace_column.at("query_id")] == "read0");
  assert(trace0[trace_column.at("query_ordinal")] == "0");
  assert(trace1[trace_column.at("query_id")] == "read1");
  assert(trace1[trace_column.at("query_ordinal")] == "1");
  assert(trace0[trace_column.at("prev_world_jaccard")] == "NA");
  assert(trace1[trace_column.at("prev_world_jaccard")] != "NA");
  assert(std::stoull(trace0[trace_column.at("world_visit_count")]) > 0);
  assert(std::stoull(trace1[trace_column.at("world_visit_count")]) > 0);

  const std::string layer_out = "/tmp/navigamer_layer_radius_smoke.csv";
  std::remove(layer_out.c_str());
  const std::string layer_command =
      "./navigamer layer-radius-experiment "
      "--ref ACGTACGTACGTACGTACGTACGTACGTACGT "
      "--length 8 --stride 4 --tolerance 1 --query-edits 1 "
      "--queries-per-cell 2 --L-values 2 --r-leaf-values 2 "
      "--alpha-values 0.5 --out " + layer_out +
      " >/tmp/navigamer_layer_radius_smoke.stdout "
      "2>/tmp/navigamer_layer_radius_smoke.stderr";
  assert(std::system(layer_command.c_str()) == 0);
  std::ifstream layer_in(layer_out);
  assert(layer_in.good());
  std::string layer_header_line;
  std::getline(layer_in, layer_header_line);
  const auto layer_header = split_csv_line(layer_header_line);
  std::map<std::string, size_t> layer_column;
  for (size_t i = 0; i < layer_header.size(); ++i) {
    layer_column[layer_header[i]] = i;
  }
  for (const std::string& required : {
           "source_id",
           "result_count",
           "source_recovered",
           "no_fn",
       }) {
    assert(layer_column.count(required) == 1);
  }
  std::string layer_row_line;
  assert(static_cast<bool>(std::getline(layer_in, layer_row_line)));
  const auto layer_row = split_csv_line(layer_row_line);
  assert(layer_row.size() == layer_header.size());
  assert(layer_row[layer_column.at("source_recovered")] == "1");
  assert(layer_row[layer_column.at("no_fn")] == "1");

  const std::string reject_command =
      "./navigamer build-scale "
      "--ref ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT "
      "--window 12 --stride 4 --prefix-lengths 24,36 "
      "--primary-radii 12,6,2 --progress-interval-seconds 0 "
      "--index " + rejected_index +
      " --out /tmp/navigamer_build_scale_rejected.csv "
      ">/tmp/navigamer_build_scale_rejected.stdout "
      "2>/tmp/navigamer_build_scale_rejected.stderr";
  assert(std::system(reject_command.c_str()) != 0);
  std::ifstream rejected_in(rejected_index, std::ios::binary);
  assert(!rejected_in.good());
  std::ifstream rejected_stderr(
      "/tmp/navigamer_build_scale_rejected.stderr");
  const std::string rejected_text(
      (std::istreambuf_iterator<char>(rejected_stderr)),
      std::istreambuf_iterator<char>());
  assert(rejected_text.find("--index requires exactly one prefix length") !=
         std::string::npos);

  const std::string multicontig_fasta =
      "/tmp/navigamer_build_scale_multicontig.fa";
  const std::string multicontig_index =
      "/tmp/navigamer_build_scale_multicontig.navidx";
  const std::string multicontig_csv =
      "/tmp/navigamer_build_scale_multicontig.csv";
  const std::string multicontig_reads =
      "/tmp/navigamer_build_scale_multicontig.fq";
  const std::string multicontig_tsv =
      "/tmp/navigamer_build_scale_multicontig.tsv";
  {
    std::ofstream fasta(multicontig_fasta);
    fasta << ">chr1\nAAAACCCC\n>chr2\nAAAANAAAA\n";
    std::ofstream reads(multicontig_reads);
    reads << "@exact\nAAAA\n+\nIIII\n";
  }
  const std::string multicontig_build_command =
      "./navigamer build-scale --ref " + multicontig_fasta +
      " --window 4 --stride 1 --prefix-lengths 17 "
      "--primary-radii 8,4,2 --progress-interval-seconds 0 "
      "--index " + multicontig_index + " --out " + multicontig_csv +
      " >/tmp/navigamer_multicontig_build.stdout "
      "2>/tmp/navigamer_multicontig_build.stderr";
  assert(std::system(multicontig_build_command.c_str()) == 0);
  {
    std::ifstream csv(multicontig_csv);
    std::string multicontig_header_line;
    std::string multicontig_row_line;
    assert(static_cast<bool>(std::getline(csv, multicontig_header_line)));
    assert(static_cast<bool>(std::getline(csv, multicontig_row_line)));
    const auto multicontig_header =
        split_csv_line(multicontig_header_line);
    const auto multicontig_row = split_csv_line(multicontig_row_line);
    std::map<std::string, size_t> multicontig_columns;
    for (size_t i = 0; i < multicontig_header.size(); ++i) {
      multicontig_columns[multicontig_header[i]] = i;
    }
    assert(multicontig_row[
               multicontig_columns.at("invalid_window_count")] == "4");
  }
  const std::string multicontig_query_command =
      "./navigamer query-index-batch --index " + multicontig_index +
      " --reads " + multicontig_reads +
      " --tolerance 0 --out " + multicontig_tsv +
      " >/tmp/navigamer_multicontig_query.stdout "
      "2>/tmp/navigamer_multicontig_query.stderr";
  assert(std::system(multicontig_query_command.c_str()) == 0);
  {
    std::ifstream tsv(multicontig_tsv);
    std::string line;
    assert(static_cast<bool>(std::getline(tsv, line)));
    const auto output_header = split_tsv_line(line);
    std::map<std::string, size_t> output_columns;
    for (size_t i = 0; i < output_header.size(); ++i) {
      output_columns[output_header[i]] = i;
    }
    std::vector<std::string> locations;
    while (std::getline(tsv, line)) {
      if (line.empty()) continue;
      const auto row = split_tsv_line(line);
      locations.push_back(
          row[output_columns.at("ref_id")] + ":" +
          row[output_columns.at("reference_start")]);
    }
    assert(locations ==
           std::vector<std::string>(
               {"chr1:0", "chr2:0", "chr2:5"}));
  }

  const std::string sharded_index =
      "/tmp/navigamer_build_scale_multicontig.navshard";
  const std::string sharded_tsv =
      "/tmp/navigamer_build_scale_multicontig_sharded.tsv";
  const std::string sharded_build_command =
      "./navigamer build-sharded --ref " + multicontig_fasta +
      " --window 4 --stride 1 --shard-windows 2 "
      "--primary-radii 8,4,2 --progress-interval-seconds 0 "
      "--index " + sharded_index +
      " >/tmp/navigamer_sharded_build.stdout "
      "2>/tmp/navigamer_sharded_build.stderr";
  assert(std::system(sharded_build_command.c_str()) == 0);
  const std::string sharded_query_command =
      "OMP_NUM_THREADS=4 ./navigamer query-index-batch --index " +
      sharded_index + " --reads " + multicontig_reads +
      " --tolerance 0 --out " + sharded_tsv +
      " >/tmp/navigamer_sharded_query.stdout "
      "2>/tmp/navigamer_sharded_query.stderr";
  assert(std::system(sharded_query_command.c_str()) == 0);
  {
    std::ifstream tsv(sharded_tsv);
    std::string line;
    assert(static_cast<bool>(std::getline(tsv, line)));
    const auto output_header = split_tsv_line(line);
    std::map<std::string, size_t> output_columns;
    for (size_t i = 0; i < output_header.size(); ++i) {
      output_columns[output_header[i]] = i;
    }
    std::vector<std::string> locations;
    while (std::getline(tsv, line)) {
      if (line.empty()) continue;
      const auto row = split_tsv_line(line);
      assert(row[output_columns.at("hit_id")] == "chr1_0");
      assert(row[output_columns.at("result_count")] == "1");
      locations.push_back(
          row[output_columns.at("ref_id")] + ":" +
          row[output_columns.at("reference_start")]);
    }
    assert(locations ==
           std::vector<std::string>(
               {"chr1:0", "chr2:0", "chr2:5"}));
  }

  // An ambiguous query must conservatively scan every shard, but the CLI
  // must never make all human-scale shards resident at once. Use enough tiny
  // shards to cross the fixed residency cap and verify a true hit survives
  // the chunked exact fallback.
  const std::string fallback_fasta =
      "/tmp/navigamer_bounded_fallback.fa";
  const std::string fallback_reads =
      "/tmp/navigamer_bounded_fallback.fq";
  const std::string fallback_index =
      "/tmp/navigamer_bounded_fallback.navshard";
  const std::string fallback_tsv =
      "/tmp/navigamer_bounded_fallback.tsv";
  std::string fallback_reference;
  fallback_reference.reserve(3000);
  uint64_t fallback_state = 0x9e3779b97f4a7c15ULL;
  constexpr char fallback_bases[] = {'A', 'C', 'G', 'T'};
  for (size_t idx = 0; idx < 3000; ++idx) {
    fallback_state =
        fallback_state * 6364136223846793005ULL + 1;
    fallback_reference.push_back(
        fallback_bases[(fallback_state >> 62) & 3]);
  }
  for (size_t motif_begin = 0;
       motif_begin + 16 <= fallback_reference.size();
       motif_begin += 100) {
    fallback_reference.replace(motif_begin, 16, 16, 'A');
  }
  {
    std::ofstream fasta(fallback_fasta);
    assert(fasta.good());
    fasta << ">fallback_ref\n" << fallback_reference << '\n';
  }
  std::string fallback_query = fallback_reference.substr(123, 150);
  fallback_query[12] = 'N';
  {
    std::ofstream reads(fallback_reads);
    assert(reads.good());
    reads << "@fallback_read\n" << fallback_query
          << "\n+\n" << std::string(150, 'I') << '\n';
    reads << "@oversized_route_read\n"
          << fallback_reference.substr(123, 150)
          << "\n+\n" << std::string(150, 'I') << '\n';
  }
  const std::string fallback_build_command =
      "OMP_NUM_THREADS=4 ./navigamer build-sharded --ref " +
      fallback_fasta +
      " --window 150 --stride 1 --shard-windows 8 "
      "--shard-build-jobs 4 --primary-radii 30,15,5 "
      "--progress-interval-seconds 0 --index " + fallback_index +
      " >/tmp/navigamer_bounded_fallback_build.stdout "
      "2>/tmp/navigamer_bounded_fallback_build.stderr";
  assert(std::system(fallback_build_command.c_str()) == 0);
  const std::string fallback_query_command =
      "OMP_NUM_THREADS=4 ./navigamer query-index-batch --index " +
      fallback_index + " --reads " + fallback_reads +
      " --tolerance 5 --out " + fallback_tsv +
      " >/tmp/navigamer_bounded_fallback_query.stdout "
      "2>/tmp/navigamer_bounded_fallback_query.stderr";
  assert(std::system(fallback_query_command.c_str()) == 0);
  {
    std::ifstream stderr_in(
        "/tmp/navigamer_bounded_fallback_query.stderr");
    assert(stderr_in.good());
    const std::string stderr_text(
        (std::istreambuf_iterator<char>(stderr_in)),
        std::istreambuf_iterator<char>());
    assert(stderr_text.find("peak_shards=64/") != std::string::npos);
    assert(stderr_text.find("routed_queries=1/2") != std::string::npos);
  }
  {
    std::ifstream tsv(fallback_tsv);
    assert(tsv.good());
    std::string line;
    assert(static_cast<bool>(std::getline(tsv, line)));
    const auto output_header = split_tsv_line(line);
    std::map<std::string, size_t> output_columns;
    for (size_t idx = 0; idx < output_header.size(); ++idx) {
      output_columns[output_header[idx]] = idx;
    }
    bool recovered_fallback_source = false;
    bool recovered_oversized_route_source = false;
    while (std::getline(tsv, line)) {
      if (line.empty()) continue;
      const auto row = split_tsv_line(line);
      if (row[output_columns.at("query_id")] == "fallback_read" &&
          row[output_columns.at("reference_start")] == "123" &&
          std::stoi(row[output_columns.at("edit_distance")]) <= 5) {
        recovered_fallback_source = true;
      }
      if (row[output_columns.at("query_id")] ==
              "oversized_route_read" &&
          row[output_columns.at("reference_start")] == "123" &&
          std::stoi(row[output_columns.at("edit_distance")]) == 0) {
        recovered_oversized_route_source = true;
      }
    }
    assert(recovered_fallback_source);
    assert(recovered_oversized_route_source);
  }
  const std::string route_budget_reads =
      "/tmp/navigamer_route_budget_queries.fq";
  const std::string route_budget_tsv =
      "/tmp/navigamer_route_budget_queries.tsv";
  constexpr size_t route_budget_query_count = 185;
  {
    std::ofstream reads(route_budget_reads);
    assert(reads.good());
    const std::string sequence = fallback_reference.substr(123, 150);
    const std::string quality(150, 'I');
    for (size_t query_idx = 0;
         query_idx < route_budget_query_count; ++query_idx) {
      reads << "@route_budget_" << query_idx << '\n'
            << sequence << "\n+\n" << quality << '\n';
    }
  }
  const std::string route_budget_command =
      "OMP_NUM_THREADS=4 ./navigamer query-index-batch --index " +
      fallback_index + " --reads " + route_budget_reads +
      " --tolerance 5 --out " + route_budget_tsv +
      " >/tmp/navigamer_route_budget.stdout "
      "2>/tmp/navigamer_route_budget.stderr";
  assert(std::system(route_budget_command.c_str()) == 0);
  {
    std::ifstream stderr_in(
        "/tmp/navigamer_route_budget.stderr");
    assert(stderr_in.good());
    const std::string stderr_text(
        (std::istreambuf_iterator<char>(stderr_in)),
        std::istreambuf_iterator<char>());
    assert(stderr_text.find("Queries: 185") != std::string::npos);
    assert(stderr_text.find("peak_route_ids=0") !=
           std::string::npos);
    assert(stderr_text.find("exact_block_direct_queries=185") !=
           std::string::npos);
    assert(stderr_text.find("sampled_qgram_direct_queries=185") !=
           std::string::npos);
    assert(stderr_text.find("routed_queries=185/185") !=
           std::string::npos);
  }
  {
    std::ifstream tsv(route_budget_tsv);
    assert(tsv.good());
    std::string line;
    assert(static_cast<bool>(std::getline(tsv, line)));
    const auto output_header = split_tsv_line(line);
    std::map<std::string, size_t> output_columns;
    for (size_t idx = 0; idx < output_header.size(); ++idx) {
      output_columns[output_header[idx]] = idx;
    }
    std::vector<uint8_t> recovered(route_budget_query_count, uint8_t{0});
    while (std::getline(tsv, line)) {
      if (line.empty()) continue;
      const auto row = split_tsv_line(line);
      if (row[output_columns.at("reference_start")] != "123" ||
          row[output_columns.at("edit_distance")] != "0") {
        continue;
      }
      const std::string& query_id = row[output_columns.at("query_id")];
      const std::string prefix = "route_budget_";
      if (query_id.compare(0, prefix.size(), prefix) != 0) continue;
      const size_t query_idx = std::stoul(query_id.substr(prefix.size()));
      assert(query_idx < recovered.size());
      recovered[query_idx] = 1;
    }
    for (uint8_t was_recovered : recovered) {
      assert(was_recovered != 0);
    }
  }
  const std::string direct_only_index =
      "/tmp/navigamer_bounded_fallback_direct.navshard";
  const std::string direct_only_build_command =
      "OMP_NUM_THREADS=4 ./navigamer build-sharded --ref " +
      fallback_fasta +
      " --window 150 --stride 1 --shard-windows 8 "
      "--shard-build-jobs 4 --router-only 1 "
      "--primary-radii 30,15,5 --progress-interval-seconds 0 "
      "--index " + direct_only_index +
      " >/tmp/navigamer_direct_only_build.stdout "
      "2>/tmp/navigamer_direct_only_build.stderr";
  assert(std::system(direct_only_build_command.c_str()) == 0);
  assert(!std::filesystem::exists(direct_only_index + ".route"));
  assert(std::filesystem::exists(direct_only_index + ".qpos"));
  const std::string direct_only_query_command =
      "OMP_NUM_THREADS=4 ./navigamer query-index-batch --index " +
      direct_only_index + " --reads " + route_budget_reads +
      " --tolerance 5 --out /tmp/navigamer_direct_only.tsv "
      ">/tmp/navigamer_direct_only.stdout "
      "2>/tmp/navigamer_direct_only.stderr";
  assert(std::system(direct_only_query_command.c_str()) == 0);
  {
    std::ifstream stderr_in("/tmp/navigamer_direct_only.stderr");
    assert(stderr_in.good());
    const std::string stderr_text(
        (std::istreambuf_iterator<char>(stderr_in)),
        std::istreambuf_iterator<char>());
    assert(stderr_text.find("sampled_qgram_direct_queries=185") !=
           std::string::npos);
    assert(stderr_text.find("peak_route_ids=0") !=
           std::string::npos);
  }
  const std::string direct_only_unsupported_command =
      "OMP_NUM_THREADS=4 ./navigamer query-index --index " +
      direct_only_index + " --query " + fallback_query +
      " --tolerance 5 >/tmp/navigamer_direct_only_unsupported.stdout "
      "2>/tmp/navigamer_direct_only_unsupported.stderr";
  assert(std::system(direct_only_unsupported_command.c_str()) != 0);
  {
    std::ifstream stderr_in(
        "/tmp/navigamer_direct_only_unsupported.stderr");
    assert(stderr_in.good());
    const std::string stderr_text(
        (std::istreambuf_iterator<char>(stderr_in)),
        std::istreambuf_iterator<char>());
    assert(stderr_text.find("graph fallback is unavailable") !=
           std::string::npos);
  }
  const std::string fallback_single_command =
      "OMP_NUM_THREADS=4 ./navigamer query-index --index " +
      fallback_index + " --query " + fallback_query +
      " --tolerance 5 "
      ">/tmp/navigamer_bounded_fallback_single.stdout "
      "2>/tmp/navigamer_bounded_fallback_single.stderr";
  assert(std::system(fallback_single_command.c_str()) == 0);
  {
    std::ifstream stderr_in(
        "/tmp/navigamer_bounded_fallback_single.stderr");
    assert(stderr_in.good());
    const std::string stderr_text(
        (std::istreambuf_iterator<char>(stderr_in)),
        std::istreambuf_iterator<char>());
    assert(stderr_text.find("peak_shards=64") != std::string::npos);
  }
  {
    std::ifstream stdout_in(
        "/tmp/navigamer_bounded_fallback_single.stdout");
    assert(stdout_in.good());
    const std::string stdout_text(
        (std::istreambuf_iterator<char>(stdout_in)),
        std::istreambuf_iterator<char>());
    assert(stdout_text.find("fallback_ref_123 dist=1") !=
           std::string::npos);
  }

  // Query input, routing tables, and batch plans are streamed in bounded
  // blocks. Cross the block boundary and verify every input record is still
  // emitted once and in order.
  const std::string streaming_reads =
      "/tmp/navigamer_streaming_queries.fq";
  const std::string streaming_tsv =
      "/tmp/navigamer_streaming_queries.tsv";
  constexpr size_t streaming_query_count = 8193;
  {
    std::ofstream reads(streaming_reads);
    assert(reads.good());
    const std::string sequence(150, 'C');
    const std::string quality(150, 'I');
    for (size_t query_idx = 0;
         query_idx < streaming_query_count; ++query_idx) {
      reads << "@stream_" << query_idx << '\n'
            << sequence << "\n+\n" << quality << '\n';
    }
  }
  const std::string streaming_query_command =
      "OMP_NUM_THREADS=4 ./navigamer query-index-batch --index " +
      fallback_index + " --reads " + streaming_reads +
      " --tolerance 0 --out " + streaming_tsv +
      " >/tmp/navigamer_streaming_query.stdout "
      "2>/tmp/navigamer_streaming_query.stderr";
  assert(std::system(streaming_query_command.c_str()) == 0);
  {
    std::ifstream stderr_in(
        "/tmp/navigamer_streaming_query.stderr");
    assert(stderr_in.good());
    const std::string stderr_text(
        (std::istreambuf_iterator<char>(stderr_in)),
        std::istreambuf_iterator<char>());
    assert(stderr_text.find("Queries: 8193") != std::string::npos);
    assert(stderr_text.find("batches=2") != std::string::npos);
  }
  {
    std::ifstream tsv(streaming_tsv);
    assert(tsv.good());
    std::string line;
    assert(static_cast<bool>(std::getline(tsv, line)));
    size_t row_count = 0;
    std::string first_query_id;
    std::string last_query_id;
    while (std::getline(tsv, line)) {
      if (line.empty()) continue;
      const auto row = split_tsv_line(line);
      if (row_count == 0) first_query_id = row.front();
      last_query_id = row.front();
      ++row_count;
    }
    assert(row_count == streaming_query_count);
    assert(first_query_id == "stream_0");
    assert(last_query_id == "stream_8192");
  }

  std::cout << "build-scale smoke tests passed\n";
  return 0;
}
