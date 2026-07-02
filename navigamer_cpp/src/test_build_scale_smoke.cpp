#include <cassert>
#include <cstdlib>
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
  while (std::getline(batch_in, batch_row)) {
    if (batch_row.find("read0\t") == 0) saw_read0 = true;
    if (batch_row.find("read1\t") == 0) saw_read1 = true;
    assert(batch_row.find("\ttrue\t") != std::string::npos);
  }
  assert(saw_read0);
  assert(saw_read1);

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

  std::cout << "build-scale smoke tests passed\n";
  return 0;
}
