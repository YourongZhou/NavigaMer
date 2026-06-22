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
