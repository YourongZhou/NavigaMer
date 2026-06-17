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
  std::remove(out_path.c_str());

  const std::string command =
      "./navigamer build-scale "
      "--ref ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT "
      "--window 12 --stride 4 --prefix-lengths 24,36 "
      "--primary-radii 12,6,2 --leaf-attach-direction world-to-seq --out " +
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
    const double phase3_distance_ms =
        std::stod(row[column.at("phase3_child_mbb_distance_ms")]);
    const double leaf_beacon_ms =
        std::stod(row[column.at("leaf_beacon_distance_ms")]);
    assert(total_ms > 0.0);
    assert(phase0_ms >= 0.0);
    assert(phase2_query_ms >= 0.0);
    assert(phase3_distance_ms >= 0.0);
    assert(leaf_beacon_ms >= 0.0);
    assert(row[column.at("leaf_attach_direction_used")] == "world_to_seq");
    row_count++;
  }
  assert(row_count == 2);

  std::cout << "build-scale smoke tests passed\n";
  return 0;
}
