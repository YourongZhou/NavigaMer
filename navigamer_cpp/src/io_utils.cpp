#include "io_utils.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>

namespace navigamer {

static bool is_file(const std::string& path) {
  std::ifstream f(path);
  return f.good();
}

static bool parse_source_pos(const std::string& header, size_t* source_pos) {
  const std::string key = "source_pos=";
  size_t pos = header.find(key);
  if (pos == std::string::npos) return false;
  pos += key.size();
  if (pos >= header.size() || !std::isdigit(static_cast<unsigned char>(header[pos]))) {
    return false;
  }
  char* end = nullptr;
  const unsigned long long value =
      std::strtoull(header.c_str() + pos, &end, 10);
  if (end == header.c_str() + pos) return false;
  *source_pos = static_cast<size_t>(value);
  return true;
}

std::pair<std::string, std::string> load_reference(const std::string& path_or_string) {
  if (!is_file(path_or_string)) {
    std::string s = path_or_string;
    auto end = std::find_if(s.begin(), s.end(), [](char c) { return c == '\n' || c == '\r'; });
    s = std::string(s.begin(), end);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(0, 1);
    return {"ref", s};
  }
  std::ifstream f(path_or_string);
  std::string ref_id = "ref";
  std::string line, seq;
  while (std::getline(f, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
    if (line.empty()) continue;
    if (line[0] == '>') {
      ref_id = line.substr(1);
      size_t sp = ref_id.find_first_of(" \t");
      if (sp != std::string::npos) ref_id = ref_id.substr(0, sp);
    } else {
      seq += line;
    }
  }
  return {ref_id, seq};
}

std::vector<std::shared_ptr<BioSequence>> load_reads(
    const std::string& path_or_string,
    const std::string& /*ref_id*/) {
  if (!is_file(path_or_string)) {
    std::string s = path_or_string;
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(0, 1);
    auto seq = std::make_shared<BioSequence>("query_0", s);
    return {seq};
  }
  std::vector<std::shared_ptr<BioSequence>> reads;
  std::ifstream f(path_or_string);
  std::string line;
  while (std::getline(f, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
    if (line.empty()) continue;
    if (line[0] != '@') continue;
    std::string header_body = line.substr(1);
    std::string seq_id = header_body;
    size_t sp = seq_id.find_first_of(" \t");
    if (sp != std::string::npos) seq_id = seq_id.substr(0, sp);
    if (!std::getline(f, line)) break;
    std::string sequence = line;
    while (!sequence.empty() && (sequence.back() == '\r' || sequence.back() == '\n')) sequence.pop_back();
    std::getline(f, line);  // +
    std::getline(f, line);   // qual
    if (!sequence.empty()) {
      auto read = std::make_shared<BioSequence>(seq_id, sequence);
      read->has_source_pos = parse_source_pos(header_body, &read->source_pos);
      reads.push_back(read);
    }
  }
  return reads;
}

void write_tsv(const std::string& output_path,
               const std::vector<std::string>& columns,
               const std::vector<std::vector<std::string>>& rows) {
  std::ofstream out(output_path);
  if (!out) {
    throw std::runtime_error("unable to open TSV output: " + output_path);
  }
  for (size_t i = 0; i < columns.size(); ++i) {
    if (i) out << '\t';
    out << columns[i];
  }
  out << '\n';
  for (const auto& row : rows) {
    for (size_t i = 0; i < row.size(); ++i) {
      if (i) out << '\t';
      out << row[i];
    }
    out << '\n';
  }
  out.close();
  if (!out) {
    throw std::runtime_error("failed to write TSV output: " + output_path);
  }
}

// Escape minimal JSON strings and format stored reference positions.
static std::string ref_positions_to_json(const std::vector<RefPosition>& pos) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < pos.size(); ++i) {
    if (i) os << ",";
    os << "[\"" << pos[i].ref_id << "\"," << pos[i].start << "," << pos[i].end
       << ",\"" << pos[i].strand << "\"]";
  }
  os << "]";
  return os.str();
}

std::vector<TsvRow> search_results_to_tsv_rows(
    const std::string& query_id, const std::string& query_seq, int query_start,
    const BioSequence& hit, int edit_distance) {
  int aligned_len = static_cast<int>(hit.seq.size());
  int score = aligned_len - edit_distance;
  std::string ref_positions_json = ref_positions_to_json(hit.ref_positions);

  std::string bwt_s = hit.bwt_interval.valid() ? std::to_string(hit.bwt_interval.start) : "-1";
  std::string bwt_e = hit.bwt_interval.valid() ? std::to_string(hit.bwt_interval.end)   : "-1";

  std::vector<TsvRow> rows;
  if (!hit.ref_positions.empty()) {
    for (const auto& occ : hit.ref_positions) {
      TsvRow r;
      r.query_id = query_id;
      r.hit_id = hit.id;
      r.distance_str = std::to_string(edit_distance);
      r.ref_positions_json = ref_positions_json;
      r.read_id = query_id;
      r.read_len = std::to_string(static_cast<int>(query_seq.size()));
      r.ref_id = occ.ref_id;
      r.strand = occ.strand;
      r.query_start = std::to_string(query_start);
      r.reference_start = std::to_string(occ.start);
      r.aligned_length = std::to_string(occ.end - occ.start);
      r.score = std::to_string(score);
      r.edit_distance = std::to_string(edit_distance);
      r.query_fragment = query_seq;
      r.reference_fragment = hit.seq;
      r.bwt_start = bwt_s;
      r.bwt_end = bwt_e;
      rows.push_back(r);
    }
  } else {
    TsvRow r;
    r.query_id = query_id;
    r.hit_id = hit.id;
    r.distance_str = std::to_string(edit_distance);
    r.ref_positions_json = ref_positions_json;
    r.read_id = query_id;
    r.read_len = std::to_string(static_cast<int>(query_seq.size()));
    r.ref_id = "";
    r.strand = "+";
    r.query_start = std::to_string(query_start);
    r.reference_start = "0";
    r.aligned_length = "0";
    r.score = std::to_string(score);
    r.edit_distance = std::to_string(edit_distance);
    r.query_fragment = query_seq;
    r.reference_fragment = hit.seq;
    r.bwt_start = bwt_s;
    r.bwt_end = bwt_e;
    rows.push_back(r);
  }
  return rows;
}

}  // namespace navigamer
