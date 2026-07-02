#include "candidate_verifier.hpp"

#include "io_utils.hpp"
#include "tools.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace navigamer {
namespace {

struct CandidateQuery {
  std::string id;
  std::string sequence;
  bool has_source_pos = false;
  size_t source_pos = 0;
};

struct CandidateRow {
  int tau = 0;
  size_t raw_candidate_count = 0;
  std::vector<size_t> window_ids;
};

std::string uppercase_dna(std::string sequence) {
  for (char& base : sequence) {
    base = static_cast<char>(
        std::toupper(static_cast<unsigned char>(base)));
  }
  return sequence;
}

std::vector<std::string> split_tab(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream ss(line);
  std::string field;
  while (std::getline(ss, field, '\t')) fields.push_back(field);
  if (!line.empty() && line.back() == '\t') fields.emplace_back();
  return fields;
}

std::vector<size_t> parse_window_ids(const std::string& csv) {
  std::vector<size_t> ids;
  std::stringstream ss(csv);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) continue;
    ids.push_back(static_cast<size_t>(std::stoull(token)));
  }
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids;
}

bool parse_source_pos(const std::string& header, size_t* source_pos) {
  const std::string key = "source_pos=";
  const size_t begin = header.find(key);
  if (begin == std::string::npos) return false;
  size_t value_begin = begin + key.size();
  size_t value_end = value_begin;
  while (value_end < header.size() &&
         std::isdigit(static_cast<unsigned char>(header[value_end]))) {
    value_end++;
  }
  if (value_end == value_begin) return false;
  *source_pos = static_cast<size_t>(
      std::stoull(header.substr(value_begin, value_end - value_begin)));
  return true;
}

std::vector<CandidateQuery> read_candidate_fastq(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("unable to open reads FASTQ: " + path);
  std::vector<CandidateQuery> queries;
  std::string header;
  while (std::getline(in, header)) {
    if (header.empty()) continue;
    if (header[0] != '@') {
      throw std::runtime_error("FASTQ record does not start with @: " + path);
    }
    std::string sequence;
    std::string plus;
    std::string quality;
    if (!std::getline(in, sequence) || !std::getline(in, plus) ||
        !std::getline(in, quality)) {
      throw std::runtime_error("truncated FASTQ record: " + path);
    }
    CandidateQuery query;
    const std::string header_body = header.substr(1);
    const size_t id_end = header_body.find_first_of(" \t");
    query.id = id_end == std::string::npos ? header_body
                                           : header_body.substr(0, id_end);
    query.sequence = uppercase_dna(sequence);
    query.has_source_pos = parse_source_pos(header_body, &query.source_pos);
    queries.push_back(std::move(query));
  }
  return queries;
}

std::unordered_map<std::string, CandidateRow> read_candidate_rows(
    const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("unable to open candidate TSV: " + path);
  std::string line;
  if (!std::getline(in, line)) {
    throw std::runtime_error("candidate TSV is empty: " + path);
  }
  std::unordered_map<std::string, size_t> column;
  const auto header = split_tab(line);
  for (size_t i = 0; i < header.size(); ++i) column[header[i]] = i;
  for (const std::string& required :
       {"read_id", "tau", "raw_candidate_count", "candidate_window_ids"}) {
    if (column.find(required) == column.end()) {
      throw std::runtime_error("candidate TSV missing column: " + required);
    }
  }

  std::unordered_map<std::string, CandidateRow> rows;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const auto fields = split_tab(line);
    auto field_at = [&fields](size_t index) -> std::string {
      return index < fields.size() ? fields[index] : std::string();
    };
    const std::string read_id = field_at(column["read_id"]);
    CandidateRow row;
    row.tau = std::stoi(field_at(column["tau"]));
    row.raw_candidate_count =
        static_cast<size_t>(std::stoull(field_at(column["raw_candidate_count"])));
    row.window_ids = parse_window_ids(field_at(column["candidate_window_ids"]));
    rows[read_id] = std::move(row);
  }
  return rows;
}

std::vector<size_t> sorted_from_set(const std::unordered_set<size_t>& values) {
  std::vector<size_t> sorted(values.begin(), values.end());
  std::sort(sorted.begin(), sorted.end());
  return sorted;
}

std::string join_ids(const std::vector<size_t>& ids) {
  std::ostringstream out;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i) out << ",";
    out << ids[i];
  }
  return out.str();
}

size_t count_intersection(const std::unordered_set<size_t>& lhs,
                          const std::unordered_set<size_t>& rhs) {
  size_t count = 0;
  for (size_t value : lhs) {
    if (rhs.find(value) != rhs.end()) count++;
  }
  return count;
}

bool window_sequence(const std::string& reference,
                     size_t window_id,
                     int window_length,
                     int stride,
                     std::string* sequence) {
  const size_t start = window_id * static_cast<size_t>(stride);
  if (window_length < 0 ||
      start + static_cast<size_t>(window_length) > reference.size()) {
    return false;
  }
  *sequence = reference.substr(start, static_cast<size_t>(window_length));
  return true;
}

std::unordered_set<size_t> verify_window_ids(
    const std::string& reference,
    const CandidateQuery& query,
    const std::vector<size_t>& window_ids,
    int tolerance,
    int window_length,
    int stride) {
  std::unordered_set<size_t> verified;
  std::string candidate;
  for (size_t window_id : window_ids) {
    if (!window_sequence(reference, window_id, window_length, stride,
                         &candidate)) {
      continue;
    }
    const int distance = compute_distance_bounded_with_mode(
        query.sequence, candidate, tolerance, DistanceMode::Myers);
    if (distance <= tolerance) verified.insert(window_id);
  }
  return verified;
}

std::unordered_set<size_t> source_truth_ids(const std::string& reference,
                                            const CandidateQuery& query,
                                            int tolerance,
                                            int window_length,
                                            int stride) {
  std::unordered_set<size_t> truth;
  if (!query.has_source_pos || stride <= 0 ||
      query.source_pos % static_cast<size_t>(stride) != 0 ||
      query.source_pos + static_cast<size_t>(window_length) >
          reference.size()) {
    return truth;
  }
  const size_t window_id = query.source_pos / static_cast<size_t>(stride);
  const std::string source =
      reference.substr(query.source_pos, static_cast<size_t>(window_length));
  const int distance = compute_distance_bounded_with_mode(
      query.sequence, source, tolerance, DistanceMode::Myers);
  if (distance <= tolerance) truth.insert(window_id);
  return truth;
}

std::unordered_set<size_t> exhaustive_truth_ids(const std::string& reference,
                                                const CandidateQuery& query,
                                                int tolerance,
                                                int window_length,
                                                int stride) {
  std::unordered_set<size_t> truth;
  if (stride <= 0 || window_length <= 0 ||
      static_cast<size_t>(window_length) > reference.size()) {
    return truth;
  }
  size_t window_id = 0;
  for (size_t start = 0;
       start + static_cast<size_t>(window_length) <= reference.size();
       start += static_cast<size_t>(stride), ++window_id) {
    const std::string candidate =
        reference.substr(start, static_cast<size_t>(window_length));
    const int distance = compute_distance_bounded_with_mode(
        query.sequence, candidate, tolerance, DistanceMode::Myers);
    if (distance <= tolerance) truth.insert(window_id);
  }
  return truth;
}

std::string format_double_local(double value) {
  std::ostringstream out;
  out.setf(std::ios::fixed);
  out.precision(6);
  out << value;
  return out.str();
}

}  // namespace

CandidateTruthMode parse_candidate_truth_mode(const std::string& value) {
  if (value == "source") return CandidateTruthMode::Source;
  if (value == "exhaustive") return CandidateTruthMode::Exhaustive;
  throw std::invalid_argument("candidate truth mode must be source or exhaustive");
}

CandidateVerifierSummary run_candidate_verifier(
    const CandidateVerifierConfig& config) {
  if (config.tolerance < 0) {
    throw std::invalid_argument("candidate verifier tolerance must be non-negative");
  }
  if (config.window_length <= 0) {
    throw std::invalid_argument("candidate verifier window length must be positive");
  }
  if (config.stride <= 0) {
    throw std::invalid_argument("candidate verifier stride must be positive");
  }

  auto [ref_id, reference] = load_reference(config.reference_input);
  (void)ref_id;
  reference = uppercase_dna(std::move(reference));
  const auto queries = read_candidate_fastq(config.reads_fastq_path);
  const auto candidate_rows = read_candidate_rows(config.candidates_tsv_path);

  CandidateVerifierSummary summary;
  summary.query_count = queries.size();
  std::vector<std::vector<std::string>> detail_rows;
  detail_rows.reserve(queries.size());

  for (const auto& query : queries) {
    const auto row_it = candidate_rows.find(query.id);
    const CandidateRow* candidate_row =
        row_it == candidate_rows.end() ? nullptr : &row_it->second;
    const int tolerance =
        candidate_row ? candidate_row->tau : config.tolerance;
    const std::vector<size_t> empty_candidates;
    const std::vector<size_t>& candidate_ids =
        candidate_row ? candidate_row->window_ids : empty_candidates;
    const size_t raw_count =
        candidate_row ? candidate_row->raw_candidate_count : size_t{0};

    const auto verify_start = std::chrono::steady_clock::now();
    const auto verified = verify_window_ids(
        reference, query, candidate_ids, tolerance, config.window_length,
        config.stride);
    const auto verify_end = std::chrono::steady_clock::now();

    const auto truth_start = std::chrono::steady_clock::now();
    const auto truth =
        config.truth_mode == CandidateTruthMode::Source
            ? source_truth_ids(reference, query, tolerance, config.window_length,
                               config.stride)
            : exhaustive_truth_ids(reference, query, tolerance,
                                   config.window_length, config.stride);
    const auto truth_end = std::chrono::steady_clock::now();

    const size_t tp = count_intersection(verified, truth);
    const size_t fp = verified.size() - tp;
    const size_t fn = truth.size() - tp;

    summary.raw_candidate_count += raw_count;
    summary.verified_match_count += verified.size();
    summary.truth_match_count += truth.size();
    summary.tp_count += tp;
    summary.fp_count += fp;
    summary.fn_count += fn;
    summary.verify_ms +=
        std::chrono::duration<double, std::milli>(verify_end - verify_start)
            .count();
    summary.truth_ms +=
        std::chrono::duration<double, std::milli>(truth_end - truth_start)
            .count();

    detail_rows.push_back({
        query.id,
        std::to_string(tolerance),
        std::to_string(raw_count),
        std::to_string(verified.size()),
        std::to_string(truth.size()),
        std::to_string(tp),
        std::to_string(fp),
        std::to_string(fn),
        join_ids(sorted_from_set(verified)),
        join_ids(sorted_from_set(truth)),
    });
  }

  if (!config.detail_tsv_path.empty()) {
    write_tsv(config.detail_tsv_path,
              {"read_id", "tau", "raw_candidate_count",
               "verified_match_count", "truth_match_count", "tp_count",
               "fp_count", "fn_count", "verified_window_ids",
               "truth_window_ids"},
              detail_rows);
  }
  if (!config.summary_tsv_path.empty()) {
    write_tsv(config.summary_tsv_path,
              {"query_count", "raw_candidate_count", "verified_match_count",
               "truth_match_count", "tp_count", "fp_count", "fn_count",
               "verify_ms", "truth_ms"},
              {{std::to_string(summary.query_count),
                std::to_string(summary.raw_candidate_count),
                std::to_string(summary.verified_match_count),
                std::to_string(summary.truth_match_count),
                std::to_string(summary.tp_count),
                std::to_string(summary.fp_count),
                std::to_string(summary.fn_count),
                format_double_local(summary.verify_ms),
                format_double_local(summary.truth_ms)}});
  }

  return summary;
}

}  // namespace navigamer
