#include "reference_windows.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

uint32_t checked_uint32(uint64_t value, const char* description) {
  if (value > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error(std::string(description) + " exceeds uint32_t");
  }
  return static_cast<uint32_t>(value);
}

std::string parse_contig_id(const std::string& header) {
  std::istringstream fields(header.substr(1));
  std::string contig_id;
  fields >> contig_id;
  if (contig_id.empty()) {
    throw std::runtime_error("FASTA record has an empty contig ID");
  }
  return contig_id;
}

}  // namespace

ReferenceWindows::ReferenceWindows(std::string contig_id, std::string sequence,
                                   uint32_t window_length, uint32_t stride,
                                   uint32_t window_count)
    : contig_id_(std::move(contig_id)),
      sequence_(std::move(sequence)),
      window_length_(window_length),
      stride_(stride),
      window_count_(window_count) {}

ReferenceWindows ReferenceWindows::from_fasta(const std::string& path,
                                              uint32_t window_length,
                                              uint32_t stride) {
  if (window_length == 0) {
    throw std::invalid_argument("window length must be greater than zero");
  }
  if (stride == 0) {
    throw std::invalid_argument("stride must be greater than zero");
  }

  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("unable to open reference FASTA: " + path);
  }

  std::string contig_id;
  std::string sequence;
  std::string line;
  bool saw_header = false;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) {
      continue;
    }
    if (line.front() == '>') {
      if (saw_header) {
        throw std::runtime_error("reference FASTA must contain exactly one contig");
      }
      saw_header = true;
      contig_id = parse_contig_id(line);
      continue;
    }
    if (!saw_header) {
      throw std::runtime_error("FASTA sequence appears before its header");
    }
    if (std::any_of(line.begin(), line.end(), [](unsigned char character) {
          return std::isspace(character) != 0;
        })) {
      throw std::runtime_error("FASTA sequence lines must not contain whitespace");
    }
    if (std::any_of(line.begin(), line.end(), [](unsigned char character) {
          return std::isalpha(character) == 0;
        })) {
      throw std::runtime_error(
          "FASTA sequence lines must contain alphabetic symbols only");
    }
    if (line.size() > std::numeric_limits<uint32_t>::max() - sequence.size()) {
      throw std::overflow_error("reference length exceeds uint32_t");
    }
    std::transform(line.begin(), line.end(), line.begin(),
                   [](unsigned char character) {
                     return static_cast<char>(std::toupper(character));
                   });
    sequence += line;
  }
  if (!input.eof()) {
    throw std::runtime_error("failed while reading reference FASTA: " + path);
  }
  if (!saw_header) {
    throw std::runtime_error("reference FASTA contains no record");
  }
  if (sequence.empty()) {
    throw std::runtime_error("reference FASTA contains an empty sequence");
  }

  const uint64_t reference_length = sequence.size();
  if (reference_length < window_length) {
    throw std::invalid_argument("reference is shorter than the window length");
  }
  const uint64_t count =
      (reference_length - static_cast<uint64_t>(window_length)) / stride + 1;
  return ReferenceWindows(std::move(contig_id), std::move(sequence),
                          window_length, stride,
                          checked_uint32(count, "number of windows"));
}

uint32_t ReferenceWindows::size() const { return window_count_; }

const std::string& ReferenceWindows::contig_id() const { return contig_id_; }

const std::string& ReferenceWindows::sequence() const { return sequence_; }

std::string_view ReferenceWindows::window(uint32_t id) const {
  const uint32_t window_start = start(id);
  return std::string_view(sequence_).substr(window_start, window_length_);
}

uint32_t ReferenceWindows::start(uint32_t id) const {
  if (id >= window_count_) {
    throw std::out_of_range("window ID is out of range");
  }
  const uint64_t window_start = static_cast<uint64_t>(id) * stride_;
  return checked_uint32(window_start, "window start");
}

uint32_t ReferenceWindows::window_id_for_start(uint32_t window_start) const {
  if (window_start % stride_ != 0) {
    throw std::invalid_argument("window start is not aligned to the stride");
  }
  const uint64_t id = static_cast<uint64_t>(window_start) / stride_;
  if (id >= window_count_) {
    throw std::out_of_range("window start is out of range");
  }
  return checked_uint32(id, "window ID");
}

std::vector<uint32_t> ReferenceWindows::covering_window_ids(
    uint32_t occurrence_start, uint32_t span) const {
  if (span == 0) {
    throw std::invalid_argument("occurrence span must be greater than zero");
  }
  if (span > window_length_) {
    throw std::invalid_argument(
        "occurrence span must not exceed the window length");
  }
  const uint64_t occurrence_end =
      static_cast<uint64_t>(occurrence_start) + span;
  if (occurrence_start >= sequence_.size() || occurrence_end > sequence_.size()) {
    throw std::out_of_range("occurrence interval is outside the reference");
  }

  const uint64_t minimum_start = occurrence_end > window_length_
                                     ? occurrence_end - window_length_
                                     : 0;
  const uint64_t maximum_window_start =
      static_cast<uint64_t>(window_count_ - 1) * stride_;
  const uint64_t maximum_start =
      std::min<uint64_t>(occurrence_start, maximum_window_start);
  if (minimum_start > maximum_start) {
    return {};
  }

  const uint64_t first_id = (minimum_start + stride_ - 1) / stride_;
  const uint64_t last_id = maximum_start / stride_;
  if (first_id > last_id) {
    return {};
  }

  std::vector<uint32_t> ids;
  ids.reserve(checked_uint32(last_id - first_id + 1, "covering window count"));
  for (uint64_t id = first_id; id <= last_id; ++id) {
    ids.push_back(checked_uint32(id, "window ID"));
  }
  return ids;
}
