#include "io_utils.hpp"
#include <array>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <stdexcept>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace navigamer {

namespace {

struct ReferenceByteTable {
  std::array<bool, 256> whitespace{};
  std::array<char, 256> uppercase{};

  ReferenceByteTable() {
    for (size_t byte = 0; byte < whitespace.size(); ++byte) {
      const auto value = static_cast<unsigned char>(byte);
      whitespace[byte] = std::isspace(value) != 0;
      uppercase[byte] = static_cast<char>(std::toupper(value));
    }
  }
};

const ReferenceByteTable& reference_byte_table() {
  static const ReferenceByteTable table;
  return table;
}

}  // namespace

class ReferenceFileMapping {
 public:
  ReferenceFileMapping(void* address, size_t size)
      : address_(address), size_(size) {}
  ~ReferenceFileMapping() {
#if defined(__unix__) || defined(__APPLE__)
    if (address_ && size_ != 0) munmap(address_, size_);
#endif
  }

  const char* data() const {
    return static_cast<const char*>(address_);
  }
  size_t size() const { return size_; }
  void discard(size_t begin, size_t end) const {
#if defined(__unix__) || defined(__APPLE__)
#if defined(MADV_DONTNEED)
    if (!address_ || begin >= end || begin >= size_) return;
    end = std::min(end, size_);
    static const size_t page_size = [] {
      const long raw_page_size = sysconf(_SC_PAGESIZE);
      return raw_page_size > 0
          ? static_cast<size_t>(raw_page_size)
          : size_t{4096};
    }();
    const size_t begin_remainder = begin % page_size;
    const size_t begin_advance = begin_remainder == 0
        ? 0
        : page_size - begin_remainder;
    const size_t aligned_begin = begin_advance > size_ - begin
        ? size_
        : begin + begin_advance;
    const size_t aligned_end = end == size_
        ? end
        : end - end % page_size;
    if (aligned_begin < aligned_end) {
      (void)madvise(
          static_cast<char*>(address_) + aligned_begin,
          aligned_end - aligned_begin, MADV_DONTNEED);
    }
#endif
#else
    (void)begin;
    (void)end;
#endif
  }

 private:
  void* address_ = nullptr;
  size_t size_ = 0;
};

static bool is_file(const std::string& path) {
  std::ifstream f(path);
  return f.good();
}

IndexedReferenceFile index_reference_genome_file(
    const std::string& path) {
  if (!is_file(path)) {
    throw std::runtime_error("reference file does not exist: " + path);
  }
  constexpr size_t kCheckpointStride = size_t{1} << 20;
  IndexedReferenceFile reference;
  reference.path = path;
  std::string current_id;
  size_t current_begin = 0;
  size_t next_checkpoint = 0;
  const auto& byte_table = reference_byte_table();

  const auto finish_contig = [&]() {
    if (current_id.empty()) return;
    if (reference.sequence_size >= static_cast<size_t>(UINT32_MAX)) {
      throw std::runtime_error(
          "reference length exceeds 32-bit coordinate storage");
    }
    reference.contigs.push_back(
        {current_id, static_cast<uint32_t>(current_begin),
         static_cast<uint32_t>(reference.sequence_size)});
  };
  const auto begin_implicit_contig = [&]() {
    current_id = "ref";
    if (reference.id.empty()) reference.id = current_id;
    current_begin = reference.sequence_size;
    next_checkpoint = current_begin;
  };

  const auto process_line = [&](std::string_view line,
                                uint64_t line_offset) {
    size_t logical_end = line.size();
    while (logical_end != 0 &&
           (line[logical_end - 1] == '\r' ||
            line[logical_end - 1] == '\n')) {
      --logical_end;
    }
    if (logical_end == 0) return;
    if (line[0] == '>') {
      finish_contig();
      current_id = std::string(line.substr(1, logical_end - 1));
      const size_t separator = current_id.find_first_of(" \t");
      if (separator != std::string::npos) {
        current_id.resize(separator);
      }
      if (current_id.empty()) current_id = "ref";
      if (reference.id.empty()) reference.id = current_id;
      current_begin = reference.sequence_size;
      next_checkpoint = current_begin;
      return;
    }
    if (current_id.empty()) begin_implicit_contig();

    for (size_t char_idx = 0; char_idx < logical_end; ++char_idx) {
      const unsigned char base =
          static_cast<unsigned char>(line[char_idx]);
      if (byte_table.whitespace[base]) continue;
      if (reference.sequence_size >= next_checkpoint) {
        reference.checkpoints.push_back(
            {reference.sequence_size, line_offset + char_idx});
        if (reference.sequence_size >
            std::numeric_limits<size_t>::max() - kCheckpointStride) {
          throw std::runtime_error("reference checkpoint overflow");
        }
        next_checkpoint = reference.sequence_size + kCheckpointStride;
      }
      ++reference.sequence_size;
    }
  };

#if defined(__unix__) || defined(__APPLE__)
  const int descriptor = open(path.c_str(), O_RDONLY);
  if (descriptor >= 0) {
    struct stat metadata {};
    if (fstat(descriptor, &metadata) == 0 && metadata.st_size > 0 &&
        static_cast<uint64_t>(metadata.st_size) <=
            std::numeric_limits<size_t>::max()) {
      const size_t mapped_size = static_cast<size_t>(metadata.st_size);
      void* address = mmap(
          nullptr, mapped_size, PROT_READ, MAP_PRIVATE, descriptor, 0);
      if (address != MAP_FAILED) {
        reference.mapping = std::make_shared<ReferenceFileMapping>(
            address, mapped_size);
      }
    }
    close(descriptor);
  }
#endif

  if (reference.mapping) {
    constexpr size_t kDiscardStride = size_t{16} << 20;
    const char* data = reference.mapping->data();
    const size_t file_size = reference.mapping->size();
    size_t line_begin = 0;
    size_t discard_begin = 0;
    while (line_begin < file_size) {
      const void* newline = std::memchr(
          data + line_begin, '\n', file_size - line_begin);
      const size_t line_end = newline
          ? static_cast<size_t>(
                static_cast<const char*>(newline) - data)
          : file_size;
      process_line(
          std::string_view(data + line_begin, line_end - line_begin),
          line_begin);
      line_begin = line_end == file_size ? file_size : line_end + 1;
      if (line_begin - discard_begin >= kDiscardStride) {
        reference.mapping->discard(discard_begin, line_begin);
        discard_begin = line_begin;
      }
    }
    reference.mapping->discard(discard_begin, file_size);
  } else {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
      throw std::runtime_error("unable to open reference file: " + path);
    }
    std::string line;
    uint64_t raw_offset = 0;
    while (std::getline(f, line)) {
      const uint64_t line_offset = raw_offset;
      const bool terminated = !f.eof();
      const uint64_t remaining =
          std::numeric_limits<uint64_t>::max() - raw_offset;
      if (line.size() > remaining ||
          (terminated && line.size() == remaining)) {
        throw std::runtime_error("reference file offset overflow");
      }
      raw_offset += line.size() + (terminated ? 1 : 0);
      process_line(line, line_offset);
    }
    if (!f.eof()) {
      throw std::runtime_error(
          "failed while reading reference file: " + path);
    }
  }
  finish_contig();
  if (reference.id.empty()) reference.id = "ref";
  return reference;
}

std::string IndexedReferenceFile::slice(size_t begin, size_t end) const {
  if (begin > end || end > sequence_size) {
    throw std::out_of_range("reference file slice is out of bounds");
  }
  if (begin == end) return {};
  const auto contig_it = std::upper_bound(
      contigs.begin(), contigs.end(), begin,
      [](size_t position, const ReferenceContig& contig) {
        return position < contig.begin;
      });
  if (contig_it == contigs.begin()) {
    throw std::out_of_range("reference file slice has no contig");
  }
  const auto& contig = *std::prev(contig_it);
  if (begin < contig.begin || begin >= contig.end || end > contig.end) {
    throw std::out_of_range(
        "reference file slice crosses a contig boundary");
  }

  const auto checkpoint_it = std::upper_bound(
      checkpoints.begin(), checkpoints.end(), begin,
      [](size_t position, const ReferenceFileCheckpoint& checkpoint) {
        return position < checkpoint.sequence_pos;
      });
  if (checkpoint_it == checkpoints.begin()) {
    throw std::runtime_error("reference file slice has no checkpoint");
  }
  const auto& checkpoint = *std::prev(checkpoint_it);
  if (checkpoint.sequence_pos < contig.begin ||
      checkpoint.sequence_pos > begin) {
    throw std::runtime_error("reference file slice checkpoint is invalid");
  }

  std::string result(end - begin, '\0');
  size_t sequence_pos = checkpoint.sequence_pos;
  bool at_line_start = false;
  const auto& byte_table = reference_byte_table();
  const auto consume_byte = [&](unsigned char byte) {
    if (byte == '\n') {
      at_line_start = true;
      return;
    }
    if (at_line_start && byte == '>') {
      throw std::runtime_error(
          "reference file slice ended before its contig boundary");
    }
    at_line_start = false;
    if (byte_table.whitespace[byte]) return;
    if (sequence_pos >= begin) {
      result[sequence_pos - begin] = byte_table.uppercase[byte];
    }
    ++sequence_pos;
  };

  if (mapping) {
    size_t raw_pos = static_cast<size_t>(checkpoint.file_pos);
    while (sequence_pos < end && raw_pos < mapping->size()) {
      consume_byte(
          static_cast<unsigned char>(mapping->data()[raw_pos++]));
    }
    mapping->discard(
        static_cast<size_t>(checkpoint.file_pos), raw_pos);
  } else {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      throw std::runtime_error("unable to reopen reference file: " + path);
    }
    in.seekg(static_cast<std::streamoff>(checkpoint.file_pos));
    if (!in) {
      throw std::runtime_error("unable to seek reference file: " + path);
    }
    std::array<char, 64 * 1024> buffer{};
    while (sequence_pos < end && in) {
      in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
      const size_t bytes_read = static_cast<size_t>(in.gcount());
      for (size_t idx = 0;
           idx < bytes_read && sequence_pos < end; ++idx) {
        consume_byte(static_cast<unsigned char>(buffer[idx]));
      }
    }
  }
  if (sequence_pos != end) {
    throw std::runtime_error("truncated reference file slice: " + path);
  }
  return result;
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

LoadedReference load_reference_genome(const std::string& path_or_string) {
  if (!is_file(path_or_string)) {
    std::string s = path_or_string;
    auto end = std::find_if(s.begin(), s.end(), [](char c) { return c == '\n' || c == '\r'; });
    s = std::string(s.begin(), end);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(0, 1);
    if (s.size() >= static_cast<size_t>(UINT32_MAX)) {
      throw std::runtime_error(
          "reference length exceeds 32-bit coordinate storage");
    }
    const auto& byte_table = reference_byte_table();
    for (char& c : s) {
      c = byte_table.uppercase[static_cast<unsigned char>(c)];
    }
    return {"ref", s,
            {{"ref", 0, static_cast<uint32_t>(s.size())}}};
  }
  std::ifstream f(path_or_string);
  LoadedReference reference;
  std::string line;
  std::string current_id;
  size_t current_begin = 0;
  const auto& byte_table = reference_byte_table();
  auto finish_contig = [&]() {
    if (current_id.empty()) return;
    if (reference.sequence.size() >= static_cast<size_t>(UINT32_MAX)) {
      throw std::runtime_error(
          "reference length exceeds 32-bit coordinate storage");
    }
    reference.contigs.push_back(
        {current_id, static_cast<uint32_t>(current_begin),
         static_cast<uint32_t>(reference.sequence.size())});
  };
  while (std::getline(f, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
    if (line.empty()) continue;
    if (line[0] == '>') {
      finish_contig();
      current_id = line.substr(1);
      size_t sp = current_id.find_first_of(" \t");
      if (sp != std::string::npos) current_id = current_id.substr(0, sp);
      if (current_id.empty()) current_id = "ref";
      if (reference.id.empty()) reference.id = current_id;
      current_begin = reference.sequence.size();
    } else {
      if (current_id.empty()) {
        current_id = "ref";
        reference.id = current_id;
        current_begin = reference.sequence.size();
      }
      for (char c : line) {
        const auto byte = static_cast<unsigned char>(c);
        if (byte_table.whitespace[byte]) continue;
        reference.sequence.push_back(byte_table.uppercase[byte]);
      }
    }
  }
  finish_contig();
  if (reference.id.empty()) reference.id = "ref";
  return reference;
}

std::pair<std::string, std::string> load_reference(
    const std::string& path_or_string) {
  auto reference = load_reference_genome(path_or_string);
  return {std::move(reference.id), std::move(reference.sequence)};
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

TsvWriter::TsvWriter(const std::string& output_path,
                     const std::vector<std::string>& columns)
    : out_(output_path) {
  if (!out_) {
    throw std::runtime_error("unable to open TSV output: " + output_path);
  }
  for (size_t i = 0; i < columns.size(); ++i) {
    if (i) out_ << '\t';
    out_ << columns[i];
  }
  out_ << '\n';
  if (!out_) {
    throw std::runtime_error("failed to write TSV header: " + output_path);
  }
}

void TsvWriter::write_row(const std::vector<std::string>& row) {
  for (size_t i = 0; i < row.size(); ++i) {
    if (i) out_ << '\t';
    out_ << row[i];
  }
  out_ << '\n';
  if (!out_) {
    throw std::runtime_error("failed to write TSV row");
  }
}

void TsvWriter::close() {
  out_.close();
  if (!out_) throw std::runtime_error("failed to finalize TSV output");
}

void write_tsv(const std::string& output_path,
               const std::vector<std::string>& columns,
               const std::vector<std::vector<std::string>>& rows) {
  TsvWriter writer(output_path, columns);
  for (const auto& row : rows) writer.write_row(row);
  writer.close();
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
