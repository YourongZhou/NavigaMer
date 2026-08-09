#ifndef NAVIGAMER_IO_UTILS_HPP
#define NAVIGAMER_IO_UTILS_HPP

#include "structure.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <fstream>

namespace navigamer {

class ReferenceFileMapping;

struct LoadedReference {
  std::string id;
  std::string sequence;
  std::vector<ReferenceContig> contigs;
};

struct ReferenceFileCheckpoint {
  size_t sequence_pos = 0;
  uint64_t file_pos = 0;
};

struct QuerySequence {
  std::string id;
  std::string seq;
};

// Incremental FASTQ/literal reader for bounded-memory query pipelines.
class QuerySequenceReader {
 public:
  explicit QuerySequenceReader(const std::string& path_or_string);

  bool next(QuerySequence* query);
  size_t read_block(size_t max_records,
                    std::vector<QuerySequence>* queries);

 private:
  std::ifstream input_;
  std::string literal_;
  std::string line_;
  bool literal_pending_ = false;
};

// Sparse random-access index over an existing FASTA/plain-sequence file.
// It preserves load_reference_genome() normalization without retaining the
// complete normalized reference in memory.
struct IndexedReferenceFile {
  std::string path;
  std::string id;
  size_t sequence_size = 0;
  std::vector<ReferenceContig> contigs;
  std::vector<ReferenceFileCheckpoint> checkpoints;

  std::string slice(size_t begin, size_t end) const;

 private:
  std::shared_ptr<const ReferenceFileMapping> mapping;

  friend IndexedReferenceFile index_reference_genome_file(
      const std::string& path);
};

// Load a reference: existing paths are parsed as FASTA, otherwise literal DNA.
LoadedReference load_reference_genome(const std::string& path_or_string);
IndexedReferenceFile index_reference_genome_file(const std::string& path);
std::pair<std::string, std::string> load_reference(const std::string& path_or_string);

// Load reads: existing paths are parsed as FASTQ, otherwise one literal read.
std::vector<std::shared_ptr<BioSequence>> load_reads(
    const std::string& path_or_string,
    const std::string& ref_id = "ref");

// Write a TSV table with an explicit header.
class TsvWriter {
 public:
  TsvWriter(const std::string& output_path,
            const std::vector<std::string>& columns);
  void write_row(const std::vector<std::string>& row);
  void close();

 private:
  std::ofstream out_;
};

void write_tsv(const std::string& output_path,
               const std::vector<std::string>& columns,
               const std::vector<std::vector<std::string>>& rows);

// Expand one hit into mapper-like TSV rows, one per stored reference position.
struct TsvRow {
  std::string query_id, hit_id, distance_str, ref_positions_json;
  std::string read_id, read_len, ref_id, strand, query_start, reference_start;
  std::string aligned_length, score, edit_distance, query_fragment, reference_fragment;
  std::string bwt_start, bwt_end;
};
std::vector<TsvRow> search_results_to_tsv_rows(
    const std::string& query_id, const std::string& query_seq, int query_start,
    const BioSequence& hit, int edit_distance);

}  // namespace navigamer

#endif  // NAVIGAMER_IO_UTILS_HPP
