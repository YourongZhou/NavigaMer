#ifndef NAVIGAMER_IO_UTILS_HPP
#define NAVIGAMER_IO_UTILS_HPP

#include "structure.hpp"
#include <string>
#include <vector>
#include <utility>
#include <fstream>

namespace navigamer {

// Load a reference: existing paths are parsed as FASTA, otherwise literal DNA.
std::pair<std::string, std::string> load_reference(const std::string& path_or_string);

// Load reads: existing paths are parsed as FASTQ, otherwise one literal read.
std::vector<std::shared_ptr<BioSequence>> load_reads(
    const std::string& path_or_string,
    const std::string& ref_id = "ref");

// Write a TSV table with an explicit header.
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
