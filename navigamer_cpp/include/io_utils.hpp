#ifndef NAVIGAMER_IO_UTILS_HPP
#define NAVIGAMER_IO_UTILS_HPP

#include "structure.hpp"
#include <string>
#include <vector>
#include <utility>
#include <fstream>

namespace navigamer {

// 加载参考序列：若 path 为文件则读 FASTA，否则视为序列字符串
std::pair<std::string, std::string> load_reference(const std::string& path_or_string);

// 加载 reads：若为文件则读 FASTQ，否则单条序列 -> 单条 BioSequence
std::vector<std::shared_ptr<BioSequence>> load_reads(
    const std::string& path_or_string,
    const std::string& ref_id = "ref");

// 写 TSV 表（列名 + 行）
void write_tsv(const std::string& output_path,
               const std::vector<std::string>& columns,
               const std::vector<std::vector<std::string>>& rows);

// 单次命中展开为多行（每条 ref_position 一行），返回行数据
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
