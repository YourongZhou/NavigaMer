#include "dna_edit_distribution.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <system_error>

#ifndef DNA_EDIT_COMPILER
#define DNA_EDIT_COMPILER "unknown"
#endif

#ifndef DNA_EDIT_WFA2_VERSION
#define DNA_EDIT_WFA2_VERSION "unknown"
#endif

#ifndef DNA_EDIT_WFA2_COMMIT
#define DNA_EDIT_WFA2_COMMIT "unknown"
#endif

namespace dna_edit_distribution {
namespace {

std::uint64_t total_count(const std::vector<std::uint64_t>& counts) {
  return std::accumulate(counts.begin(), counts.end(), std::uint64_t{0});
}

int nearest_rank(const std::vector<std::uint64_t>& counts,
                 const long double probability,
                 const std::uint64_t total) {
  const auto target = static_cast<std::uint64_t>(
      std::ceil(probability * static_cast<long double>(total)));
  std::uint64_t cumulative = 0;
  for (std::size_t distance = 0; distance < counts.size(); ++distance) {
    cumulative += counts[distance];
    if (cumulative >= target) {
      return static_cast<int>(distance);
    }
  }
  throw std::logic_error("quantile target exceeds the histogram total");
}

std::string json_escape(const std::string& value) {
  std::ostringstream escaped;
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        escaped << "\\\"";
        break;
      case '\\':
        escaped << "\\\\";
        break;
      case '\n':
        escaped << "\\n";
        break;
      case '\r':
        escaped << "\\r";
        break;
      case '\t':
        escaped << "\\t";
        break;
      default:
        if (character < 0x20U) {
          escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                  << static_cast<int>(character) << std::dec;
        } else {
          escaped << static_cast<char>(character);
        }
    }
  }
  return escaped.str();
}

std::ofstream open_output(const std::filesystem::path& path) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("cannot open output file: " + path.string());
  }
  output << std::setprecision(17);
  return output;
}

void ensure_write_succeeded(std::ofstream& output,
                            const std::filesystem::path& path) {
  output.flush();
  if (!output) {
    throw std::runtime_error("failed while writing output file: " +
                             path.string());
  }
}

}  // namespace

Summary summarize(const std::vector<std::uint64_t>& counts) {
  if (counts.empty()) {
    throw std::invalid_argument("cannot summarize an empty histogram");
  }
  const std::uint64_t total = total_count(counts);
  if (total == 0U) {
    throw std::invalid_argument("cannot summarize a zero-count histogram");
  }
  if (counts.size() - 1U >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("histogram distance exceeds integer range");
  }

  long double weighted_sum = 0.0L;
  for (std::size_t distance = 0; distance < counts.size(); ++distance) {
    weighted_sum += static_cast<long double>(distance) * counts[distance];
  }
  const long double mean = weighted_sum / static_cast<long double>(total);

  long double squared_deviation_sum = 0.0L;
  for (std::size_t distance = 0; distance < counts.size(); ++distance) {
    const long double deviation = static_cast<long double>(distance) - mean;
    squared_deviation_sum +=
        deviation * deviation * static_cast<long double>(counts[distance]);
  }

  const auto first_nonzero =
      std::find_if(counts.begin(), counts.end(), [](const std::uint64_t count) {
        return count != 0U;
      });
  const auto last_nonzero =
      std::find_if(counts.rbegin(), counts.rend(), [](const std::uint64_t count) {
        return count != 0U;
      });
  const auto mode = std::max_element(counts.begin(), counts.end());

  Summary summary;
  summary.mean = static_cast<double>(mean);
  summary.standard_deviation = static_cast<double>(
      std::sqrt(squared_deviation_sum / static_cast<long double>(total)));
  summary.min = static_cast<int>(std::distance(counts.begin(), first_nonzero));
  summary.median = nearest_rank(counts, 0.50L, total);
  summary.max = static_cast<int>(counts.size() - 1U -
                                 std::distance(counts.rbegin(), last_nonzero));
  summary.mode = static_cast<int>(std::distance(counts.begin(), mode));
  summary.q05 = nearest_rank(counts, 0.05L, total);
  summary.q95 = nearest_rank(counts, 0.95L, total);
  return summary;
}

void write_results(const RunParameters& parameters,
                   const HistogramResult& histogram,
                   const Summary& summary,
                   const RunTiming& timing) {
  const std::uint64_t total = total_count(histogram.counts);
  if (total != parameters.pairs) {
    throw std::runtime_error("histogram count does not equal --pairs");
  }
  if (histogram.counts.size() != parameters.length + 1U) {
    throw std::runtime_error("histogram bin count does not match --length");
  }

  std::error_code directory_error;
  std::filesystem::create_directories(parameters.output_dir, directory_error);
  if (directory_error) {
    throw std::runtime_error("cannot create output directory: " +
                             directory_error.message());
  }

  const auto histogram_path = parameters.output_dir / "histogram.csv";
  auto histogram_output = open_output(histogram_path);
  histogram_output
      << "edit_distance,count,probability,cumulative_probability\n";
  std::uint64_t cumulative = 0;
  for (std::size_t distance = 0; distance < histogram.counts.size();
       ++distance) {
    cumulative += histogram.counts[distance];
    histogram_output << distance << ',' << histogram.counts[distance] << ','
                     << static_cast<double>(histogram.counts[distance]) /
                            static_cast<double>(total)
                     << ','
                     << static_cast<double>(cumulative) /
                            static_cast<double>(total)
                     << '\n';
  }
  ensure_write_succeeded(histogram_output, histogram_path);

  const double pairs_per_second =
      timing.elapsed_seconds > 0.0
          ? static_cast<double>(parameters.pairs) / timing.elapsed_seconds
          : std::numeric_limits<double>::infinity();
  const auto summary_path = parameters.output_dir / "summary.csv";
  auto summary_output = open_output(summary_path);
  summary_output << "metric,value\n"
                 << "length," << parameters.length << '\n'
                 << "num_pairs," << parameters.pairs << '\n'
                 << "seed," << parameters.seed << '\n'
                 << "mean," << summary.mean << '\n'
                 << "standard_deviation," << summary.standard_deviation
                 << '\n'
                 << "min," << summary.min << '\n'
                 << "median," << summary.median << '\n'
                 << "max," << summary.max << '\n'
                 << "mode," << summary.mode << '\n'
                 << "q05," << summary.q05 << '\n'
                 << "q95," << summary.q95 << '\n'
                 << "elapsed_seconds," << timing.elapsed_seconds << '\n'
                 << "pairs_per_second," << pairs_per_second << '\n';
  ensure_write_succeeded(summary_output, summary_path);

  const auto metadata_path = parameters.output_dir / "run_metadata.json";
  auto metadata_output = open_output(metadata_path);
  metadata_output
      << "{\n"
      << "  \"length\": " << parameters.length << ",\n"
      << "  \"num_pairs\": " << parameters.pairs << ",\n"
      << "  \"seed\": " << parameters.seed << ",\n"
      << "  \"requested_threads\": " << parameters.requested_threads << ",\n"
      << "  \"actual_threads\": " << histogram.actual_threads << ",\n"
      << "  \"output_dir\": \""
      << json_escape(parameters.output_dir.string()) << "\",\n"
      << "  \"distance_metric\": \"exact global Levenshtein\",\n"
      << "  \"alignment_scope\": \"score_only\",\n"
      << "  \"heuristic\": \"none\",\n"
      << "  \"wfa2_lib_version\": \"" << DNA_EDIT_WFA2_VERSION << "\",\n"
      << "  \"wfa2_lib_commit\": \"" << DNA_EDIT_WFA2_COMMIT << "\",\n"
      << "  \"compiler\": \"" << json_escape(DNA_EDIT_COMPILER) << "\",\n"
      << "  \"compiled_at\": \"" << __DATE__ << ' ' << __TIME__ << "\",\n"
      << "  \"started_at_utc\": \""
      << json_escape(timing.started_at_utc) << "\",\n"
      << "  \"finished_at_utc\": \""
      << json_escape(timing.finished_at_utc) << "\",\n"
      << "  \"elapsed_seconds\": " << timing.elapsed_seconds << ",\n"
      << "  \"pairs_per_second\": " << pairs_per_second << "\n"
      << "}\n";
  ensure_write_succeeded(metadata_output, metadata_path);
}

}  // namespace dna_edit_distribution
