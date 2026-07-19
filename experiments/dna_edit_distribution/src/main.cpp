#include "dna_edit_distribution.hpp"

#include <charconv>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <omp.h>

namespace {

using dna_edit_distribution::RunParameters;

void print_usage(std::ostream& output, const char* program) {
  output << "Usage: " << program << " [options]\n"
         << "  --length N       sequence length (default: 150)\n"
         << "  --pairs N        number of sequence pairs (default: 1000000)\n"
         << "  --seed N         random seed (default: 20260719)\n"
         << "  --threads N      OpenMP threads (default: OpenMP maximum)\n"
         << "  --output-dir DIR output directory (default: results)\n"
         << "  --help            show this message\n";
}

std::uint64_t parse_uint64(const std::string_view text,
                           const std::string_view option) {
  std::uint64_t value = 0;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || result.ec != std::errc{} ||
      result.ptr != text.data() + text.size()) {
    throw std::invalid_argument("invalid value for " + std::string(option) +
                                ": " + std::string(text));
  }
  return value;
}

RunParameters parse_arguments(const int argc, char** argv) {
  RunParameters parameters;
  parameters.requested_threads = omp_get_max_threads();

  for (int argument = 1; argument < argc; ++argument) {
    const std::string_view option = argv[argument];
    if (option == "--help") {
      print_usage(std::cout, argv[0]);
      std::exit(0);
    }
    if (argument + 1 >= argc) {
      throw std::invalid_argument("missing value for " + std::string(option));
    }
    const std::string_view value = argv[++argument];
    if (option == "--length") {
      const std::uint64_t parsed = parse_uint64(value, option);
      if (parsed == 0U ||
          parsed > static_cast<std::uint64_t>(
                       std::numeric_limits<int>::max())) {
        throw std::invalid_argument("--length must be in [1, INT_MAX]");
      }
      parameters.length = static_cast<std::size_t>(parsed);
    } else if (option == "--pairs") {
      parameters.pairs = parse_uint64(value, option);
      if (parameters.pairs == 0U) {
        throw std::invalid_argument("--pairs must be positive");
      }
    } else if (option == "--seed") {
      parameters.seed = parse_uint64(value, option);
    } else if (option == "--threads") {
      const std::uint64_t parsed = parse_uint64(value, option);
      if (parsed == 0U ||
          parsed >
              static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("--threads must be in [1, INT_MAX]");
      }
      parameters.requested_threads = static_cast<int>(parsed);
    } else if (option == "--output-dir") {
      if (value.empty()) {
        throw std::invalid_argument("--output-dir must not be empty");
      }
      parameters.output_dir = std::string(value);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  return parameters;
}

std::string utc_timestamp_now() {
  const std::time_t now = std::time(nullptr);
  std::tm utc{};
  if (gmtime_r(&now, &utc) == nullptr) {
    throw std::runtime_error("cannot convert the current UTC time");
  }
  std::ostringstream timestamp;
  timestamp << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return timestamp.str();
}

}  // namespace

int main(const int argc, char** argv) {
  try {
    const RunParameters parameters = parse_arguments(argc, argv);
    dna_edit_distribution::RunTiming timing;
    timing.started_at_utc = utc_timestamp_now();
    const auto started = std::chrono::steady_clock::now();
    const auto histogram = dna_edit_distribution::compute_histogram(
        parameters.length, parameters.pairs, parameters.seed,
        parameters.requested_threads);
    const auto finished = std::chrono::steady_clock::now();
    timing.elapsed_seconds =
        std::chrono::duration<double>(finished - started).count();
    timing.finished_at_utc = utc_timestamp_now();

    const std::uint64_t observed_pairs = std::accumulate(
        histogram.counts.begin(), histogram.counts.end(), std::uint64_t{0});
    if (observed_pairs != parameters.pairs) {
      throw std::runtime_error("histogram count does not equal --pairs");
    }
    const auto summary = dna_edit_distribution::summarize(histogram.counts);
    dna_edit_distribution::write_results(parameters, histogram, summary,
                                         timing);

    std::cout << std::setprecision(8)
              << "pairs=" << parameters.pairs << " mean=" << summary.mean
              << " standard_deviation=" << summary.standard_deviation
              << " elapsed_seconds=" << timing.elapsed_seconds
              << " pairs_per_second="
              << static_cast<double>(parameters.pairs) / timing.elapsed_seconds
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
