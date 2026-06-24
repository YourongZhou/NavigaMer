#include "evaluation.hpp"

#include "candidate_indexes.hpp"
#include "sha256.hpp"
#include "tools.hpp"

#include "tensor_index.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("unable to open file: " + path.string());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  if (!input && !input.eof()) {
    throw std::runtime_error("unable to read file: " + path.string());
  }
  return buffer.str();
}

std::string sha256_hex_of_file(const std::filesystem::path& path) {
  return sha256_hex(read_file(path));
}

std::string current_utc_timestamp() {
  std::time_t now = std::time(nullptr);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &now);
#else
  gmtime_r(&now, &tm);
#endif
  char buffer[32] = {};
  if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &tm) == 0) {
    throw std::runtime_error("unable to format timestamp");
  }
  return buffer;
}

std::string shell_quote(const std::string& value) {
  std::string quoted = "'";
  for (char character : value) {
    if (character == '\'') {
      quoted += "'\\''";
    } else {
      quoted += character;
    }
  }
  quoted += "'";
  return quoted;
}

std::string git_commit_for_root(const std::filesystem::path& root) {
  const std::string command = "git -C " + shell_quote(root.string()) +
                              " rev-parse --short=12 HEAD 2>/dev/null";
  std::array<char, 128> buffer{};
  std::string output;
  if (FILE* pipe = popen(command.c_str(), "r")) {
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) !=
           nullptr) {
      output += buffer.data();
    }
    const int status = pclose(pipe);
    if (status != 0) {
      return {};
    }
  }
  while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) {
    output.pop_back();
  }
  return output;
}

std::filesystem::path navigamer_repo_root() {
  return std::filesystem::path(NAVIGAMER_REPO_ROOT);
}

std::filesystem::path tensor_sketch_root() {
  return std::filesystem::path(NAVIGAMER_TENSOR_SKETCH_ROOT);
}

double to_milliseconds(std::chrono::steady_clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

std::string join_parameters(const IndexManifest& manifest) {
  std::ostringstream output;
  for (std::size_t index = 0; index < manifest.parameters.size(); ++index) {
    if (index > 0) {
      output << ';';
    }
    output << manifest.parameters[index].first << '='
           << manifest.parameters[index].second;
  }
  return output.str();
}

template <typename T>
std::string format_optional_field(const std::optional<T>& value) {
  if (!value.has_value()) {
    return "NA";
  }
  std::ostringstream output;
  output << std::setprecision(17) << *value;
  return output.str();
}

template <>
std::string format_optional_field<uint32_t>(
    const std::optional<uint32_t>& value) {
  if (!value.has_value()) {
    return "NA";
  }
  return std::to_string(*value);
}

template <>
std::string format_optional_field<double>(const std::optional<double>& value) {
  if (!value.has_value()) {
    return "NA";
  }
  std::ostringstream output;
  output << std::setprecision(17) << *value;
  return output.str();
}

double mean_of(const std::vector<double>& samples) {
  if (samples.empty()) {
    throw std::invalid_argument("cannot summarize an empty sample set");
  }
  const double total =
      std::accumulate(samples.begin(), samples.end(), 0.0);
  return total / static_cast<double>(samples.size());
}

double median_of_sorted(const std::vector<double>& sorted_samples) {
  if (sorted_samples.empty()) {
    throw std::invalid_argument("cannot summarize an empty sample set");
  }
  const std::size_t mid = sorted_samples.size() / 2U;
  if (sorted_samples.size() % 2U == 1U) {
    return sorted_samples[mid];
  }
  return (sorted_samples[mid - 1] + sorted_samples[mid]) / 2.0;
}

double percentile_nearest_rank(const std::vector<double>& sorted_samples,
                               double percentile) {
  if (sorted_samples.empty()) {
    throw std::invalid_argument("cannot summarize an empty sample set");
  }
  if (!(percentile > 0.0 && percentile <= 1.0)) {
    throw std::invalid_argument("percentile must be in (0, 1]");
  }
  const std::size_t rank = std::max<std::size_t>(
      1, static_cast<std::size_t>(
             std::ceil(percentile * static_cast<double>(sorted_samples.size()))));
  return sorted_samples[std::min<std::size_t>(rank - 1, sorted_samples.size() - 1)];
}

SummaryStats summarize_sorted_samples_impl(std::vector<double> samples) {
  if (samples.empty()) {
    throw std::invalid_argument("cannot summarize an empty sample set");
  }
  std::sort(samples.begin(), samples.end());
  SummaryStats stats;
  stats.mean = mean_of(samples);
  stats.median = median_of_sorted(samples);
  stats.p95 = percentile_nearest_rank(samples, 0.95);
  stats.p99 = percentile_nearest_rank(samples, 0.99);
  return stats;
}

std::vector<uint32_t> deduplicate_preserving_order(
    const std::vector<uint32_t>& raw_candidates) {
  std::vector<uint32_t> unique;
  unique.reserve(raw_candidates.size());
  std::unordered_set<uint32_t> seen;
  seen.reserve(raw_candidates.size());
  for (uint32_t candidate : raw_candidates) {
    if (seen.insert(candidate).second) {
      unique.push_back(candidate);
    }
  }
  return unique;
}

PerReadResult evaluate_candidates_impl(
    const std::string& read_id, const std::string& query_sequence,
    const std::vector<uint32_t>& raw_candidates,
    const ReferenceWindows& reference, uint32_t tolerance,
    const DistanceVerifier& verifier, double retrieval_milliseconds) {
  PerReadResult result;
  result.read_id = read_id;
  result.tolerance = tolerance;
  result.raw_candidate_ids = deduplicate_preserving_order(raw_candidates);
  result.raw_candidate_count =
      static_cast<uint32_t>(result.raw_candidate_ids.size());
  result.retrieval_milliseconds = retrieval_milliseconds;

  const auto verification_started = std::chrono::steady_clock::now();
  result.verified_candidate_ids = result.raw_candidate_ids;
  result.verified_candidate_count =
      static_cast<uint32_t>(result.verified_candidate_ids.size());
  result.accepted_candidate_ids.reserve(result.verified_candidate_ids.size());

  const DistanceVerifier effective_verifier =
      verifier ? verifier
               : [](const std::string& lhs, const std::string& rhs, int tau) {
                   return navigamer::compute_distance_bounded_edlib(lhs, rhs,
                                                                    tau);
                 };

  for (uint32_t candidate_id : result.verified_candidate_ids) {
    if (candidate_id >= reference.size()) {
      throw std::out_of_range("candidate ID exceeds reference windows");
    }
    const std::string window(reference.window(candidate_id));
    const int distance =
        effective_verifier(query_sequence, window, static_cast<int>(tolerance));
    if (distance <= static_cast<int>(tolerance)) {
      result.accepted_candidate_ids.push_back(candidate_id);
    }
  }
  const auto verification_finished = std::chrono::steady_clock::now();
  result.accepted_candidate_count =
      static_cast<uint32_t>(result.accepted_candidate_ids.size());
  result.verification_milliseconds =
      to_milliseconds(verification_finished - verification_started);
  result.total_milliseconds =
      result.retrieval_milliseconds + result.verification_milliseconds;
  return result;
}

IndexManifest make_manifest(const std::string& method,
                            std::vector<std::pair<std::string, std::string>> parameters,
                            const std::filesystem::path& reference_path,
                            const ReferenceWindows& reference,
                            uint32_t window_length,
                            uint32_t stride,
                            const std::string& tool_version) {
  IndexManifest manifest;
  manifest.method = method;
  manifest.parameters = std::move(parameters);
  manifest.reference_path = reference_path.string();
  manifest.reference_sha256 = sha256_hex_of_file(reference_path);
  manifest.reference_length = reference.sequence().size();
  manifest.window_length = window_length;
  manifest.stride = stride;
  manifest.number_of_windows = reference.size();
  manifest.build_command = "candidate_tool build-matrix";
  manifest.created_at = current_utc_timestamp();
  manifest.git_commit = git_commit_for_root(navigamer_repo_root());
  manifest.format_version = 1;
  manifest.tool_version = tool_version;
  return manifest;
}

IndexManifest make_expected_contig_manifest(const BuildMatrixRequest& request,
                                            const ReferenceWindows& reference,
                                            uint32_t k) {
  IndexManifest manifest = make_manifest(
      "contig", {{"method", "contig"}, {"k", std::to_string(k)}},
      request.reference_path, reference, request.window_length, request.stride,
      "contig-index/1");
  manifest.build_command = "candidate_tool build --method contig --k " +
                           std::to_string(k) + " --ref " +
                           request.reference_path.string() + " --window " +
                           std::to_string(request.window_length) + " --stride " +
                           std::to_string(request.stride) + " --out-dir ...";
  return manifest;
}

IndexManifest make_expected_spaced_manifest(const BuildMatrixRequest& request,
                                            const ReferenceWindows& reference,
                                            uint32_t weight,
                                            const std::vector<SpacedMask>& masks) {
  std::vector<std::pair<std::string, std::string>> parameters{
      {"method", "spaced"},
      {"weight", std::to_string(weight)},
      {"mask_count", "4"},
  };
  for (std::size_t mask_id = 0; mask_id < masks.size(); ++mask_id) {
    parameters.emplace_back("mask_" + std::to_string(mask_id) + "_span",
                            std::to_string(masks[mask_id].span));
    std::string bits;
    bits.reserve(masks[mask_id].bits.size());
    for (uint8_t bit : masks[mask_id].bits) {
      bits.push_back(bit ? '1' : '0');
    }
    parameters.emplace_back("mask_" + std::to_string(mask_id) + "_bits", bits);
  }
  IndexManifest manifest =
      make_manifest("spaced", std::move(parameters), request.reference_path,
                    reference, request.window_length, request.stride,
                    "spaced-index/1");
  manifest.build_command = "candidate_tool build --method spaced --weight " +
                           std::to_string(weight) + " --ref " +
                           request.reference_path.string() + " --window " +
                           std::to_string(request.window_length) + " --stride " +
                           std::to_string(request.stride) + " --out-dir ...";
  return manifest;
}

IndexManifest make_expected_randstrobe_manifest(const BuildMatrixRequest& request,
                                                const ReferenceWindows& reference,
                                                uint32_t strobe_length,
                                                uint32_t w_min,
                                                uint32_t w_max,
                                                uint64_t seed) {
  IndexManifest manifest = make_manifest(
      "randstrobe",
      {{"method", "randstrobe"},
       {"order", "2"},
       {"strobe_len", std::to_string(strobe_length)},
       {"w_min", std::to_string(w_min)},
       {"w_max", std::to_string(w_max)},
       {"seed", std::to_string(seed)},
       {"hash_name", "splitmix64"}},
      request.reference_path, reference, request.window_length, request.stride,
      "randstrobe-index/1");
  manifest.build_command =
      "candidate_tool build --method randstrobe --strobe-len " +
      std::to_string(strobe_length) + " --w-min " + std::to_string(w_min) +
      " --w-max " + std::to_string(w_max) + " --seed " +
      std::to_string(seed) + " --ref " + request.reference_path.string() +
      " --window " + std::to_string(request.window_length) + " --stride " +
      std::to_string(request.stride) + " --out-dir ...";
  return manifest;
}

IndexManifest make_expected_qgram_manifest(const BuildMatrixRequest& request,
                                           const ReferenceWindows& reference,
                                           uint32_t q) {
  const uint64_t dense_cell_count =
      q == 0 ? 0 : static_cast<uint64_t>(1) << (2U * q);
  IndexManifest manifest = make_manifest(
      "qgram-safe",
      {{"method", "qgram-safe"},
       {"q", std::to_string(q)},
       {"dense_cells", std::to_string(dense_cell_count)}},
      request.reference_path, reference, request.window_length, request.stride,
      "qgram-safe-index/1");
  manifest.build_command =
      "candidate_tool build --method qgram-safe --q " + std::to_string(q) +
      " --ref " + request.reference_path.string() + " --window " +
      std::to_string(request.window_length) + " --stride " +
      std::to_string(request.stride) + " --out-dir ...";
  return manifest;
}

IndexManifest make_expected_pigeonhole_manifest(
    const BuildMatrixRequest& request, const ReferenceWindows& reference,
    uint32_t tau, uint32_t nominal_read_length) {
  const uint32_t minimum_block_length =
      (request.window_length - tau) / (tau + 1);
  IndexManifest manifest = make_manifest(
      "pigeonhole",
      {{"method", "pigeonhole"},
       {"tau", std::to_string(tau)},
       {"nominal-read-length", std::to_string(nominal_read_length)},
       {"minimum-block-length", std::to_string(minimum_block_length)},
       {"supported-query-length-min",
        std::to_string(request.window_length > tau ? request.window_length - tau
                                                   : 0)},
       {"supported-query-length-max",
        std::to_string(request.window_length + tau)}},
      request.reference_path, reference, request.window_length, request.stride,
      "pigeonhole-index/1");
  manifest.build_command =
      "candidate_tool build --method pigeonhole --tau " + std::to_string(tau) +
      " --nominal-read-length " + std::to_string(nominal_read_length) +
      " --ref " + request.reference_path.string() + " --window " +
      std::to_string(request.window_length) + " --stride " +
      std::to_string(request.stride) + " --out-dir ...";
  return manifest;
}

IndexManifest make_expected_tensor_manifest(const BuildMatrixRequest& request,
                                            const ReferenceWindows& reference,
                                            uint32_t dimension,
                                            uint32_t seed,
                                            uint32_t hnsw_M,
                                            uint32_t hnsw_ef_construction,
                                            uint32_t hnsw_ef_search) {
  IndexManifest manifest = make_manifest(
      "ts::Tensor",
      {{"algorithm", "ts::Tensor"},
       {"subsequence_length", "5"},
       {"dimension", std::to_string(dimension)},
       {"seed", std::to_string(seed)},
       {"metric", "L2"},
       {"hnsw_M", std::to_string(hnsw_M)},
       {"hnsw_ef_construction", std::to_string(hnsw_ef_construction)},
       {"hnsw_ef_search", std::to_string(hnsw_ef_search)},
       {"dependency_source_path", tensor_sketch_root().string()},
       {"dependency_git_commit", git_commit_for_root(tensor_sketch_root())}},
      request.reference_path, reference, request.window_length, request.stride,
      "tensor-index/1");
  manifest.build_command = "candidate_tool tensor-build --ref " +
                           request.reference_path.string() + " --window " +
                           std::to_string(request.window_length) + " --stride " +
                           std::to_string(request.stride) + " --dimension " +
                           std::to_string(dimension) + " --seed " +
                           std::to_string(seed) + " --hnsw-m " +
                           std::to_string(hnsw_M) + " --hnsw-ef-construction " +
                           std::to_string(hnsw_ef_construction) +
                           " --hnsw-ef-search " +
                           std::to_string(hnsw_ef_search) + " --out-dir ...";
  return manifest;
}

BuildSummaryRow make_build_row(const BuildMatrixRequest& request,
                               const std::filesystem::path& index_dir,
                               const std::filesystem::path& index_path,
                               const std::string& variant,
                               bool reused, double wall_seconds,
                               const IndexManifest& manifest) {
  BuildSummaryRow row;
  row.manifest = manifest;
  row.index_dir = index_dir;
  row.index_path = index_path;
  row.variant = variant;
  row.reused = reused;
  row.wall_seconds = wall_seconds;
  (void)request;
  return row;
}

std::string render_summary_header() {
  return "method\tvariant\treused\twall_seconds\tindex_bytes\treference_path\t"
         "reference_sha256\treference_length\twindow_length\tstride\t"
         "number_of_windows\tparameters\tbuild_command\tcreated_at\tgit_commit\t"
         "format_version\ttool_version\tindex_dir\tindex_path\n";
}

std::string render_summary_row_impl(const BuildSummaryRow& row) {
  std::ostringstream output;
  output << row.manifest.method << '\t' << row.variant << '\t'
         << (row.reused ? "true" : "false") << '\t' << std::setprecision(17)
         << row.wall_seconds << '\t' << row.manifest.index_bytes << '\t'
         << row.manifest.reference_path << '\t'
         << row.manifest.reference_sha256 << '\t'
         << row.manifest.reference_length << '\t'
         << row.manifest.window_length << '\t' << row.manifest.stride << '\t'
         << row.manifest.number_of_windows << '\t'
         << join_parameters(row.manifest) << '\t' << row.manifest.build_command
         << '\t' << row.manifest.created_at << '\t'
         << row.manifest.git_commit << '\t' << row.manifest.format_version
         << '\t' << row.manifest.tool_version << '\t' << row.index_dir.string()
         << '\t' << row.index_path.string();
  return output.str();
}

template <typename Recipe>
BuildSummaryRow materialize_recipe(const BuildMatrixRequest& request,
                                   const ReferenceWindows& reference,
                                   const Recipe& recipe) {
  const std::filesystem::path index_dir = request.out_dir / recipe.variant_dir;
  const std::filesystem::path index_path = index_dir / recipe.index_filename;
  const IndexManifest expected = recipe.make_expected_manifest(request, reference);
  bool reused = false;

  const auto started = std::chrono::steady_clock::now();
  if (std::filesystem::exists(index_path)) {
    const IndexManifest stored = recipe.load_manifest(index_dir);
    if (semantically_compatible(stored, expected)) {
      reused = true;
    } else if (!request.rebuild) {
      throw std::runtime_error("incompatible existing index; remove or rebuild " +
                               index_path.string());
    } else {
      std::filesystem::remove_all(index_dir);
    }
  }

  if (!reused) {
    std::filesystem::create_directories(index_dir);
    recipe.build(request, reference, index_dir);
  }

  const auto finished = std::chrono::steady_clock::now();
  const IndexManifest manifest = recipe.load_manifest(index_dir);
  BuildSummaryRow row = make_build_row(request, index_dir, index_path,
                                       recipe.variant, reused,
                                       to_milliseconds(finished - started),
                                       manifest);
  write_build_summary_tsv(index_dir / "build_summary.tsv", {row});
  return row;
}

struct ContigRecipe {
  std::string variant;
  std::filesystem::path variant_dir;
  std::string index_filename;
  std::function<IndexManifest(const BuildMatrixRequest&, const ReferenceWindows&)>
      make_expected_manifest;
  std::function<void(const BuildMatrixRequest&, const ReferenceWindows&,
                     const std::filesystem::path&)>
      build;
  std::function<IndexManifest(const std::filesystem::path&)> load_manifest;
};

struct SpacedRecipe {
  std::string variant;
  std::filesystem::path variant_dir;
  std::string index_filename;
  std::function<IndexManifest(const BuildMatrixRequest&, const ReferenceWindows&)>
      make_expected_manifest;
  std::function<void(const BuildMatrixRequest&, const ReferenceWindows&,
                     const std::filesystem::path&)>
      build;
  std::function<IndexManifest(const std::filesystem::path&)> load_manifest;
};

struct TensorRecipe {
  std::string variant;
  std::filesystem::path variant_dir;
  std::string index_filename;
  std::function<IndexManifest(const BuildMatrixRequest&, const ReferenceWindows&)>
      make_expected_manifest;
  std::function<void(const BuildMatrixRequest&, const ReferenceWindows&,
                     const std::filesystem::path&)>
      build;
  std::function<IndexManifest(const std::filesystem::path&)> load_manifest;
};

std::filesystem::path build_matrix_variant_root(const std::filesystem::path& root,
                                                std::string_view method,
                                                std::string_view variant) {
  return root / "indexes" / std::string(method) / std::string(variant);
}

}  // namespace

PerReadResult evaluate_candidates(const std::string& read_id,
                                  const std::string& query_sequence,
                                  const std::vector<uint32_t>& raw_candidates,
                                  const ReferenceWindows& reference,
                                  uint32_t tolerance,
                                  DistanceVerifier verifier) {
  return evaluate_candidates_impl(read_id, query_sequence, raw_candidates,
                                  reference, tolerance, verifier, 0.0);
}

PerReadResult evaluate_candidates(const std::string& read_id,
                                  const std::string& query_sequence,
                                  const CandidateRetriever& retriever,
                                  const ReferenceWindows& reference,
                                  uint32_t tolerance,
                                  DistanceVerifier verifier) {
  const auto started = std::chrono::steady_clock::now();
  const std::vector<uint32_t> raw_candidates = retriever();
  const auto finished = std::chrono::steady_clock::now();
  return evaluate_candidates_impl(read_id, query_sequence, raw_candidates,
                                  reference, tolerance, verifier,
                                  to_milliseconds(finished - started));
}

SummaryStats summarize_samples(std::vector<double> samples) {
  return summarize_sorted_samples_impl(std::move(samples));
}

std::string render_oracle_metrics_tsv(const OracleMetrics& oracle) {
  std::ostringstream output;
  output << format_optional_field(oracle.true_neighbor_count) << '\t'
         << format_optional_field(oracle.false_negative_count) << '\t'
         << format_optional_field(oracle.recall) << '\t'
         << format_optional_field(oracle.raw_candidate_blowup) << '\t'
         << format_optional_field(oracle.accepted_candidate_blowup);
  return output.str();
}

std::string render_build_summary_row_tsv(const BuildSummaryRow& row) {
  return render_summary_row_impl(row);
}

void write_build_summary_tsv(const std::filesystem::path& path,
                             const std::vector<BuildSummaryRow>& rows) {
  if (path.empty()) {
    throw std::invalid_argument("summary path must not be empty");
  }
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::trunc);
  if (!output) {
    throw std::runtime_error("unable to create build summary: " + path.string());
  }
  output << render_summary_header();
  for (const BuildSummaryRow& row : rows) {
    output << render_summary_row_impl(row) << '\n';
  }
  output.close();
  if (!output) {
    throw std::runtime_error("unable to write build summary: " + path.string());
  }
}

std::vector<BuildSummaryRow> build_candidate_matrix(
    const BuildMatrixRequest& request) {
  if (request.reference_path.empty()) {
    throw std::invalid_argument("reference path must not be empty");
  }
  if (request.window_length == 0) {
    throw std::invalid_argument("window length must be greater than zero");
  }
  if (request.stride == 0) {
    throw std::invalid_argument("stride must be greater than zero");
  }
  if (request.out_dir.empty()) {
    throw std::invalid_argument("output directory must not be empty");
  }

  const ReferenceWindows reference = ReferenceWindows::from_fasta(
      request.reference_path.string(), request.window_length, request.stride);
  std::vector<BuildSummaryRow> rows;
  rows.reserve(16);

  const auto contig_recipe = [&](uint32_t k) {
    struct Recipe {
      std::string variant;
      std::filesystem::path variant_dir;
      std::string index_filename;
      std::function<IndexManifest(const BuildMatrixRequest&, const ReferenceWindows&)>
          make_expected_manifest;
      std::function<void(const BuildMatrixRequest&, const ReferenceWindows&,
                         const std::filesystem::path&)>
          build;
      std::function<IndexManifest(const std::filesystem::path&)> load_manifest;
    };
    return Recipe{
        "k" + std::to_string(k),
        build_matrix_variant_root(request.out_dir, "contig",
                                  "k" + std::to_string(k)),
        "index.bin",
        [k](const BuildMatrixRequest& req, const ReferenceWindows& ref) {
          return make_expected_contig_manifest(req, ref, k);
        },
        [k](const BuildMatrixRequest& req, const ReferenceWindows&,
            const std::filesystem::path& out_dir) {
          ContiguousIndexConfig config;
          config.reference_path = req.reference_path;
          config.window_length = req.window_length;
          config.stride = req.stride;
          config.k = k;
          ContiguousIndex::build(config).save(out_dir);
        },
        [](const std::filesystem::path& dir) {
          return read_index_file(dir / "index.bin").manifest;
        }};
  };

  const auto spaced_recipe = [&](uint32_t weight) {
    struct Recipe {
      std::string variant;
      std::filesystem::path variant_dir;
      std::string index_filename;
      std::function<IndexManifest(const BuildMatrixRequest&, const ReferenceWindows&)>
          make_expected_manifest;
      std::function<void(const BuildMatrixRequest&, const ReferenceWindows&,
                         const std::filesystem::path&)>
          build;
      std::function<IndexManifest(const std::filesystem::path&)> load_manifest;
    };
    return Recipe{
        "w" + std::to_string(weight),
        build_matrix_variant_root(request.out_dir, "spaced",
                                  "w" + std::to_string(weight)),
        "index.bin",
        [weight](const BuildMatrixRequest& req, const ReferenceWindows& ref) {
          return make_expected_spaced_manifest(req, ref, weight,
                                               make_spaced_masks(weight));
        },
        [weight](const BuildMatrixRequest& req, const ReferenceWindows&,
            const std::filesystem::path& out_dir) {
          SpacedSeedIndexConfig config;
          config.reference_path = req.reference_path;
          config.window_length = req.window_length;
          config.stride = req.stride;
          config.weight = weight;
          SpacedSeedIndex::build(config).save(out_dir);
        },
        [](const std::filesystem::path& dir) {
          return read_index_file(dir / "index.bin").manifest;
        }};
  };

  const auto randstrobe_recipe = [&]() {
    struct Recipe {
      std::string variant;
      std::filesystem::path variant_dir;
      std::string index_filename;
      std::function<IndexManifest(const BuildMatrixRequest&, const ReferenceWindows&)>
          make_expected_manifest;
      std::function<void(const BuildMatrixRequest&, const ReferenceWindows&,
                         const std::filesystem::path&)>
          build;
      std::function<IndexManifest(const std::filesystem::path&)> load_manifest;
    };
    return Recipe{
        "order2",
        build_matrix_variant_root(request.out_dir, "randstrobe", "order2"),
        "index.bin",
        [](const BuildMatrixRequest& req, const ReferenceWindows& ref) {
          return make_expected_randstrobe_manifest(req, ref, 15, 20, 50, 0);
        },
        [](const BuildMatrixRequest& req, const ReferenceWindows&,
            const std::filesystem::path& out_dir) {
          RandstrobeIndexConfig config;
          config.reference_path = req.reference_path;
          config.window_length = req.window_length;
          config.stride = req.stride;
          config.strobe_length = 15;
          config.w_min = 20;
          config.w_max = 50;
          config.seed = 0;
          RandstrobeIndex::build(config).save(out_dir);
        },
        [](const std::filesystem::path& dir) {
          return read_index_file(dir / "index.bin").manifest;
        }};
  };

  const auto qgram_recipe = [&](uint32_t q) {
    struct Recipe {
      std::string variant;
      std::filesystem::path variant_dir;
      std::string index_filename;
      std::function<IndexManifest(const BuildMatrixRequest&, const ReferenceWindows&)>
          make_expected_manifest;
      std::function<void(const BuildMatrixRequest&, const ReferenceWindows&,
                         const std::filesystem::path&)>
          build;
      std::function<IndexManifest(const std::filesystem::path&)> load_manifest;
    };
    return Recipe{
        "q" + std::to_string(q),
        build_matrix_variant_root(request.out_dir, "qgram-safe",
                                  "q" + std::to_string(q)),
        "index.bin",
        [q](const BuildMatrixRequest& req, const ReferenceWindows& ref) {
          return make_expected_qgram_manifest(req, ref, q);
        },
        [q](const BuildMatrixRequest& req, const ReferenceWindows&,
            const std::filesystem::path& out_dir) {
          QgramSafeIndexConfig config;
          config.reference_path = req.reference_path;
          config.window_length = req.window_length;
          config.stride = req.stride;
          config.q = q;
          QgramSafeIndex::build(config).save(out_dir);
        },
        [](const std::filesystem::path& dir) {
          return read_index_file(dir / "index.bin").manifest;
        }};
  };

  const auto pigeonhole_recipe = [&](uint32_t tau) {
    struct Recipe {
      std::string variant;
      std::filesystem::path variant_dir;
      std::string index_filename;
      std::function<IndexManifest(const BuildMatrixRequest&, const ReferenceWindows&)>
          make_expected_manifest;
      std::function<void(const BuildMatrixRequest&, const ReferenceWindows&,
                         const std::filesystem::path&)>
          build;
      std::function<IndexManifest(const std::filesystem::path&)> load_manifest;
    };
    return Recipe{
        "tau" + std::to_string(tau),
        build_matrix_variant_root(request.out_dir, "pigeonhole",
                                  "tau" + std::to_string(tau)),
        "index.bin",
        [tau](const BuildMatrixRequest& req, const ReferenceWindows& ref) {
          return make_expected_pigeonhole_manifest(req, ref, tau,
                                                   req.window_length);
        },
        [tau](const BuildMatrixRequest& req, const ReferenceWindows&,
            const std::filesystem::path& out_dir) {
          PigeonholeIndexConfig config;
          config.reference_path = req.reference_path;
          config.window_length = req.window_length;
          config.stride = req.stride;
          config.tau = tau;
          config.nominal_read_length = req.window_length;
          PigeonholeIndex::build(config).save(out_dir);
        },
        [](const std::filesystem::path& dir) {
          return read_index_file(dir / "index.bin").manifest;
        }};
  };

  const auto tensor_recipe = [&](uint32_t dimension) {
    struct Recipe {
      std::string variant;
      std::filesystem::path variant_dir;
      std::string index_filename;
      std::function<IndexManifest(const BuildMatrixRequest&, const ReferenceWindows&)>
          make_expected_manifest;
      std::function<void(const BuildMatrixRequest&, const ReferenceWindows&,
                         const std::filesystem::path&)>
          build;
      std::function<IndexManifest(const std::filesystem::path&)> load_manifest;
    };
    return Recipe{
        "d" + std::to_string(dimension),
        build_matrix_variant_root(request.out_dir, "tensor",
                                  "d" + std::to_string(dimension)),
        "manifest.meta",
        [dimension](const BuildMatrixRequest& req, const ReferenceWindows& ref) {
          return make_expected_tensor_manifest(req, ref, dimension, 0, 16, 200,
                                               50);
        },
        [dimension](const BuildMatrixRequest& req, const ReferenceWindows&,
            const std::filesystem::path& out_dir) {
          tensor_index::TensorIndexConfig config;
          config.reference_path = req.reference_path;
          config.window_length = req.window_length;
          config.stride = req.stride;
          config.dimension = dimension;
          config.seed = 0;
          config.hnsw_M = 16;
          config.hnsw_ef_construction = 200;
          config.hnsw_ef_search = 50;
          config.exact_vectors = true;
          tensor_index::save_tensor_index(tensor_index::build_tensor_index(config),
                                          out_dir);
        },
        [](const std::filesystem::path& dir) {
          return tensor_index::load_tensor_index(dir).snapshot.manifest;
        }};
  };

  const auto build_and_collect = [&](const auto& recipe) {
    rows.push_back(materialize_recipe(request, reference, recipe));
  };

  for (uint32_t k : {15U, 19U, 23U}) {
    build_and_collect(contig_recipe(k));
  }
  for (uint32_t weight : {15U, 18U, 21U}) {
    build_and_collect(spaced_recipe(weight));
  }
  build_and_collect(randstrobe_recipe());
  for (uint32_t q : {5U, 6U}) {
    build_and_collect(qgram_recipe(q));
  }
  for (uint32_t tau : {2U, 3U, 5U}) {
    build_and_collect(pigeonhole_recipe(tau));
  }
  for (uint32_t dimension : {64U, 128U, 256U, 512U}) {
    build_and_collect(tensor_recipe(dimension));
  }

  write_build_summary_tsv(request.out_dir / "build_summary.tsv", rows);
  return rows;
}
