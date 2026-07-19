#include "dna_edit_distribution.hpp"

#include <algorithm>
#include <atomic>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

namespace dna_edit_distribution {
namespace {

constexpr std::uint64_t kGoldenRatio = 0x9e3779b97f4a7c15ULL;
constexpr char kBases[] = {'A', 'C', 'G', 'T'};

std::uint64_t splitmix64(std::uint64_t value) {
  value += kGoldenRatio;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

}  // namespace

void generate_pair(const std::size_t length,
                   const std::uint64_t seed,
                   const std::uint64_t pair_index,
                   std::string& first,
                   std::string& second) {
  if (length > std::numeric_limits<std::size_t>::max() / 2U) {
    throw std::invalid_argument("sequence length is too large");
  }

  first.resize(length);
  second.resize(length);
  const std::size_t total_bases = length * 2U;
  const std::uint64_t words_per_pair =
      static_cast<std::uint64_t>((total_bases + 31U) / 32U);
  const std::uint64_t first_counter = pair_index * words_per_pair;

  std::uint64_t random_word = 0;
  for (std::size_t base_index = 0; base_index < total_bases; ++base_index) {
    if ((base_index % 32U) == 0U) {
      const std::uint64_t word_index =
          static_cast<std::uint64_t>(base_index / 32U);
      random_word = splitmix64(seed +
                              (first_counter + word_index) * kGoldenRatio);
    }
    const char base = kBases[random_word & 3U];
    random_word >>= 2U;
    if (base_index < length) {
      first[base_index] = base;
    } else {
      second[base_index - length] = base;
    }
  }
}

int levenshtein_dp(const std::string_view first,
                   const std::string_view second) {
  std::vector<int> previous(second.size() + 1U);
  std::vector<int> current(second.size() + 1U);
  std::iota(previous.begin(), previous.end(), 0);

  for (std::size_t row = 1; row <= first.size(); ++row) {
    current[0] = static_cast<int>(row);
    for (std::size_t column = 1; column <= second.size(); ++column) {
      const int substitution =
          previous[column - 1U] +
          (first[row - 1U] == second[column - 1U] ? 0 : 1);
      const int deletion = previous[column] + 1;
      const int insertion = current[column - 1U] + 1;
      current[column] = std::min({substitution, deletion, insertion});
    }
    previous.swap(current);
  }
  return previous.back();
}

ExactWfaAligner::ExactWfaAligner() {
  wavefront_aligner_attr_t attributes = wavefront_aligner_attr_default;
  attributes.distance_metric = edit;
  attributes.alignment_scope = compute_score;
  attributes.alignment_form.span = alignment_end2end;
  attributes.heuristic.strategy = wf_heuristic_none;
  aligner_ = wavefront_aligner_new(&attributes);
  if (aligner_ == nullptr) {
    throw std::runtime_error("WFA2 failed to create an aligner");
  }
}

ExactWfaAligner::~ExactWfaAligner() {
  if (aligner_ != nullptr) {
    wavefront_aligner_delete(aligner_);
  }
}

int ExactWfaAligner::distance(const std::string_view first,
                              const std::string_view second) {
  if (first.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      second.size() >
          static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("sequence length exceeds the WFA2 API limit");
  }
  const int status = wavefront_align(
      aligner_, first.data(), static_cast<int>(first.size()), second.data(),
      static_cast<int>(second.size()));
  if (status != WF_STATUS_ALG_COMPLETED) {
    throw std::runtime_error("WFA2 alignment failed with status " +
                             std::to_string(status));
  }
  return aligner_->align_status.score;
}

HistogramResult compute_histogram(const std::size_t length,
                                  const std::uint64_t pairs,
                                  const std::uint64_t seed,
                                  const int threads) {
  if (length == 0U || pairs == 0U || threads <= 0) {
    throw std::invalid_argument("length, pairs, and threads must be positive");
  }
  if (length > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("sequence length exceeds the WFA2 API limit");
  }

  std::vector<std::vector<std::uint64_t>> local_counts(
      static_cast<std::size_t>(threads),
      std::vector<std::uint64_t>(length + 1U, 0));
  std::atomic<bool> failed{false};
  std::string failure_message;
  int actual_threads = 0;

#pragma omp parallel num_threads(threads) shared(actual_threads, failure_message)
  {
    const int thread_id = omp_get_thread_num();
#pragma omp single
    { actual_threads = omp_get_num_threads(); }

    try {
      ExactWfaAligner aligner;
      std::string first;
      std::string second;
      first.reserve(length);
      second.reserve(length);
      auto& counts = local_counts[static_cast<std::size_t>(thread_id)];

#pragma omp for schedule(static)
      for (std::uint64_t pair_index = 0; pair_index < pairs; ++pair_index) {
        if (failed.load(std::memory_order_relaxed)) {
          continue;
        }
        generate_pair(length, seed, pair_index, first, second);
        const int edit_distance = aligner.distance(first, second);
        if (edit_distance < 0 ||
            edit_distance > static_cast<int>(length)) {
          throw std::runtime_error("WFA2 returned an out-of-range distance");
        }
        ++counts[static_cast<std::size_t>(edit_distance)];
      }
    } catch (const std::exception& error) {
      failed.store(true, std::memory_order_relaxed);
#pragma omp critical(dna_edit_distribution_failure)
      {
        if (failure_message.empty()) {
          failure_message = error.what();
        }
      }
    }
  }

  if (failed.load(std::memory_order_relaxed)) {
    throw std::runtime_error(failure_message.empty()
                                 ? "parallel histogram computation failed"
                                 : failure_message);
  }

  HistogramResult result;
  result.counts.assign(length + 1U, 0);
  result.actual_threads = actual_threads;
  for (int thread_id = 0; thread_id < actual_threads; ++thread_id) {
    const auto& counts = local_counts[static_cast<std::size_t>(thread_id)];
    for (std::size_t distance = 0; distance < counts.size(); ++distance) {
      result.counts[distance] += counts[distance];
    }
  }
  return result;
}

}  // namespace dna_edit_distribution
