#include "reference_windows.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <typeinfo>
#include <vector>

namespace {

class TempFasta {
 public:
  TempFasta(std::string_view name, std::string_view contents)
      : directory_(make_temp_directory()),
        path_(directory_ / (std::string(name) + ".fa")) {
    std::ofstream output(path_);
    if (!output) {
      throw std::runtime_error("unable to create temporary FASTA");
    }
    output << contents;
    output.close();
    if (!output) {
      throw std::runtime_error("unable to write temporary FASTA");
    }
  }

  ~TempFasta() {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
  }

  const std::string path() const { return path_.string(); }

 private:
  static std::filesystem::path make_temp_directory() {
    std::random_device random;
    for (int attempt = 0; attempt < 64; ++attempt) {
      const auto suffix = std::to_string(random()) + "_" +
                          std::to_string(random());
      const auto path = std::filesystem::temp_directory_path() /
                        ("navigamer_reference_windows_" + suffix);
      std::error_code error;
      if (std::filesystem::create_directory(path, error)) {
        return path;
      }
      if (error) {
        throw std::runtime_error("unable to create temporary test directory: " +
                                 error.message());
      }
    }
    throw std::runtime_error("unable to allocate unique test directory");
  }

  std::filesystem::path directory_;
  std::filesystem::path path_;
};

template <typename Expected, typename Function>
void assert_throws(Function&& function, std::string_view message_substring) {
  try {
    function();
  } catch (const Expected& error) {
    if (typeid(error) != typeid(Expected)) {
      throw std::runtime_error("caught a derived exception type: " +
                               std::string(error.what()));
    }
    if (std::string_view(error.what()).find(message_substring) ==
        std::string_view::npos) {
      throw std::runtime_error("exception message did not contain '" +
                               std::string(message_substring) + "': " +
                               error.what());
    }
    return;
  } catch (const std::exception& error) {
    throw std::runtime_error("caught the wrong exception type: " +
                             std::string(error.what()));
  }
  throw std::runtime_error("expected exception was not thrown");
}

std::vector<uint32_t> naive_covering_window_ids(const ReferenceWindows& ref,
                                                uint32_t window_length,
                                                uint32_t occurrence_start,
                                                uint32_t span) {
  const uint64_t occurrence_end =
      static_cast<uint64_t>(occurrence_start) + span;
  std::vector<uint32_t> ids;
  for (uint32_t id = 0; id < ref.size(); ++id) {
    const uint64_t window_start = ref.start(id);
    if (window_start <= occurrence_start &&
        window_start + window_length >= occurrence_end) {
      ids.push_back(id);
    }
  }
  return ids;
}

void test_multiline_fasta_and_stride_one_numbering() {
  TempFasta fasta("multiline", ">ecoli description\nacgt\nACgt\n");
  const ReferenceWindows ref = ReferenceWindows::from_fasta(fasta.path(), 4, 1);

  assert(ref.contig_id() == "ecoli");
  assert(ref.sequence() == "ACGTACGT");
  assert(ref.size() == 5);
  assert(ref.window(2) == "GTAC");
  assert(ref.start(2) == 2);
  assert(ref.window_id_for_start(2) == 2);
  assert(ref.covering_window_ids(3, 2) ==
         std::vector<uint32_t>({1, 2, 3}));
}

void test_non_unit_stride_numbering_and_coverage() {
  TempFasta fasta("stride", ">ref\nACGTACGTAC\n");
  const ReferenceWindows ref = ReferenceWindows::from_fasta(fasta.path(), 4, 2);

  assert(ref.size() == 4);
  assert(ref.start(3) == 6);
  assert(ref.window_id_for_start(4) == 2);
  assert(ref.covering_window_ids(4, 2) ==
         std::vector<uint32_t>({1, 2}));
  assert_throws<std::invalid_argument>(
      [&] { ref.window_id_for_start(3); }, "not aligned");
}

void test_rejects_invalid_fasta() {
  TempFasta multiple("multiple", ">one\nACGT\n>two\nACGT\n");
  TempFasta missing_header("missing_header", "ACGT\n");
  TempFasta empty("empty", ">ref\n");
  TempFasta empty_id("empty_id", ">   \nACGT\n");
  TempFasta whitespace("whitespace", ">ref\nAC GT\n");
  TempFasta invalid_character("invalid_character", ">ref\nAC1T\n");

  assert_throws<std::runtime_error>(
      [&] { ReferenceWindows::from_fasta(multiple.path(), 4, 1); },
      "exactly one contig");
  assert_throws<std::runtime_error>(
      [&] { ReferenceWindows::from_fasta(missing_header.path(), 4, 1); },
      "before its header");
  assert_throws<std::runtime_error>(
      [&] { ReferenceWindows::from_fasta(empty.path(), 4, 1); },
      "empty sequence");
  assert_throws<std::runtime_error>(
      [&] { ReferenceWindows::from_fasta(empty_id.path(), 4, 1); },
      "empty contig ID");
  assert_throws<std::runtime_error>(
      [&] { ReferenceWindows::from_fasta(whitespace.path(), 4, 1); },
      "whitespace");
  assert_throws<std::runtime_error>(
      [&] { ReferenceWindows::from_fasta(invalid_character.path(), 4, 1); },
      "alphabetic symbols");
  assert_throws<std::runtime_error>(
      [&] {
        ReferenceWindows::from_fasta("/path/that/does/not/exist.fa", 4, 1);
      },
      "unable to open");
}

void test_rejects_invalid_parameters_and_coordinates() {
  TempFasta fasta("validation", ">ref\nACGTACGT\n");

  assert_throws<std::invalid_argument>(
      [&] { ReferenceWindows::from_fasta(fasta.path(), 0, 1); },
      "window length");
  assert_throws<std::invalid_argument>(
      [&] { ReferenceWindows::from_fasta(fasta.path(), 4, 0); }, "stride");
  assert_throws<std::invalid_argument>(
      [&] { ReferenceWindows::from_fasta(fasta.path(), 9, 1); }, "shorter");

  const ReferenceWindows ref = ReferenceWindows::from_fasta(fasta.path(), 4, 1);
  assert_throws<std::out_of_range>([&] { ref.window(ref.size()); }, "window ID");
  assert_throws<std::out_of_range>([&] { ref.start(ref.size()); }, "window ID");
  assert_throws<std::out_of_range>(
      [&] { ref.window_id_for_start(5); }, "window start");
  assert_throws<std::invalid_argument>(
      [&] { ref.covering_window_ids(0, 0); }, "greater than zero");
  assert_throws<std::invalid_argument>(
      [&] { ref.covering_window_ids(0, 5); }, "must not exceed");
  assert_throws<std::out_of_range>(
      [&] { ref.covering_window_ids(8, 1); }, "outside the reference");
  assert_throws<std::out_of_range>(
      [&] { ref.covering_window_ids(7, 2); }, "outside the reference");

  const uint32_t maximum = std::numeric_limits<uint32_t>::max();
  assert_throws<std::out_of_range>([&] { ref.window(maximum); }, "window ID");
  assert_throws<std::out_of_range>([&] { ref.start(maximum); }, "window ID");
  assert_throws<std::out_of_range>(
      [&] { ref.window_id_for_start(maximum); }, "window start");
  assert_throws<std::out_of_range>(
      [&] { ref.covering_window_ids(maximum, 1); }, "outside the reference");
  assert_throws<std::invalid_argument>(
      [&] { ref.covering_window_ids(0, maximum); }, "must not exceed");
}

void test_covering_windows_match_naive_containment_oracle() {
  TempFasta fasta("covering_oracle", ">ref\nACGTACGTAC\n");
  constexpr uint32_t window_length = 4;

  struct Case {
    uint32_t occurrence_start;
    uint32_t span;
  };
  std::vector<Case> cases;
  for (uint32_t start = 0; start < 10; ++start) {
    const uint32_t maximum_span = std::min<uint32_t>(window_length, 10 - start);
    for (uint32_t span = 1; span <= maximum_span; ++span) {
      cases.push_back({start, span});
    }
  }

  assert(cases.front().occurrence_start == 0);
  assert(cases.back().occurrence_start == 9);
  for (uint32_t stride : {1U, 2U, 3U}) {
    const ReferenceWindows ref =
        ReferenceWindows::from_fasta(fasta.path(), window_length, stride);
    for (const Case& test_case : cases) {
      assert(ref.covering_window_ids(test_case.occurrence_start,
                                     test_case.span) ==
             naive_covering_window_ids(ref, window_length,
                                       test_case.occurrence_start,
                                       test_case.span));
    }
  }
}

}  // namespace

int main() {
  test_multiline_fasta_and_stride_one_numbering();
  test_non_unit_stride_numbering_and_coverage();
  test_rejects_invalid_fasta();
  test_rejects_invalid_parameters_and_coordinates();
  test_covering_windows_match_naive_containment_oracle();
  std::cout << "reference window tests passed\n";
  return 0;
}
