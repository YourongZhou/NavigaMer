#include "reference_windows.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

class TempFasta {
 public:
  TempFasta(std::string_view name, std::string_view contents)
      : path_(std::filesystem::temp_directory_path() /
              ("navigamer_" + std::string(name) + ".fa")) {
    std::ofstream output(path_);
    if (!output) {
      throw std::runtime_error("unable to create temporary FASTA");
    }
    output << contents;
  }

  ~TempFasta() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  const std::string path() const { return path_.string(); }

 private:
  std::filesystem::path path_;
};

template <typename Function>
void assert_throws(Function&& function) {
  bool threw = false;
  try {
    function();
  } catch (const std::exception&) {
    threw = true;
  }
  assert(threw);
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
  assert_throws([&] { ref.window_id_for_start(3); });
}

void test_rejects_invalid_fasta() {
  TempFasta multiple("multiple", ">one\nACGT\n>two\nACGT\n");
  TempFasta missing_header("missing_header", "ACGT\n");
  TempFasta empty("empty", ">ref\n");
  TempFasta empty_id("empty_id", ">   \nACGT\n");
  TempFasta whitespace("whitespace", ">ref\nAC GT\n");
  TempFasta invalid_character("invalid_character", ">ref\nAC1T\n");

  assert_throws([&] { ReferenceWindows::from_fasta(multiple.path(), 4, 1); });
  assert_throws(
      [&] { ReferenceWindows::from_fasta(missing_header.path(), 4, 1); });
  assert_throws([&] { ReferenceWindows::from_fasta(empty.path(), 4, 1); });
  assert_throws([&] { ReferenceWindows::from_fasta(empty_id.path(), 4, 1); });
  assert_throws([&] { ReferenceWindows::from_fasta(whitespace.path(), 4, 1); });
  assert_throws(
      [&] { ReferenceWindows::from_fasta(invalid_character.path(), 4, 1); });
  assert_throws([&] {
    ReferenceWindows::from_fasta("/path/that/does/not/exist.fa", 4, 1);
  });
}

void test_rejects_invalid_parameters_and_coordinates() {
  TempFasta fasta("validation", ">ref\nACGTACGT\n");

  assert_throws([&] { ReferenceWindows::from_fasta(fasta.path(), 0, 1); });
  assert_throws([&] { ReferenceWindows::from_fasta(fasta.path(), 4, 0); });
  assert_throws([&] { ReferenceWindows::from_fasta(fasta.path(), 9, 1); });

  const ReferenceWindows ref = ReferenceWindows::from_fasta(fasta.path(), 4, 1);
  assert_throws([&] { ref.window(ref.size()); });
  assert_throws([&] { ref.start(ref.size()); });
  assert_throws([&] { ref.window_id_for_start(5); });
  assert_throws([&] { ref.covering_window_ids(0, 0); });
  assert_throws([&] { ref.covering_window_ids(0, 5); });
  assert_throws([&] { ref.covering_window_ids(8, 1); });
  assert_throws([&] { ref.covering_window_ids(7, 2); });
}

}  // namespace

int main() {
  test_multiline_fasta_and_stride_one_numbering();
  test_non_unit_stride_numbering_and_coverage();
  test_rejects_invalid_fasta();
  test_rejects_invalid_parameters_and_coordinates();
  std::cout << "reference window tests passed\n";
  return 0;
}
