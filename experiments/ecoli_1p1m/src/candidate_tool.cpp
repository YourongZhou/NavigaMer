#include "reference_windows.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void print_help() {
  std::cout
      << "Usage:\n"
      << "  candidate_tool --help\n"
      << "  candidate_tool inspect-reference --ref PATH --window N --stride N\n";
}

uint32_t parse_uint32(const std::string& value, const std::string& flag) {
  std::size_t parsed = 0;
  unsigned long long number = 0;
  try {
    number = std::stoull(value, &parsed);
  } catch (const std::exception&) {
    throw std::invalid_argument("invalid value for " + flag + ": " + value);
  }
  if (parsed != value.size() || number > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("invalid value for " + flag + ": " + value);
  }
  return static_cast<uint32_t>(number);
}

int inspect_reference(int argc, char** argv) {
  std::string reference_path;
  uint32_t window_length = 0;
  uint32_t stride = 0;
  bool saw_window = false;
  bool saw_stride = false;

  for (int index = 2; index < argc; index += 2) {
    const std::string flag = argv[index];
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for flag: " + flag);
    }
    const std::string value = argv[index + 1];
    if (flag == "--ref") {
      if (!reference_path.empty()) {
        throw std::invalid_argument("duplicate flag: --ref");
      }
      reference_path = value;
    } else if (flag == "--window") {
      if (saw_window) {
        throw std::invalid_argument("duplicate flag: --window");
      }
      window_length = parse_uint32(value, flag);
      saw_window = true;
    } else if (flag == "--stride") {
      if (saw_stride) {
        throw std::invalid_argument("duplicate flag: --stride");
      }
      stride = parse_uint32(value, flag);
      saw_stride = true;
    } else {
      throw std::invalid_argument("unknown flag: " + flag);
    }
  }

  if (reference_path.empty()) {
    throw std::invalid_argument("missing required flag: --ref");
  }
  if (!saw_window) {
    throw std::invalid_argument("missing required flag: --window");
  }
  if (!saw_stride) {
    throw std::invalid_argument("missing required flag: --stride");
  }

  const ReferenceWindows reference = ReferenceWindows::from_fasta(
      reference_path, window_length, stride);
  std::cout << "contig_id\treference_length\twindow_length\tstride\t"
               "number_of_windows\n"
            << reference.contig_id() << '\t' << reference.sequence().size()
            << '\t' << window_length << '\t' << stride << '\t'
            << reference.size() << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      print_help();
      return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "inspect-reference") {
      return inspect_reference(argc, argv);
    }
    if (argc < 2) {
      throw std::invalid_argument("missing command");
    }
    throw std::invalid_argument("unknown command: " + std::string(argv[1]));
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
