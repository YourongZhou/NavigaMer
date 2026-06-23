#include "tensor_index.hpp"

#include "reference_windows.hpp"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "sketch/tensor.hpp"
#include "sketch/tensor_slide.hpp"

namespace {

class TempDirectory {
 public:
  TempDirectory() {
    std::random_device random;
    for (int attempt = 0; attempt < 64; ++attempt) {
      path_ = std::filesystem::temp_directory_path() /
              ("tensor_index_" + std::to_string(random()) + "_" +
               std::to_string(random()));
      std::error_code error;
      if (std::filesystem::create_directory(path_, error)) {
        return;
      }
      if (error) {
        throw std::runtime_error("unable to create temporary directory: " +
                                 error.message());
      }
    }
    throw std::runtime_error("unable to allocate temporary directory");
  }

  ~TempDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  std::filesystem::path file(std::string_view name) const { return path_ / name; }
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

void write_text_file(const std::filesystem::path& path, std::string_view text) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("unable to create file: " + path.string());
  }
  output << text;
  output.close();
  if (!output) {
    throw std::runtime_error("unable to write file: " + path.string());
  }
}

std::vector<int> encode_sequence(std::string_view sequence) {
  std::vector<int> encoded;
  encoded.reserve(sequence.size());
  for (char base : sequence) {
    switch (base) {
      case 'A':
      case 'a':
        encoded.push_back(0);
        break;
      case 'C':
      case 'c':
        encoded.push_back(1);
        break;
      case 'G':
      case 'g':
        encoded.push_back(2);
        break;
      case 'T':
      case 't':
        encoded.push_back(3);
        break;
      default:
        throw std::runtime_error("unexpected DNA symbol");
    }
  }
  return encoded;
}

void assert_close(double lhs, double rhs, double tolerance,
                  const char* message) {
  if (std::fabs(lhs - rhs) > tolerance) {
    throw std::runtime_error(message);
  }
}

std::vector<tensor_index::QueryHit> exact_ranked(
    const tensor_index::TensorIndexSnapshot& snapshot,
    const std::vector<float>& query) {
  assert(query.size() == snapshot.dimension);
  std::vector<tensor_index::QueryHit> hits;
  hits.reserve(snapshot.labels.size());
  for (std::size_t row = 0; row < snapshot.labels.size(); ++row) {
    float distance = 0.0f;
    const std::size_t offset = row * snapshot.dimension;
    for (uint32_t column = 0; column < snapshot.dimension; ++column) {
      const float delta = snapshot.exact_vectors[offset + column] - query[column];
      distance += delta * delta;
    }
    hits.push_back({snapshot.labels[row], distance});
  }
  std::sort(hits.begin(), hits.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.distance != rhs.distance) {
      return lhs.distance < rhs.distance;
    }
    return lhs.label < rhs.label;
  });
  return hits;
}

std::vector<float> sketch_query(const std::string& sequence,
                                uint32_t dimension,
                                uint32_t seed) {
  ts::Tensor<int> tensor(4, dimension, 5, seed);
  const std::vector<int> encoded = encode_sequence(sequence);
  const std::vector<double> sketch = tensor.compute(encoded);
  std::vector<float> result;
  result.reserve(sketch.size());
  for (double value : sketch) {
    result.push_back(static_cast<float>(value));
  }
  return result;
}

void compare_hits(const std::vector<tensor_index::QueryHit>& lhs,
                  const std::vector<tensor_index::QueryHit>& rhs) {
  assert(lhs.size() == rhs.size());
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    assert(lhs[i].label == rhs[i].label);
    assert_close(lhs[i].distance, rhs[i].distance, 1e-6, "distance mismatch");
  }
}

std::vector<std::string> parameter_values(
    const IndexManifest& manifest,
    std::string_view key) {
  std::vector<std::string> values;
  for (const auto& parameter : manifest.parameters) {
    if (parameter.first == key) {
      values.push_back(parameter.second);
    }
  }
  return values;
}

std::string shell_quote(const std::filesystem::path& path) {
  std::string quoted = "'";
  for (char byte : path.string()) {
    if (byte == '\'') {
      quoted += "'\\''";
    } else {
      quoted += byte;
    }
  }
  quoted += "'";
  return quoted;
}

int run_command(const std::string& command) {
  return std::system(command.c_str());
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("unable to open file: " + path.string());
  }
  std::ostringstream content;
  content << input.rdbuf();
  return content.str();
}

void test_slide_matches_independent_windows_when_stride_is_one() {
  const std::string sequence = "ACGTACGTAC";
  const uint32_t dimension = 16;
  const uint32_t seed = 17;
  const uint32_t window_length = 6;

  const std::vector<int> encoded = encode_sequence(sequence);
  ts::TensorSlide<int> slide(4, dimension, 5, window_length, 1, seed);
  const std::vector<std::vector<double>> sliding = slide.compute(encoded);
  ts::Tensor<int> tensor(4, dimension, 5, seed);

  assert(sliding.size() == sequence.size() - window_length + 1);
  for (std::size_t window_id = 0; window_id < sliding.size(); ++window_id) {
    const std::vector<int> window =
        encode_sequence(sequence.substr(window_id, window_length));
    const std::vector<double> expected = tensor.compute(window);
    assert(sliding[window_id].size() == expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
      assert_close(sliding[window_id][index], expected[index], 1e-9,
                   "tensor slide mismatch");
    }
  }
}

void test_tensor_index_round_trip_for_dimension(uint32_t dimension) {
  TempDirectory temp;
  const auto reference = temp.file("reference.fa");
  write_text_file(reference, ">tiny\nACGTACGTACGT\n");

  tensor_index::TensorIndexConfig config;
  config.reference_path = reference;
  config.window_length = 6;
  config.stride = 1;
  config.dimension = dimension;
  config.seed = 17;
  config.hnsw_M = 16;
  config.hnsw_ef_construction = 200;
  config.hnsw_ef_search = 50;
  config.exact_vectors = true;

  tensor_index::TensorIndex built = tensor_index::build_tensor_index(config);
  assert(built.snapshot.dimension == dimension);
  assert(built.snapshot.labels.size() == 7);
  assert(built.snapshot.exact_vectors.size() ==
         built.snapshot.labels.size() * static_cast<std::size_t>(dimension));

  tensor_index::save_tensor_index(built, temp.path());
  tensor_index::TensorIndex reloaded = tensor_index::load_tensor_index(temp.path());

  assert(reloaded.snapshot.dimension == dimension);
  assert(reloaded.snapshot.seed == config.seed);
  assert(reloaded.snapshot.labels == built.snapshot.labels);
  assert(reloaded.snapshot.exact_vectors.size() == built.snapshot.exact_vectors.size());
  for (std::size_t i = 0; i < built.snapshot.exact_vectors.size(); ++i) {
    assert_close(reloaded.snapshot.exact_vectors[i],
                 built.snapshot.exact_vectors[i], 1e-9,
                 "exact vector mismatch");
  }

  const std::vector<float> query_vector = sketch_query("ACGTAC", dimension, config.seed);
  const std::vector<std::string> dependency_root_values =
      parameter_values(reloaded.snapshot.manifest, "dependency_source_path");
  assert(dependency_root_values.size() == 1);
  assert(dependency_root_values[0] == NAVIGAMER_TENSOR_SKETCH_ROOT);
  assert(parameter_values(reloaded.snapshot.manifest, "dependency_git_commit").size() == 1);

  std::vector<tensor_index::QueryHit> built_hits =
      tensor_index::query_tensor_index(built, encode_sequence("ACGTAC"), 10000);
  std::vector<tensor_index::QueryHit> reloaded_hits =
      tensor_index::query_tensor_index(reloaded, encode_sequence("ACGTAC"), 10000);
  compare_hits(built_hits, reloaded_hits);

  const std::vector<tensor_index::QueryHit> exact_hits =
      exact_ranked(reloaded.snapshot, query_vector);
  assert(!exact_hits.empty());
  for (std::size_t i = 1; i < exact_hits.size(); ++i) {
    assert(exact_hits[i - 1].distance <= exact_hits[i].distance);
    if (exact_hits[i - 1].distance == exact_hits[i].distance) {
      assert(exact_hits[i - 1].label < exact_hits[i].label);
    }
  }
}

void test_candidate_tool_tensor_commands() {
  TempDirectory temp;
  const auto reference = temp.file("reference.fa");
  const auto out_dir = temp.file("tensor-index");
  const auto query_out = temp.file("query.tsv");
  write_text_file(reference, ">tiny\nACGTACGTACGT\n");

  const std::string build_command =
      "./candidate_tool tensor-build --ref " + shell_quote(reference) +
      " --window 6 --stride 1 --dimension 16 --seed 17 --hnsw-m 16"
      " --hnsw-ef-construction 200 --hnsw-ef-search 50 --out-dir " +
      shell_quote(out_dir);
  assert(run_command(build_command.c_str()) == 0);

  const std::string query_command =
      "./candidate_tool tensor-query --index-dir " + shell_quote(out_dir) +
      " --query ACGTAC --top-k 3 > " + shell_quote(query_out);
  assert(run_command(query_command.c_str()) == 0);

  const std::string output = read_file(query_out);
  assert(output.find("label\tdistance\n") == 0);
  assert(output.find('\n', std::string("label\tdistance\n").size()) !=
         std::string::npos);
}

}  // namespace

int main() {
  test_slide_matches_independent_windows_when_stride_is_one();
  test_tensor_index_round_trip_for_dimension(16);
  test_tensor_index_round_trip_for_dimension(32);
  test_candidate_tool_tensor_commands();
  std::cout << "tensor index tests passed\n";
  return 0;
}
