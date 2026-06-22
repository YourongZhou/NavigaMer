#include "sha256.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>

namespace {

constexpr std::array<uint32_t, 64> kRoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
    0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
    0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
    0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
    0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
    0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
    0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
    0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

uint32_t rotate_right(uint32_t value, unsigned count) {
  return (value >> count) | (value << (32U - count));
}

class Sha256State {
 public:
  void update(const uint8_t* data, std::size_t size) {
    total_bytes_ += size;
    while (size != 0) {
      const std::size_t count =
          std::min<std::size_t>(size, block_.size() - block_size_);
      std::copy(data, data + count, block_.begin() + block_size_);
      block_size_ += count;
      data += count;
      size -= count;
      if (block_size_ == block_.size()) {
        compress(block_.data());
        block_size_ = 0;
      }
    }
  }

  Sha256Digest finish() {
    const uint64_t bit_length = total_bytes_ * 8U;
    block_[block_size_++] = 0x80U;
    if (block_size_ > 56) {
      std::fill(block_.begin() + block_size_, block_.end(), 0);
      compress(block_.data());
      block_size_ = 0;
    }
    std::fill(block_.begin() + block_size_, block_.begin() + 56, 0);
    for (unsigned index = 0; index < 8; ++index) {
      block_[63 - index] = static_cast<uint8_t>(bit_length >> (index * 8U));
    }
    compress(block_.data());

    Sha256Digest digest{};
    for (std::size_t index = 0; index < state_.size(); ++index) {
      for (unsigned byte = 0; byte < 4; ++byte) {
        digest[index * 4 + byte] =
            static_cast<uint8_t>(state_[index] >> (24U - byte * 8U));
      }
    }
    return digest;
  }

 private:
  void compress(const uint8_t* block) {
    std::array<uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16; ++index) {
      words[index] = (static_cast<uint32_t>(block[index * 4]) << 24U) |
                     (static_cast<uint32_t>(block[index * 4 + 1]) << 16U) |
                     (static_cast<uint32_t>(block[index * 4 + 2]) << 8U) |
                     static_cast<uint32_t>(block[index * 4 + 3]);
    }
    for (std::size_t index = 16; index < words.size(); ++index) {
      const uint32_t s0 = rotate_right(words[index - 15], 7) ^
                          rotate_right(words[index - 15], 18) ^
                          (words[index - 15] >> 3U);
      const uint32_t s1 = rotate_right(words[index - 2], 17) ^
                          rotate_right(words[index - 2], 19) ^
                          (words[index - 2] >> 10U);
      words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }

    uint32_t a = state_[0];
    uint32_t b = state_[1];
    uint32_t c = state_[2];
    uint32_t d = state_[3];
    uint32_t e = state_[4];
    uint32_t f = state_[5];
    uint32_t g = state_[6];
    uint32_t h = state_[7];
    for (std::size_t index = 0; index < words.size(); ++index) {
      const uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                            rotate_right(e, 25);
      const uint32_t choice = (e & f) ^ (~e & g);
      const uint32_t temp1 =
          h + sum1 + choice + kRoundConstants[index] + words[index];
      const uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                            rotate_right(a, 22);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temp2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<uint32_t, 8> state_ = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U,
                                    0xa54ff53aU, 0x510e527fU, 0x9b05688cU,
                                    0x1f83d9abU, 0x5be0cd19U};
  std::array<uint8_t, 64> block_{};
  std::size_t block_size_ = 0;
  uint64_t total_bytes_ = 0;
};

std::string to_hex(const Sha256Digest& digest) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (uint8_t byte : digest) {
    output << std::setw(2) << static_cast<unsigned>(byte);
  }
  return output.str();
}

}  // namespace

Sha256Digest sha256(std::string_view input) {
  Sha256State state;
  state.update(reinterpret_cast<const uint8_t*>(input.data()), input.size());
  return state.finish();
}

Sha256Digest sha256(const std::vector<uint8_t>& input) {
  Sha256State state;
  state.update(input.data(), input.size());
  return state.finish();
}

std::string sha256_hex(std::string_view input) { return to_hex(sha256(input)); }

std::string sha256_hex(const std::vector<uint8_t>& input) {
  return to_hex(sha256(input));
}
