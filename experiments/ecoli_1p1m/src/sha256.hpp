#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

using Sha256Digest = std::array<uint8_t, 32>;

Sha256Digest sha256(std::string_view input);
Sha256Digest sha256(const std::vector<uint8_t>& input);
std::string sha256_hex(std::string_view input);
std::string sha256_hex(const std::vector<uint8_t>& input);
