#pragma once
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

namespace litert::lm::checkpoint {
class Io {
 public:
  Io(bool (*transfer)(void*, size_t, void*), void* user_data, bool reading)
      : transfer_(transfer), user_data_(user_data), reading_(reading) {}
  bool Bytes(void* data, size_t count) {
    auto* p = static_cast<unsigned char*>(data);
    while (count) {
      const size_t n = std::min(count, size_t{65536});
      if (!transfer_(p, n, user_data_)) return false;
      p += n; count -= n;
    }
    return true;
  }
  template<class T> bool Value(T& value) {
    static_assert(std::is_arithmetic_v<T>);
    return Bytes(&value, sizeof(value));
  }
  template<class T> bool Expect(T value) {
    T expected = value;
    return Value(value) && value == expected;
  }
  template<class T> bool Vector(std::vector<T>& data, uint64_t limit) {
    uint64_t count = data.size();
    if (!Value(count) || count > limit) return false;
    if (reading_) data.resize(count);
    return Bytes(data.data(), count * sizeof(T));
  }
  bool Name(const std::string& expected) {
    uint64_t size = expected.size();
    if (!Expect(size)) return false;
    std::string value = expected;
    return Bytes(value.data(), value.size()) && value == expected;
  }
  bool Finish() { return transfer_(nullptr, 0, user_data_); }
 private:
  bool (*transfer_)(void*, size_t, void*);
  void* user_data_;
  bool reading_;
};
}
