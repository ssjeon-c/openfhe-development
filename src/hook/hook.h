#ifndef HOOK__
#define HOOK__

#include "openfhe.h"

#include <map>

namespace debug {

extern bool enabled;

template <typename T>
class record {
public:
    static std::map<size_t, T> values;

public:
    static void set(T value, size_t index) {
        values.emplace(index, value);
    }
    static T get(size_t index) {
        return values[index];
    }
};

template <typename T>
std::map<size_t, T> record<T>::values;

#define RECORD_GET(T, index)          \
    if (debug::enabled) {             \
        debug::record<T>::get(index); \
    }
#define RECORD_SET(val, index)                         \
    if (debug::enabled) {                              \
        debug::record<decltype(val)>::set(val, index); \
    }
#define RECORD_PRINT(T, index)                                  \
    if (debug::enabled) {                                       \
        std::cout << debug::record<T>::get(index) << std::endl; \
    }

};  // namespace debug

#endif