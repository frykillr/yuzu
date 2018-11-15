// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include "common/assert.h"

/*
 * Creates a vector with a size limit.
 *
 *
 * It's useful to avoid the following pattern without using std::vector<>:
 *
 * int foo_count = 0;
 * T foos[N];
 * foos[foo_count++] = foo;
 * foos[foo_count++] = foo;
 *
 * While also allowing multiple fields in "foos" using continuous memory stream per field. In other
 * words, for StaticVector<4, T1 T2> data is laid out as:
 * T1 T1 T1 T1
 * T2 T2 T2 T2
 *
 * Instead of:
 * T1 T2
 * T1 T2
 * T1 T2
 * T1 T2
 *
 * Sample use:
 *
 * StaticVector<SOME_SIZE_LIMIT, int> static_vector;
 * static_vector.Push(20);
 * static_vector.Push(40);
 * static_vector.Push(60);
 * Foo(static_vector.Size(), static_vector.Data());
 *
 *
 * Multi-type sample:
 *
 * StaticVector<SOME_OTHER_SIZE_LIMIT, int, float> static_vector;
 * static_vector.Push(50, 3.14f);
 * static_vector.Push(80, 3.14f / 2);
 * Bar(static_vector.Size(), static_vector.Data<int>(), static_vector.Data<float>());
 */
template <std::size_t N, typename... Types>
class StaticVector {
public:
    /// Pushes a set of values. The number of pushed elements has to be lower than N.
    void Push(const Types&... values) {
        const std::size_t index = count++;
        ASSERT_MSG(index < N, "Static vector overflow");
        InternalPush<0>(index, values...);
    }

    /// Gets current size of the vector.
    std::size_t Size() const {
        return count;
    }

    /// Gets an array of the asked member index.
    template <int V>
    auto& Get() {
        return std::get<V>(arrays);
    }

    /// Gets an array of the asked member index.
    template <int V>
    const auto& Get() const {
        return std::get<V>(arrays);
    }

    /// Gets an array of the asked type.
    template <typename T>
    std::array<T, N>& Get() {
        return std::get<std::array<T, N>>(arrays);
    }

    /// Gets an array of the asked type.
    template <typename T>
    const std::array<T, N>& Get() const {
        return std::get<std::array<T, N>>(arrays);
    }

    /// Gets a pointer to the data of the asked member index.
    template <int V>
    auto* Data() {
        return Get<V>().data();
    }

    /// Gets a pointer to the data of the asked member index.
    template <int V>
    const auto* Data() const {
        return Get<V>().data();
    }

    /// Gets a pointer to the data of the asked type.
    template <typename T>
    T* Data() {
        return Get<T>().data();
    }

    /// Gets a pointer to the data of the asked type.
    template <typename T>
    const T* Data() const {
        return Get<T>().data();
    }

    /// Gets a pointer to the data of the first member.
    auto* data() {
        return std::get<0>(arrays).data();
    }

    /// Gets a pointer to the data of the first member.
    const auto* data() const {
        return std::get<0>(arrays).data();
    }

    auto begin() const {
        return std::get<0>(arrays).begin();
    }

    auto end() const {
        return std::get<0>(arrays).begin() + count;
    }

    auto begin() {
        return std::get<0>(arrays).begin();
    }

    auto end() {
        return std::get<0>(arrays).begin() + count;
    }

    /// Returns the capacity of the vector.
    static constexpr std::size_t capacity() {
        return N;
    }

private:
    template <int V, typename T, typename... U>
    void InternalPush(std::size_t index, const T& value, const U&... values) {
        std::get<V>(arrays)[index] = value;
        if constexpr (V + 1 < sizeof...(Types)) {
            InternalPush<V + 1>(index, values...);
        }
    }

    std::size_t count = 0;
    std::tuple<std::array<Types, N>...> arrays{};

    template <std::size_t N, typename... Types>
    friend bool operator<(const StaticVector<N, Types...>& left,
                          const StaticVector<N, Types...>& right);

    template <std::size_t N, typename... Types>
    friend bool operator==(const StaticVector<N, Types...>& left,
                           const StaticVector<N, Types...>& right);
};

template <std::size_t N, typename... Types>
[[nodiscard]] bool operator<(const StaticVector<N, Types...>& left,
                             const StaticVector<N, Types...>& right) {
    return left.arrays < right.arrays;
}

template <std::size_t N, typename... Types>
[[nodiscard]] bool operator==(const StaticVector<N, Types...>& left,
                              const StaticVector<N, Types...>& right) {
    return left.count == right.count && left.arrays == right.arrays;
}

template <std::size_t N, typename... Types>
[[nodiscard]] bool operator!=(const StaticVector<N, Types...>& left,
                              const StaticVector<N, Types...>& right) {
    return !(left == right);
}