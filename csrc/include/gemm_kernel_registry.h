// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#ifdef USE_ROCM

#include <mutex>
#include <string>
#include "gemm_dispatch_utils.h"

struct GemmRegistryKey {
    std::string dtype_tag;
    std::string gfx;
    int cu;
    int M;
    int N;
    int K;

    bool operator==(const GemmRegistryKey& o) const {
        return dtype_tag == o.dtype_tag && gfx == o.gfx &&
               cu == o.cu && M == o.M && N == o.N && K == o.K;
    }
};

struct GemmRegistryKeyHash {
    std::size_t operator()(const GemmRegistryKey& k) const {
        std::size_t h = std::hash<std::string>{}(k.dtype_tag);
        h ^= std::hash<std::string>{}(k.gfx) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.cu)  + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.M)   + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.N)   + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.K)   + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

template <typename KernelFn>
class GemmKernelRegistry {
public:
    static GemmKernelRegistry& instance() {
        static GemmKernelRegistry reg;
        return reg;
    }

    void add(const std::string& dtype_tag,
             const std::string& gfx, int cu, int M, int N, int K,
             KernelFn fn) {
        std::lock_guard<std::mutex> lock(mu_);
        map_[{dtype_tag, gfx, cu, M, N, K}] = std::move(fn);
    }

    const KernelFn* find(const std::string& dtype_tag,
                         const std::string& gfx, int cu,
                         int M, int N, int K) const {
        auto it = map_.find({dtype_tag, gfx, cu, M, N, K});
        return it != map_.end() ? &it->second : nullptr;
    }

private:
    GemmKernelRegistry() = default;
    std::mutex mu_;
    std::unordered_map<GemmRegistryKey, KernelFn, GemmRegistryKeyHash> map_;
};

#endif // USE_ROCM
