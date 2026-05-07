// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <cmath>
#include <functional>
#include <unordered_map>

#include <torch/extension.h>

#include "gemm_common.h"
#include "gemm_dispatch_utils.h"
#include "gemm_kernel_registry.h"

#include "gemm_a8w8_blockscale_common.cuh"
#include "gemm_a8w8_blockscale_manifest.h"

using BlockwiseKernel = std::function<torch::Tensor(
    torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&)>;

template <typename EDataType>
static constexpr const char* etype_tag() {
    if constexpr(std::is_same_v<EDataType, BF16>) return "BF16";
    else if constexpr(std::is_same_v<EDataType, FP16>) return "FP16";
    else { static_assert(false, "unsupported EDataType"); return ""; }
}

template <typename DDataType, typename EDataType = DDataType>
static BlockwiseKernel blockscale_dispatch(int M, int N, int K)
{
    auto& reg = GemmKernelRegistry<BlockwiseKernel>::instance();
    const std::string tag = etype_tag<EDataType>();

    const int cu_num         = get_device_cu_num();
    const std::string& gfx   = get_device_gfx();

    auto* fn = reg.find(tag, gfx, cu_num, M, N, K);
    if(fn)
        return *fn;

    int padded_m = getPaddedM(M, N, K, 0);
    fn = reg.find(tag, gfx, cu_num, padded_m, N, K);
    if(fn)
        return *fn;

    padded_m = getPaddedM(M, N, K, 1);
    fn = reg.find(tag, gfx, cu_num, padded_m, N, K);
    if(fn)
        return *fn;

    return a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<
        DDataType,
        EDataType>;
}

torch::Tensor gemm_a8w8_blockscale(torch::Tensor& XQ,
                                   torch::Tensor& WQ,
                                   torch::Tensor& x_scale,
                                   torch::Tensor& w_scale,
                                   torch::Tensor& Y)
{
    TORCH_CHECK(XQ.dtype() == WQ.dtype(), "Weights and activations should have the same dtype!");
    TORCH_CHECK(x_scale.dtype() == w_scale.dtype(), "Scales should have the same dtype!");

    int M = XQ.size(0);
    int N = WQ.size(0);
    int K = XQ.size(1);

    if(x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::Half)
    {
        blockscale_dispatch<FP32, FP16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y);
    }
    else if(x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::BFloat16)
    {
        blockscale_dispatch<FP32, BF16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported scales/output dtype!");
    }
    return Y;
}
