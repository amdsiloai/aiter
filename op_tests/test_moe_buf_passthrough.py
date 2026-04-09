# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter import dtypes
from aiter.fused_moe import fused_moe_fake, fused_topk, moe_sorting
from aiter.test_common import checkAllclose

BLOCK_SIZE_M = 32


@pytest.fixture
def moe_sorting_inputs():
    device = torch.device("cuda")
    M = 32
    num_experts = 8
    topk = 2
    model_dim = 128
    dtype = dtypes.d_dtypes["bf16"]

    hidden = torch.randn((M, model_dim), dtype=dtype, device=device)
    score = torch.rand((M, num_experts), dtype=dtype, device=device)
    topk_weights, topk_ids = fused_topk(hidden, score, topk, True)

    return {
        "device": device,
        "M": M,
        "num_experts": num_experts,
        "topk": topk,
        "model_dim": model_dim,
        "dtype": dtype,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
    }


def _assert_moe_sorting_outputs_match(ref, with_buf, topk: int, num_token_rows: int):
    (
        sorted_ids_ref,
        sorted_weights_ref,
        sorted_expert_ids_ref,
        num_valid_ref,
        _moe_buf_ref,
    ) = ref
    (
        sorted_ids_b,
        sorted_weights_b,
        sorted_expert_ids_b,
        num_valid_b,
        moe_buf_b,
    ) = with_buf

    assert checkAllclose(num_valid_ref, num_valid_b, atol=0, msg="num_valid_ids") == 0

    # Padding slots use (topk << 24 | num_token_rows); see test_moe_sorting.py / native ref.
    init_pat = topk << 24 | num_token_rows
    mask = sorted_ids_ref != init_pat
    assert (
        checkAllclose(
            sorted_weights_ref[mask],
            sorted_weights_b[mask],
            msg="sorted_weights",
        )
        == 0
    )

    num_valid = int(num_valid_ref[0].item())
    assert (
        checkAllclose(
            sorted_ids_ref[:num_valid],
            sorted_ids_b[:num_valid],
            msg="sorted_ids",
            atol=0,
        )
        == 0
    )

    # CK reference compares sorted_expert_ids only for slen / block_size entries
    # (tail of the buffer may be uninitialized).
    n_expert_blocks = num_valid // BLOCK_SIZE_M
    assert (
        checkAllclose(
            sorted_expert_ids_ref[:n_expert_blocks],
            sorted_expert_ids_b[:n_expert_blocks],
            msg="sorted_expert_ids",
            atol=0,
        )
        == 0
    )
    return moe_buf_b


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_moe_sorting_preallocated_buf(moe_sorting_inputs):
    p = moe_sorting_inputs
    pre_buf = torch.empty(
        (p["M"], p["model_dim"]), dtype=p["dtype"], device=p["device"]
    )
    ptr_pre = pre_buf.data_ptr()

    out_default = moe_sorting(
        p["topk_ids"],
        p["topk_weights"],
        p["num_experts"],
        p["model_dim"],
        p["dtype"],
        BLOCK_SIZE_M,
        moe_buf=None,
    )
    out_pre = moe_sorting(
        p["topk_ids"],
        p["topk_weights"],
        p["num_experts"],
        p["model_dim"],
        p["dtype"],
        BLOCK_SIZE_M,
        moe_buf=pre_buf,
    )

    moe_buf_ret = _assert_moe_sorting_outputs_match(
        out_default,
        out_pre,
        p["topk"],
        p["topk_ids"].shape[0],
    )
    assert moe_buf_ret.data_ptr() == ptr_pre
    assert moe_buf_ret is pre_buf
    assert moe_buf_ret.shape == (p["M"], p["model_dim"])
    assert moe_buf_ret.dtype == p["dtype"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_moe_sorting_none_buf(moe_sorting_inputs):
    p = moe_sorting_inputs
    (
        _sid,
        _sw,
        _se,
        _nv,
        moe_buf,
    ) = moe_sorting(
        p["topk_ids"],
        p["topk_weights"],
        p["num_experts"],
        p["model_dim"],
        p["dtype"],
        BLOCK_SIZE_M,
        moe_buf=None,
    )
    assert moe_buf.shape == (p["M"], p["model_dim"])
    assert moe_buf.dtype == p["dtype"]
    assert moe_buf.device.type == p["device"].type
    assert moe_buf.is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_moe_fake_with_moe_buf():
    device = torch.device("cuda")
    M = 32
    num_experts = 8
    topk = 2
    model_dim = 128
    inter_dim = 64
    dtype = dtypes.d_dtypes["bf16"]

    hidden = torch.randn((M, model_dim), dtype=dtype, device=device)
    w1 = torch.randn(
        (num_experts, inter_dim * 2, model_dim), dtype=dtype, device=device
    )
    w2 = torch.randn(
        (num_experts, model_dim, inter_dim), dtype=dtype, device=device
    )
    score = torch.rand((M, num_experts), dtype=dtype, device=device)
    topk_weights, topk_ids = fused_topk(hidden, score, topk, True)

    user_buf = torch.empty((M, model_dim), dtype=dtype, device=device)
    ret = fused_moe_fake(
        hidden,
        w1,
        w2,
        topk_weights,
        topk_ids,
        moe_buf=user_buf,
    )
    assert ret is user_buf
    assert ret.data_ptr() == user_buf.data_ptr()

    a = fused_moe_fake(
        hidden,
        w1,
        w2,
        topk_weights,
        topk_ids,
        moe_buf=None,
    )
    b = fused_moe_fake(
        hidden,
        w1,
        w2,
        topk_weights,
        topk_ids,
        moe_buf=None,
    )
    assert a.data_ptr() != b.data_ptr()
