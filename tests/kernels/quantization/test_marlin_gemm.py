# SPDX-License-Identifier: Apache-2.0
"""Tests for the marlin kernel.

Run `pytest tests/kernels/marlin/test_marlin_gemm.py`.
"""
import pytest
import torch

from tests.kernels.utils import DEFAULT_OPCHECK_TEST_UTILS, opcheck
from tests.quantization.utils import is_quant_method_supported
from vllm_logits import _custom_ops as ops
from vllm_logits.model_executor.layers.quantization.gptq_marlin_24 import (
    GPTQ_MARLIN_24_MAX_PARALLEL, GPTQ_MARLIN_24_MIN_THREAD_N,
    GPTQ_MARLIN_24_SUPPORTED_GROUP_SIZES, GPTQ_MARLIN_24_SUPPORTED_QUANT_TYPES)
from vllm_logits.model_executor.layers.quantization.qqq import (
    MARLIN_QQQ_MAX_PARALLEL, MARLIN_QQQ_MIN_THREAD_N,
    MARLIN_QQQ_SUPPORTED_GROUP_SIZES, MARLIN_QQQ_SUPPORTED_NUM_BITS)
from vllm_logits.model_executor.layers.quantization.utils.marlin_utils import (
    GPTQ_MARLIN_MAX_PARALLEL, GPTQ_MARLIN_MIN_THREAD_N,
    MARLIN_SUPPORTED_GROUP_SIZES, marlin_make_empty_g_idx,
    marlin_make_workspace_new, marlin_permute_scales,
    query_marlin_supported_quant_types)
from vllm_logits.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    FP4_MARLIN_SUPPORTED_GROUP_SIZES, rand_marlin_weight_fp4_like)
from vllm_logits.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    marlin_quant_fp8_torch)
from vllm_logits.model_executor.layers.quantization.utils.marlin_utils_test import (
    MarlinWorkspace, awq_marlin_quantize, get_weight_perm, marlin_quantize,
    marlin_weights)
from vllm_logits.model_executor.layers.quantization.utils.marlin_utils_test_24 import (
    marlin_24_quantize)
from vllm_logits.model_executor.layers.quantization.utils.marlin_utils_test_qqq import (  # noqa: E501
    marlin_qqq_quantize)
from vllm_logits.model_executor.layers.quantization.utils.quant_utils import (
    awq_pack, gptq_pack, gptq_quantize_weights, quantize_weights, sort_weights)
from vllm_logits.scalar_type import scalar_types

ACT_ORDER_OPTS = [False, True]
K_FULL_OPTS = [False, True]
USE_ATOMIC_ADD_OPTS = [False, True]
USE_FP32_REDUCE_OPTS = [False, True]

MARLIN_K_CHUNKS = [128]
MARLIN_N_CHUNKS = [64, 256]

MARLIN_24_K_CHUNKS = [128]
MARLIN_24_N_CHUNKS = [512]

HQQ_SUPPORTED_GROUP_SIZES = [64]

MNK_FACTORS = [
    (1, 1, 1),
    (1, 4, 8),
    (1, 7, 5),
    (13, 17, 67),
    (26, 37, 13),
    (67, 13, 11),
    (257, 13, 11),
    (658, 13, 11),
]

DTYPES = [torch.float16, torch.bfloat16]


def compute_max_diff(output, output_ref):
    return torch.mean(torch.abs(output - output_ref)) / torch.mean(
        torch.abs(output_ref))


def rand_data(shape, dtype=torch.float16):
    return torch.randn(shape, dtype=dtype, device="cuda")


@pytest.mark.skipif(not is_quant_method_supported("gptq_marlin"),
                    reason="Marlin is not supported on this GPU type.")
@pytest.mark.parametrize("k_chunk", MARLIN_K_CHUNKS)
@pytest.mark.parametrize("n_chunk", MARLIN_N_CHUNKS)
@pytest.mark.parametrize("quant_type",
                         query_marlin_supported_quant_types(False, False))
@pytest.mark.parametrize("group_size", MARLIN_SUPPORTED_GROUP_SIZES)
@pytest.mark.parametrize("act_order", ACT_ORDER_OPTS)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_gptq_marlin_repack(k_chunk, n_chunk, quant_type, group_size,
                            act_order, mnk_factors):
    m_factor, n_factor, k_factor = mnk_factors

    size_k = k_chunk * k_factor
    size_n = n_chunk * n_factor

    # Filter act_order
    if act_order:
        if group_size == -1:
            return
        if group_size == size_k:
            return

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    # Create input
    b_weight = rand_data((size_k, size_n))

    # Quantize (and apply act_order if provided)
    w_ref, q_w, s, g_idx, rand_perm = gptq_quantize_weights(
        b_weight, quant_type, group_size, act_order)

    # Pack to GPTQ format
    q_w_gptq = gptq_pack(q_w, quant_type.size_bits, size_k, size_n)

    # For act_order, sort the "weights" and "g_idx" so that group ids are
    # increasing
    sort_indices = torch.empty(0, dtype=torch.int, device=b_weight.device)
    if act_order:
        q_w, g_idx, sort_indices = sort_weights(q_w, g_idx)

    # Pack to Marlin format
    weight_perm = get_weight_perm(quant_type.size_bits)
    marlin_q_w_1 = marlin_weights(q_w, size_k, size_n, quant_type.size_bits,
                                  weight_perm)

    opcheck(torch.ops._C.gptq_marlin_repack,
            (q_w_gptq, sort_indices, size_k, size_n, quant_type.size_bits))

    # Run Marlin repack GPU kernel
    marlin_q_w_2 = ops.gptq_marlin_repack(
        q_w_gptq,
        sort_indices,
        size_k,
        size_n,
        quant_type.size_bits,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(marlin_q_w_1, marlin_q_w_2)


@pytest.mark.skipif(not is_quant_method_supported("gptq_marlin"),
                    reason="Marlin is not supported on this GPU type.")
@pytest.mark.parametrize("k_chunk", MARLIN_K_CHUNKS)
@pytest.mark.parametrize("n_chunk", MARLIN_N_CHUNKS)
@pytest.mark.parametrize("quant_type",
                         query_marlin_supported_quant_types(True))
@pytest.mark.parametrize("group_size", MARLIN_SUPPORTED_GROUP_SIZES)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_awq_marlin_repack(k_chunk, n_chunk, quant_type, group_size,
                           mnk_factors):
    m_factor, n_factor, k_factor = mnk_factors

    size_k = k_chunk * k_factor
    size_n = n_chunk * n_factor

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    # Create input
    b_weight = rand_data((size_k, size_n))

    # Quantize
    w_ref, q_w, s, zp = quantize_weights(b_weight,
                                         quant_type,
                                         group_size,
                                         zero_points=True)

    # Pack to AWQ format
    q_w_awq = awq_pack(q_w, quant_type.size_bits, size_k, size_n)

    # Pack to Marlin format
    weight_perm = get_weight_perm(quant_type.size_bits)
    marlin_q_w_1 = marlin_weights(q_w, size_k, size_n, quant_type.size_bits,
                                  weight_perm)

    opcheck(torch.ops._C.awq_marlin_repack,
            (q_w_awq, size_k, size_n, quant_type.size_bits))

    # Run Marlin repack GPU kernel
    marlin_q_w_2 = ops.awq_marlin_repack(
        q_w_awq,
        size_k,
        size_n,
        quant_type.size_bits,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(marlin_q_w_1, marlin_q_w_2)


@pytest.mark.skipif(not is_quant_method_supported("gptq_marlin"),
                    reason="Marlin is not supported on this GPU type.")
@pytest.mark.parametrize("k_chunk", MARLIN_K_CHUNKS)
@pytest.mark.parametrize("n_chunk", MARLIN_N_CHUNKS)
@pytest.mark.parametrize("quant_type", query_marlin_supported_quant_types())
@pytest.mark.parametrize(
    "group_size",
    set(MARLIN_SUPPORTED_GROUP_SIZES + FP4_MARLIN_SUPPORTED_GROUP_SIZES))
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
@pytest.mark.parametrize("act_order", ACT_ORDER_OPTS)
@pytest.mark.parametrize("is_k_full", K_FULL_OPTS)
@pytest.mark.parametrize("use_atomic_add", USE_ATOMIC_ADD_OPTS)
@pytest.mark.parametrize("use_fp32_reduce", USE_FP32_REDUCE_OPTS)
def test_gptq_marlin_gemm(
    k_chunk,
    n_chunk,
    quant_type,
    group_size,
    mnk_factors,
    act_order,
    is_k_full,
    use_atomic_add,
    use_fp32_reduce,
):
    m_factor, n_factor, k_factor = mnk_factors
    has_zp = quant_type in [scalar_types.uint4, scalar_types.uint8]

    size_m = m_factor
    size_k = k_chunk * k_factor
    size_n = n_chunk * n_factor

    if act_order:
        if group_size == -1:
            return
        if group_size == size_k:
            return
        if has_zp:
            return

    if size_k % group_size != 0:
        return

    a_input = rand_data((size_m, size_k))
    b_weight = rand_data((size_k, size_n))

    if quant_type == scalar_types.float4_e2m1f:
        if group_size != 16 or act_order:
            return
        w_ref, marlin_q_w, marlin_s, marlin_s2 = rand_marlin_weight_fp4_like(
            b_weight.T, group_size)
        g_idx = None
        sort_indices = None
        marlin_zp = None
    elif quant_type == scalar_types.float8_e4m3fn:
        if group_size not in [-1, 128]:
            return
        if act_order:
            return
        w_ref, marlin_q_w, marlin_s = marlin_quant_fp8_torch(
            b_weight.T, group_size)
        g_idx = None
        sort_indices = None
        marlin_zp = None
        marlin_s2 = None
    elif has_zp:
        if group_size == 16:
            return
        w_ref, marlin_q_w, marlin_s, marlin_zp = awq_marlin_quantize(
            b_weight, quant_type, group_size)
        g_idx = None
        sort_indices = None
        marlin_s2 = None
    else:
        if group_size == 16:
            return
        w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
            b_weight, quant_type, group_size, act_order)
        marlin_zp = None
        marlin_s2 = None

    workspace = marlin_make_workspace_new(w_ref.device)

    opcheck(torch.ops._C.gptq_marlin_gemm,
            (a_input, None, marlin_q_w, marlin_s, marlin_s2, marlin_zp, g_idx,
             sort_indices, workspace, quant_type.id, a_input.shape[0],
             b_weight.shape[1], a_input.shape[1], is_k_full, use_atomic_add,
             use_fp32_reduce, False),
            test_utils=DEFAULT_OPCHECK_TEST_UTILS)

    output = ops.gptq_marlin_gemm(
        a_input,
        None,
        marlin_q_w,
        marlin_s,
        marlin_s2,
        marlin_zp,
        g_idx,
        sort_indices,
        workspace,
        quant_type,
        a_input.shape[0],
        b_weight.shape[1],
        a_input.shape[1],
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
        is_zp_float=False,
    )
    output_ref = torch.matmul(a_input, w_ref)

    torch.cuda.synchronize()

    max_diff = compute_max_diff(output, output_ref)

    assert max_diff < 0.04


# TODO: find better way to test this?
@torch.compile(fullgraph=True)
def marlin_24_gemm_tester(a_input, marlin_24_q_w_comp, marlin_24_meta,
                          marlin_24_s, scratch, quant_type, size_m, size_n,
                          size_k):
    return ops.gptq_marlin_24_gemm(a_input, marlin_24_q_w_comp, marlin_24_meta,
                                   marlin_24_s, scratch, quant_type, size_m,
                                   size_n, size_k)


@pytest.mark.skipif(not is_quant_method_supported("gptq_marlin"),
                    reason="Marlin is not supported on this GPU type.")
@pytest.mark.parametrize("k_chunk", MARLIN_24_K_CHUNKS)
@pytest.mark.parametrize("n_chunk", MARLIN_24_N_CHUNKS)
@pytest.mark.parametrize("quant_type", GPTQ_MARLIN_24_SUPPORTED_QUANT_TYPES)
@pytest.mark.parametrize("group_size", GPTQ_MARLIN_24_SUPPORTED_GROUP_SIZES)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_gptq_marlin_24_gemm(k_chunk, n_chunk, quant_type, group_size,
                             mnk_factors):
    m_factor, n_factor, k_factor = mnk_factors

    size_m = m_factor
    size_k = k_chunk * k_factor
    size_n = n_chunk * n_factor

    a_input = rand_data((size_m, size_k))
    b_weight = rand_data((size_k, size_n))

    (w_24_ref, marlin_24_q_w_comp, marlin_24_meta,
     marlin_24_s) = marlin_24_quantize(b_weight, quant_type, group_size)

    workspace_24 = MarlinWorkspace(size_n, GPTQ_MARLIN_24_MIN_THREAD_N,
                                   GPTQ_MARLIN_24_MAX_PARALLEL)

    output_ref = torch.matmul(a_input, w_24_ref)

    opcheck(torch.ops._C.gptq_marlin_24_gemm,
            (a_input, marlin_24_q_w_comp, marlin_24_meta, marlin_24_s,
             workspace_24.scratch, quant_type.id, a_input.shape[0],
             b_weight.shape[1], a_input.shape[1]),
            test_utils=DEFAULT_OPCHECK_TEST_UTILS)

    output = marlin_24_gemm_tester(
        a_input,
        marlin_24_q_w_comp,
        marlin_24_meta,
        marlin_24_s,
        workspace_24.scratch,
        quant_type,
        a_input.shape[0],
        b_weight.shape[1],
        a_input.shape[1],
    )

    torch.cuda.synchronize()

    max_diff = compute_max_diff(output, output_ref)

    assert max_diff < 0.04


@pytest.mark.skipif(not is_quant_method_supported("gptq_marlin"),
                    reason="Marlin is not supported on this GPU type.")
@pytest.mark.parametrize("k_chunk", MARLIN_K_CHUNKS)
@pytest.mark.parametrize("n_chunk", MARLIN_N_CHUNKS)
@pytest.mark.parametrize("group_size", HQQ_SUPPORTED_GROUP_SIZES)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
@pytest.mark.parametrize("use_fp32_reduce", USE_FP32_REDUCE_OPTS)
def test_hqq_marlin_gemm(
    k_chunk,
    n_chunk,
    group_size,
    mnk_factors,
    use_fp32_reduce,
):
    m_factor, n_factor, k_factor = mnk_factors

    size_m = m_factor
    size_k = k_chunk * k_factor
    size_n = n_chunk * n_factor

    quant_type = scalar_types.uint4

    a_input = rand_data((size_m, size_k))
    dev = a_input.device

    b_weight = torch.randint(0,
                             10, (size_n, size_k),
                             dtype=torch.uint8,
                             device=dev)
    scale = rand_data((size_n, size_k // group_size))
    zero = rand_data((size_n, size_k // group_size))

    gptq_w_q = gptq_pack(b_weight.transpose(1, 0), 4, size_k, size_n)

    sort_indices = torch.empty(0, dtype=torch.int, device=dev)
    marlin_w_q = ops.gptq_marlin_repack(gptq_w_q, sort_indices, size_k, size_n,
                                        4).to(dev)
    marlin_s = marlin_permute_scales(scale.transpose(1, 0), size_k, size_n,
                                     group_size).to(dev)
    marlin_zp = marlin_permute_scales(zero.transpose(1, 0), size_k, size_n,
                                      group_size).to(dev)

    g_idx = marlin_make_empty_g_idx(dev)
    g_idx_sort_indices = marlin_make_empty_g_idx(dev)

    workspace = marlin_make_workspace_new(b_weight.device)

    output = ops.gptq_marlin_gemm(
        a_input,
        None,
        marlin_w_q,
        marlin_s,
        None,
        marlin_zp,
        g_idx,
        g_idx_sort_indices,
        workspace,
        quant_type,
        a_input.shape[0],
        b_weight.shape[0],
        a_input.shape[1],
        is_k_full=True,
        use_fp32_reduce=use_fp32_reduce,
        is_zp_float=True,
    )

    b_flat = b_weight.reshape(-1, group_size)
    zp_flat = zero.reshape(-1, 1)
    s_flat = scale.reshape(-1, 1)
    dequant = (b_flat - zp_flat) * s_flat

    output_ref = torch.matmul(a_input,
                              dequant.reshape(b_weight.shape).transpose(1, 0))

    torch.cuda.synchronize()

    max_diff = compute_max_diff(output, output_ref)

    assert max_diff < 0.04


@pytest.mark.skipif(not is_quant_method_supported("qqq"),
                    reason="Marlin is not supported on this GPU type.")
@pytest.mark.parametrize("k_chunk", MARLIN_K_CHUNKS)
@pytest.mark.parametrize("n_chunk", MARLIN_N_CHUNKS)
@pytest.mark.parametrize("num_bits", MARLIN_QQQ_SUPPORTED_NUM_BITS)
@pytest.mark.parametrize("group_size", MARLIN_QQQ_SUPPORTED_GROUP_SIZES)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_marlin_qqq_gemm(
    k_chunk,
    n_chunk,
    num_bits,
    group_size,
    mnk_factors,
):
    int8_traits = torch.iinfo(torch.int8)
    m_factor, n_factor, k_factor = mnk_factors

    size_m = m_factor
    size_k = k_chunk * k_factor
    size_n = n_chunk * n_factor

    a_input = rand_data((size_m, size_k))
    b_weight = rand_data((size_k, size_n))

    # Quantize activations
    s_a = a_input.abs().max(dim=-1, keepdim=True)[0].div(int8_traits.max).to(
        torch.float)
    q_a = (a_input / s_a).round().clamp(int8_traits.min,
                                        int8_traits.max).to(torch.int8)

    # Quantize weights
    w_ref, marlin_qqq_q_w, marlin_qqq_s_group, marlin_qqq_s_channel = \
    marlin_qqq_quantize(b_weight, num_bits, group_size)

    workspace = MarlinWorkspace(size_n, MARLIN_QQQ_MIN_THREAD_N,
                                MARLIN_QQQ_MAX_PARALLEL)

    opcheck(torch.ops._C.marlin_qqq_gemm,
            (q_a, marlin_qqq_q_w, s_a, marlin_qqq_s_channel,
             marlin_qqq_s_group, workspace.scratch, a_input.shape[0],
             b_weight.shape[1], a_input.shape[1]))

    output = ops.marlin_qqq_gemm(
        q_a,
        marlin_qqq_q_w,
        s_a,
        marlin_qqq_s_channel,
        marlin_qqq_s_group,
        workspace.scratch,
        a_input.shape[0],
        b_weight.shape[1],
        a_input.shape[1],
    )
    output_ref = torch.matmul(q_a.half() * s_a.half(), w_ref)

    torch.cuda.synchronize()

    max_diff = compute_max_diff(output, output_ref)

    assert max_diff < 0.04


def test_marlin_gemm_subset_input():
    quant_type = scalar_types.uint4b8
    group_size = 128

    size_m, size_k, size_n = 32, 1024, 2048
    big_m = size_m * 2
    big_k = size_k * 2

    a_input = rand_data((big_m, big_k))[8:size_m + 8, 8:size_k + 8]
    b_weight = rand_data((size_k, size_n))

    w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
        b_weight, quant_type, group_size, False)

    marlin_zp = marlin_make_empty_g_idx(marlin_s.device)
    workspace = marlin_make_workspace_new(a_input.device)

    output = ops.gptq_marlin_gemm(
        a_input,
        None,
        marlin_q_w,
        marlin_s,
        None,
        marlin_zp,
        g_idx,
        sort_indices,
        workspace,
        quant_type,
        a_input.shape[0],
        b_weight.shape[1],
        a_input.shape[1],
        is_k_full=True,
        use_atomic_add=False,
        use_fp32_reduce=True,
        is_zp_float=False,
    )
    output_ref = torch.matmul(a_input, w_ref)

    torch.cuda.synchronize()

    max_diff = compute_max_diff(output, output_ref)

    assert max_diff < 0.04


def test_marlin_gemm_opcheck():
    size_m = 2048
    size_n = 4096
    size_k = 4096
    a = torch.rand((size_m, size_n), device='cuda', dtype=torch.float16)
    w = torch.randint(-5, 5, (256, 8192), device='cuda', dtype=torch.int32)
    s = torch.full((32, size_k), 0.125, device='cuda', dtype=torch.float16)
    wk = MarlinWorkspace(size_n, GPTQ_MARLIN_MIN_THREAD_N,
                         GPTQ_MARLIN_MAX_PARALLEL).scratch
    x = torch.ops._C.marlin_gemm(a, w, s, wk, size_m, size_n, size_k)
    y = torch.ops._C.marlin_gemm(a, w, s, wk, size_m, size_n, size_k)
    torch.testing.assert_close(x, y)
    opcheck(torch.ops._C.marlin_gemm, (a, w, s, wk, size_m, size_n, size_k))
