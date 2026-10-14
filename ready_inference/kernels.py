"""Triton state gathering and pointwise reader operations for BF16 inference."""
import torch
from torch.nn import functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _gather_two(A, B, Idx, Out, SA: tl.constexpr, SB: tl.constexpr,
                TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch = x // 1572864
    slot = (x // 786432) % 2
    within = x % 786432
    request = tl.load(Idx + batch, x < TOTAL, 0).to(tl.int64)
    a = tl.load(A + request * SA + within, (x < TOTAL) & (slot == 0), 0)
    b = tl.load(B + request * SB + within, (x < TOTAL) & (slot == 1), 0)
    tl.store(Out + x, tl.where(slot == 0, a, b), x < TOTAL)


@triton.jit
def _state_pointwise(S, Mean, Normalized, Log, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // 16384
    variance = tl.load(Mean + row, offsets < TOTAL, 0)
    rms = libdevice.sqrt_rn(variance + 1.0e-6)
    values = tl.load(S + offsets, offsets < TOTAL, 0)
    normal = tl.div_rn(values, rms)
    tl.store(Normalized + offsets, normal, offsets < TOTAL)
    # One lane per state matrix writes its log-RMS.
    tl.store(Log + row, libdevice.log(rms), (offsets < TOTAL) & (offsets % 16384 == 0))


@triton.jit
def _features(Read, Log, Out, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    feature = offsets % 864
    batch = offsets // (7 * 864)
    token = (offsets // 864) % 7
    head = feature // 9
    rank = feature % 9
    # Read is contiguous [batch, 7, 96, 8]; retain BF16 rounding before normalization.
    index = ((batch * 7 + token) * 96 + head) * 8 + rank
    value = tl.load(Read + index, (offsets < TOTAL) & (rank < 8), 0).to(tl.float32)
    value = value * 0.08838834764831843
    log = tl.load(Log + batch * 96 + head, offsets < TOTAL, 0).to(tl.float32)
    tl.store(Out + offsets, tl.where(rank < 8, value, log), offsets < TOTAL)


def gather_states(a, b, indices, out):
    """Gather both selected layers into stable CUDA-graph input storage."""
    assert a.is_cuda and b.is_cuda and indices.is_cuda and out.is_cuda
    assert a.shape[1:] == b.shape[1:] == (48, 128, 128)
    assert a.stride()[1:] == b.stride()[1:] == (16384, 128, 1)
    assert a.dtype == b.dtype == out.dtype == torch.float32
    assert indices.ndim == 1 and indices.is_contiguous()
    assert indices.dtype in (torch.int32, torch.int64)
    assert out.is_contiguous() and out.shape == (indices.numel(), 2, 48, 128, 128)
    _gather_two[(triton.cdiv(out.numel(), 1024),)](
        a, b, indices, out, a.stride(0), b.stride(0), out.numel(), 1024)


def _norm(x):
    return F.rms_norm(x, (x.shape[-1],), eps=torch.finfo(torch.float32).eps)


def reader_forward(reader, hidden, state):
    """Preserve the state reduction and matrix products; fuse pointwise operations."""
    b, n, width = hidden.shape
    assert hidden.is_cuda and hidden.dtype == torch.bfloat16 and (n, width) == (7, 5120)
    assert state.is_cuda and state.is_contiguous() and state.dtype == torch.float32
    assert state.shape == (b, 2, 48, 128, 128)
    mean = state.square().mean((-2, -1), keepdim=True)
    normalized = torch.empty_like(state, dtype=hidden.dtype)
    logs = torch.empty((b, 2, 48), device=hidden.device, dtype=hidden.dtype)
    _state_pointwise[(triton.cdiv(state.numel(), 1024),)](
        state, mean, normalized, logs, state.numel(), 1024, enable_fp_fusion=False)
    query = reader.query(_norm(hidden)).view(b, n, 2, 48, 8)
    compressed = torch.einsum('lhvr,blhvk->blhrk', reader.value_down, normalized)
    key = torch.einsum('bnlhr,lhrk->bnlhk', query, reader.query_up)
    raw = torch.einsum('blhrk,bnlhk->bnlhr', compressed, key).contiguous()
    features = torch.empty((b, n, 864), device=hidden.device, dtype=hidden.dtype)
    _features[(triton.cdiv(features.numel(), 256),)](
        raw, logs, features, features.numel(), 256, enable_fp_fusion=False)
    return hidden + reader.up(F.silu(reader.down(_norm(features))))
