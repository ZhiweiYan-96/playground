import triton
import triton.language as tl

@triton.jit
def safesoftmax_kernel(
    a_ptr,
    out_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)


    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # compute sum_exp of each row
    col_ptrs = tl.arange(0, K)
    row_ptr = a_ptr + (offs_am[:, None] * K + col_ptrs[None, :])
    # mask = row_ptr < (M*K)
    # mask = row
    mask_row = offs_am < M
    mask_col = col_ptrs < K
    mask_rows = mask_row[:, None] & mask_col[None, :]
    row_val = tl.load(row_ptr, mask=mask_rows)
    # print("row_val:", row_val)
    row_max = tl.max(row_val, axis=1)
    row_exp = tl.exp(row_val-row_max[:, None])
    row_exp_sum = tl.sum(row_exp, axis=1)
    
    # compute up value
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :] )
    # todo: m dim?
    # mask = a_ptrs < (M*K)
    # mask = offs_k < K
    mask_m = offs_am < M
    mask_k = offs_k < K
    mask_tile = mask_m[:, None] & mask_k[None, :]
    # a_max = tl.max(a_ptrs, mask=mask_tile)
    a_val = tl.load(a_ptrs, mask=mask_tile)
    a_exp = tl.exp(a_val-row_max[:, None])


    # store to output
    out_val = a_exp / row_exp_sum[:, None]
    out_ptrs = out_ptr + (offs_am[:, None] * K + offs_k[None, :] )
    tl.store(out_ptrs, out_val, mask=mask_tile)
    