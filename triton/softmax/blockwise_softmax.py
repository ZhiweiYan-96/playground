import triton
import triton.language as tl

@triton.jit
def softmax_1dblock_kernel(
    a_ptr,
    out_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)

    m_idx = tl.arange(0, BLOCK_SIZE_M)
    m_idx =  pid_m * BLOCK_SIZE_M + m_idx

    # m_idx_2d = m_idx[:, None] + tl.arange(0, K)
    col_ptrs = tl.arange(0, K)
    row_ptr = a_ptr + (m_idx[:, None] * K + col_ptrs[None, :])
    row_mask = m_idx < M
    col_mask = col_ptrs < K
    ele_mask = row_mask[:, None] & col_mask[None, :]
    row_val = tl.load(row_ptr, mask=ele_mask)
    x_max = tl.max(row_val, axis=1, keep_dims=True)
    ex = tl.exp(row_val - x_max)
    sum_ex = tl.sum(ex, axis=1, keep_dims=True)
    out = ex/ sum_ex
    out_ptrs = out_ptr + (m_idx[:, None] * K + col_ptrs[None, :])
    tl.store(out_ptrs, out, mask=ele_mask)
    

@triton.jit
def softmax_block_kernel_1(
    a_ptr, # [M, N]
    out_ptr, # [M , N]
    sum_ptr, # [M, NUM_BLOCKS_N]
    max_ptr, # [M, NUM_BLOCKS_N]
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCKS_SIZE_N: tl.constexpr,
    NUM_BLOCKS_N: tl.constexpr,
):
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    offsets = m * N + n * BLOCK_SIZE_N +  tl.arange(0, BLOCK_SIZE_N)
    mask = n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) < N
    val = tl.load(aptr + offsets, mask=mask)
    block_max = tl.max(val)

    # exp
    exp = tl.exp(val - block_max)
    block_sum = tl.sum(exp)

    # sub_out
    block_out = exp / block_sum

    # store block_out
    tl.store(out_ptr + offsets, block_out, mask=mask)
    
    # store sub statistics
    offsets = m * NUM_BLOCKS_N + n
    mask = offsets < M * NUM_BLOCKS_N
    tl.store(sum_ptr + offsets, block_sum, mask=mask)
    tl.store(max_ptr + offsets, block_max, mask=mask)
    

@triton.jit
def softmax_block_reduce(
    sum_ptr, # [M, NUM_BLOCK_SIZE_N]
    max_ptr, # [M, NUM_BLOCK_SIZE_N]
    NUM_BLOCK_SIZE_N: tl.constexpr,
)
    m = tl.program_id(0)

    # compute overall max
    offsets = m * NUM_BLOCK_SIZE_N +  tl.arange(0, NUM_BLOCK_SIZE_N)
    mask = offsets < M * NUM_BLOCK_SIZE_N    
    max_val = tl.load(max_ptr + offsets, mask=mask)
    overall_max = tl.max(max_val)
    
    # update overall sum
    sum_val = tl.load(sum_ptr + offsets, mask=mask)
    # naming? 
    # update_sum = sum_val * tl.exp(overall_max)
    update_sum = sum_val * tl.exp(max_val - overall_max)
    overall_sum = tl.sum(overall_sum) # bug here, should write to rowwise max

    # write to element-0
    tl.store(sum_ptr, overall_sum, mask=[True])
    tl.store(max_ptr, overall_max, mask=[True])


@triton.jit
def softmax_last_kernel(
    a_ptr, # [M, N]
    out_ptr, # [M , N]
    sum_ptr, # [M, NUM_BLOCKS_N]
    max_ptr, # [M, NUM_BLOCKS_N]
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCKS_SIZE_N: tl.constexpr,
    NUM_BLOCKS_N: tl.constexpr,
):
    m = tl.program_id(0)
    n = tl.program_id(1)

    overall_sum = tl.load(sum_ptr, mask=[True])
    overall_max = tl.load(max_ptr, mask[True])

    offsets = m * N + n * BLOCK_SIZE_N +  tl.arange(0, BLOCK_SIZE_N)
    mask = n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) < N
    val = tl.load(aptr + offsets, mask=mask)

    exp = tl.exp(val-overall_max)
    out_val = exp / overall_sum

    tl.store(out_ptr + offsets, out_val, mask=mask)

    