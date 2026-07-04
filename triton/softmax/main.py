import torch
import triton
import triton.language as tl



@triton.jit
def softmax_kernel(
    a_ptr,
    out_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # compute up value
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :] )
    # todo: m dim?
    # mask = a_ptrs < (M*K)
    # mask = offs_k < K
    mask_m = offs_am < M
    mask_k = offs_k < K
    mask_tile = mask_m[:, None] & mask_k[None, :]
    a_val = tl.load(a_ptrs, mask=mask_tile)
    a_exp = tl.exp(a_val)

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
    row_exp = tl.exp(row_val)
    row_exp_sum = tl.sum(row_exp, axis=1, keep_dims=True)
    

    # store to output
    out_val = a_exp / row_exp_sum
    out_ptrs = out_ptr + (offs_am[:, None] * K + offs_k[None, :] )
    tl.store(out_ptrs, out_val, mask=mask_tile)



def benchmark(
    M,
    K,
    kernel,
    grid,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    a = torch.randn([M, K], device="cuda")
    out = torch.randn_like(a)
    ref = torch.nn.functional.softmax(a, dim=-1)
    for i in range(5):
        # warmup
        kernel[grid](a, out, M, K, BLOCK_SIZE_M, BLOCK_SIZE_K)
        if not torch.allclose(out, ref, rtol=1e-2):
            print("out:", out)
            print("ref:", ref)
            raise "Kernel acc has issue"
    
    import time
    start_time = time.time()
    for i in range(5):
        kernel[grid](a, out, M, K, BLOCK_SIZE_M, BLOCK_SIZE_K)
    end_time = time.time()
    avg_time = (end_time - start_time) / 5
    print("avg time:", avg_time)
    
    

M=128
K=16
grid = lambda meta : (triton.cdiv(M, meta["BLOCK_SIZE_M"]),
                      triton.cdiv(K, meta['BLOCK_SIZE_K']))
benchmark(M, K, softmax_kernel, grid, 32, 32) 

from softmax1dblock import softmax_1dblock_kernel
benchmark(M, K, softmax_1dblock_kernel, grid, M, 1)

# from safesoftmax import safesoftmax_kernel
# benchmark(M, K, safesoftmax_kernel, grid, 32, 32)

# a = torch.randn([M, K], device="cuda")
# # print("a:", a)
# out = torch.randn_like(a)
# BLOCK_SIZE_M = 32
# BLOCK_SIZE_K = 32
# grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), 
#                      triton.cdiv(K, meta['BLOCK_SIZE_K']))
# safesoftmax_kernel[grid](a, out, M, K, BLOCK_SIZE_M, BLOCK_SIZE_K)

# ref = torch.nn.functional.softmax(a, dim=-1)

# print("Passed:", torch.allclose(ref, out, atol=1e-3, rtol=1e-3))
# print("ref:", ref, "\n res:", out)