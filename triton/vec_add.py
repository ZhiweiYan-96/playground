import torch

import triton
import triton.language as tl

DEVICE=triton.runtime.driver.active.get_active_torch_device()


# @triton.jit
# def add_kernel(x_ptr,
#                y_ptr,
#                output_ptr,
#                n_elements,
#                BLOCK_SIZE: tl.constexpr):
#     pid = tl.program_id(0)
#     block_start = pid * BLOCK_SIZE
#     offsets = block_start + tl.arange(0, BLOCK_SIZE)
#     mask = offsets < n_elements
#     x = tl.load(x_ptr + offsets, mask=mask)
#     y = tl.load(y_ptr + offsets, mask=mask)
#     result = x + y
#     tl.store(output_ptr + offsets, result, mask=mask)

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def add_kernel_fp8_e4m3fnuz(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               x_scale_ptr,
                y_scale_ptr,
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    pid = tl.program_id(axis=0)  
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    x_scale = tl.load(x_scale_ptr)
    y_scale = tl.load(y_scale_ptr)
    output = x.to(tl.float16) * x_scale + y.to(tl.float16) * y_scale
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape
    n_elements = x.numel()
    output = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE)
    return output

def triton_add_fp8_e4m3fnuz(x: torch.Tensor, x_scale: torch.Tensor, y: torch.Tensor, y_scale: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape
    n_elements = x.numel()
    output = torch.empty_like(x, dtype=torch.float16)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel_fp8_e4m3fnuz[grid](x, y, output, x_scale, y_scale, n_elements, BLOCK_SIZE)
    return output

def pertensor_quant(x):
    fp8_max, fp8_min = torch.finfo(torch.float8_e4m3fnuz).max, torch.finfo(torch.float8_e4m3fnuz).min
    scale = torch.max(torch.abs(x)) / torch.finfo(torch.float8_e4m3fnuz).max
    x_quant = torch.clamp(x / scale, fp8_min, fp8_max).to(torch.float8_e4m3fnuz)
    return x_quant, scale

if __name__ == "__main__":
    # breakpoint()
    # x = torch.randn(1 << 10, device=DEVICE, dtype=torch.float8_e4m3fn)
    # y = torch.randn(1 << 10, device=DEVICE, dtype=torch.float8_e4m3fn)
    # x = torch.randn(1<<10, device=DEVICE, dtype=torch.float16)
    # y = torch.randn(1<<10, device=DEVICE, dtype=torch.float16)
    x = torch.ones(1<<10, device=DEVICE, dtype=torch.float16)
    y = torch.ones(1<<10, device=DEVICE, dtype=torch.float16)
    # fp8_x, x_scale = pertensor_quant(x)
    # fp8_y, y_scale = pertensor_quant(y)

    # output = triton_add_fp8_e4m3fnuz(fp8_x, x_scale.to(DEVICE), fp8_y, y_scale.to(DEVICE))
    # print("Output:", output)
    
    output = triton_add(x, y)
    # print("Output:", output)