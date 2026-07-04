import torch
import triton
import triton.language as tl


@triton.jit
def test_atomic_add_kernel(a_ptr, out_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    ptr = a_ptr + offs
    val = tl.load(ptr, mask=mask, other=0.0)

    # atomic_add returns the old value before accumulation.
    old = tl.atomic_add(ptr, val, mask=mask)
    tl.store(out_ptr + offs, old, mask=mask)


@triton.jit
def reduce_atomic_add_1st(a_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    stride = 1

    while stride <= n_elements / 2:
        mask = (offsets < n_elements) & (offsets % (2 * stride) == 0)

        offsets_adj = offsets + stride
        mask_1 = (offsets_adj < n_elements) & (offsets % (2 * stride) == 0)
    
        ptr_a = a_ptr + offsets
        ptr_b = a_ptr + offsets_adj
        val_b = tl.load(ptr_b, mask=mask_1)
        # zero = tl.zeros([BLOCK], tl.float32)
        # tl.store(ptr_b, zero, mask=mask_1)
        tl.atomic_add(ptr_a, val_b, mask=mask)
        stride = stride * 2


def main():
    device = "cuda"
    n = 128
    block = 128

    x = torch.arange(0, n, device=device, dtype=torch.float32)
    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    test_atomic_add_kernel[grid](x, out, n_elements=n, BLOCK=block)

    print("after atomic_add, x:", x)
    print("returned old values, out:", out)

def main1():
    device = "cuda"
    n = 128
    block=128
    x = torch.arange(0, n, device=device, dtype=torch.float32)
    print("input x:", x)
    print("ref:", x.sum())

    
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)

    reduce_atomic_add_1st[grid](x, n_elements=n, BLOCK=block)

    print("result:", x)



if __name__ == "__main__":
    main1()
