#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

__global__ void test_mfma_kernel(float* C, const float* A, const float* B, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane_id = threadIdx.x % 64;
    int wave_id = tid / 64;
    
    if (lane_id < 32) {
        float a_val = A[wave_id];
        float b_val = B[wave_id];
        
        // 方法1: 使用 AccVGPR 数组（推荐）
        float acc[32];
        for (int i = 0; i < 32; i++) {
            acc[i] = 0.0f;
        }
        
        // 关键：告诉编译器这是一个连续的寄存器块
        asm volatile(
            "v_mfma_f32_32x32x1_2b_f32 a[0:31], %0, %1, a[0:31]"
            : 
            : "v"(a_val), "v"(b_val)
            : "a0","a1","a2","a3","a4","a5","a6","a7",
              "a8","a9","a10","a11","a12","a13","a14","a15",
              "a16","a17","a18","a19","a20","a21","a22","a23",
              "a24","a25","a26","a27","a28","a29","a30","a31"
        );
        
        // 从 AGPR 读回到 VGPR
        for (int i = 0; i < 32; i++) {
            asm volatile(
                "v_accvgpr_read_b32 %0, a%1"
                : "=v"(acc[i])
                : "n"(i)
            );
        }
        
        // 写回结果
        for (int i = 0; i < 32; i++) {
            C[wave_id * 32 + i] = acc[i];
        }
    }
}

int main() {
    const int N = 1024;
    const int NUM_WAVES = N / 64;
    
    std::vector<float> h_A(NUM_WAVES, 1.0f);
    std::vector<float> h_B(NUM_WAVES, 2.0f);
    std::vector<float> h_C(NUM_WAVES * 32, 0.0f);
    
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, NUM_WAVES * sizeof(float));
    hipMalloc(&d_B, NUM_WAVES * sizeof(float));
    hipMalloc(&d_C, NUM_WAVES * 32 * sizeof(float));
    
    hipMemcpy(d_A, h_A.data(), NUM_WAVES * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), NUM_WAVES * sizeof(float), hipMemcpyHostToDevice);
    
    hipLaunchKernelGGL(test_mfma_kernel, dim3(NUM_WAVES), dim3(64), 0, 0, 
                       d_C, d_A, d_B, N);
    
    hipDeviceSynchronize();
    
    hipMemcpy(h_C.data(), d_C, NUM_WAVES * 32 * sizeof(float), hipMemcpyDeviceToHost);
    
    std::cout << "First 10 results: ";
    for (int i = 0; i < 10; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;
    
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    
    return 0;
}