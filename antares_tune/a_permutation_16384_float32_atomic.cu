// GLOBALS: input0:float32[16384, 16384] -> output0:float32[1024, 2048, 16, 8]
// BACKEND: c-cuda (default)
// CONFIG: {"Foutput0:D0": [-1, 1, 1, 1], "Foutput0:D1": [-1, 64, 4, 1], "Foutput0:D2": [-1, 1, 2, 2], "Foutput0:D3": [-1, 2], "Foutput0:O": [3, 0, 2, 1], "Foutput0:S": 3, "Foutput0:R": 0}
// COMPUTE_V1: - einstein_v2("output0[MM, NN, M, N] = input0[MM * 16 + M, NN * 8 + N] where MM in 1024, NN in 2048, M in 16, N in 8", input_dict={"input0": {"dtype": "float32", "shape": [16384, 16384]}, "output0": {"dtype": "float32", "shape": [1024, 2048, 16, 8]}})


// ---------------------------------------------------------------------------
// LOCAL: template_op_kernel0 -- input0:float32[16384, 16384] -> output0:float32[1024, 2048, 16, 8]

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#ifndef __CUDA_COMMON_MACRO__
#define __CUDA_COMMON_MACRO__

#if (__CUDA_ARCH__ >= 600)
__forceinline__ __device__ __half hmax(const __half &a, const __half &b) { return a > b ? a : b; }
__forceinline__ __device__ __half hmin(const __half &a, const __half &b) { return a < b ? a : b; }
#endif

#endif


extern "C" __global__ __launch_bounds__(8) void template_op_kernel0(float* __restrict__ input0, float* __restrict__ output0) {
  // [thread_extent] blockIdx.x = 1024
  // [thread_extent] threadIdx.x = 1
  // [thread_extent] blockIdx.y = 8
  // [thread_extent] threadIdx.y = 4
  // [thread_extent] blockIdx.z = 4
  // [thread_extent] threadIdx.z = 2
  for (int vthread_s = 0; vthread_s < 64; ++vthread_s) {
    output0[(((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.y) * 32768)) + (vthread_s * 512)) + (((int)threadIdx.y) * 128)) + (((int)blockIdx.z) * 32)) + (((int)threadIdx.z) * 16)))] = input0[(((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.z) * 65536)) + (((int)threadIdx.z) * 32768)) + (((int)blockIdx.y) * 2048)) + (vthread_s * 32)) + (((int)threadIdx.y) * 8)))];
    output0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.y) * 32768)) + (vthread_s * 512)) + (((int)threadIdx.y) * 128)) + (((int)blockIdx.z) * 32)) + (((int)threadIdx.z) * 16)) + 1))] = input0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.z) * 65536)) + (((int)threadIdx.z) * 32768)) + (((int)blockIdx.y) * 2048)) + (vthread_s * 32)) + (((int)threadIdx.y) * 8)) + 1))];
    output0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.y) * 32768)) + (vthread_s * 512)) + (((int)threadIdx.y) * 128)) + (((int)blockIdx.z) * 32)) + (((int)threadIdx.z) * 16)) + 2))] = input0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.z) * 65536)) + (((int)threadIdx.z) * 32768)) + (((int)blockIdx.y) * 2048)) + (vthread_s * 32)) + (((int)threadIdx.y) * 8)) + 2))];
    output0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.y) * 32768)) + (vthread_s * 512)) + (((int)threadIdx.y) * 128)) + (((int)blockIdx.z) * 32)) + (((int)threadIdx.z) * 16)) + 3))] = input0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.z) * 65536)) + (((int)threadIdx.z) * 32768)) + (((int)blockIdx.y) * 2048)) + (vthread_s * 32)) + (((int)threadIdx.y) * 8)) + 3))];
    output0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.y) * 32768)) + (vthread_s * 512)) + (((int)threadIdx.y) * 128)) + (((int)blockIdx.z) * 32)) + (((int)threadIdx.z) * 16)) + 4))] = input0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.z) * 65536)) + (((int)threadIdx.z) * 32768)) + (((int)blockIdx.y) * 2048)) + (vthread_s * 32)) + (((int)threadIdx.y) * 8)) + 4))];
    output0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.y) * 32768)) + (vthread_s * 512)) + (((int)threadIdx.y) * 128)) + (((int)blockIdx.z) * 32)) + (((int)threadIdx.z) * 16)) + 5))] = input0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.z) * 65536)) + (((int)threadIdx.z) * 32768)) + (((int)blockIdx.y) * 2048)) + (vthread_s * 32)) + (((int)threadIdx.y) * 8)) + 5))];
    output0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.y) * 32768)) + (vthread_s * 512)) + (((int)threadIdx.y) * 128)) + (((int)blockIdx.z) * 32)) + (((int)threadIdx.z) * 16)) + 6))] = input0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.z) * 65536)) + (((int)threadIdx.z) * 32768)) + (((int)blockIdx.y) * 2048)) + (vthread_s * 32)) + (((int)threadIdx.y) * 8)) + 6))];
    output0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.y) * 32768)) + (vthread_s * 512)) + (((int)threadIdx.y) * 128)) + (((int)blockIdx.z) * 32)) + (((int)threadIdx.z) * 16)) + 7))] = input0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.z) * 65536)) + (((int)threadIdx.z) * 32768)) + (((int)blockIdx.y) * 2048)) + (vthread_s * 32)) + (((int)threadIdx.y) * 8)) + 7))];
    output0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.y) * 32768)) + (vthread_s * 512)) + (((int)threadIdx.y) * 128)) + (((int)blockIdx.z) * 32)) + (((int)threadIdx.z) * 16)) + 8))] = input0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.z) * 65536)) + (((int)threadIdx.z) * 32768)) + (((int)blockIdx.y) * 2048)) + (vthread_s * 32)) + (((int)threadIdx.y) * 8)) + 16384))];
    output0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.y) * 32768)) + (vthread_s * 512)) + (((int)threadIdx.y) * 128)) + (((int)blockIdx.z) * 32)) + (((int)threadIdx.z) * 16)) + 9))] = input0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.z) * 65536)) + (((int)threadIdx.z) * 32768)) + (((int)blockIdx.y) * 2048)) + (vthread_s * 32)) + (((int)threadIdx.y) * 8)) + 16385))];
    output0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.y) * 32768)) + (vthread_s * 512)) + (((int)threadIdx.y) * 128)) + (((int)blockIdx.z) * 32)) + (((int)threadIdx.z) * 16)) + 10))] = input0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.z) * 65536)) + (((int)threadIdx.z) * 32768)) + (((int)blockIdx.y) * 2048)) + (vthread_s * 32)) + (((int)threadIdx.y) * 8)) + 16386))];
    output0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.y) * 32768)) + (vthread_s * 512)) + (((int)threadIdx.y) * 128)) + (((int)blockIdx.z) * 32)) + (((int)threadIdx.z) * 16)) + 11))] = input0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.z) * 65536)) + (((int)threadIdx.z) * 32768)) + (((int)blockIdx.y) * 2048)) + (vthread_s * 32)) + (((int)threadIdx.y) * 8)) + 16387))];
    output0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.y) * 32768)) + (vthread_s * 512)) + (((int)threadIdx.y) * 128)) + (((int)blockIdx.z) * 32)) + (((int)threadIdx.z) * 16)) + 12))] = input0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.z) * 65536)) + (((int)threadIdx.z) * 32768)) + (((int)blockIdx.y) * 2048)) + (vthread_s * 32)) + (((int)threadIdx.y) * 8)) + 16388))];
    output0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.y) * 32768)) + (vthread_s * 512)) + (((int)threadIdx.y) * 128)) + (((int)blockIdx.z) * 32)) + (((int)threadIdx.z) * 16)) + 13))] = input0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.z) * 65536)) + (((int)threadIdx.z) * 32768)) + (((int)blockIdx.y) * 2048)) + (vthread_s * 32)) + (((int)threadIdx.y) * 8)) + 16389))];
    output0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.y) * 32768)) + (vthread_s * 512)) + (((int)threadIdx.y) * 128)) + (((int)blockIdx.z) * 32)) + (((int)threadIdx.z) * 16)) + 14))] = input0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.z) * 65536)) + (((int)threadIdx.z) * 32768)) + (((int)blockIdx.y) * 2048)) + (vthread_s * 32)) + (((int)threadIdx.y) * 8)) + 16390))];
    output0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.y) * 32768)) + (vthread_s * 512)) + (((int)threadIdx.y) * 128)) + (((int)blockIdx.z) * 32)) + (((int)threadIdx.z) * 16)) + 15))] = input0[((((((((((int)blockIdx.x) * 262144) + (((int)blockIdx.z) * 65536)) + (((int)threadIdx.z) * 32768)) + (((int)blockIdx.y) * 2048)) + (vthread_s * 32)) + (((int)threadIdx.y) * 8)) + 16391))];
  }
}

// Saved Perf = 2.612440e-03 sec / run; Step Produced = 463; Planned Steps = 1000;