// GLOBALS: input0:int8[16384, 16384] -> output0:int8[1024, 1024, 16, 16]
// BACKEND: c-cuda (default)
// CONFIG: {"F___output0:D0": [-1, 2, 2, 2], "F___output0:D1": [-1, 32, 128, 1], "F___output0:O": [0, 1], "F___output0:S": 4, "F___output0:R": 1}
// COMPUTE_V1: - einstein_v2("output0[M / 16, N / 16, M % 16, N % 16] =. input0[M, N]", input_dict={"input0": {"dtype": "int8", "shape": [16384, 16384]}, "output0": {"dtype": "int8", "shape": [1024, 1024, 16, 16]}})


// ---------------------------------------------------------------------------
// LOCAL: template_op_kernel0 -- input0:int8[16384, 16384] -> output0:int8[1024, 1024, 16, 16]

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


extern "C" __global__ __launch_bounds__(256) void template_op_kernel0(char* __restrict__ input0, char* __restrict__ output0) {
  // [thread_extent] blockIdx.x = 2048
  // [thread_extent] threadIdx.x = 2
  // [thread_extent] blockIdx.y = 4
  // [thread_extent] threadIdx.y = 128
  for (int vthread_s = 0; vthread_s < 32; ++vthread_s) {
    ((output0[((((((((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) * 2)) / 16) * 262144) + (((int)blockIdx.y) * 65536)) + (vthread_s * 2048)) + ((((int)threadIdx.y) / 16) * 256)) + ((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) * 2)) & 15) * 16)) + (((int)threadIdx.y) & 15)))]) = (input0[((((((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 32768)) + (((int)blockIdx.y) * 4096)) + (vthread_s * 128)) + ((int)threadIdx.y)))]));
    ((output0[(((((((((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) * 2)) + 1) / 16) * 262144) + (((int)blockIdx.y) * 65536)) + (vthread_s * 2048)) + ((((int)threadIdx.y) / 16) * 256)) + (((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) * 2)) + 1) & 15) * 16)) + (((int)threadIdx.y) & 15)))]) = (input0[(((((((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 32768)) + (((int)blockIdx.y) * 4096)) + (vthread_s * 128)) + ((int)threadIdx.y)) + 16384))]));
    ((output0[(((((((((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) * 2)) + 4) / 16) * 262144) + (((int)blockIdx.y) * 65536)) + (vthread_s * 2048)) + ((((int)threadIdx.y) / 16) * 256)) + (((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) * 2)) + 4) & 15) * 16)) + (((int)threadIdx.y) & 15)))]) = (input0[(((((((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 32768)) + (((int)blockIdx.y) * 4096)) + (vthread_s * 128)) + ((int)threadIdx.y)) + 65536))]));
    ((output0[(((((((((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) * 2)) + 5) / 16) * 262144) + (((int)blockIdx.y) * 65536)) + (vthread_s * 2048)) + ((((int)threadIdx.y) / 16) * 256)) + (((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) * 2)) + 5) & 15) * 16)) + (((int)threadIdx.y) & 15)))]) = (input0[(((((((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 32768)) + (((int)blockIdx.y) * 4096)) + (vthread_s * 128)) + ((int)threadIdx.y)) + 81920))]));
  }
}

// Saved Perf = 7.146940e-04 sec / run; Step Produced = 30; Planned Steps = 1000;