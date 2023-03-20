// GLOBALS: input0:int8[128, 42, 42, 1024] -> output0:int8[42, 42, 8, 24, 16, 16]
// BACKEND: c-cuda (default)
// CONFIG: {"F___output0:D0": [-1, 1, 1, 1], "F___output0:D1": [-1, 1, 1, 1], "F___output0:D2": [-1, 1, 8, 1], "F___output0:D3": [-1, 32], "F___output0:O": [3, 0, 1, 2], "F___output0:S": 1, "F___output0:R": 1}
// COMPUTE_V1: - einstein_v2("output0[H, W, N // 16, C // 16, N % 16, C % 16] =. input0[N, H, W, C]", input_dict={"input0": {"dtype": "int8", "shape": [128, 42, 42, 1024]}, "output0": {"dtype": "int8", "shape": [42, 42, 8, 24, 16, 16]}})


// ---------------------------------------------------------------------------
// LOCAL: template_op_kernel0 -- input0:int8[128, 42, 42, 1024] -> output0:int8[42, 42, 8, 24, 16, 16]

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


extern "C" __global__ __launch_bounds__(8) void template_op_kernel0(char* __restrict__ input0, char* __restrict__ output0) {
  // [thread_extent] blockIdx.x = 42
  // [thread_extent] threadIdx.x = 1
  // [thread_extent] blockIdx.y = 42
  // [thread_extent] threadIdx.y = 1
  // [thread_extent] blockIdx.z = 16
  // [thread_extent] threadIdx.z = 8
  for (int vthread_s = 0; vthread_s < 32; ++vthread_s) {
    for (int vthread_s1 = 0; vthread_s1 < 32; ++vthread_s1) {
      ((output0[((((((((((int)blockIdx.y) * 2064384) + (((int)blockIdx.x) * 49152)) + ((((((int)blockIdx.z) * 8) + ((int)threadIdx.z)) / 16) * 6144)) + (vthread_s * 512)) + ((vthread_s1 / 16) * 256)) + ((((((int)blockIdx.z) * 8) + ((int)threadIdx.z)) & 15) * 16)) + (vthread_s1 & 15)))]) = (input0[(((((((((int)blockIdx.z) * 14450688) + (((int)threadIdx.z) * 1806336)) + (((int)blockIdx.y) * 43008)) + (((int)blockIdx.x) * 1024)) + (vthread_s * 32)) + vthread_s1))]));
    }
  }
}

// Saved Perf = 1.732690e-03 sec / run; Step Produced = 777; Planned Steps = 1000;