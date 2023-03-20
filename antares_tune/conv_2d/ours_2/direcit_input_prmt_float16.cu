// GLOBALS: input0:float16[128, 56, 56, 128] -> output0:float16[8, 56, 56, 32, 16, 16]
// BACKEND: c-cuda (default)
// CONFIG: {"F___output0:D0": [-1, 1, 1, 1], "F___output0:D1": [-1, 1, 1, 1], "F___output0:D2": [-1, 2, 32, 1], "F___output0:D3": [-1, 1], "F___output0:O": [2, 1, 0, 3], "F___output0:S": 1, "F___output0:R": 1}
// COMPUTE_V1: - einstein_v2("output0[H, W, N // 16, C // 16, N % 16, C % 16] =. input0[N, H, W, C]", input_dict={"input0": {"dtype": "float16", "shape": [128, 56, 56, 128]}, "output0": {"dtype": "float16", "shape": [8, 56, 56, 32, 16, 16]}})


// ---------------------------------------------------------------------------
// LOCAL: template_op_kernel0 -- input0:float16[128, 56, 56, 128] -> output0:float16[8, 56, 56, 32, 16, 16]

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


extern "C" __global__ __launch_bounds__(32) void template_op_kernel0(half* __restrict__ input0, half* __restrict__ output0) {
  // [thread_extent] blockIdx.x = 56
  // [thread_extent] threadIdx.x = 1
  // [thread_extent] blockIdx.y = 8
  // [thread_extent] threadIdx.y = 1
  // [thread_extent] blockIdx.z = 2
  // [thread_extent] threadIdx.z = 32
  for (int vthread_s = 0; vthread_s < 128; ++vthread_s) {
    ((output0[((((((((((int)blockIdx.y) * 25690112) + (((int)blockIdx.x) * 458752)) + ((vthread_s / 16) * 8192)) + (((int)blockIdx.z) * 1024)) + ((((int)threadIdx.z) / 16) * 256)) + ((vthread_s & 15) * 16)) + (((int)threadIdx.z) & 15)))]) = (input0[((((((vthread_s * 401408) + (((int)blockIdx.y) * 7168)) + (((int)blockIdx.x) * 128)) + (((int)blockIdx.z) * 64)) + ((int)threadIdx.z)))]));
    ((output0[(((((((((((int)blockIdx.y) * 25690112) + (((int)blockIdx.x) * 458752)) + ((vthread_s / 16) * 8192)) + (((int)blockIdx.z) * 1024)) + ((((int)threadIdx.z) / 16) * 256)) + ((vthread_s & 15) * 16)) + (((int)threadIdx.z) & 15)) + 512))]) = (input0[(((((((vthread_s * 401408) + (((int)blockIdx.y) * 7168)) + (((int)blockIdx.x) * 128)) + (((int)blockIdx.z) * 64)) + ((int)threadIdx.z)) + 32))]));
  }
}

// Saved Perf = 3.758740e-05 sec / run; Step Produced = 832; Planned Steps = 1000;
// Antares Tuning Completed in 1000 steps.