// GLOBALS: input0:float32[16384, 16384] -> output0:float32[1024, 2048, 16, 8]
// BACKEND: c-cuda (default)
// CONFIG: {"F___output0:D0": [-1, 2, 1, 1], "F___output0:D1": [-1, 1, 1024, 1], "F___output0:O": [1, 0], "F___output0:S": 0, "F___output0:R": 1}
// COMPUTE_V1: - einstein_v2("output0[M // 16, N // 8, M % 16, N % 8] =. input0[M, N]", input_dict={"input0": {"dtype": "float32", "shape": [16384, 16384]}, "output0": {"dtype": "float32", "shape": [1024, 2048, 16, 8]}})


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


extern "C" __global__ __launch_bounds__(1024) void template_op_kernel0(float* __restrict__ input0, float* __restrict__ output0) {
  // [thread_extent] blockIdx.x = 8192
  // [thread_extent] threadIdx.x = 1
  // [thread_extent] blockIdx.y = 16
  // [thread_extent] threadIdx.y = 1024
  ((output0[(((((((((int)blockIdx.x) / 8) * 262144) + (((int)blockIdx.y) * 16384)) + ((((int)threadIdx.y) / 8) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.y) & 7)))]) = (input0[((((((int)blockIdx.x) * 32768) + (((int)blockIdx.y) * 1024)) + ((int)threadIdx.y)))]));
  ((output0[(((((((((((int)blockIdx.x) * 2) + 1) / 16) * 262144) + (((int)blockIdx.y) * 16384)) + ((((int)threadIdx.y) / 8) * 128)) + ((((((int)blockIdx.x) * 2) + 1) & 15) * 8)) + (((int)threadIdx.y) & 7)))]) = (input0[(((((((int)blockIdx.x) * 32768) + (((int)blockIdx.y) * 1024)) + ((int)threadIdx.y)) + 16384))]));
}

// Saved Perf = 2.526300e-03 sec / run; Step Produced = 314; Planned Steps = 1000;