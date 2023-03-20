// GLOBALS: input0:float16[16384, 16384] -> output0:float16[1024, 1024, 16, 16]
// BACKEND: c-cuda (default)
// CONFIG: {"F___output0:D0": [-1, 16, 8, 4], "F___output0:D1": [-1, 2, 1, 1], "F___output0:O": [1, 0], "F___output0:S": 1, "F___output0:R": 1}
// COMPUTE_V1: - einstein_v2("output0[M / 16, N / 16, M % 16, N % 16] =. input0[M, N]", input_dict={"input0": {"dtype": "float16", "shape": [16384, 16384]}, "output0": {"dtype": "float16", "shape": [1024, 1024, 16, 16]}})


// ---------------------------------------------------------------------------
// LOCAL: template_op_kernel0 -- input0:float16[16384, 16384] -> output0:float16[1024, 1024, 16, 16]

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


extern "C" __global__ __launch_bounds__(8) void template_op_kernel0(half* __restrict__ input0, half* __restrict__ output0) {
  // [thread_extent] blockIdx.x = 32
  // [thread_extent] threadIdx.x = 8
  // [thread_extent] blockIdx.y = 8192
  // [thread_extent] threadIdx.y = 1
  for (int vthread_s = 0; vthread_s < 16; ++vthread_s) {
    ((output0[((((((((((int)blockIdx.y) / 8) * 262144) + (((int)blockIdx.x) * 8192)) + (vthread_s * 512)) + ((((int)threadIdx.x) / 4) * 256)) + ((((int)blockIdx.y) & 7) * 32)) + ((((int)threadIdx.x) & 3) * 4)))]) = (input0[(((((((int)blockIdx.y) * 32768) + (((int)blockIdx.x) * 512)) + (vthread_s * 32)) + (((int)threadIdx.x) * 4)))]));
    ((output0[((((((((((((int)blockIdx.y) * 2) + 1) / 16) * 262144) + (((int)blockIdx.x) * 8192)) + (vthread_s * 512)) + ((((int)threadIdx.x) / 4) * 256)) + ((((((int)blockIdx.y) * 2) + 1) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)))]) = (input0[((((((((int)blockIdx.y) * 32768) + (((int)blockIdx.x) * 512)) + (vthread_s * 32)) + (((int)threadIdx.x) * 4)) + 16384))]));
    ((output0[((((((((((int)blockIdx.y) / 8) * 262144) + (((int)blockIdx.x) * 8192)) + (vthread_s * 512)) + ((((((int)threadIdx.x) * 4) + 1) / 16) * 256)) + ((((int)blockIdx.y) & 7) * 32)) + (((((int)threadIdx.x) * 4) + 1) & 15)))]) = (input0[((((((((int)blockIdx.y) * 32768) + (((int)blockIdx.x) * 512)) + (vthread_s * 32)) + (((int)threadIdx.x) * 4)) + 1))]));
    ((output0[((((((((((((int)blockIdx.y) * 2) + 1) / 16) * 262144) + (((int)blockIdx.x) * 8192)) + (vthread_s * 512)) + ((((((int)threadIdx.x) * 4) + 1) / 16) * 256)) + ((((((int)blockIdx.y) * 2) + 1) & 15) * 16)) + (((((int)threadIdx.x) * 4) + 1) & 15)))]) = (input0[((((((((int)blockIdx.y) * 32768) + (((int)blockIdx.x) * 512)) + (vthread_s * 32)) + (((int)threadIdx.x) * 4)) + 16385))]));
    ((output0[((((((((((int)blockIdx.y) / 8) * 262144) + (((int)blockIdx.x) * 8192)) + (vthread_s * 512)) + ((((((int)threadIdx.x) * 4) + 2) / 16) * 256)) + ((((int)blockIdx.y) & 7) * 32)) + (((((int)threadIdx.x) * 4) + 2) & 15)))]) = (input0[((((((((int)blockIdx.y) * 32768) + (((int)blockIdx.x) * 512)) + (vthread_s * 32)) + (((int)threadIdx.x) * 4)) + 2))]));
    ((output0[((((((((((((int)blockIdx.y) * 2) + 1) / 16) * 262144) + (((int)blockIdx.x) * 8192)) + (vthread_s * 512)) + ((((((int)threadIdx.x) * 4) + 2) / 16) * 256)) + ((((((int)blockIdx.y) * 2) + 1) & 15) * 16)) + (((((int)threadIdx.x) * 4) + 2) & 15)))]) = (input0[((((((((int)blockIdx.y) * 32768) + (((int)blockIdx.x) * 512)) + (vthread_s * 32)) + (((int)threadIdx.x) * 4)) + 16386))]));
    ((output0[((((((((((int)blockIdx.y) / 8) * 262144) + (((int)blockIdx.x) * 8192)) + (vthread_s * 512)) + ((((((int)threadIdx.x) * 4) + 3) / 16) * 256)) + ((((int)blockIdx.y) & 7) * 32)) + (((((int)threadIdx.x) * 4) + 3) & 15)))]) = (input0[((((((((int)blockIdx.y) * 32768) + (((int)blockIdx.x) * 512)) + (vthread_s * 32)) + (((int)threadIdx.x) * 4)) + 3))]));
    ((output0[((((((((((((int)blockIdx.y) * 2) + 1) / 16) * 262144) + (((int)blockIdx.x) * 8192)) + (vthread_s * 512)) + ((((((int)threadIdx.x) * 4) + 3) / 16) * 256)) + ((((((int)blockIdx.y) * 2) + 1) & 15) * 16)) + (((((int)threadIdx.x) * 4) + 3) & 15)))]) = (input0[((((((((int)blockIdx.y) * 32768) + (((int)blockIdx.x) * 512)) + (vthread_s * 32)) + (((int)threadIdx.x) * 4)) + 16387))]));
  }
}

// Saved Perf = 1.256140e-03 sec / run; Step Produced = 473; Planned Steps = 1000;