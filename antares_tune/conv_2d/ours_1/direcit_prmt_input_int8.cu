// GLOBALS: input0:int8[128, 56, 56, 256] -> output0:int8[8, 56, 56, 16, 16, 16]
// BACKEND: c-cuda (default)
// CONFIG: {"F___output0:D0": [-1, 1, 1, 2], "F___output0:D1": [-1, 2, 16, 1], "F___output0:D2": [-1, 1, 1, 1], "F___output0:D3": [-1, 16], "F___output0:O": [2, 3, 0, 1], "F___output0:S": 1, "F___output0:R": 0}
// COMPUTE_V1: - einstein_v2("output0[H, W, N // 16, C // 16, N % 16, C % 16] =. input0[N, H, W, C]", input_dict={"input0": {"dtype": "int8", "shape": [128, 56, 56, 256]}, "output0": {"dtype": "int8", "shape": [8, 56, 56, 16, 16, 16]}})


// ---------------------------------------------------------------------------
// LOCAL: template_op_kernel0 -- input0:int8[128, 56, 56, 256] -> output0:int8[8, 56, 56, 16, 16, 16]

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


extern "C" __global__ __launch_bounds__(16) void template_op_kernel0(char* __restrict__ input0, char* __restrict__ output0) {
  // [thread_extent] blockIdx.x = 4
  // [thread_extent] threadIdx.x = 1
  // [thread_extent] blockIdx.y = 4
  // [thread_extent] threadIdx.y = 16
  // [thread_extent] blockIdx.z = 56
  // [thread_extent] threadIdx.z = 1
  for (int vthread_s = 0; vthread_s < 16; ++vthread_s) {
    for (int vthread_s1 = 0; vthread_s1 < 16; ++vthread_s1) {
      ((output0[(((((((((int)blockIdx.x) * 25690112) + (((int)blockIdx.z) * 229376)) + (((int)blockIdx.y) * 8192)) + (vthread_s * 256)) + (((int)threadIdx.y) * 16)) + vthread_s1))]) = (input0[(((((((((int)blockIdx.y) * 25690112) + (((int)threadIdx.y) * 802816)) + (((int)blockIdx.x) * 28672)) + (((int)blockIdx.z) * 256)) + (vthread_s * 16)) + vthread_s1))]));
    }
  }
  for (int vthread_s2 = 0; vthread_s2 < 16; ++vthread_s2) {
    for (int vthread_s3 = 0; vthread_s3 < 16; ++vthread_s3) {
      ((output0[((((((((((int)blockIdx.x) * 25690112) + (((int)blockIdx.z) * 229376)) + (((int)blockIdx.y) * 8192)) + (vthread_s2 * 256)) + (((int)threadIdx.y) * 16)) + vthread_s3) + 12845056))]) = (input0[((((((((((int)blockIdx.y) * 25690112) + (((int)threadIdx.y) * 802816)) + (((int)blockIdx.x) * 28672)) + (((int)blockIdx.z) * 256)) + (vthread_s2 * 16)) + vthread_s3) + 14336))]));
    }
  }
  for (int vthread_s4 = 0; vthread_s4 < 16; ++vthread_s4) {
    for (int vthread_s5 = 0; vthread_s5 < 16; ++vthread_s5) {
      ((output0[((((((((((int)blockIdx.x) * 25690112) + (((int)blockIdx.z) * 229376)) + (((int)blockIdx.y) * 8192)) + (vthread_s4 * 256)) + (((int)threadIdx.y) * 16)) + vthread_s5) + 4096))]) = (input0[((((((((((int)blockIdx.y) * 25690112) + (((int)threadIdx.y) * 802816)) + (((int)blockIdx.x) * 28672)) + (((int)blockIdx.z) * 256)) + (vthread_s4 * 16)) + vthread_s5) + 12845056))]));
    }
  }
  for (int vthread_s6 = 0; vthread_s6 < 16; ++vthread_s6) {
    for (int vthread_s7 = 0; vthread_s7 < 16; ++vthread_s7) {
      ((output0[((((((((((int)blockIdx.x) * 25690112) + (((int)blockIdx.z) * 229376)) + (((int)blockIdx.y) * 8192)) + (vthread_s6 * 256)) + (((int)threadIdx.y) * 16)) + vthread_s7) + 12849152))]) = (input0[((((((((((int)blockIdx.y) * 25690112) + (((int)threadIdx.y) * 802816)) + (((int)blockIdx.x) * 28672)) + (((int)blockIdx.z) * 256)) + (vthread_s6 * 16)) + vthread_s7) + 12859392))]));
    }
  }
}

// Saved Perf = 1.160520e-04 sec / run; Step Produced = 685; Planned Steps = 1000;
// Antares Tuning Completed in 1000 steps.