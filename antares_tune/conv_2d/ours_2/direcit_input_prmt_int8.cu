// GLOBALS: input0:int8[128, 56, 56, 128] -> output0:int8[8, 56, 56, 32, 16, 16]
// BACKEND: c-cuda (default)
// CONFIG: {"F___output0:D0": [-1, 1, 16, 1], "F___output0:D1": [-1, 1, 1, 1], "F___output0:D2": [-1, 4, 2, 1], "F___output0:D3": [-1, 2], "F___output0:O": [3, 0, 2, 1], "F___output0:S": 2, "F___output0:R": 0}
// COMPUTE_V1: - einstein_v2("output0[H, W, N // 16, C // 16, N % 16, C % 16] =. input0[N, H, W, C]", input_dict={"input0": {"dtype": "int8", "shape": [128, 56, 56, 128]}, "output0": {"dtype": "int8", "shape": [8, 56, 56, 32, 16, 16]}})


// ---------------------------------------------------------------------------
// LOCAL: template_op_kernel0 -- input0:int8[128, 56, 56, 128] -> output0:int8[8, 56, 56, 32, 16, 16]

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


extern "C" __global__ __launch_bounds__(32) void template_op_kernel0(char* __restrict__ input0, char* __restrict__ output0) {
  // [thread_extent] blockIdx.x = 8
  // [thread_extent] threadIdx.x = 16
  // [thread_extent] blockIdx.y = 8
  // [thread_extent] threadIdx.y = 1
  // [thread_extent] blockIdx.z = 7
  // [thread_extent] threadIdx.z = 2
  for (int vthread_s = 0; vthread_s < 64; ++vthread_s) {
    ((output0[((((((((((int)blockIdx.y) * 25690112) + (((int)blockIdx.z) * 3670016)) + (((int)threadIdx.z) * 458752)) + ((vthread_s / 8) * 8192)) + (((int)blockIdx.x) * 256)) + ((vthread_s & 7) * 32)) + ((int)threadIdx.x)))]) = (input0[(((((((vthread_s * 802816) + (((int)blockIdx.y) * 7168)) + (((int)blockIdx.z) * 1024)) + (((int)threadIdx.z) * 128)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)))]));
    ((output0[((((((((((int)blockIdx.y) * 25690112) + (((int)blockIdx.z) * 3670016)) + (((int)threadIdx.z) * 458752)) + ((((vthread_s * 2) + 1) / 16) * 8192)) + (((int)blockIdx.x) * 256)) + ((((vthread_s * 2) + 1) & 15) * 16)) + ((int)threadIdx.x)))]) = (input0[((((((((vthread_s * 802816) + (((int)blockIdx.y) * 7168)) + (((int)blockIdx.z) * 1024)) + (((int)threadIdx.z) * 128)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 401408))]));
  }
  for (int vthread_s1 = 0; vthread_s1 < 64; ++vthread_s1) {
    ((output0[(((((((((((int)blockIdx.y) * 25690112) + (((int)blockIdx.z) * 3670016)) + (((int)threadIdx.z) * 458752)) + ((vthread_s1 / 8) * 8192)) + (((int)blockIdx.x) * 256)) + ((vthread_s1 & 7) * 32)) + ((int)threadIdx.x)) + 917504))]) = (input0[((((((((vthread_s1 * 802816) + (((int)blockIdx.y) * 7168)) + (((int)blockIdx.z) * 1024)) + (((int)threadIdx.z) * 128)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 256))]));
    ((output0[(((((((((((int)blockIdx.y) * 25690112) + (((int)blockIdx.z) * 3670016)) + (((int)threadIdx.z) * 458752)) + ((((vthread_s1 * 2) + 1) / 16) * 8192)) + (((int)blockIdx.x) * 256)) + ((((vthread_s1 * 2) + 1) & 15) * 16)) + ((int)threadIdx.x)) + 917504))]) = (input0[((((((((vthread_s1 * 802816) + (((int)blockIdx.y) * 7168)) + (((int)blockIdx.z) * 1024)) + (((int)threadIdx.z) * 128)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 401664))]));
  }
  for (int vthread_s2 = 0; vthread_s2 < 64; ++vthread_s2) {
    ((output0[(((((((((((int)blockIdx.y) * 25690112) + (((int)blockIdx.z) * 3670016)) + (((int)threadIdx.z) * 458752)) + ((vthread_s2 / 8) * 8192)) + (((int)blockIdx.x) * 256)) + ((vthread_s2 & 7) * 32)) + ((int)threadIdx.x)) + 1835008))]) = (input0[((((((((vthread_s2 * 802816) + (((int)blockIdx.y) * 7168)) + (((int)blockIdx.z) * 1024)) + (((int)threadIdx.z) * 128)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 512))]));
    ((output0[(((((((((((int)blockIdx.y) * 25690112) + (((int)blockIdx.z) * 3670016)) + (((int)threadIdx.z) * 458752)) + ((((vthread_s2 * 2) + 1) / 16) * 8192)) + (((int)blockIdx.x) * 256)) + ((((vthread_s2 * 2) + 1) & 15) * 16)) + ((int)threadIdx.x)) + 1835008))]) = (input0[((((((((vthread_s2 * 802816) + (((int)blockIdx.y) * 7168)) + (((int)blockIdx.z) * 1024)) + (((int)threadIdx.z) * 128)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 401920))]));
  }
  for (int vthread_s3 = 0; vthread_s3 < 64; ++vthread_s3) {
    ((output0[(((((((((((int)blockIdx.y) * 25690112) + (((int)blockIdx.z) * 3670016)) + (((int)threadIdx.z) * 458752)) + ((vthread_s3 / 8) * 8192)) + (((int)blockIdx.x) * 256)) + ((vthread_s3 & 7) * 32)) + ((int)threadIdx.x)) + 2752512))]) = (input0[((((((((vthread_s3 * 802816) + (((int)blockIdx.y) * 7168)) + (((int)blockIdx.z) * 1024)) + (((int)threadIdx.z) * 128)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 768))]));
    ((output0[(((((((((((int)blockIdx.y) * 25690112) + (((int)blockIdx.z) * 3670016)) + (((int)threadIdx.z) * 458752)) + ((((vthread_s3 * 2) + 1) / 16) * 8192)) + (((int)blockIdx.x) * 256)) + ((((vthread_s3 * 2) + 1) & 15) * 16)) + ((int)threadIdx.x)) + 2752512))]) = (input0[((((((((vthread_s3 * 802816) + (((int)blockIdx.y) * 7168)) + (((int)blockIdx.z) * 1024)) + (((int)threadIdx.z) * 128)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 402176))]));
  }
}

// Saved Perf = 2.077610e-05 sec / run; Step Produced = 996; Planned Steps = 1000;
// Antares Tuning Completed in 1000 steps.