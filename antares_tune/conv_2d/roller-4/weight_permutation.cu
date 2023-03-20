// GLOBALS: input0:float32[384, 1024] -> output0:float32[24, 64, 16, 16]
// BACKEND: c-cuda (default)
// CONFIG: {"F___output0:D0": [-1, 4, 2, 2], "F___output0:D1": [-1, 2, 32, 1], "F___output0:O": [0, 1], "F___output0:S": 4, "F___output0:R": 0}
// COMPUTE_V1: - einstein_v2("output0[M // 16, N // 16, M % 16, N % 16] =. input0[M, N]", input_dict={"input0": {"dtype": "float32", "shape": [384, 1024]}, "output0": {"dtype": "float32", "shape": [24, 64, 16, 16]}})


// ---------------------------------------------------------------------------
// LOCAL: template_op_kernel0 -- input0:float32[384, 1024] -> output0:float32[24, 64, 16, 16]

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


extern "C" __global__ __launch_bounds__(64) void template_op_kernel0(float* __restrict__ input0, float* __restrict__ output0) {
  // [thread_extent] blockIdx.x = 24
  // [thread_extent] threadIdx.x = 2
  // [thread_extent] blockIdx.y = 16
  // [thread_extent] threadIdx.y = 32
  ((output0[((((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + (((int)threadIdx.x) * 32)) + (((int)threadIdx.y) & 15)))]) = (input0[(((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]));
  ((output0[(((((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + (((int)threadIdx.x) * 32)) + (((int)threadIdx.y) & 15)) + 16))]) = (input0[((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 1024))]));
  ((output0[(((((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + (((int)threadIdx.x) * 32)) + (((int)threadIdx.y) & 15)) + 64))]) = (input0[((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 4096))]));
  ((output0[(((((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + (((int)threadIdx.x) * 32)) + (((int)threadIdx.y) & 15)) + 80))]) = (input0[((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 5120))]));
  ((output0[(((((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + (((int)threadIdx.x) * 32)) + (((int)threadIdx.y) & 15)) + 128))]) = (input0[((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 8192))]));
  ((output0[(((((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + (((int)threadIdx.x) * 32)) + (((int)threadIdx.y) & 15)) + 144))]) = (input0[((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 9216))]));
  ((output0[(((((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + (((int)threadIdx.x) * 32)) + (((int)threadIdx.y) & 15)) + 192))]) = (input0[((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 12288))]));
  ((output0[(((((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + (((int)threadIdx.x) * 32)) + (((int)threadIdx.y) & 15)) + 208))]) = (input0[((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 13312))]));
  ((output0[(((((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + (((int)threadIdx.x) * 32)) + (((int)threadIdx.y) & 15)) + 512))]) = (input0[((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]));
  ((output0[(((((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + (((int)threadIdx.x) * 32)) + (((int)threadIdx.y) & 15)) + 528))]) = (input0[((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 1056))]));
  ((output0[(((((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + (((int)threadIdx.x) * 32)) + (((int)threadIdx.y) & 15)) + 576))]) = (input0[((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 4128))]));
  ((output0[(((((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + (((int)threadIdx.x) * 32)) + (((int)threadIdx.y) & 15)) + 592))]) = (input0[((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 5152))]));
  ((output0[(((((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + (((int)threadIdx.x) * 32)) + (((int)threadIdx.y) & 15)) + 640))]) = (input0[((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 8224))]));
  ((output0[(((((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + (((int)threadIdx.x) * 32)) + (((int)threadIdx.y) & 15)) + 656))]) = (input0[((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 9248))]));
  ((output0[(((((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + (((int)threadIdx.x) * 32)) + (((int)threadIdx.y) & 15)) + 704))]) = (input0[((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 12320))]));
  ((output0[(((((((((int)blockIdx.x) * 16384) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + (((int)threadIdx.x) * 32)) + (((int)threadIdx.y) & 15)) + 720))]) = (input0[((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 2048)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 13344))]));
}

// Saved Perf = 5.031400e-06 sec / run; Step Produced = 821; Planned Steps = 1000;