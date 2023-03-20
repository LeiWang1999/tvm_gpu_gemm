// GLOBALS: input0:int8[18966528, 25], input1:int8[1, 25] -> output0:int32[18966528, 25]
// BACKEND: c-cuda (default)
// CONFIG: {"Moutput0T": 0, "Moutput0:D0": [-1, 1, 128, 6], "Moutput0:D1": [-1, 5, 1, 5], "Moutput0:R0": [-1, 1, 5], "Moutput0:RA": 0, "Moutput0:AL0": 1, "Moutput0:AL1": 1, "Moutput0:S": 2, "Moutput0:U": 1}
// COMPUTE_V1: - einstein_v2("output0[N, M] +=! input0[N, K].cast(`int32`) * input1[K, M].cast(`int32`)", { "input0": {"dtype": "int8", "shape": [18966528, 25]}, "input1": {"dtype": "int8", "shape": [1, 25]}})


// ---------------------------------------------------------------------------
// LOCAL: template_op_kernel0 -- input0:int8[18966528, 25], input1:int8[1, 25] -> output0:int32[18966528, 25]

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


extern "C" __global__ __launch_bounds__(128) void template_op_kernel0(char* __restrict__ input0, char* __restrict__ input1, int* __restrict__ output0) {
  // [thread_extent] blockIdx.x = 24696
  // [thread_extent] threadIdx.x = 128
  int output0_local[150];
  for (int N_c_inner_init = 0; N_c_inner_init < 6; ++N_c_inner_init) {
    output0_local[((N_c_inner_init * 5))] = 0;
    output0_local[(((N_c_inner_init * 5) + 30))] = 0;
    output0_local[(((N_c_inner_init * 5) + 60))] = 0;
    output0_local[(((N_c_inner_init * 5) + 90))] = 0;
    output0_local[(((N_c_inner_init * 5) + 120))] = 0;
    output0_local[(((N_c_inner_init * 5) + 1))] = 0;
    output0_local[(((N_c_inner_init * 5) + 31))] = 0;
    output0_local[(((N_c_inner_init * 5) + 61))] = 0;
    output0_local[(((N_c_inner_init * 5) + 91))] = 0;
    output0_local[(((N_c_inner_init * 5) + 121))] = 0;
    output0_local[(((N_c_inner_init * 5) + 2))] = 0;
    output0_local[(((N_c_inner_init * 5) + 32))] = 0;
    output0_local[(((N_c_inner_init * 5) + 62))] = 0;
    output0_local[(((N_c_inner_init * 5) + 92))] = 0;
    output0_local[(((N_c_inner_init * 5) + 122))] = 0;
    output0_local[(((N_c_inner_init * 5) + 3))] = 0;
    output0_local[(((N_c_inner_init * 5) + 33))] = 0;
    output0_local[(((N_c_inner_init * 5) + 63))] = 0;
    output0_local[(((N_c_inner_init * 5) + 93))] = 0;
    output0_local[(((N_c_inner_init * 5) + 123))] = 0;
    output0_local[(((N_c_inner_init * 5) + 4))] = 0;
    output0_local[(((N_c_inner_init * 5) + 34))] = 0;
    output0_local[(((N_c_inner_init * 5) + 64))] = 0;
    output0_local[(((N_c_inner_init * 5) + 94))] = 0;
    output0_local[(((N_c_inner_init * 5) + 124))] = 0;
  }
  for (int K_outer_outer = 0; K_outer_outer < 5; ++K_outer_outer) {
    __shared__ char input0_shared[3840];
  // [thread_extent] threadIdx.x = 128
    __syncthreads();
    input0_shared[(((int)threadIdx.x))] = input0[(((((((int)blockIdx.x) * 19200) + ((((int)threadIdx.x) / 5) * 25)) + (K_outer_outer * 5)) + (((int)threadIdx.x) % 5)))];
    input0_shared[((((int)threadIdx.x) + 128))] = input0[(((((((int)blockIdx.x) * 19200) + (((((int)threadIdx.x) + 128) / 5) * 25)) + (K_outer_outer * 5)) + ((((int)threadIdx.x) + 3) % 5)))];
    input0_shared[((((int)threadIdx.x) + 256))] = input0[(((((((int)blockIdx.x) * 19200) + (((((int)threadIdx.x) + 256) / 5) * 25)) + (K_outer_outer * 5)) + ((((int)threadIdx.x) + 1) % 5)))];
    input0_shared[((((int)threadIdx.x) + 384))] = input0[(((((((int)blockIdx.x) * 19200) + (((((int)threadIdx.x) + 384) / 5) * 25)) + (K_outer_outer * 5)) + ((((int)threadIdx.x) + 4) % 5)))];
    input0_shared[((((int)threadIdx.x) + 512))] = input0[(((((((int)blockIdx.x) * 19200) + (((((int)threadIdx.x) + 512) / 5) * 25)) + (K_outer_outer * 5)) + ((((int)threadIdx.x) + 2) % 5)))];
    input0_shared[((((int)threadIdx.x) + 640))] = input0[((((((((int)blockIdx.x) * 19200) + ((((int)threadIdx.x) / 5) * 25)) + (K_outer_outer * 5)) + (((int)threadIdx.x) % 5)) + 3200))];
    input0_shared[((((int)threadIdx.x) + 768))] = input0[(((((((int)blockIdx.x) * 19200) + (((((int)threadIdx.x) + 768) / 5) * 25)) + (K_outer_outer * 5)) + ((((int)threadIdx.x) + 3) % 5)))];
    input0_shared[((((int)threadIdx.x) + 896))] = input0[(((((((int)blockIdx.x) * 19200) + (((((int)threadIdx.x) + 896) / 5) * 25)) + (K_outer_outer * 5)) + ((((int)threadIdx.x) + 1) % 5)))];
    input0_shared[((((int)threadIdx.x) + 1024))] = input0[(((((((int)blockIdx.x) * 19200) + (((((int)threadIdx.x) + 1024) / 5) * 25)) + (K_outer_outer * 5)) + ((((int)threadIdx.x) + 4) % 5)))];
    input0_shared[((((int)threadIdx.x) + 1152))] = input0[(((((((int)blockIdx.x) * 19200) + (((((int)threadIdx.x) + 1152) / 5) * 25)) + (K_outer_outer * 5)) + ((((int)threadIdx.x) + 2) % 5)))];
    input0_shared[((((int)threadIdx.x) + 1280))] = input0[((((((((int)blockIdx.x) * 19200) + ((((int)threadIdx.x) / 5) * 25)) + (K_outer_outer * 5)) + (((int)threadIdx.x) % 5)) + 6400))];
    input0_shared[((((int)threadIdx.x) + 1408))] = input0[(((((((int)blockIdx.x) * 19200) + (((((int)threadIdx.x) + 1408) / 5) * 25)) + (K_outer_outer * 5)) + ((((int)threadIdx.x) + 3) % 5)))];
    input0_shared[((((int)threadIdx.x) + 1536))] = input0[(((((((int)blockIdx.x) * 19200) + (((((int)threadIdx.x) + 1536) / 5) * 25)) + (K_outer_outer * 5)) + ((((int)threadIdx.x) + 1) % 5)))];
    input0_shared[((((int)threadIdx.x) + 1664))] = input0[(((((((int)blockIdx.x) * 19200) + (((((int)threadIdx.x) + 1664) / 5) * 25)) + (K_outer_outer * 5)) + ((((int)threadIdx.x) + 4) % 5)))];
    input0_shared[((((int)threadIdx.x) + 1792))] = input0[(((((((int)blockIdx.x) * 19200) + (((((int)threadIdx.x) + 1792) / 5) * 25)) + (K_outer_outer * 5)) + ((((int)threadIdx.x) + 2) % 5)))];
    input0_shared[((((int)threadIdx.x) + 1920))] = input0[((((((((int)blockIdx.x) * 19200) + ((((int)threadIdx.x) / 5) * 25)) + (K_outer_outer * 5)) + (((int)threadIdx.x) % 5)) + 9600))];
    input0_shared[((((int)threadIdx.x) + 2048))] = input0[(((((((int)blockIdx.x) * 19200) + (((((int)threadIdx.x) + 2048) / 5) * 25)) + (K_outer_outer * 5)) + ((((int)threadIdx.x) + 3) % 5)))];
    input0_shared[((((int)threadIdx.x) + 2176))] = input0[(((((((int)blockIdx.x) * 19200) + (((((int)threadIdx.x) + 2176) / 5) * 25)) + (K_outer_outer * 5)) + ((((int)threadIdx.x) + 1) % 5)))];
    input0_shared[((((int)threadIdx.x) + 2304))] = input0[(((((((int)blockIdx.x) * 19200) + (((((int)threadIdx.x) + 2304) / 5) * 25)) + (K_outer_outer * 5)) + ((((int)threadIdx.x) + 4) % 5)))];
    input0_shared[((((int)threadIdx.x) + 2432))] = input0[(((((((int)blockIdx.x) * 19200) + (((((int)threadIdx.x) + 2432) / 5) * 25)) + (K_outer_outer * 5)) + ((((int)threadIdx.x) + 2) % 5)))];
    input0_shared[((((int)threadIdx.x) + 2560))] = input0[((((((((int)blockIdx.x) * 19200) + ((((int)threadIdx.x) / 5) * 25)) + (K_outer_outer * 5)) + (((int)threadIdx.x) % 5)) + 12800))];
    input0_shared[((((int)threadIdx.x) + 2688))] = input0[(((((((int)blockIdx.x) * 19200) + (((((int)threadIdx.x) + 2688) / 5) * 25)) + (K_outer_outer * 5)) + ((((int)threadIdx.x) + 3) % 5)))];
    input0_shared[((((int)threadIdx.x) + 2816))] = input0[(((((((int)blockIdx.x) * 19200) + (((((int)threadIdx.x) + 2816) / 5) * 25)) + (K_outer_outer * 5)) + ((((int)threadIdx.x) + 1) % 5)))];
    input0_shared[((((int)threadIdx.x) + 2944))] = input0[(((((((int)blockIdx.x) * 19200) + (((((int)threadIdx.x) + 2944) / 5) * 25)) + (K_outer_outer * 5)) + ((((int)threadIdx.x) + 4) % 5)))];
    input0_shared[((((int)threadIdx.x) + 3072))] = input0[(((((((int)blockIdx.x) * 19200) + (((((int)threadIdx.x) + 3072) / 5) * 25)) + (K_outer_outer * 5)) + ((((int)threadIdx.x) + 2) % 5)))];
    input0_shared[((((int)threadIdx.x) + 3200))] = input0[((((((((int)blockIdx.x) * 19200) + ((((int)threadIdx.x) / 5) * 25)) + (K_outer_outer * 5)) + (((int)threadIdx.x) % 5)) + 16000))];
    input0_shared[((((int)threadIdx.x) + 3328))] = input0[(((((((int)blockIdx.x) * 19200) + (((((int)threadIdx.x) + 3328) / 5) * 25)) + (K_outer_outer * 5)) + ((((int)threadIdx.x) + 3) % 5)))];
    input0_shared[((((int)threadIdx.x) + 3456))] = input0[(((((((int)blockIdx.x) * 19200) + (((((int)threadIdx.x) + 3456) / 5) * 25)) + (K_outer_outer * 5)) + ((((int)threadIdx.x) + 1) % 5)))];
    input0_shared[((((int)threadIdx.x) + 3584))] = input0[(((((((int)blockIdx.x) * 19200) + (((((int)threadIdx.x) + 3584) / 5) * 25)) + (K_outer_outer * 5)) + ((((int)threadIdx.x) + 4) % 5)))];
    input0_shared[((((int)threadIdx.x) + 3712))] = input0[(((((((int)blockIdx.x) * 19200) + (((((int)threadIdx.x) + 3712) / 5) * 25)) + (K_outer_outer * 5)) + ((((int)threadIdx.x) + 2) % 5)))];
    __shared__ char input1_shared[125];
  // [thread_extent] threadIdx.x = 128
    if (((int)threadIdx.x) < 125) {
      if (((K_outer_outer * 5) + (((int)threadIdx.x) / 25)) < 1) {
        input1_shared[(((int)threadIdx.x))] = input1[(((K_outer_outer * 125) + ((int)threadIdx.x)))];
      }
    }
    __syncthreads();
    for (int K_inner = 0; K_inner < 5; ++K_inner) {
      for (int N_c_inner = 0; N_c_inner < 6; ++N_c_inner) {
        output0_local[((N_c_inner * 5))] = (output0_local[((N_c_inner * 5))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[((K_inner * 25))])));
        output0_local[(((N_c_inner * 5) + 30))] = (output0_local[(((N_c_inner * 5) + 30))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[(((K_inner * 25) + 5))])));
        output0_local[(((N_c_inner * 5) + 60))] = (output0_local[(((N_c_inner * 5) + 60))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[(((K_inner * 25) + 10))])));
        output0_local[(((N_c_inner * 5) + 90))] = (output0_local[(((N_c_inner * 5) + 90))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[(((K_inner * 25) + 15))])));
        output0_local[(((N_c_inner * 5) + 120))] = (output0_local[(((N_c_inner * 5) + 120))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[(((K_inner * 25) + 20))])));
        output0_local[(((N_c_inner * 5) + 1))] = (output0_local[(((N_c_inner * 5) + 1))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[(((K_inner * 25) + 1))])));
        output0_local[(((N_c_inner * 5) + 31))] = (output0_local[(((N_c_inner * 5) + 31))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[(((K_inner * 25) + 6))])));
        output0_local[(((N_c_inner * 5) + 61))] = (output0_local[(((N_c_inner * 5) + 61))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[(((K_inner * 25) + 11))])));
        output0_local[(((N_c_inner * 5) + 91))] = (output0_local[(((N_c_inner * 5) + 91))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[(((K_inner * 25) + 16))])));
        output0_local[(((N_c_inner * 5) + 121))] = (output0_local[(((N_c_inner * 5) + 121))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[(((K_inner * 25) + 21))])));
        output0_local[(((N_c_inner * 5) + 2))] = (output0_local[(((N_c_inner * 5) + 2))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[(((K_inner * 25) + 2))])));
        output0_local[(((N_c_inner * 5) + 32))] = (output0_local[(((N_c_inner * 5) + 32))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[(((K_inner * 25) + 7))])));
        output0_local[(((N_c_inner * 5) + 62))] = (output0_local[(((N_c_inner * 5) + 62))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[(((K_inner * 25) + 12))])));
        output0_local[(((N_c_inner * 5) + 92))] = (output0_local[(((N_c_inner * 5) + 92))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[(((K_inner * 25) + 17))])));
        output0_local[(((N_c_inner * 5) + 122))] = (output0_local[(((N_c_inner * 5) + 122))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[(((K_inner * 25) + 22))])));
        output0_local[(((N_c_inner * 5) + 3))] = (output0_local[(((N_c_inner * 5) + 3))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[(((K_inner * 25) + 3))])));
        output0_local[(((N_c_inner * 5) + 33))] = (output0_local[(((N_c_inner * 5) + 33))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[(((K_inner * 25) + 8))])));
        output0_local[(((N_c_inner * 5) + 63))] = (output0_local[(((N_c_inner * 5) + 63))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[(((K_inner * 25) + 13))])));
        output0_local[(((N_c_inner * 5) + 93))] = (output0_local[(((N_c_inner * 5) + 93))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[(((K_inner * 25) + 18))])));
        output0_local[(((N_c_inner * 5) + 123))] = (output0_local[(((N_c_inner * 5) + 123))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[(((K_inner * 25) + 23))])));
        output0_local[(((N_c_inner * 5) + 4))] = (output0_local[(((N_c_inner * 5) + 4))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[(((K_inner * 25) + 4))])));
        output0_local[(((N_c_inner * 5) + 34))] = (output0_local[(((N_c_inner * 5) + 34))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[(((K_inner * 25) + 9))])));
        output0_local[(((N_c_inner * 5) + 64))] = (output0_local[(((N_c_inner * 5) + 64))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[(((K_inner * 25) + 14))])));
        output0_local[(((N_c_inner * 5) + 94))] = (output0_local[(((N_c_inner * 5) + 94))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[(((K_inner * 25) + 19))])));
        output0_local[(((N_c_inner * 5) + 124))] = (output0_local[(((N_c_inner * 5) + 124))] + (((int)input0_shared[((((((int)threadIdx.x) * 30) + (N_c_inner * 5)) + K_inner))]) * ((int)input1_shared[(((K_inner * 25) + 24))])));
      }
    }
  }
  for (int N_inner = 0; N_inner < 6; ++N_inner) {
    for (int M_inner = 0; M_inner < 5; ++M_inner) {
      output0[(((((((int)blockIdx.x) * 19200) + (((int)threadIdx.x) * 150)) + (N_inner * 25)) + M_inner))] = output0_local[(((N_inner * 5) + M_inner))];
      output0[((((((((int)blockIdx.x) * 19200) + (((int)threadIdx.x) * 150)) + (N_inner * 25)) + M_inner) + 5))] = output0_local[((((N_inner * 5) + M_inner) + 30))];
      output0[((((((((int)blockIdx.x) * 19200) + (((int)threadIdx.x) * 150)) + (N_inner * 25)) + M_inner) + 10))] = output0_local[((((N_inner * 5) + M_inner) + 60))];
      output0[((((((((int)blockIdx.x) * 19200) + (((int)threadIdx.x) * 150)) + (N_inner * 25)) + M_inner) + 15))] = output0_local[((((N_inner * 5) + M_inner) + 90))];
      output0[((((((((int)blockIdx.x) * 19200) + (((int)threadIdx.x) * 150)) + (N_inner * 25)) + M_inner) + 20))] = output0_local[((((N_inner * 5) + M_inner) + 120))];
    }
  }
}

// Saved Perf = 3.298590e-04 sec / run; Step Produced = 95; Planned Steps = 1000;
// Antares Tuning Completed in 1000 steps.