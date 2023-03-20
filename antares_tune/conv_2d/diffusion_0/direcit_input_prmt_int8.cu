// GLOBALS: input0:int8[128, 32, 32, 1280] -> output0:int8[8, 32, 32, 32, 80, 16]
// BACKEND: c-cuda (default)
// CONFIG: {"F___output0:D0": [-1, 1, 4, 1], "F___output0:D1": [-1, 1, 1, 2], "F___output0:D2": [-1, 32, 40, 1], "F___output0:D3": [-1, 1], "F___output0:O": [2, 3, 1, 0], "F___output0:S": 2, "F___output0:R": 0}
// COMPUTE_V1: - einstein_v2("output0[H, W, N // 16, C // 16, N % 16, C % 16] =. input0[N, H, W, C]", input_dict={"input0": {"dtype": "int8", "shape": [128, 32, 32, 1280]}, "output0": {"dtype": "int8", "shape": [8, 32, 32, 32, 80, 16]}})


// ---------------------------------------------------------------------------
// LOCAL: template_op_kernel0 -- input0:int8[128, 32, 32, 1280] -> output0:int8[8, 32, 32, 32, 80, 16]

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


extern "C" __global__ __launch_bounds__(160) void template_op_kernel0(char* __restrict__ input0, char* __restrict__ output0) {
  // [thread_extent] blockIdx.x = 32
  // [thread_extent] threadIdx.x = 4
  // [thread_extent] blockIdx.y = 16
  // [thread_extent] threadIdx.y = 1
  // [thread_extent] blockIdx.z = 1
  // [thread_extent] threadIdx.z = 40
  for (int vthread_s = 0; vthread_s < 32; ++vthread_s) {
    ((output0[((((((((int)blockIdx.y) * 2621440) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) / 16) * 40960)) + ((((vthread_s * 40) + ((int)threadIdx.z)) / 16) * 1280)) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) & 15) * 16)) + (((vthread_s * 40) + ((int)threadIdx.z)) & 15)))]) = (input0[((((((((int)blockIdx.x) * 5242880) + (((int)threadIdx.x) * 1310720)) + (((int)blockIdx.y) * 2560)) + (vthread_s * 40)) + ((int)threadIdx.z)))]));
  }
  for (int vthread_s1 = 0; vthread_s1 < 32; ++vthread_s1) {
    ((output0[(((((((((int)blockIdx.y) * 2621440) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) / 16) * 40960)) + ((((vthread_s1 * 40) + ((int)threadIdx.z)) / 16) * 1280)) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) & 15) * 16)) + (((vthread_s1 * 40) + ((int)threadIdx.z)) & 15)) + 41943040))]) = (input0[(((((((((int)blockIdx.x) * 5242880) + (((int)threadIdx.x) * 1310720)) + (((int)blockIdx.y) * 2560)) + (vthread_s1 * 40)) + ((int)threadIdx.z)) + 40960))]));
  }
  for (int vthread_s2 = 0; vthread_s2 < 32; ++vthread_s2) {
    ((output0[(((((((((int)blockIdx.y) * 2621440) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) / 16) * 40960)) + ((((vthread_s2 * 40) + ((int)threadIdx.z)) / 16) * 1280)) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) & 15) * 16)) + (((vthread_s2 * 40) + ((int)threadIdx.z)) & 15)) + 83886080))]) = (input0[(((((((((int)blockIdx.x) * 5242880) + (((int)threadIdx.x) * 1310720)) + (((int)blockIdx.y) * 2560)) + (vthread_s2 * 40)) + ((int)threadIdx.z)) + 81920))]));
  }
  for (int vthread_s3 = 0; vthread_s3 < 32; ++vthread_s3) {
    ((output0[(((((((((int)blockIdx.y) * 2621440) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) / 16) * 40960)) + ((((vthread_s3 * 40) + ((int)threadIdx.z)) / 16) * 1280)) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) & 15) * 16)) + (((vthread_s3 * 40) + ((int)threadIdx.z)) & 15)) + 125829120))]) = (input0[(((((((((int)blockIdx.x) * 5242880) + (((int)threadIdx.x) * 1310720)) + (((int)blockIdx.y) * 2560)) + (vthread_s3 * 40)) + ((int)threadIdx.z)) + 122880))]));
  }
  for (int vthread_s4 = 0; vthread_s4 < 32; ++vthread_s4) {
    ((output0[(((((((((int)blockIdx.y) * 2621440) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) / 16) * 40960)) + ((((vthread_s4 * 40) + ((int)threadIdx.z)) / 16) * 1280)) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) & 15) * 16)) + (((vthread_s4 * 40) + ((int)threadIdx.z)) & 15)) + 167772160))]) = (input0[(((((((((int)blockIdx.x) * 5242880) + (((int)threadIdx.x) * 1310720)) + (((int)blockIdx.y) * 2560)) + (vthread_s4 * 40)) + ((int)threadIdx.z)) + 163840))]));
  }
  for (int vthread_s5 = 0; vthread_s5 < 32; ++vthread_s5) {
    ((output0[(((((((((int)blockIdx.y) * 2621440) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) / 16) * 40960)) + ((((vthread_s5 * 40) + ((int)threadIdx.z)) / 16) * 1280)) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) & 15) * 16)) + (((vthread_s5 * 40) + ((int)threadIdx.z)) & 15)) + 209715200))]) = (input0[(((((((((int)blockIdx.x) * 5242880) + (((int)threadIdx.x) * 1310720)) + (((int)blockIdx.y) * 2560)) + (vthread_s5 * 40)) + ((int)threadIdx.z)) + 204800))]));
  }
  for (int vthread_s6 = 0; vthread_s6 < 32; ++vthread_s6) {
    ((output0[(((((((((int)blockIdx.y) * 2621440) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) / 16) * 40960)) + ((((vthread_s6 * 40) + ((int)threadIdx.z)) / 16) * 1280)) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) & 15) * 16)) + (((vthread_s6 * 40) + ((int)threadIdx.z)) & 15)) + 251658240))]) = (input0[(((((((((int)blockIdx.x) * 5242880) + (((int)threadIdx.x) * 1310720)) + (((int)blockIdx.y) * 2560)) + (vthread_s6 * 40)) + ((int)threadIdx.z)) + 245760))]));
  }
  for (int vthread_s7 = 0; vthread_s7 < 32; ++vthread_s7) {
    ((output0[(((((((((int)blockIdx.y) * 2621440) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) / 16) * 40960)) + ((((vthread_s7 * 40) + ((int)threadIdx.z)) / 16) * 1280)) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) & 15) * 16)) + (((vthread_s7 * 40) + ((int)threadIdx.z)) & 15)) + 293601280))]) = (input0[(((((((((int)blockIdx.x) * 5242880) + (((int)threadIdx.x) * 1310720)) + (((int)blockIdx.y) * 2560)) + (vthread_s7 * 40)) + ((int)threadIdx.z)) + 286720))]));
  }
  for (int vthread_s8 = 0; vthread_s8 < 32; ++vthread_s8) {
    ((output0[(((((((((int)blockIdx.y) * 2621440) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) / 16) * 40960)) + ((((vthread_s8 * 40) + ((int)threadIdx.z)) / 16) * 1280)) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) & 15) * 16)) + (((vthread_s8 * 40) + ((int)threadIdx.z)) & 15)) + 1310720))]) = (input0[(((((((((int)blockIdx.x) * 5242880) + (((int)threadIdx.x) * 1310720)) + (((int)blockIdx.y) * 2560)) + (vthread_s8 * 40)) + ((int)threadIdx.z)) + 1280))]));
  }
  for (int vthread_s9 = 0; vthread_s9 < 32; ++vthread_s9) {
    ((output0[(((((((((int)blockIdx.y) * 2621440) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) / 16) * 40960)) + ((((vthread_s9 * 40) + ((int)threadIdx.z)) / 16) * 1280)) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) & 15) * 16)) + (((vthread_s9 * 40) + ((int)threadIdx.z)) & 15)) + 43253760))]) = (input0[(((((((((int)blockIdx.x) * 5242880) + (((int)threadIdx.x) * 1310720)) + (((int)blockIdx.y) * 2560)) + (vthread_s9 * 40)) + ((int)threadIdx.z)) + 42240))]));
  }
  for (int vthread_s10 = 0; vthread_s10 < 32; ++vthread_s10) {
    ((output0[(((((((((int)blockIdx.y) * 2621440) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) / 16) * 40960)) + ((((vthread_s10 * 40) + ((int)threadIdx.z)) / 16) * 1280)) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) & 15) * 16)) + (((vthread_s10 * 40) + ((int)threadIdx.z)) & 15)) + 85196800))]) = (input0[(((((((((int)blockIdx.x) * 5242880) + (((int)threadIdx.x) * 1310720)) + (((int)blockIdx.y) * 2560)) + (vthread_s10 * 40)) + ((int)threadIdx.z)) + 83200))]));
  }
  for (int vthread_s11 = 0; vthread_s11 < 32; ++vthread_s11) {
    ((output0[(((((((((int)blockIdx.y) * 2621440) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) / 16) * 40960)) + ((((vthread_s11 * 40) + ((int)threadIdx.z)) / 16) * 1280)) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) & 15) * 16)) + (((vthread_s11 * 40) + ((int)threadIdx.z)) & 15)) + 127139840))]) = (input0[(((((((((int)blockIdx.x) * 5242880) + (((int)threadIdx.x) * 1310720)) + (((int)blockIdx.y) * 2560)) + (vthread_s11 * 40)) + ((int)threadIdx.z)) + 124160))]));
  }
  for (int vthread_s12 = 0; vthread_s12 < 32; ++vthread_s12) {
    ((output0[(((((((((int)blockIdx.y) * 2621440) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) / 16) * 40960)) + ((((vthread_s12 * 40) + ((int)threadIdx.z)) / 16) * 1280)) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) & 15) * 16)) + (((vthread_s12 * 40) + ((int)threadIdx.z)) & 15)) + 169082880))]) = (input0[(((((((((int)blockIdx.x) * 5242880) + (((int)threadIdx.x) * 1310720)) + (((int)blockIdx.y) * 2560)) + (vthread_s12 * 40)) + ((int)threadIdx.z)) + 165120))]));
  }
  for (int vthread_s13 = 0; vthread_s13 < 32; ++vthread_s13) {
    ((output0[(((((((((int)blockIdx.y) * 2621440) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) / 16) * 40960)) + ((((vthread_s13 * 40) + ((int)threadIdx.z)) / 16) * 1280)) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) & 15) * 16)) + (((vthread_s13 * 40) + ((int)threadIdx.z)) & 15)) + 211025920))]) = (input0[(((((((((int)blockIdx.x) * 5242880) + (((int)threadIdx.x) * 1310720)) + (((int)blockIdx.y) * 2560)) + (vthread_s13 * 40)) + ((int)threadIdx.z)) + 206080))]));
  }
  for (int vthread_s14 = 0; vthread_s14 < 32; ++vthread_s14) {
    ((output0[(((((((((int)blockIdx.y) * 2621440) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) / 16) * 40960)) + ((((vthread_s14 * 40) + ((int)threadIdx.z)) / 16) * 1280)) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) & 15) * 16)) + (((vthread_s14 * 40) + ((int)threadIdx.z)) & 15)) + 252968960))]) = (input0[(((((((((int)blockIdx.x) * 5242880) + (((int)threadIdx.x) * 1310720)) + (((int)blockIdx.y) * 2560)) + (vthread_s14 * 40)) + ((int)threadIdx.z)) + 247040))]));
  }
  for (int vthread_s15 = 0; vthread_s15 < 32; ++vthread_s15) {
    ((output0[(((((((((int)blockIdx.y) * 2621440) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) / 16) * 40960)) + ((((vthread_s15 * 40) + ((int)threadIdx.z)) / 16) * 1280)) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) & 15) * 16)) + (((vthread_s15 * 40) + ((int)threadIdx.z)) & 15)) + 294912000))]) = (input0[(((((((((int)blockIdx.x) * 5242880) + (((int)threadIdx.x) * 1310720)) + (((int)blockIdx.y) * 2560)) + (vthread_s15 * 40)) + ((int)threadIdx.z)) + 288000))]));
  }
}

// Saved Perf = 9.504470e-05 sec / run; Step Produced = 593; Planned Steps = 1000;
// Antares Tuning Completed in 1000 steps.