// GLOBALS: input0:float16[128, 42, 42, 1008] -> output0:float16[128, 42, 42, 1024]
// BACKEND: c-cuda (default)
// CONFIG: {"Foutput0:D0": [-1, 2, 1, 1], "Foutput0:D1": [-1, 6, 1, 1], "Foutput0:D2": [-1, 1, 2, 1], "Foutput0:D3": [-1, 256], "Foutput0:O": [3, 0, 1, 2], "Foutput0:S": 1, "Foutput0:R": 0}
// COMPUTE_V1: - einstein_v2("output0[N, H, W, C] = input0[N, H, W, C].when([N < 128, H < 42, W < 42, C < 1008], const(0.0).cast(`float16`)) where N in 128, H in 42, W in 42, C in 1024 ", input_dict={"input0": {"dtype": "float16", "shape": [128, 42, 42, 1008]}})


// ---------------------------------------------------------------------------
// LOCAL: template_op_kernel0 -- input0:float16[128, 42, 42, 1008] -> output0:float16[128, 42, 42, 1024]

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


extern "C" __global__ __launch_bounds__(2) void template_op_kernel0(half* __restrict__ input0, half* __restrict__ output0) {
  // [thread_extent] blockIdx.x = 64
  // [thread_extent] threadIdx.x = 1
  // [thread_extent] blockIdx.y = 7
  // [thread_extent] threadIdx.y = 1
  // [thread_extent] blockIdx.z = 21
  // [thread_extent] threadIdx.z = 2
  for (int vthread_s = 0; vthread_s < 256; ++vthread_s) {
    output0[((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s))] = input0[((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s))];
  }
  for (int vthread_s1 = 0; vthread_s1 < 256; ++vthread_s1) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s1) + 256))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s1) + 256))];
  }
  for (int vthread_s2 = 0; vthread_s2 < 256; ++vthread_s2) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s2) + 512))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s2) + 512))];
  }
  for (int vthread_s3 = 0; vthread_s3 < 256; ++vthread_s3) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s3) + 768))] = ((vthread_s3 < 240) ? input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s3) + 768))] : __float2half_rn(0.000000e+00f));
  }
  for (int vthread_s4 = 0; vthread_s4 < 256; ++vthread_s4) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s4) + 1806336))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s4) + 1778112))];
  }
  for (int vthread_s5 = 0; vthread_s5 < 256; ++vthread_s5) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s5) + 1806592))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s5) + 1778368))];
  }
  for (int vthread_s6 = 0; vthread_s6 < 256; ++vthread_s6) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s6) + 1806848))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s6) + 1778624))];
  }
  for (int vthread_s7 = 0; vthread_s7 < 256; ++vthread_s7) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s7) + 1807104))] = ((vthread_s7 < 240) ? input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s7) + 1778880))] : __float2half_rn(0.000000e+00f));
  }
  for (int vthread_s8 = 0; vthread_s8 < 256; ++vthread_s8) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s8) + 43008))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s8) + 42336))];
  }
  for (int vthread_s9 = 0; vthread_s9 < 256; ++vthread_s9) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s9) + 43264))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s9) + 42592))];
  }
  for (int vthread_s10 = 0; vthread_s10 < 256; ++vthread_s10) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s10) + 43520))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s10) + 42848))];
  }
  for (int vthread_s11 = 0; vthread_s11 < 256; ++vthread_s11) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s11) + 43776))] = ((vthread_s11 < 240) ? input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s11) + 43104))] : __float2half_rn(0.000000e+00f));
  }
  for (int vthread_s12 = 0; vthread_s12 < 256; ++vthread_s12) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s12) + 1849344))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s12) + 1820448))];
  }
  for (int vthread_s13 = 0; vthread_s13 < 256; ++vthread_s13) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s13) + 1849600))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s13) + 1820704))];
  }
  for (int vthread_s14 = 0; vthread_s14 < 256; ++vthread_s14) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s14) + 1849856))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s14) + 1820960))];
  }
  for (int vthread_s15 = 0; vthread_s15 < 256; ++vthread_s15) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s15) + 1850112))] = ((vthread_s15 < 240) ? input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s15) + 1821216))] : __float2half_rn(0.000000e+00f));
  }
  for (int vthread_s16 = 0; vthread_s16 < 256; ++vthread_s16) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s16) + 86016))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s16) + 84672))];
  }
  for (int vthread_s17 = 0; vthread_s17 < 256; ++vthread_s17) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s17) + 86272))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s17) + 84928))];
  }
  for (int vthread_s18 = 0; vthread_s18 < 256; ++vthread_s18) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s18) + 86528))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s18) + 85184))];
  }
  for (int vthread_s19 = 0; vthread_s19 < 256; ++vthread_s19) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s19) + 86784))] = ((vthread_s19 < 240) ? input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s19) + 85440))] : __float2half_rn(0.000000e+00f));
  }
  for (int vthread_s20 = 0; vthread_s20 < 256; ++vthread_s20) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s20) + 1892352))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s20) + 1862784))];
  }
  for (int vthread_s21 = 0; vthread_s21 < 256; ++vthread_s21) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s21) + 1892608))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s21) + 1863040))];
  }
  for (int vthread_s22 = 0; vthread_s22 < 256; ++vthread_s22) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s22) + 1892864))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s22) + 1863296))];
  }
  for (int vthread_s23 = 0; vthread_s23 < 256; ++vthread_s23) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s23) + 1893120))] = ((vthread_s23 < 240) ? input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s23) + 1863552))] : __float2half_rn(0.000000e+00f));
  }
  for (int vthread_s24 = 0; vthread_s24 < 256; ++vthread_s24) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s24) + 129024))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s24) + 127008))];
  }
  for (int vthread_s25 = 0; vthread_s25 < 256; ++vthread_s25) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s25) + 129280))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s25) + 127264))];
  }
  for (int vthread_s26 = 0; vthread_s26 < 256; ++vthread_s26) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s26) + 129536))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s26) + 127520))];
  }
  for (int vthread_s27 = 0; vthread_s27 < 256; ++vthread_s27) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s27) + 129792))] = ((vthread_s27 < 240) ? input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s27) + 127776))] : __float2half_rn(0.000000e+00f));
  }
  for (int vthread_s28 = 0; vthread_s28 < 256; ++vthread_s28) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s28) + 1935360))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s28) + 1905120))];
  }
  for (int vthread_s29 = 0; vthread_s29 < 256; ++vthread_s29) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s29) + 1935616))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s29) + 1905376))];
  }
  for (int vthread_s30 = 0; vthread_s30 < 256; ++vthread_s30) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s30) + 1935872))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s30) + 1905632))];
  }
  for (int vthread_s31 = 0; vthread_s31 < 256; ++vthread_s31) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s31) + 1936128))] = ((vthread_s31 < 240) ? input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s31) + 1905888))] : __float2half_rn(0.000000e+00f));
  }
  for (int vthread_s32 = 0; vthread_s32 < 256; ++vthread_s32) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s32) + 172032))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s32) + 169344))];
  }
  for (int vthread_s33 = 0; vthread_s33 < 256; ++vthread_s33) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s33) + 172288))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s33) + 169600))];
  }
  for (int vthread_s34 = 0; vthread_s34 < 256; ++vthread_s34) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s34) + 172544))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s34) + 169856))];
  }
  for (int vthread_s35 = 0; vthread_s35 < 256; ++vthread_s35) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s35) + 172800))] = ((vthread_s35 < 240) ? input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s35) + 170112))] : __float2half_rn(0.000000e+00f));
  }
  for (int vthread_s36 = 0; vthread_s36 < 256; ++vthread_s36) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s36) + 1978368))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s36) + 1947456))];
  }
  for (int vthread_s37 = 0; vthread_s37 < 256; ++vthread_s37) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s37) + 1978624))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s37) + 1947712))];
  }
  for (int vthread_s38 = 0; vthread_s38 < 256; ++vthread_s38) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s38) + 1978880))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s38) + 1947968))];
  }
  for (int vthread_s39 = 0; vthread_s39 < 256; ++vthread_s39) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s39) + 1979136))] = ((vthread_s39 < 240) ? input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s39) + 1948224))] : __float2half_rn(0.000000e+00f));
  }
  for (int vthread_s40 = 0; vthread_s40 < 256; ++vthread_s40) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s40) + 215040))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s40) + 211680))];
  }
  for (int vthread_s41 = 0; vthread_s41 < 256; ++vthread_s41) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s41) + 215296))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s41) + 211936))];
  }
  for (int vthread_s42 = 0; vthread_s42 < 256; ++vthread_s42) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s42) + 215552))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s42) + 212192))];
  }
  for (int vthread_s43 = 0; vthread_s43 < 256; ++vthread_s43) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s43) + 215808))] = ((vthread_s43 < 240) ? input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s43) + 212448))] : __float2half_rn(0.000000e+00f));
  }
  for (int vthread_s44 = 0; vthread_s44 < 256; ++vthread_s44) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s44) + 2021376))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s44) + 1989792))];
  }
  for (int vthread_s45 = 0; vthread_s45 < 256; ++vthread_s45) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s45) + 2021632))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s45) + 1990048))];
  }
  for (int vthread_s46 = 0; vthread_s46 < 256; ++vthread_s46) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s46) + 2021888))] = input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s46) + 1990304))];
  }
  for (int vthread_s47 = 0; vthread_s47 < 256; ++vthread_s47) {
    output0[(((((((((int)blockIdx.x) * 3612672) + (((int)blockIdx.y) * 258048)) + (((int)blockIdx.z) * 2048)) + (((int)threadIdx.z) * 1024)) + vthread_s47) + 2022144))] = ((vthread_s47 < 240) ? input0[(((((((((int)blockIdx.x) * 3556224) + (((int)blockIdx.y) * 254016)) + (((int)blockIdx.z) * 2016)) + (((int)threadIdx.z) * 1008)) + vthread_s47) + 1990560))] : __float2half_rn(0.000000e+00f));
  }
}

// Saved Perf = 3.275290e-03 sec / run; Step Produced = 841; Planned Steps = 1000;
// Antares Tuning Completed in 1000 steps.