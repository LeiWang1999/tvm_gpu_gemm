// GLOBALS: input0:float16[128, 42, 42, 1024] -> output0:float16[8, 42, 42, 24, 16, 16]
// BACKEND: c-cuda (default)
// CONFIG: {"F___output0:D0": [-1, 3, 1, 1], "F___output0:D1": [-1, 2, 32, 1], "F___output0:D2": [-1, 2, 1, 2], "F___output0:D3": [-1, 64], "F___output0:O": [1, 3, 0, 2], "F___output0:S": 2, "F___output0:R": 1}
// COMPUTE_V1: - einstein_v2("output0[H, W, N // 16, C // 16, N % 16, C % 16] =. input0[N, H, W, C]", input_dict={"input0": {"dtype": "float16", "shape": [128, 42, 42, 1024]}, "output0": {"dtype": "float16", "shape": [8, 42, 42, 24, 16, 16]}})


// ---------------------------------------------------------------------------
// LOCAL: template_op_kernel0 -- input0:float16[128, 42, 42, 1024] -> output0:float16[8, 42, 42, 24, 16, 16]

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
  // [thread_extent] blockIdx.x = 14
  // [thread_extent] threadIdx.x = 1
  // [thread_extent] blockIdx.y = 16
  // [thread_extent] threadIdx.y = 32
  // [thread_extent] blockIdx.z = 2
  // [thread_extent] threadIdx.z = 1
  for (int vthread_s = 0; vthread_s < 64; ++vthread_s) {
    ((output0[((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s & 15) * 16)) + (((int)threadIdx.y) & 15)))]) = (input0[((((((vthread_s * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)))]));
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s & 15) * 16)) + (((int)threadIdx.y) & 15)) + 512))]) = (input0[(((((((vthread_s * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 32))]));
  }
  for (int vthread_s1 = 0; vthread_s1 < 64; ++vthread_s1) {
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s1 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s1 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 24576))]) = (input0[(((((((vthread_s1 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 115605504))]));
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s1 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s1 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 25088))]) = (input0[(((((((vthread_s1 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 115605536))]));
  }
  for (int vthread_s2 = 0; vthread_s2 < 64; ++vthread_s2) {
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s2 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s2 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 258048))]) = (input0[(((((((vthread_s2 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 1024))]));
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s2 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s2 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 258560))]) = (input0[(((((((vthread_s2 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 1056))]));
  }
  for (int vthread_s3 = 0; vthread_s3 < 64; ++vthread_s3) {
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s3 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s3 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 282624))]) = (input0[(((((((vthread_s3 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 115606528))]));
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s3 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s3 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 283136))]) = (input0[(((((((vthread_s3 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 115606560))]));
  }
  for (int vthread_s4 = 0; vthread_s4 < 64; ++vthread_s4) {
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s4 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s4 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 516096))]) = (input0[(((((((vthread_s4 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 2048))]));
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s4 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s4 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 516608))]) = (input0[(((((((vthread_s4 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 2080))]));
  }
  for (int vthread_s5 = 0; vthread_s5 < 64; ++vthread_s5) {
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s5 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s5 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 540672))]) = (input0[(((((((vthread_s5 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 115607552))]));
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s5 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s5 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 541184))]) = (input0[(((((((vthread_s5 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 115607584))]));
  }
  for (int vthread_s6 = 0; vthread_s6 < 64; ++vthread_s6) {
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s6 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s6 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 10838016))]) = (input0[(((((((vthread_s6 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 43008))]));
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s6 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s6 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 10838528))]) = (input0[(((((((vthread_s6 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 43040))]));
  }
  for (int vthread_s7 = 0; vthread_s7 < 64; ++vthread_s7) {
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s7 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s7 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 10862592))]) = (input0[(((((((vthread_s7 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 115648512))]));
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s7 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s7 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 10863104))]) = (input0[(((((((vthread_s7 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 115648544))]));
  }
  for (int vthread_s8 = 0; vthread_s8 < 64; ++vthread_s8) {
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s8 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s8 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 11096064))]) = (input0[(((((((vthread_s8 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 44032))]));
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s8 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s8 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 11096576))]) = (input0[(((((((vthread_s8 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 44064))]));
  }
  for (int vthread_s9 = 0; vthread_s9 < 64; ++vthread_s9) {
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s9 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s9 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 11120640))]) = (input0[(((((((vthread_s9 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 115649536))]));
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s9 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s9 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 11121152))]) = (input0[(((((((vthread_s9 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 115649568))]));
  }
  for (int vthread_s10 = 0; vthread_s10 < 64; ++vthread_s10) {
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s10 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s10 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 11354112))]) = (input0[(((((((vthread_s10 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 45056))]));
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s10 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s10 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 11354624))]) = (input0[(((((((vthread_s10 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 45088))]));
  }
  for (int vthread_s11 = 0; vthread_s11 < 64; ++vthread_s11) {
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s11 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s11 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 11378688))]) = (input0[(((((((vthread_s11 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 115650560))]));
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s11 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s11 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 11379200))]) = (input0[(((((((vthread_s11 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 115650592))]));
  }
  for (int vthread_s12 = 0; vthread_s12 < 64; ++vthread_s12) {
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s12 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s12 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 21676032))]) = (input0[(((((((vthread_s12 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 86016))]));
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s12 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s12 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 21676544))]) = (input0[(((((((vthread_s12 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 86048))]));
  }
  for (int vthread_s13 = 0; vthread_s13 < 64; ++vthread_s13) {
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s13 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s13 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 21700608))]) = (input0[(((((((vthread_s13 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 115691520))]));
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s13 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s13 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 21701120))]) = (input0[(((((((vthread_s13 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 115691552))]));
  }
  for (int vthread_s14 = 0; vthread_s14 < 64; ++vthread_s14) {
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s14 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s14 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 21934080))]) = (input0[(((((((vthread_s14 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 87040))]));
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s14 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s14 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 21934592))]) = (input0[(((((((vthread_s14 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 87072))]));
  }
  for (int vthread_s15 = 0; vthread_s15 < 64; ++vthread_s15) {
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s15 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s15 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 21958656))]) = (input0[(((((((vthread_s15 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 115692544))]));
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s15 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s15 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 21959168))]) = (input0[(((((((vthread_s15 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 115692576))]));
  }
  for (int vthread_s16 = 0; vthread_s16 < 64; ++vthread_s16) {
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s16 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s16 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 22192128))]) = (input0[(((((((vthread_s16 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 88064))]));
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s16 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s16 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 22192640))]) = (input0[(((((((vthread_s16 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 88096))]));
  }
  for (int vthread_s17 = 0; vthread_s17 < 64; ++vthread_s17) {
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s17 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s17 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 22216704))]) = (input0[(((((((vthread_s17 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 115693568))]));
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s17 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s17 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 22217216))]) = (input0[(((((((vthread_s17 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 115693600))]));
  }
  for (int vthread_s18 = 0; vthread_s18 < 64; ++vthread_s18) {
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s18 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s18 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 32514048))]) = (input0[(((((((vthread_s18 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 129024))]));
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s18 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s18 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 32514560))]) = (input0[(((((((vthread_s18 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 129056))]));
  }
  for (int vthread_s19 = 0; vthread_s19 < 64; ++vthread_s19) {
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s19 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s19 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 32538624))]) = (input0[(((((((vthread_s19 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 115734528))]));
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s19 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s19 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 32539136))]) = (input0[(((((((vthread_s19 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 115734560))]));
  }
  for (int vthread_s20 = 0; vthread_s20 < 64; ++vthread_s20) {
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s20 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s20 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 32772096))]) = (input0[(((((((vthread_s20 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 130048))]));
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s20 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s20 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 32772608))]) = (input0[(((((((vthread_s20 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 130080))]));
  }
  for (int vthread_s21 = 0; vthread_s21 < 64; ++vthread_s21) {
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s21 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s21 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 32796672))]) = (input0[(((((((vthread_s21 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 115735552))]));
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s21 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s21 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 32797184))]) = (input0[(((((((vthread_s21 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 115735584))]));
  }
  for (int vthread_s22 = 0; vthread_s22 < 64; ++vthread_s22) {
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s22 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s22 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 33030144))]) = (input0[(((((((vthread_s22 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 131072))]));
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s22 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s22 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 33030656))]) = (input0[(((((((vthread_s22 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 131104))]));
  }
  for (int vthread_s23 = 0; vthread_s23 < 64; ++vthread_s23) {
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s23 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s23 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 33054720))]) = (input0[(((((((vthread_s23 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 115736576))]));
    ((output0[(((((((((((int)blockIdx.z) * 43352064) + (((int)blockIdx.x) * 774144)) + ((vthread_s23 / 16) * 6144)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.y) / 16) * 256)) + ((vthread_s23 & 15) * 16)) + (((int)threadIdx.y) & 15)) + 33055232))]) = (input0[(((((((vthread_s23 * 1806336) + (((int)blockIdx.z) * 172032)) + (((int)blockIdx.x) * 3072)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.y)) + 115736608))]));
  }
}

// Saved Perf = 1.590430e-04 sec / run; Step Produced = 661; Planned Steps = 1000;
// Antares Tuning Completed in 1000 steps.