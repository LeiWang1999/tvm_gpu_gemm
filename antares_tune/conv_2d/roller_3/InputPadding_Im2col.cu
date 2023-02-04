// GLOBALS: input0:float16[128, 42, 42, 1008] -> temp0:float16[225792, 1008]
// BACKEND: c-cuda (default)
// CONFIG: {"Ftemp0:D0": [-1, 3, 2, 1], "Ftemp0:D1": [-1, 2, 504, 1], "Ftemp0:O": [1, 0], "Ftemp0:S": 3, "Ftemp0:R": 0}
// COMPUTE_V1: - N, C, F = 128, 1008, 336; HI = WI = 42; KW = KH = 1; SH = SW = 1; PH = PW = 0; HO = (HI - KH + PH * 2) // SH + 1; WO = (WI - KW + PW * 2) // SW + 1; einstein_v2(f"temp0[I, K] = input0[I / alter(`HOWO:{HO * WO}`), (I / alter(`WO:{WO}`) % alter(`HO:{HO}`) * alter(`SH:{SH}`) + K / alter(`KWC:{KW * C}`) - alter(`PH:{PH}`)), (I % alter(`WO:{WO}`) * alter(`SW:{SW}`) + K / alter(`C:{C}`) % alter(`KW:{KW}`) - alter(`PW:{PW}`)), K % alter(`C:{C}`)].when([I / alter(`WO:{WO}`) % alter(`HO:{HO}`) * alter(`SH:{SH}`) + K / alter(`KWC:{KW * C}`) - alter(`PH:{PH}`) >= 0, I / alter(`WO:{WO}`) % alter(`HO:{HO}`) * alter(`SH:{SH}`) + K / alter(`KWC:{KW * C}`) - alter(`PH:{PH}`) < alter(`HI:{HI}`), I % alter(`WO:{WO}`) * alter(`SW:{SW}`) + K / alter(`C:{C}`) % alter(`KW:{KW}`) - alter(`PW:{PW}`) >= 0, I % alter(`WO:{WO}`) * alter(`SW:{SW}`) + K / alter(`C:{C}`) % alter(`KW:{KW}`) - alter(`PW:{PW}`) < alter(`WI:{WI}`)], const(0.0).cast(`float16`)) where I in I:{N * HO * WO}, K in K:{KH * KW * C}", { "input0": {"dtype": "float16", "shape": [f"N:{N}", f"HI:{HI}", f"WI:{WI}", f"C:{C}"]}})


// ---------------------------------------------------------------------------
// LOCAL: template_op_kernel0 -- input0:float16[128, 42, 42, 1008] -> temp0:float16[225792, 1008]

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


extern "C" __global__ __launch_bounds__(1008) void template_op_kernel0(half* __restrict__ input0, half* __restrict__ temp0) {
  // [thread_extent] blockIdx.x = 37632
  // [thread_extent] threadIdx.x = 2
  // [thread_extent] blockIdx.y = 1
  // [thread_extent] threadIdx.y = 504
  temp0[((((((int)blockIdx.x) * 6048) + (((int)threadIdx.x) * 1008)) + ((int)threadIdx.y)))] = input0[((((((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) / 1764) * 1778112) + (((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) / 42) % 42) * 42336)) + ((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) % 42) * 1008)) + ((int)threadIdx.y)))];
  temp0[(((((((int)blockIdx.x) * 6048) + (((int)threadIdx.x) * 1008)) + ((int)threadIdx.y)) + 504))] = input0[(((((((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) / 1764) * 1778112) + (((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) / 42) % 42) * 42336)) + ((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) % 42) * 1008)) + ((int)threadIdx.y)) + 504))];
  temp0[(((((((int)blockIdx.x) * 6048) + (((int)threadIdx.x) * 1008)) + ((int)threadIdx.y)) + 2016))] = input0[(((((((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) + 2) / 1764) * 1778112) + ((((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) + 2) / 42) % 42) * 42336)) + (((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) + 2) % 42) * 1008)) + ((int)threadIdx.y)))];
  temp0[(((((((int)blockIdx.x) * 6048) + (((int)threadIdx.x) * 1008)) + ((int)threadIdx.y)) + 2520))] = input0[((((((((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) + 2) / 1764) * 1778112) + ((((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) + 2) / 42) % 42) * 42336)) + (((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) + 2) % 42) * 1008)) + ((int)threadIdx.y)) + 504))];
  temp0[(((((((int)blockIdx.x) * 6048) + (((int)threadIdx.x) * 1008)) + ((int)threadIdx.y)) + 4032))] = input0[(((((((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) + 4) / 1764) * 1778112) + ((((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) + 4) / 42) % 42) * 42336)) + (((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) + 4) % 42) * 1008)) + ((int)threadIdx.y)))];
  temp0[(((((((int)blockIdx.x) * 6048) + (((int)threadIdx.x) * 1008)) + ((int)threadIdx.y)) + 4536))] = input0[((((((((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) + 4) / 1764) * 1778112) + ((((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) + 4) / 42) % 42) * 42336)) + (((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) + 4) % 42) * 1008)) + ((int)threadIdx.y)) + 504))];
}

// Saved Perf = 1.066720e-03 sec / run; Step Produced = 491; Planned Steps = 1000;
// Antares Tuning Completed in 1000 steps.