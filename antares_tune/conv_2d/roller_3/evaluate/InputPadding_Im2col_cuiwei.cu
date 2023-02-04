// GLOBALS: input0:float16[128, 42, 42, 1008] -> temp0:float16[225792, 1008]
// BACKEND: c-cuda (default)
// CONFIG: null
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


extern "C" __global__ __launch_bounds__(1) void template_op_kernel0(half* __restrict__ input0, half* __restrict__ temp0) {
  // [thread_extent] blockIdx.x = 225792
  // [thread_extent] threadIdx.x = 1
  // [thread_extent] blockIdx.y = 1008
  // [thread_extent] threadIdx.y = 1
  temp0[(((((int)blockIdx.x) * 1008) + ((int)blockIdx.y)))] = input0[((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) / 42) % 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + ((int)blockIdx.y)))];
}
