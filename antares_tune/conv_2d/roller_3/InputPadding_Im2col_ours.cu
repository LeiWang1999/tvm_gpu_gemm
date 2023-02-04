// GLOBALS: input0:float16[128, 42, 42, 1008] -> output0:float16[225792, 1008]
// BACKEND: c-cuda (default)
// CONFIG: {"Foutput0:D0": [-1, 1, 1, 1], "Foutput0:D1": [-1, 9, 14, 8], "Foutput0:O": [1, 0], "Foutput0:S": 3, "Foutput0:R": 1}
// COMPUTE_V1: - _N, _CI, _H, _W, _CO, _KH, _KW, _SH, _SW, _PH, _PW = 128, 1008, 42, 42, 336, 1, 1, 1, 1, 0, 0;_HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1;_PHI, _PWI = _H + _PH * 2, _W + _PW * 2; _GM, _GN, _GK = _N * _HO * _WO, _CO, _CI * _KH * _KW; einstein_v2(f"temp0[N, PHI, PWI, C] = input0[N, -{_PH} + PHI, -{_PW} + PWI, C].when([-{_PH} + PHI >= 0, -{_PH} + PHI < {_H}, -{_PW} + PWI >= 0, -{_PW} + PWI < {_W}], const(0.0).cast(`float16`)) where PHI in {_PHI}, PWI in {_PWI};output0[GM, GK] = temp0[GM // ({_HO} * {_WO}), {_SH} * ((GM % ({_HO} * {_WO})) // {_WO}) + (GK // {_CI}) // {_KW}, {_SW} * ((GM % ({_HO} * {_WO})) % {_WO}) + (GK // {_CI}) % {_KW}, GK % {_CI}] where GM in {_GM}, GK in {_GK}", input_dict={"input0": {"dtype": "float16", "shape": [_N, _H, _W, _CI]}})


// ---------------------------------------------------------------------------
// LOCAL: template_op_kernel0 -- input0:float16[128, 42, 42, 1008] -> output0:float16[225792, 1008]

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


extern "C" __global__ __launch_bounds__(14) void template_op_kernel0(half* __restrict__ input0, half* __restrict__ output0) {
  // [thread_extent] blockIdx.x = 225792
  // [thread_extent] threadIdx.x = 1
  // [thread_extent] blockIdx.y = 1
  // [thread_extent] threadIdx.y = 14
  output0[(((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)))] = input0[((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 1))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 1))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 2))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 2))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 3))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 3))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 4))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 4))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 5))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 5))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 6))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 6))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 7))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 7))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 112))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 112))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 113))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 113))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 114))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 114))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 115))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 115))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 116))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 116))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 117))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 117))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 118))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 118))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 119))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 119))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 224))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 224))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 225))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 225))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 226))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 226))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 227))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 227))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 228))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 228))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 229))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 229))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 230))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 230))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 231))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 231))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 336))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 336))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 337))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 337))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 338))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 338))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 339))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 339))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 340))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 340))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 341))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 341))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 342))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 342))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 343))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 343))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 448))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 448))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 449))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 449))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 450))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 450))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 451))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 451))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 452))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 452))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 453))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 453))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 454))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 454))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 455))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 455))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 560))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 560))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 561))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 561))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 562))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 562))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 563))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 563))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 564))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 564))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 565))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 565))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 566))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 566))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 567))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 567))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 672))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 672))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 673))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 673))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 674))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 674))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 675))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 675))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 676))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 676))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 677))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 677))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 678))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 678))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 679))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 679))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 784))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 784))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 785))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 785))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 786))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 786))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 787))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 787))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 788))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 788))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 789))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 789))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 790))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 790))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 791))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 791))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 896))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 896))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 897))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 897))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 898))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 898))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 899))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 899))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 900))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 900))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 901))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 901))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 902))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 902))];
  output0[((((((int)blockIdx.x) * 1008) + (((int)threadIdx.y) * 8)) + 903))] = input0[(((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + (((int)threadIdx.y) * 8)) + 903))];
}

// Saved Perf = 1.065600e-03 sec / run; Step Produced = 1413; Planned Steps = 10000;