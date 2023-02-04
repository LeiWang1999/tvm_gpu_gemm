// GLOBALS: input0:float16[128, 42, 42, 1008] -> output0:float16[225792, 1008]
// BACKEND: c-cuda (default)
// CONFIG: null
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


extern "C" __global__ __launch_bounds__(1) void template_op_kernel0(half* __restrict__ input0, half* __restrict__ output0) {
  // [thread_extent] blockIdx.x = 225792
  // [thread_extent] threadIdx.x = 1
  // [thread_extent] blockIdx.y = 1008
  // [thread_extent] threadIdx.y = 1
  output0[(((((int)blockIdx.x) * 1008) + ((int)blockIdx.y)))] = input0[((((((((int)blockIdx.x) / 1764) * 1778112) + (((((int)blockIdx.x) % 1764) / 42) * 42336)) + ((((int)blockIdx.x) % 42) * 1008)) + ((int)blockIdx.y)))];
}
