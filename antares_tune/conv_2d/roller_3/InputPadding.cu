// GLOBALS: input0:float16[128, 42, 42, 1008] -> output0:float16[128, 42, 42, 42]
// BACKEND: c-cuda (default)
// CONFIG: {"Foutput0:D0": [-1, 1, 1, 1], "Foutput0:D1": [-1, 1, 1, 1], "Foutput0:D2": [-1, 1, 3, 1], "Foutput0:D3": [-1, 1], "Foutput0:O": [3, 1, 2, 0], "Foutput0:S": 0, "Foutput0:R": 1}
// COMPUTE_V1: - _N, _CI, _H, _W, _CO, _KH, _KW, _SH, _SW, _PH, _PW = 128, 1008, 42, 42, 336, 1, 1, 1, 1, 0, 0; _PHI, _PWI = _H + _PH * 2, _W + _PW * 2; einstein_v2(f"output0[N, C, PHI, PWI] = input0[N, C, -{_PH} + PHI, -{_PW} + PWI].when([-{_PH} + PHI >= 0, -{_PH} + PHI < {_H}, -{_PW} + PWI >= 0, -{_PW} + PWI < {_W}], const(0.0).cast(`float16`)) where PHI in {_PHI}, PWI in {_PWI}", input_dict={"input0": {"dtype": "float16", "shape": [_N, _H, _W, _CI]}})


// ---------------------------------------------------------------------------
// LOCAL: template_op_kernel0 -- input0:float16[128, 42, 42, 1008] -> output0:float16[128, 42, 42, 42]

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


extern "C" __global__ __launch_bounds__(3) void template_op_kernel0(half* __restrict__ input0, half* __restrict__ output0) {
  // [thread_extent] blockIdx.x = 128
  // [thread_extent] threadIdx.x = 1
  // [thread_extent] blockIdx.y = 42
  // [thread_extent] threadIdx.y = 1
  // [thread_extent] blockIdx.z = 14
  // [thread_extent] threadIdx.z = 3
  for (int vthread_s = 0; vthread_s < 42; ++vthread_s) {
    output0[((((((((int)blockIdx.x) * 74088) + (((int)blockIdx.y) * 1764)) + (((int)blockIdx.z) * 126)) + (((int)threadIdx.z) * 42)) + vthread_s))] = input0[((((((((int)blockIdx.x) * 1778112) + (((int)blockIdx.y) * 42336)) + (((int)blockIdx.z) * 3024)) + (((int)threadIdx.z) * 1008)) + vthread_s))];
  }
}

// Saved Perf = 1.140860e-04 sec / run; Step Produced = 260; Planned Steps = 1000;