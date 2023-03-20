// GLOBALS: input0:float16[128, 28, 28, 128] -> output0:float16[128, 28, 28, 128]
// BACKEND: c-cuda (default)
// CONFIG: {"Foutput0:D0": [-1, 1, 1, 2], "Foutput0:D1": [-1, 1, 1, 1], "Foutput0:D2": [-1, 1, 2, 1], "Foutput0:D3": [-1, 2], "Foutput0:O": [2, 3, 0, 1], "Foutput0:S": 2, "Foutput0:R": 1}
// COMPUTE_V1: - _N, _CI, _H, _W, _CO, _KH, _KW, _SH, _SW, _PH, _PW = 128, 128, 28, 28, 128, 3, 3, 1, 1, 0, 0;_HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1;_PHI, _PWI = _H + _PH * 2, _W + _PW * 2; _GM, _GN, _GK = _N * _HO * _WO, _CO, _CI * _KH * _KW; einstein_v2(f"output0[N, PHI, PWI, C] = input0[N, -{_PH} + PHI, -{_PW} + PWI, C].when([-{_PH} + PHI >= 0, -{_PH} + PHI < {_H}, -{_PW} + PWI >= 0, -{_PW} + PWI < {_W}], const(0.0).cast(`float16`)) where PHI in {_PHI}, PWI in {_PWI};", input_dict={"input0": {"dtype": "float16", "shape": [_N, _H, _W, _CI]}})


// ---------------------------------------------------------------------------
// LOCAL: template_op_kernel0 -- input0:float16[128, 28, 28, 128] -> output0:float16[128, 28, 28, 128]

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
  // [thread_extent] blockIdx.y = 28
  // [thread_extent] threadIdx.y = 1
  // [thread_extent] blockIdx.z = 14
  // [thread_extent] threadIdx.z = 2
  for (int vthread_s = 0; vthread_s < 64; ++vthread_s) {
    output0[((((((((int)blockIdx.x) * 200704) + (((int)blockIdx.y) * 3584)) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 128)) + (vthread_s * 2)))] = input0[((((((((int)blockIdx.x) * 200704) + (((int)blockIdx.y) * 3584)) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 128)) + (vthread_s * 2)))];
    output0[(((((((((int)blockIdx.x) * 200704) + (((int)blockIdx.y) * 3584)) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 128)) + (vthread_s * 2)) + 1))] = input0[(((((((((int)blockIdx.x) * 200704) + (((int)blockIdx.y) * 3584)) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 128)) + (vthread_s * 2)) + 1))];
  }
  for (int vthread_s1 = 0; vthread_s1 < 64; ++vthread_s1) {
    output0[(((((((((int)blockIdx.x) * 200704) + (((int)blockIdx.y) * 3584)) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 128)) + (vthread_s1 * 2)) + 100352))] = input0[(((((((((int)blockIdx.x) * 200704) + (((int)blockIdx.y) * 3584)) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 128)) + (vthread_s1 * 2)) + 100352))];
    output0[(((((((((int)blockIdx.x) * 200704) + (((int)blockIdx.y) * 3584)) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 128)) + (vthread_s1 * 2)) + 100353))] = input0[(((((((((int)blockIdx.x) * 200704) + (((int)blockIdx.y) * 3584)) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 128)) + (vthread_s1 * 2)) + 100353))];
  }
}

// Saved Perf = 1.918960e-04 sec / run; Step Produced = 832; Planned Steps = 1000;
// Antares Tuning Completed in 1000 steps.