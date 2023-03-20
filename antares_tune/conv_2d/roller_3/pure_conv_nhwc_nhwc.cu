// GLOBALS: input0:float16[128, 42, 42, 1024], input1:float16[384, 1, 1, 1024] -> output0:float16[128, 42, 42, 384]
// BACKEND: c-cuda (default)
// CONFIG: {"Moutput0T": 0, "Moutput0:D0": [-1, 4, 2, 2], "Moutput0:D1": [-1, 1, 2, 1], "Moutput0:D2": [-1, 1, 2, 1], "Moutput0:D3": [-1, 3, 16, 4], "Moutput0:R0": [-1, 2, 1], "Moutput0:R1": [-1, 1, 1], "Moutput0:R2": [-1, 1, 1], "Moutput0:RA": 0, "Moutput0:AL0": 0, "Moutput0:AL1": 0, "Moutput0:S": 2, "Moutput0:U": 1}
// COMPUTE_V1: - einstein_v2("output0[N, HO, WO, F] +=! input0[N, HO + KH, WO + KW, C] * input1[F, KH, KW, C] where HO in 42, WO in 42", { "input0": {"dtype": "float16", "shape": [128, 42, 42, 1024]}, "input1": {"dtype": "float16", "shape": [384, 1, 1, 1024]}})


// ---------------------------------------------------------------------------
// LOCAL: template_op_kernel0 -- input0:float16[128, 42, 42, 1024], input1:float16[384, 1, 1, 1024] -> output0:float16[128, 42, 42, 384]

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


extern "C" __global__ __launch_bounds__(128) void template_op_kernel0(half* __restrict__ input0, half* __restrict__ input1, half* __restrict__ output0) {
  // [thread_extent] blockIdx.x = 7056
  // [thread_extent] threadIdx.x = 128
  half output0_local[96];
  for (int N_c_inner_init = 0; N_c_inner_init < 2; ++N_c_inner_init) {
    for (int F_c_inner_init = 0; F_c_inner_init < 4; ++F_c_inner_init) {
      output0_local[(((N_c_inner_init * 4) + F_c_inner_init))] = __float2half_rn(0.000000e+00f);
      output0_local[((((N_c_inner_init * 4) + F_c_inner_init) + 8))] = __float2half_rn(0.000000e+00f);
      output0_local[((((N_c_inner_init * 4) + F_c_inner_init) + 16))] = __float2half_rn(0.000000e+00f);
      output0_local[((((N_c_inner_init * 4) + F_c_inner_init) + 24))] = __float2half_rn(0.000000e+00f);
      output0_local[((((N_c_inner_init * 4) + F_c_inner_init) + 32))] = __float2half_rn(0.000000e+00f);
      output0_local[((((N_c_inner_init * 4) + F_c_inner_init) + 40))] = __float2half_rn(0.000000e+00f);
      output0_local[((((N_c_inner_init * 4) + F_c_inner_init) + 48))] = __float2half_rn(0.000000e+00f);
      output0_local[((((N_c_inner_init * 4) + F_c_inner_init) + 56))] = __float2half_rn(0.000000e+00f);
      output0_local[((((N_c_inner_init * 4) + F_c_inner_init) + 64))] = __float2half_rn(0.000000e+00f);
      output0_local[((((N_c_inner_init * 4) + F_c_inner_init) + 72))] = __float2half_rn(0.000000e+00f);
      output0_local[((((N_c_inner_init * 4) + F_c_inner_init) + 80))] = __float2half_rn(0.000000e+00f);
      output0_local[((((N_c_inner_init * 4) + F_c_inner_init) + 88))] = __float2half_rn(0.000000e+00f);
    }
  }
  for (int C_outer_outer = 0; C_outer_outer < 512; ++C_outer_outer) {
    __shared__ half input0_shared[128];
  // [thread_extent] threadIdx.x = 128
    __syncthreads();
    input0_shared[(((int)threadIdx.x))] = input0[((((((((((((int)blockIdx.x) / 882) * 28901376) + ((((int)threadIdx.x) >> 3) * 1806336)) + (((((int)blockIdx.x) % 882) / 42) * 86016)) + (((((int)threadIdx.x) & 7) >> 2) * 43008)) + (((((int)blockIdx.x) % 42) >> 1) * 2048)) + (((((int)threadIdx.x) & 3) >> 1) * 1024)) + (C_outer_outer * 2)) + (((int)threadIdx.x) & 1)))];
    __shared__ half input1_shared[384];
  // [thread_extent] threadIdx.x = 128
    input1_shared[(((int)threadIdx.x))] = input1[((((((((int)blockIdx.x) & 1) * 196608) + ((((int)threadIdx.x) >> 1) * 1024)) + (C_outer_outer * 2)) + (((int)threadIdx.x) & 1)))];
    input1_shared[((((int)threadIdx.x) + 128))] = input1[(((((((((int)blockIdx.x) & 1) * 196608) + ((((int)threadIdx.x) >> 1) * 1024)) + (C_outer_outer * 2)) + (((int)threadIdx.x) & 1)) + 65536))];
    input1_shared[((((int)threadIdx.x) + 256))] = input1[(((((((((int)blockIdx.x) & 1) * 196608) + ((((int)threadIdx.x) >> 1) * 1024)) + (C_outer_outer * 2)) + (((int)threadIdx.x) & 1)) + 131072))];
    __syncthreads();
    for (int C_inner = 0; C_inner < 2; ++C_inner) {
      for (int N_c_inner = 0; N_c_inner < 2; ++N_c_inner) {
        for (int F_c_inner = 0; F_c_inner < 4; ++F_c_inner) {
          output0_local[(((N_c_inner * 4) + F_c_inner))] = (output0_local[(((N_c_inner * 4) + F_c_inner))] + (input0_shared[((((((((int)threadIdx.x) >> 6) * 16) + (N_c_inner * 8)) + (((((int)threadIdx.x) & 63) >> 4) * 2)) + C_inner))] * input1_shared[(((((((int)threadIdx.x) & 15) * 8) + (F_c_inner * 2)) + C_inner))]));
          output0_local[((((N_c_inner * 4) + F_c_inner) + 8))] = (output0_local[((((N_c_inner * 4) + F_c_inner) + 8))] + (input0_shared[((((((((int)threadIdx.x) >> 6) * 16) + (N_c_inner * 8)) + (((((int)threadIdx.x) & 63) >> 4) * 2)) + C_inner))] * input1_shared[((((((((int)threadIdx.x) & 15) * 8) + (F_c_inner * 2)) + C_inner) + 128))]));
          output0_local[((((N_c_inner * 4) + F_c_inner) + 16))] = (output0_local[((((N_c_inner * 4) + F_c_inner) + 16))] + (input0_shared[((((((((int)threadIdx.x) >> 6) * 16) + (N_c_inner * 8)) + (((((int)threadIdx.x) & 63) >> 4) * 2)) + C_inner))] * input1_shared[((((((((int)threadIdx.x) & 15) * 8) + (F_c_inner * 2)) + C_inner) + 256))]));
          output0_local[((((N_c_inner * 4) + F_c_inner) + 24))] = (output0_local[((((N_c_inner * 4) + F_c_inner) + 24))] + (input0_shared[(((((((((int)threadIdx.x) >> 6) * 16) + (N_c_inner * 8)) + (((((int)threadIdx.x) & 63) >> 4) * 2)) + C_inner) + 32))] * input1_shared[(((((((int)threadIdx.x) & 15) * 8) + (F_c_inner * 2)) + C_inner))]));
          output0_local[((((N_c_inner * 4) + F_c_inner) + 32))] = (output0_local[((((N_c_inner * 4) + F_c_inner) + 32))] + (input0_shared[(((((((((int)threadIdx.x) >> 6) * 16) + (N_c_inner * 8)) + (((((int)threadIdx.x) & 63) >> 4) * 2)) + C_inner) + 32))] * input1_shared[((((((((int)threadIdx.x) & 15) * 8) + (F_c_inner * 2)) + C_inner) + 128))]));
          output0_local[((((N_c_inner * 4) + F_c_inner) + 40))] = (output0_local[((((N_c_inner * 4) + F_c_inner) + 40))] + (input0_shared[(((((((((int)threadIdx.x) >> 6) * 16) + (N_c_inner * 8)) + (((((int)threadIdx.x) & 63) >> 4) * 2)) + C_inner) + 32))] * input1_shared[((((((((int)threadIdx.x) & 15) * 8) + (F_c_inner * 2)) + C_inner) + 256))]));
          output0_local[((((N_c_inner * 4) + F_c_inner) + 48))] = (output0_local[((((N_c_inner * 4) + F_c_inner) + 48))] + (input0_shared[(((((((((int)threadIdx.x) >> 6) * 16) + (N_c_inner * 8)) + (((((int)threadIdx.x) & 63) >> 4) * 2)) + C_inner) + 64))] * input1_shared[(((((((int)threadIdx.x) & 15) * 8) + (F_c_inner * 2)) + C_inner))]));
          output0_local[((((N_c_inner * 4) + F_c_inner) + 56))] = (output0_local[((((N_c_inner * 4) + F_c_inner) + 56))] + (input0_shared[(((((((((int)threadIdx.x) >> 6) * 16) + (N_c_inner * 8)) + (((((int)threadIdx.x) & 63) >> 4) * 2)) + C_inner) + 64))] * input1_shared[((((((((int)threadIdx.x) & 15) * 8) + (F_c_inner * 2)) + C_inner) + 128))]));
          output0_local[((((N_c_inner * 4) + F_c_inner) + 64))] = (output0_local[((((N_c_inner * 4) + F_c_inner) + 64))] + (input0_shared[(((((((((int)threadIdx.x) >> 6) * 16) + (N_c_inner * 8)) + (((((int)threadIdx.x) & 63) >> 4) * 2)) + C_inner) + 64))] * input1_shared[((((((((int)threadIdx.x) & 15) * 8) + (F_c_inner * 2)) + C_inner) + 256))]));
          output0_local[((((N_c_inner * 4) + F_c_inner) + 72))] = (output0_local[((((N_c_inner * 4) + F_c_inner) + 72))] + (input0_shared[(((((((((int)threadIdx.x) >> 6) * 16) + (N_c_inner * 8)) + (((((int)threadIdx.x) & 63) >> 4) * 2)) + C_inner) + 96))] * input1_shared[(((((((int)threadIdx.x) & 15) * 8) + (F_c_inner * 2)) + C_inner))]));
          output0_local[((((N_c_inner * 4) + F_c_inner) + 80))] = (output0_local[((((N_c_inner * 4) + F_c_inner) + 80))] + (input0_shared[(((((((((int)threadIdx.x) >> 6) * 16) + (N_c_inner * 8)) + (((((int)threadIdx.x) & 63) >> 4) * 2)) + C_inner) + 96))] * input1_shared[((((((((int)threadIdx.x) & 15) * 8) + (F_c_inner * 2)) + C_inner) + 128))]));
          output0_local[((((N_c_inner * 4) + F_c_inner) + 88))] = (output0_local[((((N_c_inner * 4) + F_c_inner) + 88))] + (input0_shared[(((((((((int)threadIdx.x) >> 6) * 16) + (N_c_inner * 8)) + (((((int)threadIdx.x) & 63) >> 4) * 2)) + C_inner) + 96))] * input1_shared[((((((((int)threadIdx.x) & 15) * 8) + (F_c_inner * 2)) + C_inner) + 256))]));
        }
      }
    }
  }
  for (int N_inner = 0; N_inner < 2; ++N_inner) {
    for (int F_inner = 0; F_inner < 4; ++F_inner) {
      output0[((((((((((((((int)blockIdx.x) / 882) * 10838016) + ((((int)threadIdx.x) >> 6) * 1354752)) + (N_inner * 677376)) + (((((int)blockIdx.x) % 882) / 42) * 32256)) + (((((int)threadIdx.x) & 63) >> 5) * 16128)) + (((((int)blockIdx.x) % 42) >> 1) * 768)) + (((((int)threadIdx.x) & 31) >> 4) * 384)) + ((((int)blockIdx.x) & 1) * 192)) + ((((int)threadIdx.x) & 15) * 4)) + F_inner))] = output0_local[(((N_inner * 4) + F_inner))];
      output0[(((((((((((((((int)blockIdx.x) / 882) * 10838016) + ((((int)threadIdx.x) >> 6) * 1354752)) + (N_inner * 677376)) + (((((int)blockIdx.x) % 882) / 42) * 32256)) + (((((int)threadIdx.x) & 63) >> 5) * 16128)) + (((((int)blockIdx.x) % 42) >> 1) * 768)) + (((((int)threadIdx.x) & 31) >> 4) * 384)) + ((((int)blockIdx.x) & 1) * 192)) + ((((int)threadIdx.x) & 15) * 4)) + F_inner) + 64))] = output0_local[((((N_inner * 4) + F_inner) + 8))];
      output0[(((((((((((((((int)blockIdx.x) / 882) * 10838016) + ((((int)threadIdx.x) >> 6) * 1354752)) + (N_inner * 677376)) + (((((int)blockIdx.x) % 882) / 42) * 32256)) + (((((int)threadIdx.x) & 63) >> 5) * 16128)) + (((((int)blockIdx.x) % 42) >> 1) * 768)) + (((((int)threadIdx.x) & 31) >> 4) * 384)) + ((((int)blockIdx.x) & 1) * 192)) + ((((int)threadIdx.x) & 15) * 4)) + F_inner) + 128))] = output0_local[((((N_inner * 4) + F_inner) + 16))];
      output0[(((((((((((((((int)blockIdx.x) / 882) * 10838016) + ((((int)threadIdx.x) >> 6) * 1354752)) + (N_inner * 677376)) + (((((int)blockIdx.x) % 882) / 42) * 32256)) + (((((int)threadIdx.x) & 63) >> 5) * 16128)) + (((((int)blockIdx.x) % 42) >> 1) * 768)) + (((((int)threadIdx.x) & 31) >> 4) * 384)) + ((((int)blockIdx.x) & 1) * 192)) + ((((int)threadIdx.x) & 15) * 4)) + F_inner) + 2709504))] = output0_local[((((N_inner * 4) + F_inner) + 24))];
      output0[(((((((((((((((int)blockIdx.x) / 882) * 10838016) + ((((int)threadIdx.x) >> 6) * 1354752)) + (N_inner * 677376)) + (((((int)blockIdx.x) % 882) / 42) * 32256)) + (((((int)threadIdx.x) & 63) >> 5) * 16128)) + (((((int)blockIdx.x) % 42) >> 1) * 768)) + (((((int)threadIdx.x) & 31) >> 4) * 384)) + ((((int)blockIdx.x) & 1) * 192)) + ((((int)threadIdx.x) & 15) * 4)) + F_inner) + 2709568))] = output0_local[((((N_inner * 4) + F_inner) + 32))];
      output0[(((((((((((((((int)blockIdx.x) / 882) * 10838016) + ((((int)threadIdx.x) >> 6) * 1354752)) + (N_inner * 677376)) + (((((int)blockIdx.x) % 882) / 42) * 32256)) + (((((int)threadIdx.x) & 63) >> 5) * 16128)) + (((((int)blockIdx.x) % 42) >> 1) * 768)) + (((((int)threadIdx.x) & 31) >> 4) * 384)) + ((((int)blockIdx.x) & 1) * 192)) + ((((int)threadIdx.x) & 15) * 4)) + F_inner) + 2709632))] = output0_local[((((N_inner * 4) + F_inner) + 40))];
      output0[(((((((((((((((int)blockIdx.x) / 882) * 10838016) + ((((int)threadIdx.x) >> 6) * 1354752)) + (N_inner * 677376)) + (((((int)blockIdx.x) % 882) / 42) * 32256)) + (((((int)threadIdx.x) & 63) >> 5) * 16128)) + (((((int)blockIdx.x) % 42) >> 1) * 768)) + (((((int)threadIdx.x) & 31) >> 4) * 384)) + ((((int)blockIdx.x) & 1) * 192)) + ((((int)threadIdx.x) & 15) * 4)) + F_inner) + 5419008))] = output0_local[((((N_inner * 4) + F_inner) + 48))];
      output0[(((((((((((((((int)blockIdx.x) / 882) * 10838016) + ((((int)threadIdx.x) >> 6) * 1354752)) + (N_inner * 677376)) + (((((int)blockIdx.x) % 882) / 42) * 32256)) + (((((int)threadIdx.x) & 63) >> 5) * 16128)) + (((((int)blockIdx.x) % 42) >> 1) * 768)) + (((((int)threadIdx.x) & 31) >> 4) * 384)) + ((((int)blockIdx.x) & 1) * 192)) + ((((int)threadIdx.x) & 15) * 4)) + F_inner) + 5419072))] = output0_local[((((N_inner * 4) + F_inner) + 56))];
      output0[(((((((((((((((int)blockIdx.x) / 882) * 10838016) + ((((int)threadIdx.x) >> 6) * 1354752)) + (N_inner * 677376)) + (((((int)blockIdx.x) % 882) / 42) * 32256)) + (((((int)threadIdx.x) & 63) >> 5) * 16128)) + (((((int)blockIdx.x) % 42) >> 1) * 768)) + (((((int)threadIdx.x) & 31) >> 4) * 384)) + ((((int)blockIdx.x) & 1) * 192)) + ((((int)threadIdx.x) & 15) * 4)) + F_inner) + 5419136))] = output0_local[((((N_inner * 4) + F_inner) + 64))];
      output0[(((((((((((((((int)blockIdx.x) / 882) * 10838016) + ((((int)threadIdx.x) >> 6) * 1354752)) + (N_inner * 677376)) + (((((int)blockIdx.x) % 882) / 42) * 32256)) + (((((int)threadIdx.x) & 63) >> 5) * 16128)) + (((((int)blockIdx.x) % 42) >> 1) * 768)) + (((((int)threadIdx.x) & 31) >> 4) * 384)) + ((((int)blockIdx.x) & 1) * 192)) + ((((int)threadIdx.x) & 15) * 4)) + F_inner) + 8128512))] = output0_local[((((N_inner * 4) + F_inner) + 72))];
      output0[(((((((((((((((int)blockIdx.x) / 882) * 10838016) + ((((int)threadIdx.x) >> 6) * 1354752)) + (N_inner * 677376)) + (((((int)blockIdx.x) % 882) / 42) * 32256)) + (((((int)threadIdx.x) & 63) >> 5) * 16128)) + (((((int)blockIdx.x) % 42) >> 1) * 768)) + (((((int)threadIdx.x) & 31) >> 4) * 384)) + ((((int)blockIdx.x) & 1) * 192)) + ((((int)threadIdx.x) & 15) * 4)) + F_inner) + 8128576))] = output0_local[((((N_inner * 4) + F_inner) + 80))];
      output0[(((((((((((((((int)blockIdx.x) / 882) * 10838016) + ((((int)threadIdx.x) >> 6) * 1354752)) + (N_inner * 677376)) + (((((int)blockIdx.x) % 882) / 42) * 32256)) + (((((int)threadIdx.x) & 63) >> 5) * 16128)) + (((((int)blockIdx.x) % 42) >> 1) * 768)) + (((((int)threadIdx.x) & 31) >> 4) * 384)) + ((((int)blockIdx.x) & 1) * 192)) + ((((int)threadIdx.x) & 15) * 4)) + F_inner) + 8128640))] = output0_local[((((N_inner * 4) + F_inner) + 88))];
    }
  }
}

// Saved Perf = 1.648990e-02 sec / run; Step Produced = 69; Planned Steps = 1000;