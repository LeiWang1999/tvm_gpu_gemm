// GLOBALS: input0:int8[18966528, 32], input1:int8[1, 32] -> output0:int32[18966528, 32]
// BACKEND: c-cuda (default)
// CONFIG: {"Moutput0T": 0, "Moutput0:D0": [-1, 2, 16, 7], "Moutput0:D1": [-1, 2, 8, 2], "Moutput0:R0": [-1, 2, 16], "Moutput0:RA": 0, "Moutput0:AL0": 0, "Moutput0:AL1": 1, "Moutput0:S": 3, "Moutput0:U": 1}
// COMPUTE_V1: - einstein_v2("output0[N, M] +=! input0[N, K].cast(`int32`) * input1[K, M].cast(`int32`)", { "input0": {"dtype": "int8", "shape": [18966528, 32]}, "input1": {"dtype": "int8", "shape": [1, 32]}})


// ---------------------------------------------------------------------------
// LOCAL: template_op_kernel0 -- input0:int8[18966528, 32], input1:int8[1, 32] -> output0:int32[18966528, 32]

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


extern "C" __global__ __launch_bounds__(128) void template_op_kernel0(char* __restrict__ input0, char* __restrict__ input1, int* __restrict__ output0) {
  // [thread_extent] blockIdx.x = 84672
  // [thread_extent] threadIdx.x = 128
  int output0_local[56];
  output0_local[(0)] = 0;
  output0_local[(14)] = 0;
  output0_local[(28)] = 0;
  output0_local[(42)] = 0;
  output0_local[(1)] = 0;
  output0_local[(15)] = 0;
  output0_local[(29)] = 0;
  output0_local[(43)] = 0;
  output0_local[(2)] = 0;
  output0_local[(16)] = 0;
  output0_local[(30)] = 0;
  output0_local[(44)] = 0;
  output0_local[(3)] = 0;
  output0_local[(17)] = 0;
  output0_local[(31)] = 0;
  output0_local[(45)] = 0;
  output0_local[(4)] = 0;
  output0_local[(18)] = 0;
  output0_local[(32)] = 0;
  output0_local[(46)] = 0;
  output0_local[(5)] = 0;
  output0_local[(19)] = 0;
  output0_local[(33)] = 0;
  output0_local[(47)] = 0;
  output0_local[(6)] = 0;
  output0_local[(20)] = 0;
  output0_local[(34)] = 0;
  output0_local[(48)] = 0;
  output0_local[(7)] = 0;
  output0_local[(21)] = 0;
  output0_local[(35)] = 0;
  output0_local[(49)] = 0;
  output0_local[(8)] = 0;
  output0_local[(22)] = 0;
  output0_local[(36)] = 0;
  output0_local[(50)] = 0;
  output0_local[(9)] = 0;
  output0_local[(23)] = 0;
  output0_local[(37)] = 0;
  output0_local[(51)] = 0;
  output0_local[(10)] = 0;
  output0_local[(24)] = 0;
  output0_local[(38)] = 0;
  output0_local[(52)] = 0;
  output0_local[(11)] = 0;
  output0_local[(25)] = 0;
  output0_local[(39)] = 0;
  output0_local[(53)] = 0;
  output0_local[(12)] = 0;
  output0_local[(26)] = 0;
  output0_local[(40)] = 0;
  output0_local[(54)] = 0;
  output0_local[(13)] = 0;
  output0_local[(27)] = 0;
  output0_local[(41)] = 0;
  output0_local[(55)] = 0;
  __shared__ char input0_shared[7168];
  // [thread_extent] threadIdx.x = 128
  input0_shared[(((int)threadIdx.x))] = input0[(((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)))];
  input0_shared[((((int)threadIdx.x) + 128))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 128))];
  input0_shared[((((int)threadIdx.x) + 256))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 256))];
  input0_shared[((((int)threadIdx.x) + 384))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 384))];
  input0_shared[((((int)threadIdx.x) + 512))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 512))];
  input0_shared[((((int)threadIdx.x) + 640))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 640))];
  input0_shared[((((int)threadIdx.x) + 768))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 768))];
  input0_shared[((((int)threadIdx.x) + 896))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 896))];
  input0_shared[((((int)threadIdx.x) + 1024))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 1024))];
  input0_shared[((((int)threadIdx.x) + 1152))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 1152))];
  input0_shared[((((int)threadIdx.x) + 1280))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 1280))];
  input0_shared[((((int)threadIdx.x) + 1408))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 1408))];
  input0_shared[((((int)threadIdx.x) + 1536))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 1536))];
  input0_shared[((((int)threadIdx.x) + 1664))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 1664))];
  input0_shared[((((int)threadIdx.x) + 1792))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 1792))];
  input0_shared[((((int)threadIdx.x) + 1920))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 1920))];
  input0_shared[((((int)threadIdx.x) + 2048))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 2048))];
  input0_shared[((((int)threadIdx.x) + 2176))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 2176))];
  input0_shared[((((int)threadIdx.x) + 2304))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 2304))];
  input0_shared[((((int)threadIdx.x) + 2432))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 2432))];
  input0_shared[((((int)threadIdx.x) + 2560))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 2560))];
  input0_shared[((((int)threadIdx.x) + 2688))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 2688))];
  input0_shared[((((int)threadIdx.x) + 2816))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 2816))];
  input0_shared[((((int)threadIdx.x) + 2944))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 2944))];
  input0_shared[((((int)threadIdx.x) + 3072))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 3072))];
  input0_shared[((((int)threadIdx.x) + 3200))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 3200))];
  input0_shared[((((int)threadIdx.x) + 3328))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 3328))];
  input0_shared[((((int)threadIdx.x) + 3456))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 3456))];
  input0_shared[((((int)threadIdx.x) + 3584))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 3584))];
  input0_shared[((((int)threadIdx.x) + 3712))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 3712))];
  input0_shared[((((int)threadIdx.x) + 3840))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 3840))];
  input0_shared[((((int)threadIdx.x) + 3968))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 3968))];
  input0_shared[((((int)threadIdx.x) + 4096))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 4096))];
  input0_shared[((((int)threadIdx.x) + 4224))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 4224))];
  input0_shared[((((int)threadIdx.x) + 4352))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 4352))];
  input0_shared[((((int)threadIdx.x) + 4480))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 4480))];
  input0_shared[((((int)threadIdx.x) + 4608))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 4608))];
  input0_shared[((((int)threadIdx.x) + 4736))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 4736))];
  input0_shared[((((int)threadIdx.x) + 4864))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 4864))];
  input0_shared[((((int)threadIdx.x) + 4992))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 4992))];
  input0_shared[((((int)threadIdx.x) + 5120))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 5120))];
  input0_shared[((((int)threadIdx.x) + 5248))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 5248))];
  input0_shared[((((int)threadIdx.x) + 5376))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 5376))];
  input0_shared[((((int)threadIdx.x) + 5504))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 5504))];
  input0_shared[((((int)threadIdx.x) + 5632))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 5632))];
  input0_shared[((((int)threadIdx.x) + 5760))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 5760))];
  input0_shared[((((int)threadIdx.x) + 5888))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 5888))];
  input0_shared[((((int)threadIdx.x) + 6016))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 6016))];
  input0_shared[((((int)threadIdx.x) + 6144))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 6144))];
  input0_shared[((((int)threadIdx.x) + 6272))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 6272))];
  input0_shared[((((int)threadIdx.x) + 6400))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 6400))];
  input0_shared[((((int)threadIdx.x) + 6528))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 6528))];
  input0_shared[((((int)threadIdx.x) + 6656))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 6656))];
  input0_shared[((((int)threadIdx.x) + 6784))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 6784))];
  input0_shared[((((int)threadIdx.x) + 6912))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 6912))];
  input0_shared[((((int)threadIdx.x) + 7040))] = input0[((((((int)blockIdx.x) * 7168) + ((int)threadIdx.x)) + 7040))];
  __shared__ char input1_shared[33];
  // [thread_extent] threadIdx.x = 128
  if (((int)threadIdx.x) < 32) {
    input1_shared[(((int)threadIdx.x))] = input1[(((int)threadIdx.x))];
  }
  __syncthreads();
  for (int K_inner = 0; K_inner < 32; ++K_inner) {
    output0_local[(0)] = (output0_local[(0)] + (((int)input0_shared[((((((int)threadIdx.x) >> 3) * 224) + K_inner))]) * ((int)input1_shared[(((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)))])));
    output0_local[(14)] = (output0_local[(14)] + (((int)input0_shared[((((((int)threadIdx.x) >> 3) * 224) + K_inner))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 16))])));
    output0_local[(28)] = (output0_local[(28)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3584))]) * ((int)input1_shared[(((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)))])));
    output0_local[(42)] = (output0_local[(42)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3584))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 16))])));
    output0_local[(1)] = (output0_local[(1)] + (((int)input0_shared[((((((int)threadIdx.x) >> 3) * 224) + K_inner))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 1))])));
    output0_local[(15)] = (output0_local[(15)] + (((int)input0_shared[((((((int)threadIdx.x) >> 3) * 224) + K_inner))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 17))])));
    output0_local[(29)] = (output0_local[(29)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3584))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 1))])));
    output0_local[(43)] = (output0_local[(43)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3584))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 17))])));
    output0_local[(2)] = (output0_local[(2)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 32))]) * ((int)input1_shared[(((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)))])));
    output0_local[(16)] = (output0_local[(16)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 32))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 16))])));
    output0_local[(30)] = (output0_local[(30)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3616))]) * ((int)input1_shared[(((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)))])));
    output0_local[(44)] = (output0_local[(44)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3616))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 16))])));
    output0_local[(3)] = (output0_local[(3)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 32))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 1))])));
    output0_local[(17)] = (output0_local[(17)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 32))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 17))])));
    output0_local[(31)] = (output0_local[(31)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3616))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 1))])));
    output0_local[(45)] = (output0_local[(45)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3616))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 17))])));
    output0_local[(4)] = (output0_local[(4)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 64))]) * ((int)input1_shared[(((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)))])));
    output0_local[(18)] = (output0_local[(18)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 64))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 16))])));
    output0_local[(32)] = (output0_local[(32)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3648))]) * ((int)input1_shared[(((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)))])));
    output0_local[(46)] = (output0_local[(46)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3648))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 16))])));
    output0_local[(5)] = (output0_local[(5)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 64))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 1))])));
    output0_local[(19)] = (output0_local[(19)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 64))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 17))])));
    output0_local[(33)] = (output0_local[(33)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3648))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 1))])));
    output0_local[(47)] = (output0_local[(47)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3648))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 17))])));
    output0_local[(6)] = (output0_local[(6)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 96))]) * ((int)input1_shared[(((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)))])));
    output0_local[(20)] = (output0_local[(20)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 96))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 16))])));
    output0_local[(34)] = (output0_local[(34)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3680))]) * ((int)input1_shared[(((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)))])));
    output0_local[(48)] = (output0_local[(48)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3680))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 16))])));
    output0_local[(7)] = (output0_local[(7)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 96))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 1))])));
    output0_local[(21)] = (output0_local[(21)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 96))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 17))])));
    output0_local[(35)] = (output0_local[(35)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3680))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 1))])));
    output0_local[(49)] = (output0_local[(49)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3680))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 17))])));
    output0_local[(8)] = (output0_local[(8)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 128))]) * ((int)input1_shared[(((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)))])));
    output0_local[(22)] = (output0_local[(22)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 128))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 16))])));
    output0_local[(36)] = (output0_local[(36)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3712))]) * ((int)input1_shared[(((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)))])));
    output0_local[(50)] = (output0_local[(50)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3712))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 16))])));
    output0_local[(9)] = (output0_local[(9)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 128))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 1))])));
    output0_local[(23)] = (output0_local[(23)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 128))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 17))])));
    output0_local[(37)] = (output0_local[(37)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3712))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 1))])));
    output0_local[(51)] = (output0_local[(51)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3712))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 17))])));
    output0_local[(10)] = (output0_local[(10)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 160))]) * ((int)input1_shared[(((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)))])));
    output0_local[(24)] = (output0_local[(24)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 160))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 16))])));
    output0_local[(38)] = (output0_local[(38)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3744))]) * ((int)input1_shared[(((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)))])));
    output0_local[(52)] = (output0_local[(52)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3744))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 16))])));
    output0_local[(11)] = (output0_local[(11)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 160))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 1))])));
    output0_local[(25)] = (output0_local[(25)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 160))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 17))])));
    output0_local[(39)] = (output0_local[(39)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3744))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 1))])));
    output0_local[(53)] = (output0_local[(53)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3744))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 17))])));
    output0_local[(12)] = (output0_local[(12)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 192))]) * ((int)input1_shared[(((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)))])));
    output0_local[(26)] = (output0_local[(26)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 192))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 16))])));
    output0_local[(40)] = (output0_local[(40)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3776))]) * ((int)input1_shared[(((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)))])));
    output0_local[(54)] = (output0_local[(54)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3776))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 16))])));
    output0_local[(13)] = (output0_local[(13)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 192))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 1))])));
    output0_local[(27)] = (output0_local[(27)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 192))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 17))])));
    output0_local[(41)] = (output0_local[(41)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3776))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 1))])));
    output0_local[(55)] = (output0_local[(55)] + (((int)input0_shared[(((((((int)threadIdx.x) >> 3) * 224) + K_inner) + 3776))]) * ((int)input1_shared[((((K_inner * 33) + ((((int)threadIdx.x) & 7) * 2)) + 17))])));
  }
  for (int N_inner = 0; N_inner < 7; ++N_inner) {
    for (int M_inner = 0; M_inner < 2; ++M_inner) {
      output0[((((((((int)blockIdx.x) * 7168) + ((((int)threadIdx.x) >> 3) * 224)) + (N_inner * 32)) + ((((int)threadIdx.x) & 7) * 2)) + M_inner))] = output0_local[(((N_inner * 2) + M_inner))];
      output0[(((((((((int)blockIdx.x) * 7168) + ((((int)threadIdx.x) >> 3) * 224)) + (N_inner * 32)) + ((((int)threadIdx.x) & 7) * 2)) + M_inner) + 16))] = output0_local[((((N_inner * 2) + M_inner) + 14))];
      output0[(((((((((int)blockIdx.x) * 7168) + ((((int)threadIdx.x) >> 3) * 224)) + (N_inner * 32)) + ((((int)threadIdx.x) & 7) * 2)) + M_inner) + 3584))] = output0_local[((((N_inner * 2) + M_inner) + 28))];
      output0[(((((((((int)blockIdx.x) * 7168) + ((((int)threadIdx.x) >> 3) * 224)) + (N_inner * 32)) + ((((int)threadIdx.x) & 7) * 2)) + M_inner) + 3600))] = output0_local[((((N_inner * 2) + M_inner) + 42))];
    }
  }
}

// Saved Perf = 4.736940e-03 sec / run; Step Produced = 984; Planned Steps = 1000;
// Antares Tuning Completed in 1000 steps.