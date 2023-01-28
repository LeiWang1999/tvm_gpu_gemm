#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
#include <sm_61_intrinsics.h>
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(256) main_kernel0(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ C) {
  signed char C_local[64];
  __shared__ int A_shared[1024];
  __shared__ signed char B_shared[4096];
  signed char A_shared_local[32];
  signed char B_shared_local[32];
  for (int j_1_1_0_1_init = 0; j_1_1_0_1_init < 2; ++j_1_1_0_1_init) {
    for (int i_1_1_0_1_init = 0; i_1_1_0_1_init < 2; ++i_1_1_0_1_init) {
      *(float*)(C_local + ((i_1_1_0_1_init * 2) + j_1_1_0_1_init)) = 0.000000e+00f;
      *(float*)(C_local + (((i_1_1_0_1_init * 2) + j_1_1_0_1_init) + 16)) = 0.000000e+00f;
      *(float*)(C_local + (((i_1_1_0_1_init * 2) + j_1_1_0_1_init) + 32)) = 0.000000e+00f;
      *(float*)(C_local + (((i_1_1_0_1_init * 2) + j_1_1_0_1_init) + 48)) = 0.000000e+00f;
      *(float*)(C_local + (((i_1_1_0_1_init * 2) + j_1_1_0_1_init) + 4)) = 0.000000e+00f;
      *(float*)(C_local + (((i_1_1_0_1_init * 2) + j_1_1_0_1_init) + 20)) = 0.000000e+00f;
      *(float*)(C_local + (((i_1_1_0_1_init * 2) + j_1_1_0_1_init) + 36)) = 0.000000e+00f;
      *(float*)(C_local + (((i_1_1_0_1_init * 2) + j_1_1_0_1_init) + 52)) = 0.000000e+00f;
      *(float*)(C_local + (((i_1_1_0_1_init * 2) + j_1_1_0_1_init) + 8)) = 0.000000e+00f;
      *(float*)(C_local + (((i_1_1_0_1_init * 2) + j_1_1_0_1_init) + 24)) = 0.000000e+00f;
      *(float*)(C_local + (((i_1_1_0_1_init * 2) + j_1_1_0_1_init) + 40)) = 0.000000e+00f;
      *(float*)(C_local + (((i_1_1_0_1_init * 2) + j_1_1_0_1_init) + 56)) = 0.000000e+00f;
      *(float*)(C_local + (((i_1_1_0_1_init * 2) + j_1_1_0_1_init) + 12)) = 0.000000e+00f;
      *(float*)(C_local + (((i_1_1_0_1_init * 2) + j_1_1_0_1_init) + 28)) = 0.000000e+00f;
      *(float*)(C_local + (((i_1_1_0_1_init * 2) + j_1_1_0_1_init) + 44)) = 0.000000e+00f;
      *(float*)(C_local + (((i_1_1_0_1_init * 2) + j_1_1_0_1_init) + 60)) = 0.000000e+00f;
    }
  }
  for (int k_0 = 0; k_0 < 512; ++k_0) {
    __syncthreads();
    for (int ax0_ax1_fused_2 = 0; ax0_ax1_fused_2 < 4; ++ax0_ax1_fused_2) {
      A_shared[(((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_2)] = *(int*)(A + ((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.y) * 131072)) + ((((int)threadIdx.x) >> 1) * 16384)) + (k_0 * 32)) + ((((int)threadIdx.x) & 1) * 16)) + (ax0_ax1_fused_2 * 4)));
    }
    for (int ax0_ax1_fused_2_1 = 0; ax0_ax1_fused_2_1 < 4; ++ax0_ax1_fused_2_1) {
      *(int*)(B_shared + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 16)) + (ax0_ax1_fused_2_1 * 4))) = *(int*)(B + ((((((k_0 * 524288) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 3) * 16384)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + (ax0_ax1_fused_2_1 * 4)));
    }
    __syncthreads();
    for (int k_1_1_0 = 0; k_1_1_0 < 8; ++k_1_1_0) {
      for (int ax0 = 0; ax0 < 2; ++ax0) {
        *(int*)(A_shared_local + (ax0 * 4)) = A_shared[(((((int)threadIdx.x) * 16) + (ax0 * 8)) + k_1_1_0)];
        *(int*)(A_shared_local + ((ax0 * 4) + 8)) = A_shared[((((((int)threadIdx.x) * 16) + (ax0 * 8)) + k_1_1_0) + 256)];
        *(int*)(A_shared_local + ((ax0 * 4) + 16)) = A_shared[((((((int)threadIdx.x) * 16) + (ax0 * 8)) + k_1_1_0) + 512)];
        *(int*)(A_shared_local + ((ax0 * 4) + 24)) = A_shared[((((((int)threadIdx.x) * 16) + (ax0 * 8)) + k_1_1_0) + 768)];
      }
      for (int ax0_1 = 0; ax0_1 < 4; ++ax0_1) {
        *(char2*)(B_shared_local + (ax0_1 * 2)) = *(char2*)(B_shared + (((k_1_1_0 * 512) + (ax0_1 * 128)) + (((int)threadIdx.y) * 2)));
        *(char2*)(B_shared_local + ((ax0_1 * 2) + 8)) = *(char2*)(B_shared + ((((k_1_1_0 * 512) + (ax0_1 * 128)) + (((int)threadIdx.y) * 2)) + 32));
        *(char2*)(B_shared_local + ((ax0_1 * 2) + 16)) = *(char2*)(B_shared + ((((k_1_1_0 * 512) + (ax0_1 * 128)) + (((int)threadIdx.y) * 2)) + 64));
        *(char2*)(B_shared_local + ((ax0_1 * 2) + 24)) = *(char2*)(B_shared + ((((k_1_1_0 * 512) + (ax0_1 * 128)) + (((int)threadIdx.y) * 2)) + 96));
      }
      for (int j_1_1_0_1 = 0; j_1_1_0_1 < 2; ++j_1_1_0_1) {
        for (int i_1_1_0_1 = 0; i_1_1_0_1 < 2; ++i_1_1_0_1) {
          for (int k_1_1_1 = 0; k_1_1_1 < 4; ++k_1_1_1) {
            C_local[((i_1_1_0_1 * 2) + j_1_1_0_1)] = (C_local[((i_1_1_0_1 * 2) + j_1_1_0_1)] + (A_shared_local[((i_1_1_0_1 * 4) + k_1_1_1)] * B_shared_local[((k_1_1_1 * 2) + j_1_1_0_1)]));
            C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 16)] = (C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 16)] + (A_shared_local[((i_1_1_0_1 * 4) + k_1_1_1)] * B_shared_local[(((k_1_1_1 * 2) + j_1_1_0_1) + 8)]));
            C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 32)] = (C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 32)] + (A_shared_local[((i_1_1_0_1 * 4) + k_1_1_1)] * B_shared_local[(((k_1_1_1 * 2) + j_1_1_0_1) + 16)]));
            C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 48)] = (C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 48)] + (A_shared_local[((i_1_1_0_1 * 4) + k_1_1_1)] * B_shared_local[(((k_1_1_1 * 2) + j_1_1_0_1) + 24)]));
            C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 4)] = (C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 4)] + (A_shared_local[(((i_1_1_0_1 * 4) + k_1_1_1) + 8)] * B_shared_local[((k_1_1_1 * 2) + j_1_1_0_1)]));
            C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 20)] = (C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 20)] + (A_shared_local[(((i_1_1_0_1 * 4) + k_1_1_1) + 8)] * B_shared_local[(((k_1_1_1 * 2) + j_1_1_0_1) + 8)]));
            C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 36)] = (C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 36)] + (A_shared_local[(((i_1_1_0_1 * 4) + k_1_1_1) + 8)] * B_shared_local[(((k_1_1_1 * 2) + j_1_1_0_1) + 16)]));
            C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 52)] = (C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 52)] + (A_shared_local[(((i_1_1_0_1 * 4) + k_1_1_1) + 8)] * B_shared_local[(((k_1_1_1 * 2) + j_1_1_0_1) + 24)]));
            C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 8)] = (C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 8)] + (A_shared_local[(((i_1_1_0_1 * 4) + k_1_1_1) + 16)] * B_shared_local[((k_1_1_1 * 2) + j_1_1_0_1)]));
            C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 24)] = (C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 24)] + (A_shared_local[(((i_1_1_0_1 * 4) + k_1_1_1) + 16)] * B_shared_local[(((k_1_1_1 * 2) + j_1_1_0_1) + 8)]));
            C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 40)] = (C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 40)] + (A_shared_local[(((i_1_1_0_1 * 4) + k_1_1_1) + 16)] * B_shared_local[(((k_1_1_1 * 2) + j_1_1_0_1) + 16)]));
            C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 56)] = (C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 56)] + (A_shared_local[(((i_1_1_0_1 * 4) + k_1_1_1) + 16)] * B_shared_local[(((k_1_1_1 * 2) + j_1_1_0_1) + 24)]));
            C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 12)] = (C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 12)] + (A_shared_local[(((i_1_1_0_1 * 4) + k_1_1_1) + 24)] * B_shared_local[((k_1_1_1 * 2) + j_1_1_0_1)]));
            C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 28)] = (C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 28)] + (A_shared_local[(((i_1_1_0_1 * 4) + k_1_1_1) + 24)] * B_shared_local[(((k_1_1_1 * 2) + j_1_1_0_1) + 8)]));
            C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 44)] = (C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 44)] + (A_shared_local[(((i_1_1_0_1 * 4) + k_1_1_1) + 24)] * B_shared_local[(((k_1_1_1 * 2) + j_1_1_0_1) + 16)]));
            C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 60)] = (C_local[(((i_1_1_0_1 * 2) + j_1_1_0_1) + 60)] + (A_shared_local[(((i_1_1_0_1 * 4) + k_1_1_1) + 24)] * B_shared_local[(((k_1_1_1 * 2) + j_1_1_0_1) + 24)]));
          }
        }
      }
    }
  }
  for (int ax0_2 = 0; ax0_2 < 2; ++ax0_2) {
    for (int ax1 = 0; ax1 < 2; ++ax1) {
      C[((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.x) * 32768)) + (ax0_2 * 16384)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 2)) + ax1)] = C_local[((ax0_2 * 2) + ax1)];
      C[(((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.x) * 32768)) + (ax0_2 * 16384)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 2)) + ax1) + 32)] = C_local[(((ax0_2 * 2) + ax1) + 16)];
      C[(((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.x) * 32768)) + (ax0_2 * 16384)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 2)) + ax1) + 64)] = C_local[(((ax0_2 * 2) + ax1) + 32)];
      C[(((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.x) * 32768)) + (ax0_2 * 16384)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 2)) + ax1) + 96)] = C_local[(((ax0_2 * 2) + ax1) + 48)];
      C[(((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.x) * 32768)) + (ax0_2 * 16384)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 2)) + ax1) + 524288)] = C_local[(((ax0_2 * 2) + ax1) + 4)];
      C[(((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.x) * 32768)) + (ax0_2 * 16384)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 2)) + ax1) + 524320)] = C_local[(((ax0_2 * 2) + ax1) + 20)];
      C[(((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.x) * 32768)) + (ax0_2 * 16384)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 2)) + ax1) + 524352)] = C_local[(((ax0_2 * 2) + ax1) + 36)];
      C[(((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.x) * 32768)) + (ax0_2 * 16384)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 2)) + ax1) + 524384)] = C_local[(((ax0_2 * 2) + ax1) + 52)];
      C[(((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.x) * 32768)) + (ax0_2 * 16384)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 2)) + ax1) + 1048576)] = C_local[(((ax0_2 * 2) + ax1) + 8)];
      C[(((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.x) * 32768)) + (ax0_2 * 16384)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 2)) + ax1) + 1048608)] = C_local[(((ax0_2 * 2) + ax1) + 24)];
      C[(((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.x) * 32768)) + (ax0_2 * 16384)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 2)) + ax1) + 1048640)] = C_local[(((ax0_2 * 2) + ax1) + 40)];
      C[(((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.x) * 32768)) + (ax0_2 * 16384)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 2)) + ax1) + 1048672)] = C_local[(((ax0_2 * 2) + ax1) + 56)];
      C[(((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.x) * 32768)) + (ax0_2 * 16384)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 2)) + ax1) + 1572864)] = C_local[(((ax0_2 * 2) + ax1) + 12)];
      C[(((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.x) * 32768)) + (ax0_2 * 16384)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 2)) + ax1) + 1572896)] = C_local[(((ax0_2 * 2) + ax1) + 28)];
      C[(((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.x) * 32768)) + (ax0_2 * 16384)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 2)) + ax1) + 1572928)] = C_local[(((ax0_2 * 2) + ax1) + 44)];
      C[(((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.x) * 32768)) + (ax0_2 * 16384)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 2)) + ax1) + 1572960)] = C_local[(((ax0_2 * 2) + ax1) + 60)];
    }
  }
}

