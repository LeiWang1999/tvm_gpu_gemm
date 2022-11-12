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
extern "C" __global__ void __launch_bounds__(256) default_function_kernel0(signed char* __restrict__ A, signed char* __restrict__ B, int* __restrict__ C) {
  int C_local[64];
  __shared__ signed char A_shared[4096];
  __shared__ signed char B_shared[4096];
  for (int i_c_outer_init = 0; i_c_outer_init < 8; ++i_c_outer_init) {
    for (int j_c_outer_init = 0; j_c_outer_init < 8; ++j_c_outer_init) {
      C_local[((i_c_outer_init * 8) + j_c_outer_init)] = 0;
    }
  }
  for (int k_outer = 0; k_outer < 512; ++k_outer) {
    __syncthreads();
    *(int4*)(A_shared + ((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 16))) = *(int4*)(A + (((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.y) * 131072)) + ((((int)threadIdx.x) >> 1) * 16384)) + (k_outer * 32)) + ((((int)threadIdx.x) & 1) * 16)));
    *(int4*)(B_shared + ((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 16))) = *(int4*)(B + (((((((int)blockIdx.y) * 2097152) + (((int)threadIdx.y) * 131072)) + ((((int)threadIdx.x) >> 1) * 16384)) + (k_outer * 32)) + ((((int)threadIdx.x) & 1) * 16)));
    __syncthreads();
    for (int k_inner_outer = 0; k_inner_outer < 8; ++k_inner_outer) {
      for (int i_c_outer = 0; i_c_outer < 8; ++i_c_outer) {
        for (int j_c_outer = 0; j_c_outer < 8; ++j_c_outer) {
          C_local[((i_c_outer * 8) + j_c_outer)] = __dp4a(*(int*)(A_shared + (((((int)threadIdx.x) * 256) + (i_c_outer * 32)) + (k_inner_outer * 4))), *(int*)(B_shared + (((((int)threadIdx.y) * 256) + (j_c_outer * 32)) + (k_inner_outer * 4))), C_local[((i_c_outer * 8) + j_c_outer)]);
        }
      }
    }
  }
  for (int i_inner_inner = 0; i_inner_inner < 8; ++i_inner_inner) {
    for (int j_inner_inner = 0; j_inner_inner < 8; ++j_inner_inner) {
      C[((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.x) * 131072)) + (i_inner_inner * 16384)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 8)) + j_inner_inner)] = C_local[((i_inner_inner * 8) + j_inner_inner)];
    }
  }
}

