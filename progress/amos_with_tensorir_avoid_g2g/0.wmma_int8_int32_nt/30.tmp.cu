#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
#include <sm_61_intrinsics.h>
#endif
#include <mma.h>

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
extern "C" __global__ void __launch_bounds__(128) main_kernel2(signed char* __restrict__ A_global, signed char* __restrict__ B_global, int* __restrict__ C) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int> C_wmma_accumulator[16];
  __shared__ signed char A_global_shared[2048];
  __shared__ signed char B_global_shared[8192];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, signed char, nvcuda::wmma::row_major> A_global_shared_wmma_matrix_a[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, signed char, nvcuda::wmma::col_major> B_global_shared_wmma_matrix_b[8];
  for (int i_0_2_init = 0; i_0_2_init < 2; ++i_0_2_init) {
    for (int j_0_2_init = 0; j_0_2_init < 8; ++j_0_2_init) {
      nvcuda::wmma::fill_fragment(C_wmma_accumulator[((i_0_2_init * 8) + j_0_2_init)], 0.000000e+00f);
    }
  }
  for (int k_0_0 = 0; k_0_0 < 512; ++k_0_0) {
    __syncthreads();
    *(int4*)(A_global_shared + (((((int)threadIdx.y) * 1024) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(A_global + (((((((int)blockIdx.y) * 1048576) + (((int)threadIdx.y) * 524288)) + (((int)threadIdx.z) * 262144)) + (k_0_0 * 512)) + (((int)threadIdx.x) * 16)));
    for (int ax0_0_ax1_0_ax0_1_ax1_1_fused_2 = 0; ax0_0_ax1_0_ax0_1_ax1_1_fused_2 < 4; ++ax0_0_ax1_0_ax0_1_ax1_1_fused_2) {
      *(int4*)(B_global_shared + ((((((int)threadIdx.y) * 4096) + (((int)threadIdx.z) * 2048)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(B_global + ((((((((int)blockIdx.z) * 4194304) + (((int)threadIdx.y) * 2097152)) + (((int)threadIdx.z) * 1048576)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 262144)) + (k_0_0 * 512)) + (((int)threadIdx.x) * 16)));
    }
    __syncthreads();
    for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
      for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0) {
        nvcuda::wmma::load_matrix_sync(A_global_shared_wmma_matrix_a[ax0_0], (&(A_global_shared[(((((int)threadIdx.y) * 1024) + (ax0_0 * 512)) + (k_0_1 * 256))])), 16);
      }
      for (int ax0_0_1 = 0; ax0_0_1 < 8; ++ax0_0_1) {
        nvcuda::wmma::load_matrix_sync(B_global_shared_wmma_matrix_b[ax0_0_1], (&(B_global_shared[(((((int)threadIdx.z) * 4096) + (ax0_0_1 * 512)) + (k_0_1 * 256))])), 16);
      }
      for (int i_0_2 = 0; i_0_2 < 2; ++i_0_2) {
        for (int j_0_2 = 0; j_0_2 < 8; ++j_0_2) {
          nvcuda::wmma::mma_sync(C_wmma_accumulator[((i_0_2 * 8) + j_0_2)], A_global_shared_wmma_matrix_a[i_0_2], B_global_shared_wmma_matrix_b[j_0_2], C_wmma_accumulator[((i_0_2 * 8) + j_0_2)]);
        }
      }
    }
  }
  for (int ax0_0_2 = 0; ax0_0_2 < 2; ++ax0_0_2) {
    for (int ax1_0 = 0; ax1_0 < 8; ++ax1_0) {
      nvcuda::wmma::store_matrix_sync((&(C[((((((((int)blockIdx.y) * 1048576) + (((int)threadIdx.y) * 524288)) + (ax0_0_2 * 262144)) + (((int)blockIdx.z) * 256)) + (((int)threadIdx.z) * 128)) + (ax1_0 * 16))])), C_wmma_accumulator[((ax0_0_2 * 8) + ax1_0)], 16384, nvcuda::wmma::mem_row_major);
    }
  }
}

extern "C" __global__ void __launch_bounds__(8) main_kernel0(signed char* __restrict__ B_global, signed char* __restrict__ B) {
  for (int ax0_ax1_fused_2 = 0; ax0_ax1_fused_2 < 4; ++ax0_ax1_fused_2) {
    *(int4*)(B_global + (((((((((((int)blockIdx.y) * 32768) + (((int)blockIdx.x) * 512)) + (ax0_ax1_fused_2 * 128)) + (((int)threadIdx.x) * 16)) >> 18) * 262144) + (((((((int)blockIdx.x) * 32) + (ax0_ax1_fused_2 * 8)) + ((int)threadIdx.x)) & 1023) * 256)) + ((((int)blockIdx.y) & 7) * 32)) + ((((int)blockIdx.x) >> 5) * 16))) = *(int4*)(B + ((((((int)blockIdx.y) * 32768) + (((int)blockIdx.x) * 512)) + (ax0_ax1_fused_2 * 128)) + (((int)threadIdx.x) * 16)));
  }
}

extern "C" __global__ void __launch_bounds__(8) main_kernel1(signed char* __restrict__ A_global, signed char* __restrict__ A) {
  for (int ax0_ax1_fused_2 = 0; ax0_ax1_fused_2 < 4; ++ax0_ax1_fused_2) {
    *(int4*)(A_global + (((((((((((int)blockIdx.y) * 32768) + (((int)blockIdx.x) * 512)) + (ax0_ax1_fused_2 * 128)) + (((int)threadIdx.x) * 16)) >> 18) * 262144) + (((((((int)blockIdx.x) * 32) + (ax0_ax1_fused_2 * 8)) + ((int)threadIdx.x)) & 1023) * 256)) + ((((int)blockIdx.y) & 7) * 32)) + ((((int)blockIdx.x) >> 5) * 16))) = *(int4*)(A + ((((((int)blockIdx.y) * 32768) + (((int)blockIdx.x) * 512)) + (ax0_ax1_fused_2 * 128)) + (((int)threadIdx.x) * 16)));
  }
}

