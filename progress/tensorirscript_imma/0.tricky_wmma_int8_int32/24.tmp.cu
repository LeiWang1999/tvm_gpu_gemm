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
extern "C" __global__ void __launch_bounds__(128) main_kernel0(signed char* __restrict__ A, signed char* __restrict__ B, int* __restrict__ C) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int> C_wmma_accumulator[16];
  __shared__ signed char A_shared[16384];
  __shared__ signed char B_shared[4096];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, signed char, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, signed char, nvcuda::wmma::col_major> B_shared_wmma_matrix_b[4];
  for (int ii_2_init = 0; ii_2_init < 4; ++ii_2_init) {
    for (int jj_2_init = 0; jj_2_init < 4; ++jj_2_init) {
      nvcuda::wmma::fill_fragment(C_wmma_accumulator[((ii_2_init * 4) + jj_2_init)], 0.000000e+00f);
    }
  }
  for (int kk_0 = 0; kk_0 < 256; ++kk_0) {
    __syncthreads();
    for (int ax0_ax1_ax2_ax3_fused_2 = 0; ax0_ax1_ax2_ax3_fused_2 < 8; ++ax0_ax1_ax2_ax3_fused_2) {
      *(int4*)(A_shared + (((((int)threadIdx.y) * 4096) + (ax0_ax1_ax2_ax3_fused_2 * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(A + ((((((((int)blockIdx.y) * 4194304) + (((int)threadIdx.y) * 1048576)) + ((ax0_ax1_ax2_ax3_fused_2 >> 1) * 262144)) + (kk_0 * 1024)) + ((ax0_ax1_ax2_ax3_fused_2 & 1) * 512)) + (((int)threadIdx.x) * 16)));
    }
    for (int ax0_ax1_ax2_ax3_fused_2_1 = 0; ax0_ax1_ax2_ax3_fused_2_1 < 2; ++ax0_ax1_ax2_ax3_fused_2_1) {
      *(int4*)(B_shared + (((((int)threadIdx.y) * 1024) + (ax0_ax1_ax2_ax3_fused_2_1 * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(B + ((((((((int)blockIdx.z) * 33554432) + (((int)blockIdx.x) * 1048576)) + (((int)threadIdx.y) * 262144)) + (kk_0 * 1024)) + (ax0_ax1_ax2_ax3_fused_2_1 * 512)) + (((int)threadIdx.x) * 16)));
    }
    __syncthreads();
    for (int kk_1 = 0; kk_1 < 4; ++kk_1) {
      for (int ax0 = 0; ax0 < 4; ++ax0) {
        nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[ax0], (&(A_shared[(((((int)threadIdx.y) * 4096) + (ax0 * 1024)) + (kk_1 * 256))])), 16);
      }
      for (int ax0_1 = 0; ax0_1 < 4; ++ax0_1) {
        nvcuda::wmma::load_matrix_sync(B_shared_wmma_matrix_b[ax0_1], (&(B_shared[((ax0_1 * 1024) + (kk_1 * 256))])), 16);
      }
      for (int ii_2 = 0; ii_2 < 4; ++ii_2) {
        for (int jj_2 = 0; jj_2 < 4; ++jj_2) {
          nvcuda::wmma::mma_sync(C_wmma_accumulator[((ii_2 * 4) + jj_2)], A_shared_wmma_matrix_a[ii_2], B_shared_wmma_matrix_b[jj_2], C_wmma_accumulator[((ii_2 * 4) + jj_2)]);
        }
      }
    }
  }
  for (int ax0_2 = 0; ax0_2 < 4; ++ax0_2) {
    for (int ax1 = 0; ax1 < 4; ++ax1) {
      nvcuda::wmma::store_matrix_sync((&(C[((((((((int)blockIdx.y) * 4194304) + (((int)threadIdx.y) * 1048576)) + (ax0_2 * 262144)) + (((int)blockIdx.z) * 32768)) + (((int)blockIdx.x) * 1024)) + (ax1 * 256))])), C_wmma_accumulator[((ax0_2 * 4) + ax1)], 16, nvcuda::wmma::mem_row_major);
    }
  }
}

