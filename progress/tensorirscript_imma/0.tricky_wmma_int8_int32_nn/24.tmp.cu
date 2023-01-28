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
  __shared__ signed char A_shared[8192];
  __shared__ signed char B_shared[2048];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, signed char, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[16];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, signed char, nvcuda::wmma::row_major> B_shared_wmma_matrix_b[1];
  for (int ii_2_init = 0; ii_2_init < 16; ++ii_2_init) {
    nvcuda::wmma::fill_fragment(C_wmma_accumulator[ii_2_init], 0.000000e+00f);
  }
  for (int kk_0 = 0; kk_0 < 512; ++kk_0) {
    __syncthreads();
    for (int ax0_ax1_ax2_ax3_fused_2 = 0; ax0_ax1_ax2_ax3_fused_2 < 4; ++ax0_ax1_ax2_ax3_fused_2) {
      *(int4*)(A_shared + (((((int)threadIdx.z) * 2048) + (ax0_ax1_ax2_ax3_fused_2 * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(A + (((((((int)blockIdx.y) * 4194304) + (((int)threadIdx.z) * 1048576)) + (ax0_ax1_ax2_ax3_fused_2 * 262144)) + (kk_0 * 512)) + (((int)threadIdx.x) * 16)));
    }
    *(int4*)(B_shared + ((((int)threadIdx.z) * 512) + (((int)threadIdx.x) * 16))) = *(int4*)(B + ((((((kk_0 * 524288) + ((((int)threadIdx.z) >> 1) * 262144)) + (((int)blockIdx.z) * 16384)) + (((int)blockIdx.x) * 1024)) + ((((int)threadIdx.z) & 1) * 512)) + (((int)threadIdx.x) * 16)));
    __syncthreads();
    for (int kk_1 = 0; kk_1 < 2; ++kk_1) {
      for (int ax0 = 0; ax0 < 16; ++ax0) {
        nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[ax0], (&(A_shared[((ax0 * 512) + (kk_1 * 256))])), 16);
      }
      nvcuda::wmma::load_matrix_sync(B_shared_wmma_matrix_b[0], (&(B_shared[((kk_1 * 1024) + (((int)threadIdx.z) * 256))])), 16);
      for (int ii_2 = 0; ii_2 < 16; ++ii_2) {
        nvcuda::wmma::mma_sync(C_wmma_accumulator[ii_2], A_shared_wmma_matrix_a[ii_2], B_shared_wmma_matrix_b[0], C_wmma_accumulator[ii_2]);
      }
    }
  }
  for (int ax0_1 = 0; ax0_1 < 16; ++ax0_1) {
    nvcuda::wmma::store_matrix_sync((&(C[(((((((int)blockIdx.y) * 4194304) + (ax0_1 * 262144)) + (((int)blockIdx.z) * 16384)) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.z) * 256))])), C_wmma_accumulator[ax0_1], 16, nvcuda::wmma::mem_row_major);
  }
}

