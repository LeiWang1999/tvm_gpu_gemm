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
extern "C" __global__ void __launch_bounds__(128) default_function_kernel0(signed char* __restrict__ A, signed char* __restrict__ B, int* __restrict__ C) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int> C_wmma_accumulator[16];
  __shared__ signed char A_shared[4096];
  __shared__ signed char B_shared[16384];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, signed char, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, signed char, nvcuda::wmma::col_major> B_shared_wmma_matrix_b[8];
  for (int i_c_init = 0; i_c_init < 2; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 8; ++j_c_init) {
      nvcuda::wmma::fill_fragment(C_wmma_accumulator[((i_c_init * 8) + j_c_init)], 0.000000e+00f);
    }
  }
  for (int k1_outer = 0; k1_outer < 256; ++k1_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_outer) {
      *(int4*)(A_shared + ((((((int)threadIdx.y) * 2048) + (((int)threadIdx.z) * 1024)) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_outer * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(A + ((((((((int)blockIdx.x) * 1048576) + (((int)threadIdx.y) * 524288)) + (((int)threadIdx.z) * 262144)) + (k1_outer * 1024)) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_outer * 512)) + (((int)threadIdx.x) * 16)));
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_outer_1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_outer_1 < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_outer_1) {
      *(int4*)(B_shared + ((((((int)threadIdx.y) * 8192) + (((int)threadIdx.z) * 4096)) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_outer_1 * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(B + (((((((((int)blockIdx.y) * 4194304) + (((int)threadIdx.y) * 2097152)) + (((int)threadIdx.z) * 1048576)) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_outer_1 >> 1) * 262144)) + (k1_outer * 1024)) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_outer_1 & 1) * 512)) + (((int)threadIdx.x) * 16)));
    }
    __syncthreads();
    for (int k1_inner = 0; k1_inner < 4; ++k1_inner) {
      for (int ax0 = 0; ax0 < 2; ++ax0) {
        nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[ax0], (&(A_shared[(((((int)threadIdx.y) * 2048) + (ax0 * 1024)) + (k1_inner * 256))])), 16);
      }
      for (int ax0_1 = 0; ax0_1 < 8; ++ax0_1) {
        nvcuda::wmma::load_matrix_sync(B_shared_wmma_matrix_b[ax0_1], (&(B_shared[(((((int)threadIdx.z) * 8192) + (ax0_1 * 1024)) + (k1_inner * 256))])), 16);
      }
      for (int i_c = 0; i_c < 2; ++i_c) {
        for (int j_c = 0; j_c < 8; ++j_c) {
          nvcuda::wmma::mma_sync(C_wmma_accumulator[((i_c * 8) + j_c)], A_shared_wmma_matrix_a[i_c], B_shared_wmma_matrix_b[j_c], C_wmma_accumulator[((i_c * 8) + j_c)]);
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 2; ++i_inner) {
    for (int j_inner = 0; j_inner < 8; ++j_inner) {
      nvcuda::wmma::store_matrix_sync((&(C[((((((((int)blockIdx.x) * 1048576) + (((int)threadIdx.y) * 524288)) + (i_inner * 262144)) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.z) * 2048)) + (j_inner * 256))])), C_wmma_accumulator[((i_inner * 8) + j_inner)], 16, nvcuda::wmma::mem_row_major);
    }
  }
}

