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
extern "C" __global__ void __launch_bounds__(128) main_kernel2(signed char* __restrict__ A_global, signed char* __restrict__ B_global, int* __restrict__ PB, int* __restrict__ C) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int> QC_wmma_accumulator[16];
  __shared__ signed char A_global_shared[8192];
  __shared__ signed char B_global_shared[2048];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, signed char, nvcuda::wmma::row_major> A_global_shared_wmma_matrix_a[16];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, signed char, nvcuda::wmma::row_major> B_global_shared_wmma_matrix_b[1];
  int C_local[256];
  for (int i_0_2_init = 0; i_0_2_init < 16; ++i_0_2_init) {
    nvcuda::wmma::fill_fragment(QC_wmma_accumulator[i_0_2_init], 0.000000e+00f);
  }
  for (int k_0_0 = 0; k_0_0 < 512; ++k_0_0) {
    __syncthreads();
    for (int ax0_0_ax1_0_ax0_1_ax1_1_fused_2 = 0; ax0_0_ax1_0_ax0_1_ax1_1_fused_2 < 4; ++ax0_0_ax1_0_ax0_1_ax1_1_fused_2) {
      *(int4*)(A_global_shared + (((((int)threadIdx.z) * 2048) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(A_global + (((((((int)blockIdx.y) * 4194304) + (((int)threadIdx.z) * 1048576)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 262144)) + (k_0_0 * 512)) + (((int)threadIdx.x) * 16)));
    }
    *(int4*)(B_global_shared + ((((int)threadIdx.z) * 512) + (((int)threadIdx.x) * 16))) = *(int4*)(B_global + ((((((k_0_0 * 524288) + ((((int)threadIdx.z) >> 1) * 262144)) + (((int)blockIdx.z) * 16384)) + (((int)blockIdx.x) * 1024)) + ((((int)threadIdx.z) & 1) * 512)) + (((int)threadIdx.x) * 16)));
    __syncthreads();
    for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
      for (int ax0_0 = 0; ax0_0 < 16; ++ax0_0) {
        nvcuda::wmma::load_matrix_sync(A_global_shared_wmma_matrix_a[ax0_0], (&(A_global_shared[((ax0_0 * 512) + (k_0_1 * 256))])), 16);
      }
      nvcuda::wmma::load_matrix_sync(B_global_shared_wmma_matrix_b[0], (&(B_global_shared[((k_0_1 * 1024) + (((int)threadIdx.z) * 256))])), 16);
      for (int i_0_2 = 0; i_0_2 < 16; ++i_0_2) {
        nvcuda::wmma::mma_sync(QC_wmma_accumulator[i_0_2], A_global_shared_wmma_matrix_a[i_0_2], B_global_shared_wmma_matrix_b[0], QC_wmma_accumulator[i_0_2]);
      }
    }
  }
  for (int ax0_0_1 = 0; ax0_0_1 < 16; ++ax0_0_1) {
    __syncthreads();
    nvcuda::wmma::store_matrix_sync((&(((int*)A_global_shared)[(((int)threadIdx.z) * 16)])), QC_wmma_accumulator[ax0_0_1], 64, nvcuda::wmma::mem_row_major);
    __syncthreads();
    for (int ax0 = 0; ax0 < 16; ++ax0) {
      for (int ax1 = 0; ax1 < 16; ++ax1) {
        C_local[((ax0 * 16) + ax1)] = ((((int*)A_global_shared)[(((ax0 * 64) + (((int)threadIdx.z) * 16)) + ax1)] + PB[((((((int)blockIdx.z) * 1024) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.z) * 16)) + ax1)]) + 12);
      }
    }
    for (int ax0_1 = 0; ax0_1 < 16; ++ax0_1) {
      for (int ax1_1 = 0; ax1_1 < 16; ++ax1_1) {
        C[(((((((((int)blockIdx.y) * 4194304) + (ax0_0_1 * 262144)) + (ax0_1 * 16384)) + (((int)blockIdx.z) * 1024)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.z) * 16)) + ax1_1)] = C_local[((ax0_1 * 16) + ax1_1)];
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) main_kernel0(signed char* __restrict__ B_global, signed char* __restrict__ B) {
  for (int ax0_ax1_fused_5 = 0; ax0_ax1_fused_5 < 2; ++ax0_ax1_fused_5) {
    *(int4*)(B_global + (((((((int)blockIdx.y) * 67108864) + ((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 32)) + (ax0_ax1_fused_5 * 16)) >> 18) * 262144)) + (((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_5) & 1023) * 256)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.y) >> 6) * 16))) = *(int4*)(B + (((((((int)blockIdx.y) * 67108864) + (((int)blockIdx.x) * 32768)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 32)) + (ax0_ax1_fused_5 * 16)));
  }
}

extern "C" __global__ void __launch_bounds__(1024) main_kernel1(signed char* __restrict__ A_global, signed char* __restrict__ A) {
  for (int ax0_ax1_fused_5 = 0; ax0_ax1_fused_5 < 2; ++ax0_ax1_fused_5) {
    *(int4*)(A_global + (((((((int)blockIdx.y) * 67108864) + ((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 32)) + (ax0_ax1_fused_5 * 16)) >> 18) * 262144)) + (((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_5) & 1023) * 256)) + ((((int)blockIdx.x) & 7) * 32)) + ((((int)threadIdx.y) >> 6) * 16))) = *(int4*)(A + (((((((int)blockIdx.y) * 67108864) + (((int)blockIdx.x) * 32768)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 32)) + (ax0_ax1_fused_5 * 16)));
  }
}

