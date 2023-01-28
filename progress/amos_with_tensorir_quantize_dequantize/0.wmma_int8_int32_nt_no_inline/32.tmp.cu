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
extern "C" __global__ void __launch_bounds__(128) main_kernel2(signed char* __restrict__ QA, signed char* __restrict__ A_global, signed char* __restrict__ QB, signed char* __restrict__ B_global, int* __restrict__ QC, signed char* __restrict__ C) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int> QC_wmma_accumulator[16];
  __shared__ signed char QA_shared[16384];
  __shared__ signed char QB_shared[4096];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, signed char, nvcuda::wmma::row_major> QA_shared_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, signed char, nvcuda::wmma::col_major> QB_shared_wmma_matrix_b[4];
  for (int ax0 = 0; ax0 < 256; ++ax0) {
    for (int ax1 = 0; ax1 < 16384; ++ax1) {
      QA[(((((int)blockIdx.y) * 4194304) + (ax0 * 16384)) + ax1)] = ((signed char)roundf((((float)A_global[(((((((int)blockIdx.y) * 4194304) + ((ax0 >> 4) * 262144)) + ((ax1 >> 4) * 256)) + ((ax0 & 15) * 16)) + (ax1 & 15))]) * 5.000000e-01f)));
    }
  }
  for (int ax0_1 = 0; ax0_1 < 64; ++ax0_1) {
    for (int ax1_1 = 0; ax1_1 < 16384; ++ax1_1) {
      QB[((((((int)blockIdx.z) * 33554432) + (((int)blockIdx.x) * 1048576)) + (ax0_1 * 16384)) + ax1_1)] = ((signed char)roundf((((float)B_global[((((((((int)blockIdx.z) * 33554432) + (((int)blockIdx.x) * 1048576)) + ((ax0_1 >> 4) * 262144)) + ((ax1_1 >> 4) * 256)) + ((ax0_1 & 15) * 16)) + (ax1_1 & 15))]) * 1.000000e-01f)));
    }
  }
  for (int i_0_2_init = 0; i_0_2_init < 4; ++i_0_2_init) {
    for (int j_0_2_init = 0; j_0_2_init < 4; ++j_0_2_init) {
      nvcuda::wmma::fill_fragment(QC_wmma_accumulator[((i_0_2_init * 4) + j_0_2_init)], 0.000000e+00f);
    }
  }
  for (int k_0_0 = 0; k_0_0 < 256; ++k_0_0) {
    __syncthreads();
    for (int ax0_0_ax1_0_ax0_1_ax1_1_fused_2 = 0; ax0_0_ax1_0_ax0_1_ax1_1_fused_2 < 8; ++ax0_0_ax1_0_ax0_1_ax1_1_fused_2) {
      *(int4*)(QA_shared + (((((int)threadIdx.y) * 4096) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(QA + (((((((((int)blockIdx.y) * 4194304) + (((int)threadIdx.y) * 1048576)) + ((ax0_0_ax1_0_ax0_1_ax1_1_fused_2 >> 1) * 262144)) + ((((int)threadIdx.x) & 15) * 16384)) + (k_0_0 * 64)) + ((ax0_0_ax1_0_ax0_1_ax1_1_fused_2 & 1) * 32)) + ((((int)threadIdx.x) >> 4) * 16)));
    }
    for (int ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1 = 0; ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1 < 2; ++ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1) {
      *(int4*)(QB_shared + (((((int)threadIdx.y) * 1024) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1 * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(QB + (((((((((int)blockIdx.z) * 33554432) + (((int)blockIdx.x) * 1048576)) + (((int)threadIdx.y) * 262144)) + ((((int)threadIdx.x) & 15) * 16384)) + (k_0_0 * 64)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1 * 32)) + ((((int)threadIdx.x) >> 4) * 16)));
    }
    __syncthreads();
    for (int k_0_1 = 0; k_0_1 < 4; ++k_0_1) {
      for (int ax0_0 = 0; ax0_0 < 4; ++ax0_0) {
        nvcuda::wmma::load_matrix_sync(QA_shared_wmma_matrix_a[ax0_0], (&(QA_shared[(((((int)threadIdx.y) * 4096) + (ax0_0 * 1024)) + (k_0_1 * 256))])), 16);
      }
      for (int ax0_0_1 = 0; ax0_0_1 < 4; ++ax0_0_1) {
        nvcuda::wmma::load_matrix_sync(QB_shared_wmma_matrix_b[ax0_0_1], (&(QB_shared[((ax0_0_1 * 1024) + (k_0_1 * 256))])), 16);
      }
      for (int i_0_2 = 0; i_0_2 < 4; ++i_0_2) {
        for (int j_0_2 = 0; j_0_2 < 4; ++j_0_2) {
          nvcuda::wmma::mma_sync(QC_wmma_accumulator[((i_0_2 * 4) + j_0_2)], QA_shared_wmma_matrix_a[i_0_2], QB_shared_wmma_matrix_b[j_0_2], QC_wmma_accumulator[((i_0_2 * 4) + j_0_2)]);
        }
      }
    }
  }
  for (int ax0_0_2 = 0; ax0_0_2 < 4; ++ax0_0_2) {
    for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
      nvcuda::wmma::store_matrix_sync((&(QC[((((((((int)blockIdx.y) * 4194304) + (((int)threadIdx.y) * 1048576)) + (ax0_0_2 * 262144)) + (((int)blockIdx.z) * 2048)) + (((int)blockIdx.x) * 64)) + (ax1_0 * 16))])), QC_wmma_accumulator[((ax0_0_2 * 4) + ax1_0)], 16384, nvcuda::wmma::mem_row_major);
    }
  }
  for (int ax0_2 = 0; ax0_2 < 64; ++ax0_2) {
    for (int ax1_2 = 0; ax1_2 < 64; ++ax1_2) {
      C[((((((((int)blockIdx.y) * 4194304) + (((int)threadIdx.y) * 1048576)) + (ax0_2 * 16384)) + (((int)blockIdx.z) * 2048)) + (((int)blockIdx.x) * 64)) + ax1_2)] = ((signed char)(((float)QC[((((((((int)blockIdx.y) * 4194304) + (((int)threadIdx.y) * 1048576)) + (ax0_2 * 16384)) + (((int)blockIdx.z) * 2048)) + (((int)blockIdx.x) * 64)) + ax1_2)]) * 1.000000e+02f));
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

