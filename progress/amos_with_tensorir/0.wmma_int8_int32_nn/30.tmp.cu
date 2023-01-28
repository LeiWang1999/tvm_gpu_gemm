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
  __shared__ signed char A_global_shared[8192];
  __shared__ signed char B_global_shared[8192];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, signed char, nvcuda::wmma::row_major> A_global_shared_wmma_matrix_a[8];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, signed char, nvcuda::wmma::row_major> B_global_shared_wmma_matrix_b[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, signed char, nvcuda::wmma::row_major> A_global_shared_wmma_matrix_a_1[8];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, signed char, nvcuda::wmma::row_major> B_global_shared_wmma_matrix_b_1[2];

  const int MAX_BLOCK_N = 1;
  const auto baseBlockIdx = blockIdx.x + gridDim.x *blockIdx.y;
  const auto totalPanel = (gridDim.x * gridDim.y +MAX_BLOCK_N * gridDim.x - 1) / (MAX_BLOCK_N * gridDim.x);
  const auto totalBlock = gridDim.x * gridDim.y;
  const auto panelIdx = baseBlockIdx / (MAX_BLOCK_N *gridDim.x);
  const auto strideLd = panelIdx + 1 < totalPanel ?MAX_BLOCK_N : (totalBlock - panelIdx * (MAX_BLOCK_N *gridDim.x)) / gridDim.x;
  const auto bx = (panelIdx & 1) ? gridDim.x -(baseBlockIdx - panelIdx * MAX_BLOCK_N * gridDim.x) /strideLd - 1 : (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) / strideLd;
  const auto by = (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) % strideLd + panelIdx * MAX_BLOCK_N;
  const auto bz = blockIdx.z;
  const dim3 blockIdx(bx, by, bz);
  
  for (int i_0_2_init = 0; i_0_2_init < 8; ++i_0_2_init) {
    for (int j_0_2_init = 0; j_0_2_init < 2; ++j_0_2_init) {
      nvcuda::wmma::fill_fragment(C_wmma_accumulator[((i_0_2_init * 2) + j_0_2_init)], 0.000000e+00f);
    }
  }
  for (int ax0_0_ax1_0_ax0_1_ax1_1_fused_2 = 0; ax0_0_ax1_0_ax0_1_ax1_1_fused_2 < 2; ++ax0_0_ax1_0_ax0_1_ax1_1_fused_2) {
    *(int4*)(A_global_shared + (((((int)threadIdx.z) * 1024) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(A_global + ((((((int)blockIdx.y) * 2097152) + (((int)threadIdx.z) * 524288)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 262144)) + (((int)threadIdx.x) * 16)));
  }
  for (int ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1 = 0; ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1 < 2; ++ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1) {
    *(int4*)(B_global_shared + (((((int)threadIdx.z) * 1024) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1 * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(B_global + (((((((int)threadIdx.z) >> 1) * 262144) + (((int)blockIdx.x) * 2048)) + (((((((int)threadIdx.z) * 4) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1 * 2)) + (((int)threadIdx.x) >> 4)) & 7) * 256)) + ((((int)threadIdx.x) & 15) * 16)));
  }
  for (int k_0_0 = 0; k_0_0 < 511; ++k_0_0) {
    __syncthreads();
    for (int ax0_0_ax1_0_ax0_1_ax1_1_fused_2_2 = 0; ax0_0_ax1_0_ax0_1_ax1_1_fused_2_2 < 2; ++ax0_0_ax1_0_ax0_1_ax1_1_fused_2_2) {
      *(int4*)(A_global_shared + ((((((k_0_0 + 1) & 1) * 4096) + (((int)threadIdx.z) * 1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_2 * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(A_global + ((((((((int)blockIdx.y) * 2097152) + (((int)threadIdx.z) * 524288)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_2 * 262144)) + (k_0_0 * 512)) + (((int)threadIdx.x) * 16)) + 512));
    }
    for (int ax0_0_ax1_0_ax0_1_ax1_1_fused_2_3 = 0; ax0_0_ax1_0_ax0_1_ax1_1_fused_2_3 < 2; ++ax0_0_ax1_0_ax0_1_ax1_1_fused_2_3) {
      *(int4*)(B_global_shared + ((((((k_0_0 + 1) & 1) * 4096) + (((int)threadIdx.z) * 1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_3 * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(B_global + ((((((k_0_0 * 524288) + ((((int)threadIdx.z) >> 1) * 262144)) + (((int)blockIdx.x) * 2048)) + (((((((int)threadIdx.z) * 4) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_3 * 2)) + (((int)threadIdx.x) >> 4)) & 7) * 256)) + ((((int)threadIdx.x) & 15) * 16)) + 524288));
    }
    __syncthreads();
    for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
      for (int ax0_0 = 0; ax0_0 < 8; ++ax0_0) {
        nvcuda::wmma::load_matrix_sync(A_global_shared_wmma_matrix_a[ax0_0], (&(A_global_shared[((((k_0_0 & 1) * 4096) + (ax0_0 * 512)) + (k_0_1 * 256))])), 16);
      }
      for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0) {
        nvcuda::wmma::load_matrix_sync(B_global_shared_wmma_matrix_b[ax1_0], (&(B_global_shared[(((((k_0_0 & 1) * 4096) + (k_0_1 * 2048)) + (((int)threadIdx.z) * 512)) + (ax1_0 * 256))])), 16);
      }
      for (int i_0_2 = 0; i_0_2 < 8; ++i_0_2) {
        for (int j_0_2 = 0; j_0_2 < 2; ++j_0_2) {
          nvcuda::wmma::mma_sync(C_wmma_accumulator[((i_0_2 * 2) + j_0_2)], A_global_shared_wmma_matrix_a[i_0_2], B_global_shared_wmma_matrix_b[j_0_2], C_wmma_accumulator[((i_0_2 * 2) + j_0_2)]);
        }
      }
    }
  }
  for (int k_0_1_1 = 0; k_0_1_1 < 2; ++k_0_1_1) {
    for (int ax0_0_1 = 0; ax0_0_1 < 8; ++ax0_0_1) {
      nvcuda::wmma::load_matrix_sync(A_global_shared_wmma_matrix_a_1[ax0_0_1], (&(A_global_shared[(((ax0_0_1 * 512) + (k_0_1_1 * 256)) + 4096)])), 16);
    }
    for (int ax1_0_1 = 0; ax1_0_1 < 2; ++ax1_0_1) {
      nvcuda::wmma::load_matrix_sync(B_global_shared_wmma_matrix_b_1[ax1_0_1], (&(B_global_shared[((((k_0_1_1 * 2048) + (((int)threadIdx.z) * 512)) + (ax1_0_1 * 256)) + 4096)])), 16);
    }
    for (int i_0_2_1 = 0; i_0_2_1 < 8; ++i_0_2_1) {
      for (int j_0_2_1 = 0; j_0_2_1 < 2; ++j_0_2_1) {
        nvcuda::wmma::mma_sync(C_wmma_accumulator[((i_0_2_1 * 2) + j_0_2_1)], A_global_shared_wmma_matrix_a_1[i_0_2_1], B_global_shared_wmma_matrix_b_1[j_0_2_1], C_wmma_accumulator[((i_0_2_1 * 2) + j_0_2_1)]);
      }
    }
  }
  for (int ax0_0_2 = 0; ax0_0_2 < 8; ++ax0_0_2) {
    for (int ax1_0_2 = 0; ax1_0_2 < 2; ++ax1_0_2) {
      nvcuda::wmma::store_matrix_sync((&(C[(((((((int)blockIdx.y) * 2097152) + (ax0_0_2 * 262144)) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.z) * 32)) + (ax1_0_2 * 16))])), C_wmma_accumulator[((ax0_0_2 * 2) + ax1_0_2)], 16384, nvcuda::wmma::mem_row_major);
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

