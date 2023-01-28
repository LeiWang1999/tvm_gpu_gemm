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
extern "C" __global__ void __launch_bounds__(128) main_kernel2(signed char* __restrict__ APad_global, signed char* __restrict__ BPad_global, int* __restrict__ C) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int> CPad_shared_wmma_accumulator[16];
  __shared__ signed char APad_global_shared[4096];
  __shared__ signed char BPad_global_shared[4096];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, signed char, nvcuda::wmma::row_major> APad_global_shared_wmma_matrix_a[8];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, signed char, nvcuda::wmma::row_major> BPad_global_shared_wmma_matrix_b[2];
  __shared__ int CPad_shared[2048];
  for (int i_0_2_init = 0; i_0_2_init < 8; ++i_0_2_init) {
    for (int j_0_2_init = 0; j_0_2_init < 2; ++j_0_2_init) {
      nvcuda::wmma::fill_fragment(CPad_shared_wmma_accumulator[((i_0_2_init * 2) + j_0_2_init)], 0.000000e+00f);
    }
  }

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
  
  for (int k_0_0 = 0; k_0_0 < 20; ++k_0_0) {
    __syncthreads();
    for (int ax0_0_ax1_0_ax0_1_ax1_1_fused_2 = 0; ax0_0_ax1_0_ax0_1_ax1_1_fused_2 < 2; ++ax0_0_ax1_0_ax0_1_ax1_1_fused_2) {
      *(int4*)(APad_global_shared + (((((int)threadIdx.z) * 1024) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(APad_global + (((((((int)blockIdx.y) * 81920) + (((int)threadIdx.z) * 20480)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 10240)) + (k_0_0 * 512)) + (((int)threadIdx.x) * 16)));
    }
    for (int ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1 = 0; ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1 < 2; ++ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1) {
      *(int4*)(BPad_global_shared + (((((int)threadIdx.z) * 1024) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1 * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(BPad_global + ((((k_0_0 * 4096) + (((int)threadIdx.z) * 1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1 * 512)) + (((int)threadIdx.x) * 16)));
    }
    __syncthreads();
    for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
      for (int ax0_0 = 0; ax0_0 < 8; ++ax0_0) {
        nvcuda::wmma::load_matrix_sync(APad_global_shared_wmma_matrix_a[ax0_0], (&(APad_global_shared[((ax0_0 * 512) + (k_0_1 * 256))])), 16);
      }
      for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0) {
        nvcuda::wmma::load_matrix_sync(BPad_global_shared_wmma_matrix_b[ax1_0], (&(BPad_global_shared[(((k_0_1 * 2048) + (((int)threadIdx.z) * 512)) + (ax1_0 * 256))])), 16);
      }
      for (int i_0_2 = 0; i_0_2 < 8; ++i_0_2) {
        for (int j_0_2 = 0; j_0_2 < 2; ++j_0_2) {
          nvcuda::wmma::mma_sync(CPad_shared_wmma_accumulator[((i_0_2 * 2) + j_0_2)], APad_global_shared_wmma_matrix_a[i_0_2], BPad_global_shared_wmma_matrix_b[j_0_2], CPad_shared_wmma_accumulator[((i_0_2 * 2) + j_0_2)]);
        }
      }
    }
  }
  for (int ax0_0_1 = 0; ax0_0_1 < 8; ++ax0_0_1) {
    __syncthreads();
    for (int ax1_0_1 = 0; ax1_0_1 < 2; ++ax1_0_1) {
      nvcuda::wmma::store_matrix_sync((&(CPad_shared[((((int)threadIdx.z) * 32) + (ax1_0_1 * 16))])), CPad_shared_wmma_accumulator[((ax0_0_1 * 2) + ax1_0_1)], 128, nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0) {
      C[(((((((int)blockIdx.y) * 8192) + (ax0_0_1 * 1024)) + (ax0_ax1_fused_0 * 256)) + (((int)threadIdx.z) * 96)) + ((int)threadIdx.x))] = CPad_shared[(((ax0_ax1_fused_0 * 512) + (((int)threadIdx.z) * 160)) + ((int)threadIdx.x))];
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) main_kernel0(signed char* __restrict__ APad_global, signed char* __restrict__ A) {
  for (int ax0_ax1_fused_6_s = 0; ax0_ax1_fused_6_s < 16; ++ax0_ax1_fused_6_s) {
    if ((((((((int)blockIdx.y) * 33554432) + (((int)blockIdx.x) * 16384)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 16)) + ax0_ax1_fused_6_s) < 2048000) {
      APad_global[((((((((((((int)blockIdx.y) * 33554432) + (((int)blockIdx.x) * 16384)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 16)) + ax0_ax1_fused_6_s) / 10240) * 10240) + ((((((((int)blockIdx.y) * 2097152) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) % 40) * 256)) + ((((((((((int)blockIdx.y) * 33554432) + (((int)blockIdx.x) * 16384)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 16)) + ax0_ax1_fused_6_s) % 10240) / 640) * 16)) + ax0_ax1_fused_6_s)] = ((((((((((int)blockIdx.y) * 33554432) + (((int)blockIdx.x) * 16384)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 16)) + ax0_ax1_fused_6_s) < 2006976) && (((((((((int)blockIdx.y) * 33554432) + (((int)blockIdx.x) * 16384)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 16)) + ax0_ax1_fused_6_s) % 640) < 576)) ? A[((((((((((int)blockIdx.y) * 33554432) + (((int)blockIdx.x) * 16384)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 16)) + ax0_ax1_fused_6_s) / 640) * 576) + ((((((((int)blockIdx.y) * 33554432) + (((int)blockIdx.x) * 16384)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 16)) + ax0_ax1_fused_6_s) % 640))] : (signed char)0);
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) main_kernel1(signed char* __restrict__ BPad_global, signed char* __restrict__ B) {
  for (int ax0_ax1_fused_6_s = 0; ax0_ax1_fused_6_s < 16; ++ax0_ax1_fused_6_s) {
    if ((((((((int)blockIdx.y) * 33554432) + (((int)blockIdx.x) * 16384)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 16)) + ax0_ax1_fused_6_s) < 81920) {
      BPad_global[((((((((int)blockIdx.y) * 33554432) + (((int)blockIdx.x) * 16384)) + ((((int)threadIdx.y) >> 4) * 2048)) + (((int)threadIdx.x) * 256)) + ((((int)threadIdx.y) & 15) * 16)) + ax0_ax1_fused_6_s)] = ((((((((((int)blockIdx.y) * 33554432) + (((int)blockIdx.x) * 16384)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 16)) + ax0_ax1_fused_6_s) < 73664) && (((int)threadIdx.x) < 4)) ? B[(((((((int)blockIdx.y) * 16777216) + (((int)blockIdx.x) * 8192)) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 16)) + ax0_ax1_fused_6_s)] : (signed char)0);
    }
  }
}

