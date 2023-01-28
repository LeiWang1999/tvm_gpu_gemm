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
extern "C" __global__ void __launch_bounds__(128) main_kernel0(signed char* __restrict__ A, signed char* __restrict__ B, int* __restrict__ C) {
  int C_warp[128];
  __shared__ signed char A_shared[16384];
  __shared__ signed char B_shared[4096];
  signed char A_shared_warp[64];
  signed char B_shared_warp[64];
  for (int ii_2_init = 0; ii_2_init < 4; ++ii_2_init) {
    for (int jj_2_init = 0; jj_2_init < 4; ++jj_2_init) {
      for (int i = 0; i < 8; ++i) {
C_warp[((ii_2_init * 32) + (jj_2_init * 8)) + i] = 0.0;}
;
    }
  }

  const int MAX_BLOCK_N = 16;
  const auto baseBlockIdx = blockIdx.x + gridDim.x *blockIdx.y;
  const auto totalPanel = (gridDim.x * gridDim.y +MAX_BLOCK_N * gridDim.x - 1) / (MAX_BLOCK_N * gridDim.x);
  const auto totalBlock = gridDim.x * gridDim.y;
  const auto panelIdx = baseBlockIdx / (MAX_BLOCK_N *gridDim.x);
  const auto strideLd = panelIdx + 1 < totalPanel ?MAX_BLOCK_N : (totalBlock - panelIdx * (MAX_BLOCK_N *gridDim.x)) / gridDim.x;
  const auto bx = (panelIdx & 1) ? gridDim.x -(baseBlockIdx - panelIdx * MAX_BLOCK_N * gridDim.x) /strideLd - 1 : (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) / strideLd;
  const auto by = (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) % strideLd + panelIdx * MAX_BLOCK_N;
  const auto bz = blockIdx.z;
  const dim3 blockIdx(bx, by, bz);
  
  for (int kk_0 = 0; kk_0 < 256; ++kk_0) {
    __syncthreads();
    for (int ax0_ax1_ax2_ax3_fused_2 = 0; ax0_ax1_ax2_ax3_fused_2 < 8; ++ax0_ax1_ax2_ax3_fused_2) {
      *(int4*)(A_shared + (((((int)threadIdx.y) * 4096) + (ax0_ax1_ax2_ax3_fused_2 * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(A + (((((((((int)blockIdx.x) * 4194304) + (((int)threadIdx.y) * 1048576)) + ((ax0_ax1_ax2_ax3_fused_2 >> 1) * 262144)) + (kk_0 * 1024)) + ((ax0_ax1_ax2_ax3_fused_2 & 1) * 512)) + ((((int)threadIdx.x) & 15) * 32)) + ((((int)threadIdx.x) >> 4) * 16)));
    }
    for (int ax0_ax1_ax2_ax3_fused_2_1 = 0; ax0_ax1_ax2_ax3_fused_2_1 < 2; ++ax0_ax1_ax2_ax3_fused_2_1) {
      *(int4*)(B_shared + (((((int)threadIdx.y) * 1024) + (ax0_ax1_ax2_ax3_fused_2_1 * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(B + (((((((((int)blockIdx.y) * 1048576) + (((int)threadIdx.y) * 262144)) + (kk_0 * 1024)) + (ax0_ax1_ax2_ax3_fused_2_1 * 512)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((int)threadIdx.x) & 15) >> 3) * 16)));
    }
    __syncthreads();
    for (int kk_1 = 0; kk_1 < 2; ++kk_1) {
      for (int ax0 = 0; ax0 < 4; ++ax0) {

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_shared[(((((int)threadIdx.y) * 4096) + (ax0 * 1024)) + (kk_1 * 512))])) + (((int)threadIdx.x) * 16)))
    );
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_shared_warp + (ax0 * 16)))[0]), "=r"(((unsigned *)(A_shared_warp + (ax0 * 16)))[1]), "=r"(((unsigned *)(A_shared_warp + (ax0 * 16)))[2]), "=r"(((unsigned *)(A_shared_warp + (ax0 * 16)))[3])
      : "r"(addr)
    );
  }
      }
      for (int ax0_1 = 0; ax0_1 < 4; ++ax0_1) {

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(B_shared[((ax0_1 * 1024) + (kk_1 * 512))])) + (((int)threadIdx.x) * 16)))
    );
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_shared_warp + (ax0_1 * 16)))[0]), "=r"(((unsigned *)(B_shared_warp + (ax0_1 * 16)))[1]), "=r"(((unsigned *)(B_shared_warp + (ax0_1 * 16)))[2]), "=r"(((unsigned *)(B_shared_warp + (ax0_1 * 16)))[3])
      : "r"(addr)
    );
  }
      }
      for (int ii_2 = 0; ii_2 < 4; ++ii_2) {
        for (int jj_2 = 0; jj_2 < 4; ++jj_2) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=r"(((int *)(C_warp + ((ii_2 * 32) + (jj_2 * 8))))[0]), "=r"(((int *)(C_warp + ((ii_2 * 32) + (jj_2 * 8))))[1]), "=r"(((int *)(C_warp + ((ii_2 * 32) + (jj_2 * 8))))[2]), "=r"(((int *)(C_warp + ((ii_2 * 32) + (jj_2 * 8))))[3])
      : "r"(((unsigned *)(A_shared_warp + (ii_2 * 16)))[0]), "r"(((unsigned *)(A_shared_warp + (ii_2 * 16)))[1]), "r"(((unsigned *)(A_shared_warp + (ii_2 * 16)))[2]), "r"(((unsigned *)(A_shared_warp + (ii_2 * 16)))[3]), "r"(((unsigned *)(B_shared_warp + (jj_2 * 16)))[0]), "r"(((unsigned *)(B_shared_warp + (jj_2 * 16)))[1]), "r"(((int *)(C_warp + ((ii_2 * 32) + (jj_2 * 8))))[0]), "r"(((int *)(C_warp + ((ii_2 * 32) + (jj_2 * 8))))[1]), "r"(((int *)(C_warp + ((ii_2 * 32) + (jj_2 * 8))))[2]), "r"(((int *)(C_warp + ((ii_2 * 32) + (jj_2 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=r"(((int *)(C_warp + (((ii_2 * 32) + (jj_2 * 8)) + 4)))[0]), "=r"(((int *)(C_warp + (((ii_2 * 32) + (jj_2 * 8)) + 4)))[1]), "=r"(((int *)(C_warp + (((ii_2 * 32) + (jj_2 * 8)) + 4)))[2]), "=r"(((int *)(C_warp + (((ii_2 * 32) + (jj_2 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_shared_warp + (ii_2 * 16)))[0]), "r"(((unsigned *)(A_shared_warp + (ii_2 * 16)))[1]), "r"(((unsigned *)(A_shared_warp + (ii_2 * 16)))[2]), "r"(((unsigned *)(A_shared_warp + (ii_2 * 16)))[3]), "r"(((unsigned *)(B_shared_warp + ((jj_2 * 16) + 8)))[0]), "r"(((unsigned *)(B_shared_warp + ((jj_2 * 16) + 8)))[1]), "r"(((int *)(C_warp + (((ii_2 * 32) + (jj_2 * 8)) + 4)))[0]), "r"(((int *)(C_warp + (((ii_2 * 32) + (jj_2 * 8)) + 4)))[1]), "r"(((int *)(C_warp + (((ii_2 * 32) + (jj_2 * 8)) + 4)))[2]), "r"(((int *)(C_warp + (((ii_2 * 32) + (jj_2 * 8)) + 4)))[3]));
  }
        }
      }
    }
  }
  for (int ax0_2 = 0; ax0_2 < 4; ++ax0_2) {
    for (int ax1 = 0; ax1 < 4; ++ax1) {
      for (int local_id = 0; local_id < 8; ++local_id) {
(&(C[(((((((int)blockIdx.x) * 4194304) + (((int)threadIdx.y) * 1048576)) + (ax0_2 * 262144)) + (((int)blockIdx.y) * 1024)) + (ax1 * 256))]))[((((((local_id % 4) / 2) * 8) + (threadIdx.x / 4)) * 16) + ((((local_id / 4) * 8) + ((threadIdx.x % 4) * 2)) + (local_id % 2)))] = C_warp[((ax0_2 * 32) + (ax1 * 8)) + local_id];
}
;
    }
  }
}

