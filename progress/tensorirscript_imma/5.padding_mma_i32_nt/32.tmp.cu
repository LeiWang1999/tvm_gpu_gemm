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
extern "C" __global__ void __launch_bounds__(256) main_kernel0(signed char* __restrict__ A, signed char* __restrict__ B, int* __restrict__ C) {
  int C_warp[128];
  __shared__ signed char A_shared[8192];
  __shared__ signed char B_shared[16384];
  signed char A_shared_warp[16];
  signed char B_shared_warp[16];
  for (int i_1_1_0_init = 0; i_1_1_0_init < 4; ++i_1_1_0_init) {
    for (int j_1_1_0_init = 0; j_1_1_0_init < 4; ++j_1_1_0_init) {
      for (int i = 0; i < 8; ++i) {
C_warp[((i_1_1_0_init * 32) + (j_1_1_0_init * 8)) + i] = 0.0;}
;
    }
  }
  for (int k_0 = 0; k_0 < 256; ++k_0) {
    __syncthreads();
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0) {
      *(int4*)(A_shared + (((ax0_ax1_fused_0 * 2048) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(A + ((((((((int)blockIdx.y) * 2097152) + (ax0_ax1_fused_0 * 524288)) + (((int)threadIdx.y) * 131072)) + ((((int)threadIdx.x) >> 2) * 16384)) + (k_0 * 64)) + ((((int)threadIdx.x) & 3) * 16)));
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 8; ++ax0_ax1_fused_0_1) {
      *(int4*)(B_shared + (((ax0_ax1_fused_0_1 * 2048) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(B + ((((((((int)blockIdx.x) * 4194304) + (ax0_ax1_fused_0_1 * 524288)) + (((int)threadIdx.y) * 131072)) + ((((int)threadIdx.x) >> 2) * 16384)) + (k_0 * 64)) + ((((int)threadIdx.x) & 3) * 16)));
    }
    __syncthreads();
    for (int i_1_1_0 = 0; i_1_1_0 < 4; ++i_1_1_0) {
      for (int j_1_1_0 = 0; j_1_1_0 < 4; ++j_1_1_0) {
        for (int k_1_0 = 0; k_1_0 < 2; ++k_1_0) {

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_shared[(((((int)threadIdx.z) * 4096) + (i_1_1_0 * 1024)) + (k_1_0 * 32))])) + (((((int)threadIdx.x) & 15) * 64) + ((((int)threadIdx.x) >> 4) * 16))))
    );
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_shared_warp + 0))[0]), "=r"(((unsigned *)(A_shared_warp + 0))[1]), "=r"(((unsigned *)(A_shared_warp + 0))[2]), "=r"(((unsigned *)(A_shared_warp + 0))[3])
      : "r"(addr)
    );
  }

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(B_shared[(((((int)threadIdx.y) * 4096) + (j_1_1_0 * 1024)) + (k_1_0 * 32))])) + ((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((int)threadIdx.x) & 15) >> 3) * 16))))
    );
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_shared_warp + 0))[0]), "=r"(((unsigned *)(B_shared_warp + 0))[1]), "=r"(((unsigned *)(B_shared_warp + 0))[2]), "=r"(((unsigned *)(B_shared_warp + 0))[3])
      : "r"(addr)
    );
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=r"(((int *)(C_warp + ((i_1_1_0 * 32) + (j_1_1_0 * 8))))[0]), "=r"(((int *)(C_warp + ((i_1_1_0 * 32) + (j_1_1_0 * 8))))[1]), "=r"(((int *)(C_warp + ((i_1_1_0 * 32) + (j_1_1_0 * 8))))[2]), "=r"(((int *)(C_warp + ((i_1_1_0 * 32) + (j_1_1_0 * 8))))[3])
      : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + 0))[0]), "r"(((unsigned *)(B_shared_warp + 0))[1]), "r"(((int *)(C_warp + ((i_1_1_0 * 32) + (j_1_1_0 * 8))))[0]), "r"(((int *)(C_warp + ((i_1_1_0 * 32) + (j_1_1_0 * 8))))[1]), "r"(((int *)(C_warp + ((i_1_1_0 * 32) + (j_1_1_0 * 8))))[2]), "r"(((int *)(C_warp + ((i_1_1_0 * 32) + (j_1_1_0 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=r"(((int *)(C_warp + (((i_1_1_0 * 32) + (j_1_1_0 * 8)) + 4)))[0]), "=r"(((int *)(C_warp + (((i_1_1_0 * 32) + (j_1_1_0 * 8)) + 4)))[1]), "=r"(((int *)(C_warp + (((i_1_1_0 * 32) + (j_1_1_0 * 8)) + 4)))[2]), "=r"(((int *)(C_warp + (((i_1_1_0 * 32) + (j_1_1_0 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + 8))[0]), "r"(((unsigned *)(B_shared_warp + 8))[1]), "r"(((int *)(C_warp + (((i_1_1_0 * 32) + (j_1_1_0 * 8)) + 4)))[0]), "r"(((int *)(C_warp + (((i_1_1_0 * 32) + (j_1_1_0 * 8)) + 4)))[1]), "r"(((int *)(C_warp + (((i_1_1_0 * 32) + (j_1_1_0 * 8)) + 4)))[2]), "r"(((int *)(C_warp + (((i_1_1_0 * 32) + (j_1_1_0 * 8)) + 4)))[3]));
  }
        }
      }
    }
  }
  for (int ax0_0 = 0; ax0_0 < 4; ++ax0_0) {
    for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
      for (int local_id = 0; local_id < 8; ++local_id) {
(&(C[((((((((int)blockIdx.y) * 2097152) + (((int)threadIdx.z) * 1048576)) + (ax0_0 * 262144)) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.y) * 64)) + (ax1_0 * 16))]))[((((((local_id % 4) / 2) * 8) + (threadIdx.x / 4)) * 16384) + ((((local_id / 4) * 8) + ((threadIdx.x % 4) * 2)) + (local_id % 2)))] = C_warp[((ax0_0 * 32) + (ax1_0 * 8)) + local_id];
}
;
    }
  }
}

