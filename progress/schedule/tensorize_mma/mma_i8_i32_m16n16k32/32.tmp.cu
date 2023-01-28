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
extern "C" __global__ void __launch_bounds__(32) main_kernel0(signed char* __restrict__ A, signed char* __restrict__ B, int* __restrict__ C) {
  __shared__ signed char A_shared[512];
  signed char A_shared_warp[16];
  signed char B_shared_warp[16];
  int C_warp[8];
  *(int4*)(A_shared + (((int)threadIdx.x) * 16)) = *(int4*)(A + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 16)));
  __syncthreads();

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_shared[0])) + (((int)threadIdx.x) * 16)))
    );
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_shared_warp + 0))[0]), "=r"(((unsigned *)(A_shared_warp + 0))[1]), "=r"(((unsigned *)(A_shared_warp + 0))[2]), "=r"(((unsigned *)(A_shared_warp + 0))[3])
      : "r"(addr)
    );
  }
  __syncthreads();
  *(int4*)(A_shared + (((int)threadIdx.x) * 16)) = *(int4*)(B + (((((int)threadIdx.x) & 15) * 32) + ((((int)threadIdx.x) >> 4) * 16)));
  __syncthreads();

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_shared[0])) + (((int)threadIdx.x) * 16)))
    );
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_shared_warp + 0))[0]), "=r"(((unsigned *)(B_shared_warp + 0))[1]), "=r"(((unsigned *)(B_shared_warp + 0))[2]), "=r"(((unsigned *)(B_shared_warp + 0))[3])
      : "r"(addr)
    );
  }
  for (int i = 0; i < 8; ++i) {
C_warp[0 + i] = 0.0;}
;

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=r"(((int *)(C_warp + 0))[0]), "=r"(((int *)(C_warp + 0))[1]), "=r"(((int *)(C_warp + 0))[2]), "=r"(((int *)(C_warp + 0))[3])
      : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + 0))[0]), "r"(((unsigned *)(B_shared_warp + 0))[1]), "r"(((int *)(C_warp + 0))[0]), "r"(((int *)(C_warp + 0))[1]), "r"(((int *)(C_warp + 0))[2]), "r"(((int *)(C_warp + 0))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=r"(((int *)(C_warp + 4))[0]), "=r"(((int *)(C_warp + 4))[1]), "=r"(((int *)(C_warp + 4))[2]), "=r"(((int *)(C_warp + 4))[3])
      : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + 8))[0]), "r"(((unsigned *)(B_shared_warp + 8))[1]), "r"(((int *)(C_warp + 4))[0]), "r"(((int *)(C_warp + 4))[1]), "r"(((int *)(C_warp + 4))[2]), "r"(((int *)(C_warp + 4))[3]));
  }
  for (int local_id = 0; local_id < 8; ++local_id) {
(&(C[0]))[((((((local_id % 4) / 2) * 8) + (threadIdx.x / 4)) * 16) + ((((local_id / 4) * 8) + ((threadIdx.x % 4) * 2)) + (local_id % 2)))] = C_warp[0 + local_id];
}
;
}

