
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#define __shfl_sync(mask, var, lane, width) \
        __shfl((var), (lane), (width))

#define __shfl_down_sync(mask, var, offset, width) \
        __shfl_down((var), (offset), (width))

#define __shfl_up_sync(mask, var, offset, width) \
        __shfl_up((var), (offset), (width))
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
  int in_thread_C[1];
  signed char A_local[12];
  signed char B_local[12];
  int red_buf0[1];
  for (int i_2 = 0; i_2 < 3; ++i_2) {
    in_thread_C[0] = 0;
    for (int k_0 = 0; k_0 < 2; ++k_0) {
      for (int ax0_ax1_fused_1_s = 0; ax0_ax1_fused_1_s < 12; ++ax0_ax1_fused_1_s) {
        if (((k_0 * 3) + (((((int)threadIdx.x) * 3) + (ax0_ax1_fused_1_s >> 2)) >> 1)) < 4) {
          A_local[ax0_ax1_fused_1_s] = A[(((((((((int)blockIdx.x) * 12288) + (((int)threadIdx.z) * 1536)) + (i_2 * 512)) + (((int)threadIdx.y) * 32)) + (k_0 * 24)) + (((int)threadIdx.x) * 12)) + ax0_ax1_fused_1_s)];
        }
      }
      for (int ax0_ax1_fused_1_s_1 = 0; ax0_ax1_fused_1_s_1 < 12; ++ax0_ax1_fused_1_s_1) {
        if (((k_0 * 3) + (((((int)threadIdx.x) * 3) + (ax0_ax1_fused_1_s_1 >> 2)) >> 1)) < 4) {
          B_local[ax0_ax1_fused_1_s_1] = B[(((k_0 * 24) + (((int)threadIdx.x) * 12)) + ax0_ax1_fused_1_s_1)];
        }
      }
      for (int k_2 = 0; k_2 < 3; ++k_2) {
        for (int k_3 = 0; k_3 < 4; ++k_3) {
          if (((((k_0 * 24) + (((int)threadIdx.x) * 12)) + (k_2 * 4)) + k_3) < 32) {
            in_thread_C[0] = (in_thread_C[0] + (((int)A_local[((k_2 * 4) + k_3)]) * ((int)B_local[((k_2 * 4) + k_3)])));
          }
        }
      }
    }
    uint mask[1];
    int t0[1];
    red_buf0[0] = in_thread_C[0];
    mask[0] = (__activemask() & ((uint)(3 << ((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 2)))));
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
    red_buf0[0] = (red_buf0[0] + t0[0]);
    red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], ((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 2)), 32);
    C[((((((int)blockIdx.x) * 384) + (((int)threadIdx.z) * 48)) + (i_2 * 16)) + ((int)threadIdx.y))] = red_buf0[0];
  }
}

