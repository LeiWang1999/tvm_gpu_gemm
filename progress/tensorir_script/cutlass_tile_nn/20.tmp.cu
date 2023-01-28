
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
extern "C" __global__ void __launch_bounds__(128) main_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  float C_local[64];
  __shared__ float A_local_shared[1024];
  float A_local[4];
  __shared__ float4 B_shared[2048];
  float A_local_shared_local[8];
  float B_shared_local[8];
  for (int i_3_init = 0; i_3_init < 4; ++i_3_init) {
    for (int j_3_init = 0; j_3_init < 4; ++j_3_init) {
      C_local[((i_3_init * 4) + j_3_init)] = 0.000000e+00f;
      C_local[(((i_3_init * 4) + j_3_init) + 32)] = 0.000000e+00f;
      C_local[(((i_3_init * 4) + j_3_init) + 16)] = 0.000000e+00f;
      C_local[(((i_3_init * 4) + j_3_init) + 48)] = 0.000000e+00f;
    }
  }
  for (int k_0 = 0; k_0 < 512; ++k_0) {
    __syncthreads();
    for (int ax0_ax1_0_fused_0 = 0; ax0_ax1_0_fused_0 < 2; ++ax0_ax1_0_fused_0) {
      *(float4*)(A_local + 0) = *(float4*)(A + ((((((((int)blockIdx.y) * 524288) + (ax0_ax1_0_fused_0 * 262144)) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (k_0 * 32)) + ((((int)threadIdx.x) & 7) * 4)));
      for (int ax0 = 0; ax0 < 4; ++ax0) {
        A_local_shared[((((((((int)threadIdx.x) & 7) * 128) + (ax0 * 32)) + (ax0_ax1_0_fused_0 * 16)) + (((int)threadIdx.y) * 4)) + (((int)threadIdx.x) >> 3))] = A_local[ax0];
      }
    }
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 16; ++ax0_ax1_fused_0) {
      B_shared[(((ax0_ax1_fused_0 * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] = *(float4*)(B + ((((((k_0 * 524288) + (ax0_ax1_fused_0 * 32768)) + ((((int)threadIdx.y) >> 1) * 16384)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.y) & 1) * 128)) + (((int)threadIdx.x) * 4)));
    }
    __syncthreads();
    for (int k_1 = 0; k_1 < 32; ++k_1) {
      *(float4*)(A_local_shared_local + 0) = *(float4*)(A_local_shared + ((k_1 * 32) + (((int)threadIdx.y) * 4)));
      *(float4*)(A_local_shared_local + 4) = *(float4*)(A_local_shared + (((k_1 * 32) + (((int)threadIdx.y) * 4)) + 16));
      *(float4*)(B_shared_local + 0) = B_shared[((k_1 * 64) + ((int)threadIdx.x))];
      *(float4*)(B_shared_local + 4) = B_shared[(((k_1 * 64) + ((int)threadIdx.x)) + 32)];
      for (int i_3 = 0; i_3 < 4; ++i_3) {
        for (int j_3 = 0; j_3 < 4; ++j_3) {
          C_local[((i_3 * 4) + j_3)] = (C_local[((i_3 * 4) + j_3)] + (A_local_shared_local[i_3] * B_shared_local[j_3]));
          C_local[(((i_3 * 4) + j_3) + 32)] = (C_local[(((i_3 * 4) + j_3) + 32)] + (A_local_shared_local[(i_3 + 4)] * B_shared_local[j_3]));
          C_local[(((i_3 * 4) + j_3) + 16)] = (C_local[(((i_3 * 4) + j_3) + 16)] + (A_local_shared_local[i_3] * B_shared_local[(j_3 + 4)]));
          C_local[(((i_3 * 4) + j_3) + 48)] = (C_local[(((i_3 * 4) + j_3) + 48)] + (A_local_shared_local[(i_3 + 4)] * B_shared_local[(j_3 + 4)]));
        }
      }
    }
  }
  for (int ax0_1 = 0; ax0_1 < 4; ++ax0_1) {
    for (int ax1 = 0; ax1 < 4; ++ax1) {
      C[((((((((int)blockIdx.y) * 524288) + (((int)threadIdx.y) * 65536)) + (ax0_1 * 16384)) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) * 4)) + ax1)] = C_local[((ax0_1 * 4) + ax1)];
      C[(((((((((int)blockIdx.y) * 524288) + (((int)threadIdx.y) * 65536)) + (ax0_1 * 16384)) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) * 4)) + ax1) + 262144)] = C_local[(((ax0_1 * 4) + ax1) + 32)];
      C[(((((((((int)blockIdx.y) * 524288) + (((int)threadIdx.y) * 65536)) + (ax0_1 * 16384)) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) * 4)) + ax1) + 128)] = C_local[(((ax0_1 * 4) + ax1) + 16)];
      C[(((((((((int)blockIdx.y) * 524288) + (((int)threadIdx.y) * 65536)) + (ax0_1 * 16384)) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) * 4)) + ax1) + 262272)] = C_local[(((ax0_1 * 4) + ax1) + 48)];
    }
  }
}

