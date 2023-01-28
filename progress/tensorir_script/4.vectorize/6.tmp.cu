
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
extern "C" __global__ void __launch_bounds__(256) main_kernel0(float* __restrict__ AT, float* __restrict__ B, float* __restrict__ C) {
  float C_local[64];
  __shared__ float4 AT_shared[512];
  __shared__ float4 B_shared[512];
  float AT_shared_local[8];
  float B_shared_local[8];
  for (int i_1_1_1_init = 0; i_1_1_1_init < 4; ++i_1_1_1_init) {
    for (int j_1_1_1_init = 0; j_1_1_1_init < 4; ++j_1_1_1_init) {
      C_local[((i_1_1_1_init * 4) + j_1_1_1_init)] = 0.000000e+00f;
      C_local[(((i_1_1_1_init * 4) + j_1_1_1_init) + 32)] = 0.000000e+00f;
      C_local[(((i_1_1_1_init * 4) + j_1_1_1_init) + 16)] = 0.000000e+00f;
      C_local[(((i_1_1_1_init * 4) + j_1_1_1_init) + 48)] = 0.000000e+00f;
    }
  }
  for (int k_0 = 0; k_0 < 1024; ++k_0) {
    __syncthreads();
    for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0) {
      AT_shared[(((((int)threadIdx.y) * 32) + (ax1_0 * 16)) + ((int)threadIdx.x))] = *(float4*)(AT + (((((k_0 * 262144) + (((int)threadIdx.y) * 16384)) + (((int)blockIdx.y) * 128)) + (ax1_0 * 64)) + (((int)threadIdx.x) * 4)));
    }
    for (int ax1_0_1 = 0; ax1_0_1 < 2; ++ax1_0_1) {
      B_shared[(((((int)threadIdx.y) * 32) + (ax1_0_1 * 16)) + ((int)threadIdx.x))] = *(float4*)(B + (((((k_0 * 262144) + (((int)threadIdx.y) * 16384)) + (((int)blockIdx.x) * 128)) + (ax1_0_1 * 64)) + (((int)threadIdx.x) * 4)));
    }
    __syncthreads();
    for (int k_1 = 0; k_1 < 16; ++k_1) {
      *(float4*)(AT_shared_local + 0) = AT_shared[((k_1 * 32) + ((int)threadIdx.y))];
      *(float4*)(AT_shared_local + 4) = AT_shared[(((k_1 * 32) + ((int)threadIdx.y)) + 16)];
      *(float4*)(B_shared_local + 0) = B_shared[((k_1 * 32) + ((int)threadIdx.x))];
      *(float4*)(B_shared_local + 4) = B_shared[(((k_1 * 32) + ((int)threadIdx.x)) + 16)];
      for (int i_1_1_1 = 0; i_1_1_1 < 4; ++i_1_1_1) {
        for (int j_1_1_1 = 0; j_1_1_1 < 4; ++j_1_1_1) {
          C_local[((i_1_1_1 * 4) + j_1_1_1)] = (C_local[((i_1_1_1 * 4) + j_1_1_1)] + (AT_shared_local[i_1_1_1] * B_shared_local[j_1_1_1]));
          C_local[(((i_1_1_1 * 4) + j_1_1_1) + 32)] = (C_local[(((i_1_1_1 * 4) + j_1_1_1) + 32)] + (AT_shared_local[(i_1_1_1 + 4)] * B_shared_local[j_1_1_1]));
          C_local[(((i_1_1_1 * 4) + j_1_1_1) + 16)] = (C_local[(((i_1_1_1 * 4) + j_1_1_1) + 16)] + (AT_shared_local[i_1_1_1] * B_shared_local[(j_1_1_1 + 4)]));
          C_local[(((i_1_1_1 * 4) + j_1_1_1) + 48)] = (C_local[(((i_1_1_1 * 4) + j_1_1_1) + 48)] + (AT_shared_local[(i_1_1_1 + 4)] * B_shared_local[(j_1_1_1 + 4)]));
        }
      }
    }
  }
  for (int ax0 = 0; ax0 < 4; ++ax0) {
    for (int ax1 = 0; ax1 < 4; ++ax1) {
      C[((((((((int)blockIdx.y) * 2097152) + (((int)threadIdx.y) * 65536)) + (ax0 * 16384)) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) * 4)) + ax1)] = C_local[((ax0 * 4) + ax1)];
      C[(((((((((int)blockIdx.y) * 2097152) + (((int)threadIdx.y) * 65536)) + (ax0 * 16384)) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) * 4)) + ax1) + 1048576)] = C_local[(((ax0 * 4) + ax1) + 32)];
      C[(((((((((int)blockIdx.y) * 2097152) + (((int)threadIdx.y) * 65536)) + (ax0 * 16384)) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) * 4)) + ax1) + 64)] = C_local[(((ax0 * 4) + ax1) + 16)];
      C[(((((((((int)blockIdx.y) * 2097152) + (((int)threadIdx.y) * 65536)) + (ax0 * 16384)) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) * 4)) + ax1) + 1048640)] = C_local[(((ax0 * 4) + ax1) + 48)];
    }
  }
}

