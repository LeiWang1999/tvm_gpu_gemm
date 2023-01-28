
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
  for (int i_3_init = 0; i_3_init < 4; ++i_3_init) {
    for (int j_3_init = 0; j_3_init < 4; ++j_3_init) {
      C_local[((i_3_init * 4) + j_3_init)] = 0.000000e+00f;
      C_local[(((i_3_init * 4) + j_3_init) + 32)] = 0.000000e+00f;
      C_local[(((i_3_init * 4) + j_3_init) + 16)] = 0.000000e+00f;
      C_local[(((i_3_init * 4) + j_3_init) + 48)] = 0.000000e+00f;
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
  
  for (int k_0 = 0; k_0 < 1024; ++k_0) {
    __syncthreads();
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 2; ++ax0_ax1_fused_0) {
      AT_shared[(((ax0_ax1_fused_0 * 256) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x))] = *(float4*)(AT + ((((((k_0 * 262144) + (ax0_ax1_fused_0 * 131072)) + ((((int)threadIdx.y) >> 1) * 16384)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 4)));
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 2; ++ax0_ax1_fused_0_1) {
      B_shared[(((ax0_ax1_fused_0_1 * 256) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x))] = *(float4*)(B + ((((((k_0 * 262144) + (ax0_ax1_fused_0_1 * 131072)) + ((((int)threadIdx.y) >> 1) * 16384)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 4)));
    }
    __syncthreads();
    for (int k_1 = 0; k_1 < 16; ++k_1) {
      *(float4*)(AT_shared_local + 0) = AT_shared[((k_1 * 32) + ((int)threadIdx.y))];
      *(float4*)(AT_shared_local + 4) = AT_shared[(((k_1 * 32) + ((int)threadIdx.y)) + 16)];
      *(float4*)(B_shared_local + 0) = B_shared[((k_1 * 32) + ((int)threadIdx.x))];
      *(float4*)(B_shared_local + 4) = B_shared[(((k_1 * 32) + ((int)threadIdx.x)) + 16)];
      for (int i_3 = 0; i_3 < 4; ++i_3) {
        for (int j_3 = 0; j_3 < 4; ++j_3) {
          C_local[((i_3 * 4) + j_3)] = (C_local[((i_3 * 4) + j_3)] + (AT_shared_local[i_3] * B_shared_local[j_3]));
          C_local[(((i_3 * 4) + j_3) + 32)] = (C_local[(((i_3 * 4) + j_3) + 32)] + (AT_shared_local[(i_3 + 4)] * B_shared_local[j_3]));
          C_local[(((i_3 * 4) + j_3) + 16)] = (C_local[(((i_3 * 4) + j_3) + 16)] + (AT_shared_local[i_3] * B_shared_local[(j_3 + 4)]));
          C_local[(((i_3 * 4) + j_3) + 48)] = (C_local[(((i_3 * 4) + j_3) + 48)] + (AT_shared_local[(i_3 + 4)] * B_shared_local[(j_3 + 4)]));
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

