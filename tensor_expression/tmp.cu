
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
extern "C" __global__ void __launch_bounds__(256) default_function_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  float C_local[64];
  __shared__ float A_shared[2048];
  __shared__ float B_shared[2048];
  float A_shared_local[8];
  float B_shared_local[8];
  for (int ii_c_init = 0; ii_c_init < 8; ++ii_c_init) {
    for (int jj_c_init = 0; jj_c_init < 8; ++jj_c_init) {
      C_local[((ii_c_init * 8) + jj_c_init)] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 1024; ++k_outer) {
    __syncthreads();
    for (int ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
      A_shared[(((((int)threadIdx.x) * 128) + (((int)threadIdx.y) * 8)) + ax1_inner)] = A[(((((k_outer * 262144) + (((int)threadIdx.x) * 16384)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 8)) + ax1_inner)];
    }
    for (int ax1_inner1 = 0; ax1_inner1 < 8; ++ax1_inner1) {
      B_shared[(((((int)threadIdx.x) * 128) + (((int)threadIdx.y) * 8)) + ax1_inner1)] = B[(((((k_outer * 262144) + (((int)threadIdx.x) * 16384)) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.y) * 8)) + ax1_inner1)];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 16; ++k_inner) {
      for (int ax1 = 0; ax1 < 8; ++ax1) {
        A_shared_local[ax1] = A_shared[(((k_inner * 128) + (((int)threadIdx.y) * 8)) + ax1)];
      }
      for (int ax11 = 0; ax11 < 8; ++ax11) {
        B_shared_local[ax11] = B_shared[(((k_inner * 128) + (((int)threadIdx.x) * 8)) + ax11)];
      }
      for (int ii_c = 0; ii_c < 8; ++ii_c) {
        for (int jj_c = 0; jj_c < 8; ++jj_c) {
          C_local[((ii_c * 8) + jj_c)] = (C_local[((ii_c * 8) + jj_c)] + (A_shared_local[jj_c] * B_shared_local[ii_c]));
        }
      }
    }
  }
  for (int ii_inner_inner = 0; ii_inner_inner < 8; ++ii_inner_inner) {
    for (int jj_inner_inner = 0; jj_inner_inner < 8; ++jj_inner_inner) {
      C[((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.x) * 131072)) + (ii_inner_inner * 16384)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 8)) + jj_inner_inner)] = C_local[((ii_inner_inner * 8) + jj_inner_inner)];
    }
  }
}

