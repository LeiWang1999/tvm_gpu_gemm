
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
  __shared__ float4 A_shared[512];
  __shared__ float4 B_shared[512];
  float A_shared_local[8];
  float B_shared_local[8];
  for (int ii_c_init = 0; ii_c_init < 4; ++ii_c_init) {
    for (int jj_c_init = 0; jj_c_init < 4; ++jj_c_init) {
      C_local[((ii_c_init * 4) + jj_c_init)] = 0.000000e+00f;
      C_local[(((ii_c_init * 4) + jj_c_init) + 32)] = 0.000000e+00f;
      C_local[(((ii_c_init * 4) + jj_c_init) + 16)] = 0.000000e+00f;
      C_local[(((ii_c_init * 4) + jj_c_init) + 48)] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 1024; ++k_outer) {
    __syncthreads();
    for (int ax1_outer = 0; ax1_outer < 2; ++ax1_outer) {
      A_shared[(((((int)threadIdx.y) * 32) + (ax1_outer * 16)) + ((int)threadIdx.x))] = *(float4*)(A + (((((k_outer * 262144) + (((int)threadIdx.y) * 16384)) + (((int)blockIdx.y) * 128)) + (ax1_outer * 64)) + (((int)threadIdx.x) * 4)));
    }
    for (int ax1_outer1 = 0; ax1_outer1 < 2; ++ax1_outer1) {
      B_shared[(((((int)threadIdx.y) * 32) + (ax1_outer1 * 16)) + ((int)threadIdx.x))] = *(float4*)(B + (((((k_outer * 262144) + (((int)threadIdx.y) * 16384)) + (((int)blockIdx.x) * 128)) + (ax1_outer1 * 64)) + (((int)threadIdx.x) * 4)));
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 16; ++k_inner) {
      *(float4*)(A_shared_local + 0) = A_shared[((k_inner * 32) + ((int)threadIdx.y))];
      *(float4*)(A_shared_local + 4) = A_shared[(((k_inner * 32) + ((int)threadIdx.y)) + 16)];
      *(float4*)(B_shared_local + 0) = B_shared[((k_inner * 32) + ((int)threadIdx.x))];
      *(float4*)(B_shared_local + 4) = B_shared[(((k_inner * 32) + ((int)threadIdx.x)) + 16)];
      for (int ii_c = 0; ii_c < 4; ++ii_c) {
        for (int jj_c = 0; jj_c < 4; ++jj_c) {
          C_local[((ii_c * 4) + jj_c)] = (C_local[((ii_c * 4) + jj_c)] + (A_shared_local[jj_c] * B_shared_local[ii_c]));
          C_local[(((ii_c * 4) + jj_c) + 32)] = (C_local[(((ii_c * 4) + jj_c) + 32)] + (A_shared_local[(jj_c + 4)] * B_shared_local[ii_c]));
          C_local[(((ii_c * 4) + jj_c) + 16)] = (C_local[(((ii_c * 4) + jj_c) + 16)] + (A_shared_local[jj_c] * B_shared_local[(ii_c + 4)]));
          C_local[(((ii_c * 4) + jj_c) + 48)] = (C_local[(((ii_c * 4) + jj_c) + 48)] + (A_shared_local[(jj_c + 4)] * B_shared_local[(ii_c + 4)]));
        }
      }
    }
  }
  for (int ii_inner_inner_inner = 0; ii_inner_inner_inner < 4; ++ii_inner_inner_inner) {
    for (int jj_inner_inner_inner = 0; jj_inner_inner_inner < 4; ++jj_inner_inner_inner) {
      C[((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.x) * 65536)) + (ii_inner_inner_inner * 16384)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 4)) + jj_inner_inner_inner)] = C_local[((ii_inner_inner_inner * 4) + jj_inner_inner_inner)];
      C[(((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.x) * 65536)) + (ii_inner_inner_inner * 16384)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 4)) + jj_inner_inner_inner) + 64)] = C_local[(((ii_inner_inner_inner * 4) + jj_inner_inner_inner) + 32)];
      C[(((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.x) * 65536)) + (ii_inner_inner_inner * 16384)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 4)) + jj_inner_inner_inner) + 1048576)] = C_local[(((ii_inner_inner_inner * 4) + jj_inner_inner_inner) + 16)];
      C[(((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.x) * 65536)) + (ii_inner_inner_inner * 16384)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 4)) + jj_inner_inner_inner) + 1048640)] = C_local[(((ii_inner_inner_inner * 4) + jj_inner_inner_inner) + 48)];
    }
  }
}

