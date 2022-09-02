
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
extern "C" __global__ void __launch_bounds__(1024) default_function_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  float C_local[1];
  C_local[0] = 0.000000e+00f;
  for (int k = 0; k < 16384; ++k) {
    C_local[0] = (C_local[0] + (A[(((k * 16384) + (((int)blockIdx.y) * 32)) + ((int)threadIdx.y))] * B[(((k * 16384) + (((int)blockIdx.x) * 32)) + ((int)threadIdx.x))]));
  }
  C[((((((int)blockIdx.x) * 524288) + (((int)threadIdx.x) * 16384)) + (((int)blockIdx.y) * 32)) + ((int)threadIdx.y))] = C_local[0];
}

